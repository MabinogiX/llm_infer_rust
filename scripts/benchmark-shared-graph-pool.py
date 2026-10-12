#!/usr/bin/env python3
"""Same-GPU HTTP benchmark of Rust pool stages and upstream SGLang.

Run on sglang-test with the Rust .venv Python. Each service is isolated and
stopped before the next one starts. No source/dependency changes in SGLang.
Raw request timings, token usage, output hashes, allocator logs and NVML
samples are saved. NVML sampled startup peak is a lower bound, not exact.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import re
from pathlib import Path
import signal
import statistics
import subprocess
import tempfile
import threading
import time
import urllib.request

BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 320, 384, 448, 512,
           640, 768, 896, 1024, 1280, 1536, 1792, 2048]
CASES = [
    ('cold64', 64, 1, 1, 12),
    ('cold512', 512, 1, 1, 12),
    ('cold2048', 2048, 1, 1, 12),
    ('fallback4096', 4096, 1, 1, 12),
    ('generate64c1', 64, 128, 1, 12),
    ('generate512c4', 512, 64, 4, 32),
    ('generate1024c4', 1024, 32, 4, 32),
    ('prefix1024', 1040, 32, 1, 12),
]

# Import hook patches only the capture entry point after its normal import,
# avoiding premature CUDA initialization in the launcher's parent process.
SG_HOOK = '''import importlib.abc, importlib.machinery, sys
class CaptureLoader(importlib.abc.Loader):
    def __init__(self, original): self.original = original
    def create_module(self, spec):
        return self.original.create_module(spec) if hasattr(self.original, 'create_module') else None
    def exec_module(self, module):
        self.original.exec_module(module)
        original = module.ModelRunner.init_cuda_graphs
        def capture(self, *args, **kwargs):
            import json, time, torch
            torch.cuda.synchronize()
            before = dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved())
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            result = original(self, *args, **kwargs)
            torch.cuda.synchronize()
            stats = dict(before_allocated_bytes=before['allocated'], before_reserved_bytes=before['reserved'],
                allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                capture_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                capture_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                capture_ms=(time.monotonic()-start)*1000)
            print('SHARED_POOL_BENCH_CAPTURE ' + json.dumps(stats), flush=True)
            return result
        module.ModelRunner.init_cuda_graphs = capture
class CaptureFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'sglang.srt.model_executor.model_runner': return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None: spec.loader = CaptureLoader(spec.loader)
        return spec
sys.meta_path.insert(0, CaptureFinder())
'''


def request(base, row):
    payload = dict(model='default', prompt=row['prompt'], temperature=0,
                   max_tokens=row['output'], ignore_eos=True, stream=True,
                   stream_options={'include_usage': True})
    started = time.perf_counter()
    arrivals, texts, usage = [], [], None
    req = urllib.request.Request(base + '/v1/completions',
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=180) as response:
        for line in response:
            if not line.startswith(b'data: ') or line.strip() == b'data: [DONE]':
                continue
            event = json.loads(line[6:])
            if event.get('error'):
                raise RuntimeError(event['error'])
            if event.get('usage'):
                usage = event['usage']
            choices = event.get('choices') or []
            if choices and choices[0].get('text'):
                arrivals.append(time.perf_counter())
                texts.append(choices[0]['text'])
    ended = time.perf_counter()
    if not arrivals or not usage:
        raise RuntimeError(f"missing content/usage: {row['key']}")
    if (usage['prompt_tokens'], usage['completion_tokens']) != (row['input'], row['output']):
        raise RuntimeError(f"token counts differ: {row['key']} {usage}")
    ttft = (arrivals[0] - started) * 1000
    e2e = (ended - started) * 1000
    return dict(key=row['key'], input=row['input'], output=row['output'], usage=usage,
        ttft_ms=ttft, e2e_ms=e2e,
        tpot_ms=(e2e - ttft)/(row['output'] - 1) if row['output'] > 1 else None,
        content_chunks=len(arrivals),
        output_sha256=hashlib.sha256(''.join(texts).encode()).hexdigest())


def group(base, rows, concurrency):
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        samples = list(executor.map(lambda row: request(base, row), rows))
    elapsed = time.perf_counter() - started
    def median(field):
        values = [row[field] for row in samples if row[field] is not None]
        return statistics.median(values) if values else None
    return dict(samples=samples, wall_s=elapsed,
        output_tokens_per_s=sum(row['output'] for row in samples)/elapsed,
        ttft_ms=median('ttft_ms'), tpot_ms=median('tpot_ms'), e2e_ms=median('e2e_ms'))


def make_workload(args):
    # Use the same local model tokenizer for both servers. No downloads.
    script = '''import hashlib,json,sys
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained(sys.argv[1],local_files_only=True)
cases=json.loads(sys.argv[2]); result=[]
def prompt(key, length):
 marker=hashlib.sha256(key.encode()).hexdigest()[:32] + ' '
 count=length-len(t.encode(marker,add_special_tokens=False))
 text=marker+' sky'*count
 while len(t.encode(text,add_special_tokens=False)) > length: text=text[:-4]
 while len(t.encode(text,add_special_tokens=False)) < length: text+=' sky'
 assert len(t.encode(text,add_special_tokens=False)) == length
 return text
for trial in range(int(sys.argv[3])):
 trial_rows={}
 for name,isl,osl,c,n in cases:
  rows=[]
  prefix=prompt(f'{trial}-prefix-seed',1024) if name=='prefix1024' else None
  for index in range(n+max(2,c)):
   key=f'{trial}-{name}-{index}'
   text=prompt(key,isl) if prefix is None else prefix+' '+hashlib.sha256(key.encode()).hexdigest()[:20]
   rows.append(dict(key=key,prompt=text,input=len(t.encode(text,add_special_tokens=False)),output=osl))
  trial_rows[name]=dict(rows=rows,prefix=prefix)
 result.append(trial_rows)
print(json.dumps(result))
'''
    result = subprocess.check_output([str(args.sglang_repo / '.venv/bin/python'), '-c',
        script, str(args.model), json.dumps(CASES), str(args.rounds)], text=True)
    return json.loads(result)


def run(args, mode, trial, workload):
    name = f'{trial}-{mode}'
    path = args.output_dir / f'{name}.json'
    if args.resume and path.exists():
        return json.loads(path.read_text())
    import pynvml
    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
    upstream = mode.startswith('sglang')
    root = args.sglang_repo if upstream else args.repo
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
        SGLANG_PROFILE_STEPS='0', NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost',
        VIRTUAL_ENV=str(root / '.venv'))
    env['PATH'] = str(root / '.venv/bin') + ':' + env['PATH']
    env['LD_LIBRARY_PATH'] = '/usr/local/cuda/compat:' + str(root / '.venv/lib/python3.12/site-packages/torch/lib')
    if upstream:
        command = [str(root / '.venv/bin/python'), '-m', 'sglang.launch_server',
            '--model-path', str(args.model), '--host', '127.0.0.1', '--port', str(args.port),
            '--served-model-name', 'default', '--dtype', 'bfloat16', '--attention-backend', 'flashinfer',
            '--max-running-requests', '4', '--context-length', '40960', '--page-size', '16',
            '--mem-fraction-static', '0.4', '--max-total-tokens', '163904',
            '--chunked-prefill-size', '-1', '--cuda-graph-backend-decode', 'full',
            '--cuda-graph-backend-prefill', 'breakable', '--cuda-graph-bs-decode', '1', '2', '4',
            '--cuda-graph-bs-prefill'] + list(map(str, BUCKETS))
        if mode == 'sglang-serial':
            command.append('--disable-overlap-schedule')
    else:
        binary = (args.stage2_binary if mode == 'rust-stage2' else args.baseline_binary if mode == 'rust-before-threads' else args.repo / 'target/release/sglang-rust')
        enabled = mode != 'eager'
        command = [str(binary), '--model-path', str(args.model), '--port', str(args.port),
            '--device', 'cuda', '--dtype', 'bfloat16', '--attention-backend', 'flashinfer',
            '--max-running-req', '4', '--max-seq-len', '40960', '--page-size', '16',
            '--memory-ratio', '0.4', '--cuda-graph-bs', '4' if enabled else '0',
            '--prefill-cuda-graph-max-tokens', '2048' if enabled else '0']
        if mode == 'rust-serial':
            command.append('--disable-cpu-gpu-overlap')
        if mode == 'rust-no-metadata':
            command.append('--disable-async-metadata')
        env['LD_PRELOAD'] = str(root / '.venv/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so')
    samples, done = [], threading.Event()
    def sample():
        while not done.is_set():
            try:
                samples.append(dict(t=time.monotonic(), gpu_bytes=pynvml.nvmlDeviceGetMemoryInfo(gpu).used))
            except pynvml.NVMLError:
                pass
            done.wait(.02)
    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    base = f'http://127.0.0.1:{args.port}'
    started = time.monotonic()
    proc = None
    try:
        with tempfile.TemporaryDirectory(prefix='sg-capture-hook-', dir=args.output_dir) as hook_dir:
            if upstream:
                Path(hook_dir, 'sitecustomize.py').write_text(SG_HOOK)
                env['PYTHONPATH'] = hook_dir + os.pathsep + env.get('PYTHONPATH', '')
            with (args.output_dir / f'{name}-server.log').open('w') as log:
                proc = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
                deadline = started + 420
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError(f'{name} exited {proc.returncode}; see server log')
                    try:
                        with urllib.request.urlopen(base + '/health', timeout=1) as response:
                            if response.status == 200:
                                break
                    except Exception:
                        pass
                    if time.monotonic() > deadline:
                        raise RuntimeError(f'{name} readiness timeout')
                    time.sleep(.1)
                ready = time.monotonic()
                result = dict(mode=mode, trial=trial, command=command, startup_s=ready-started,
                    startup_sampled_gpu_peak_bytes=max(row['gpu_bytes'] for row in samples),
                    ready_gpu_bytes=pynvml.nvmlDeviceGetMemoryInfo(gpu).used, cases={})
                for case, _, _, concurrency, count in CASES:
                    data = workload[case]
                    if data['prefix']:
                        seed = dict(key=f'{trial}-prefix-seed', prompt=data['prefix'], input=1024, output=1)
                        request(base, seed)
                    warm = max(2, concurrency)
                    group(base, data['rows'][:warm], concurrency)
                    result['cases'][case] = group(base, data['rows'][warm:], concurrency)
                    print(name, case, json.dumps({key: value for key, value in result['cases'][case].items()
                          if key != 'samples'}), flush=True)
                result['steady_gpu_bytes'] = pynvml.nvmlDeviceGetMemoryInfo(gpu).used
                contents = (args.output_dir / f'{name}-server.log').read_text()
                result['capture_log'] = [line for line in contents.splitlines()
                    if 'shared CUDA graph catalog memory' in line or 'SHARED_POOL_BENCH_CAPTURE ' in line]
                if mode in ('rust', 'rust-serial', 'rust-no-metadata', 'sglang', 'sglang-serial') and not result['capture_log']:
                    raise RuntimeError('capture instrumentation missing')
                path.write_text(json.dumps(result, indent=2))
                return result
    finally:
        if proc and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        done.set()
        monitor.join(timeout=2)
        (args.output_dir / f'{name}-gpu.json').write_text(json.dumps(samples))
        pynvml.nvmlShutdown()


def binary_hashes(args):
    """Only binaries actually executed by the selected modes are dependencies."""
    binaries = {}
    if set(args.modes) & {'rust', 'rust-serial', 'rust-no-metadata', 'eager'}:
        binaries['rust_binary'] = args.repo / 'target/release/sglang-rust'
    if 'rust-stage2' in args.modes:
        binaries['stage2_binary'] = args.stage2_binary
    if 'rust-before-threads' in args.modes:
        binaries['before_threads_binary'] = args.baseline_binary
    return {label: hashlib.sha256(binary.read_bytes()).hexdigest()
            for label, binary in binaries.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path('/sjtu/yaosikai/llm_infer_rust'))
    parser.add_argument('--sglang-repo', type=Path, default=Path('/sjtu/yaosikai/sglang'))
    parser.add_argument('--model', type=Path, default=Path('/sjtu/yaosikai/Qwen3-0.6B'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--stage2-binary', type=Path, default=Path('/tmp/shared-pool-stage2-bin'))
    parser.add_argument('--baseline-binary', type=Path, default=Path('/tmp/shared-pool-stage4-before-latency-bin'))
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--port', type=int, default=18114)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--modes', nargs='+', choices=['rust-stage2', 'rust-before-threads', 'rust', 'rust-serial', 'rust-no-metadata', 'eager', 'sglang', 'sglang-serial'],
        default=['rust-stage2', 'rust-before-threads', 'rust', 'rust-serial', 'rust-no-metadata', 'eager', 'sglang', 'sglang-serial'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error('--rounds must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configuration = dict(model=str(args.model), gpu=args.gpu, rounds=args.rounds,
        cases=CASES, buckets=BUCKETS, modes=args.modes)
    metadata_path = args.output_dir / 'metadata.json'
    recorded = json.loads(metadata_path.read_text()) if args.resume and metadata_path.exists() else None
    if recorded:
        for key, value in configuration.items():
            if recorded[key] != json.loads(json.dumps(value)):
                parser.error(f'--resume configuration differs: {key}')
    incomplete = any(not (args.output_dir / f'{trial}-{mode}.json').exists()
        for trial in range(args.rounds) for mode in args.modes)
    if recorded and not incomplete:
        # Completed datasets need neither historical binaries nor live GPU/model
        # dependencies. Keep their original provenance and exact input fixture.
        metadata = recorded
        workload = json.loads((args.output_dir / 'workload.json').read_text())
    else:
        metadata = configuration.copy()
        for label, root in [('rust', args.repo), ('sglang', args.sglang_repo)]:
            metadata[label] = dict(git_head=subprocess.check_output(
                ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
                versions=json.loads(subprocess.check_output([str(root / '.venv/bin/python'), '-c',
                    "import json,importlib.metadata as m; print(json.dumps({p:m.version(p) for p in ['torch','flashinfer-python']}))"], text=True)))
        metadata.update(binary_hashes(args))
        metadata['nvml_v1_includes_driver_reserved_mib'] = int(subprocess.check_output(
            ['nvidia-smi', f'--id={args.gpu}', '--query-gpu=memory.reserved',
             '--format=csv,noheader,nounits'], text=True).strip())
        metadata['gpu_info'] = subprocess.check_output(['nvidia-smi', f'--id={args.gpu}',
            '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], text=True).strip()
        if recorded:
            if any(recorded.get(key) != value for key, value in metadata.items()
                    if key.endswith('_binary')):
                parser.error('--resume cannot mix different binaries in unfinished results')
            workload = json.loads((args.output_dir / 'workload.json').read_text())
        else:
            metadata_path.write_text(json.dumps(metadata, indent=2))
            workload = make_workload(args)
            (args.output_dir / 'workload.json').write_text(json.dumps(workload))
    driver_reserved_mib = metadata['nvml_v1_includes_driver_reserved_mib']
    results = []
    for trial in range(args.rounds):
        # Rotate order so implementation and trial order are not identical.
        modes = args.modes[trial % len(args.modes):] + args.modes[:trial % len(args.modes)]
        for mode in modes:
            results.append(run(args, mode, trial, workload[trial]))
    (args.output_dir / 'all.json').write_text(json.dumps(results, indent=2))
    summary = {}
    for mode in args.modes:
        runs = [row for row in results if row['mode'] == mode]
        summary[mode] = dict(cases={case: {metric: statistics.median(
            run['cases'][case][metric] for run in runs) for metric in
            ('ttft_ms', 'e2e_ms', 'output_tokens_per_s')} for case, *_ in CASES},
            ready_nvml_used_mib=statistics.median(row['ready_gpu_bytes']/2**20 for row in runs),
            steady_nvml_used_mib=statistics.median(row['steady_gpu_bytes']/2**20 for row in runs),
            # Approximate to within 1 MiB because reserved is rounded by CLI.
            ready_resident_mib_approx=statistics.median(int(row['ready_gpu_bytes']/2**20)-driver_reserved_mib for row in runs),
            steady_resident_mib_approx=statistics.median(int(row['steady_gpu_bytes']/2**20)-driver_reserved_mib for row in runs),
            startup_s=statistics.median(row['startup_s'] for row in runs))
        for case, *_ in CASES:
            values = [row['cases'][case]['tpot_ms'] for row in runs if row['cases'][case]['tpot_ms'] is not None]
            summary[mode]['cases'][case]['tpot_ms'] = statistics.median(values) if values else None
    capture = {}
    for row in results:
        metrics = []
        for line in row['capture_log']:
            if 'SHARED_POOL_BENCH_CAPTURE ' in line:
                values = json.loads(line.split('SHARED_POOL_BENCH_CAPTURE ', 1)[1])
            else:
                clean = re.sub(r'\x1b\[[0-9;]*m', '', line)
                values = {key: int(value) for key, value in re.findall(r'(\w+)=(\d+)', clean)}
            for field in ('allocated', 'reserved'):
                peak = values.get(f'capture_peak_{field}_bytes')
                if peak is not None:
                    assert peak >= values[f'{field}_bytes'], 'invalid capture peak counter'
                    before = values.get(f'before_{field}_bytes')
                    if before is not None:
                        values[f'capture_peak_{field}_delta_bytes'] = peak - before
            metrics.append(values)
        capture[f"{row['trial']}-{row['mode']}"] = metrics
    (args.output_dir / 'capture.json').write_text(json.dumps(capture, indent=2))
    comparisons = []
    for trial in range(args.rounds):
        by_mode = {row['mode']: row for row in results if row['trial'] == trial}
        baselines = [name for name in ('rust-no-metadata', 'rust-serial') if name in by_mode]
        if not baselines:
            baselines = ['rust-before-threads' if 'rust-before-threads' in by_mode else 'rust-stage2']
        if 'rust' not in by_mode:
            continue
        for baseline in baselines:
            if baseline not in by_mode:
                continue
            for case, _, _, concurrency, _ in CASES:
                old = by_mode[baseline]['cases'][case]['samples']
                new = by_mode['rust']['cases'][case]['samples']
                assert [x['key'] for x in old] == [x['key'] for x in new]
                comparisons.append(dict(trial=trial, baseline=baseline, case=case, concurrency=concurrency,
                    total=len(old), equal=sum(a['output_sha256'] == b['output_sha256'] for a, b in zip(old, new)),
                    differing_keys=[a['key'] for a, b in zip(old, new) if a['output_sha256'] != b['output_sha256']]))
    (args.output_dir / 'output-comparisons.json').write_text(json.dumps(comparisons, indent=2))
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__':
    main()
