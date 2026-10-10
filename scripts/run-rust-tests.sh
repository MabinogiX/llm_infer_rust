#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-cpu}"
case "$mode" in
    cpu|cuda) ;;
    *) echo "用法：$0 [cpu|cuda]" >&2; exit 2 ;;
esac
if [[ "$mode" == cpu ]]; then
    export CUDA_VISIBLE_DEVICES=""
fi
venv_dir="${VENV_DIR:-${repo_dir}/.venv}"
venv_dir="$(cd "$venv_dir" && pwd)"
export VIRTUAL_ENV="$venv_dir"
export PATH="${venv_dir}/bin:${PATH}"
export LIBTORCH_USE_PYTORCH=1
torch_lib="$("${venv_dir}/bin/python" -c 'from pathlib import Path; import torch; print(Path(torch.__file__).parent / "lib")')"
if [[ "$(uname -s)" == Darwin ]]; then
    export DYLD_LIBRARY_PATH="${torch_lib}${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="${torch_lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    if [[ "$mode" == cuda && -f "${torch_lib}/libtorch_cuda.so" ]]; then
        export LD_PRELOAD="${torch_lib}/libtorch_cuda.so${LD_PRELOAD:+:$LD_PRELOAD}"
    fi
fi
cargo_args=(test --locked)
case "${SGLANG_BUILD_PROFILE:-debug}" in
    debug) ;;
    release) cargo_args+=(--release) ;;
    *) echo "SGLANG_BUILD_PROFILE 必须是 debug 或 release" >&2; exit 2 ;;
esac
cd "$repo_dir"
if [[ "$mode" == cuda ]]; then
    "${venv_dir}/bin/python" -c 'import torch; assert torch.cuda.is_available(), "CUDA tests require an available GPU"'
    # A usable GPU alone does not prove native kernels were compiled. Check one
    # representative test for each native cfg before allowing a partial green run.
    catalog="$(cargo "${cargo_args[@]}" --lib -- --list)"
    for test_name in \
        fused_cuda_ops_match_eager_bfloat16 \
        cuda_kv_store_preserves_packed_strides_padding_and_unwritten_slots \
        flashinfer_sampling_preserves_support_distribution_and_rng \
        flashinfer_graph_replays_dynamic_pages_lengths_and_padded_batches; do
        if [[ "$catalog" != *"${test_name}: test"* ]]; then
            echo "CUDA 测试未编入：${test_name}；请检查 CUDA toolkit 和 FlashInfer 环境。" >&2
            exit 1
        fi
    done
    exec cargo "${cargo_args[@]}" -- --ignored --skip benchmark_sampling_cuda --test-threads=1
fi
# Global RNG seeds and graph capture require serial execution in the CUDA suite;
# keep the same deterministic execution order for the default suite.
exec cargo "${cargo_args[@]}" -- --test-threads=1
