#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

input="${SMOKE_ISL:-64}"
output="${SMOKE_OSL:-32}"
requests="${SMOKE_REQUESTS:-100}"
positive_int SMOKE_ISL "$input"
positive_int SMOKE_OSL "$output"
positive_int SMOKE_REQUESTS "$requests"
check_context "$input" "$output"

python="${SGLANG_BENCH_PYTHON:-python3}"
if [[ "${BENCH_DRY_RUN:-0}" != 1 ]]; then
    require_command "$python"
    "$python" -c 'import sglang.bench_serving' >/dev/null || die "无法导入 sglang.bench_serving"
    check_server
    prepare_results
fi

for concurrency in 1 8 32; do
    echo "SGLang smoke: ${requests} requests, concurrency=${concurrency}, ISL/OSL=${input}/${output}"
    run_or_print "$python" -m sglang.bench_serving \
        --backend vllm-chat --base-url "$base_url" \
        --model "$model" --tokenizer "$tokenizer" \
        --dataset-name random --random-input-len "$input" \
        --random-output-len "$output" --random-range-ratio 1 \
        --num-prompts "$requests" --request-rate inf \
        --max-concurrency "$concurrency" --seed 42 \
        --output-file "${results_dir}/sglang-smoke.jsonl" \
        --tag "c${concurrency}-i${input}-o${output}"
done
