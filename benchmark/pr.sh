#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
export BENCH_RESULTS_DIR="$results_dir"
protocol_python="${PROTOCOL_PYTHON:-${repo_dir}/.venv/bin/python}"
if [[ ! -x "$protocol_python" ]]; then protocol_python=python3; fi

if [[ "${BENCH_DRY_RUN:-0}" != 1 ]]; then
    require_command "$protocol_python"
    check_server
fi

echo "OpenAI SDK protocol tests"
run_or_print env SGLANG_E2E_BASE_URL="${base_url}/v1" SGLANG_E2E_MODEL="$model" \
    "$protocol_python" -m unittest discover -s "${repo_dir}/tests" -p test_api_e2e.py -v

echo "GSM8K small correctness check"
"${benchmark_dir}/lm_eval/gsm8k.sh"

echo "SGLang serving smoke"
"${benchmark_dir}/sglang_bench/smoke.sh"
