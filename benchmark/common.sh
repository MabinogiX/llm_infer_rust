#!/usr/bin/env bash

benchmark_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${benchmark_dir}/.." && pwd)"
base_url="${BENCH_BASE_URL:-http://127.0.0.1:8000}"
base_url="${base_url%/}"
model="${BENCH_MODEL:-default}"
tokenizer="${BENCH_TOKENIZER:-${MINISGL_MODEL_PATH:-${repo_dir}/../mini-sglang/Qwen/Qwen3-0.6B}}"
results_dir="${BENCH_RESULTS_DIR:-${benchmark_dir}/results/$(date +%Y%m%d-%H%M%S)}"

die() { echo "benchmark: $*" >&2; exit 1; }

positive_int() {
    [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 必须是正整数：$2"
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "找不到 $1；请先按 benchmark/README.md 安装依赖"
}

check_server() {
    require_command curl
    curl --fail --silent --show-error --max-time 5 "${base_url}/health" >/dev/null ||
        die "服务未就绪：${base_url}/health"
}

prepare_results() { mkdir -p "${results_dir}"; }

check_context() {
    local input="$1" output="$2" max_seq="${BENCH_MAX_SEQ_LEN:-512}"
    positive_int BENCH_MAX_SEQ_LEN "$max_seq"
    # Chat template and tokenizer special tokens need a small margin.
    (( input + output + 16 <= max_seq )) ||
        die "ISL ${input} + OSL ${output} + 16 超过 BENCH_MAX_SEQ_LEN=${max_seq}；请提高服务 --max-seq-len 并同步设置该变量"
}

run_or_print() {
    if [[ "${BENCH_DRY_RUN:-0}" == 1 ]]; then
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}
