#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

limit="${GSM8K_LIMIT:-50}"
concurrency="${GSM8K_CONCURRENCY:-1}"
positive_int GSM8K_CONCURRENCY "$concurrency"
if [[ -n "$limit" ]]; then positive_int GSM8K_LIMIT "$limit"; fi

if [[ "${BENCH_DRY_RUN:-0}" != 1 ]]; then
    require_command lm-eval
    check_server
    prepare_results
fi

args=(run --model local-chat-completions
    --model_args "model=${model},base_url=${base_url}/v1/chat/completions,num_concurrent=${concurrency},max_retries=0,timeout=120,tokenizer=${tokenizer}"
    --tasks gsm8k --apply_chat_template --num_fewshot 0
    --output_path "${results_dir}/gsm8k")
if [[ -n "$limit" ]]; then args+=(--limit "$limit"); fi
run_or_print lm-eval "${args[@]}"
