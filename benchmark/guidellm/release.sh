#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

requests="${RELEASE_REQUESTS:-1000}"
timeout="${RELEASE_TIMEOUT:-3600}"
positive_int RELEASE_REQUESTS "$requests"
positive_int RELEASE_TIMEOUT "$timeout"
shapes=("128:128" "1024:128" "4096:256" "16384:256" "65536:256")
concurrencies=(1 8 32 128)

# Check the complete matrix before any long run starts.
for shape in "${shapes[@]}"; do
    IFS=: read -r input output <<< "$shape"
    check_context "$input" "$output"
done

if [[ "${BENCH_DRY_RUN:-0}" != 1 ]]; then
    require_command guidellm
    check_server
    prepare_results
fi

for shape in "${shapes[@]}"; do
    IFS=: read -r input output <<< "$shape"
    for concurrency in "${concurrencies[@]}"; do
        echo "GuideLLM release: ${requests} requests, concurrency=${concurrency}, ISL/OSL=${input}/${output}"
        run_or_print guidellm run \
            --backend "kind=openai_http,target=${base_url},model=${model},request_format=/v1/chat/completions,timeout=${timeout}" \
            --tokenizer "kind=huggingface_auto,model=${tokenizer}" \
            --data "kind=synthetic_text,prompt_tokens=${input},output_tokens=${output}" \
            --profile "kind=concurrent,streams=${concurrency}" \
            --constraint "kind=max_requests,count=${requests}" \
            --output "kind=json,path=${results_dir}/guidellm-i${input}-o${output}-c${concurrency}.json"
    done
done
