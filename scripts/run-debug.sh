#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${VENV_DIR:-${repo_dir}/.venv}"

if [[ ! -x "${venv_dir}/bin/python" ]]; then
    echo "找不到 Python 虚拟环境：${venv_dir}（可通过 VENV_DIR 指定）" >&2
    exit 1
fi
venv_dir="$(cd "${venv_dir}" && pwd)"

torch_lib=""
for candidate in "${venv_dir}"/lib/python*/site-packages/torch/lib; do
    if [[ -d "${candidate}" ]]; then
        torch_lib="${candidate}"
        break
    fi
done
if [[ -z "${torch_lib}" ]]; then
    echo "在 ${venv_dir} 中找不到 PyTorch/libtorch，请先安装与 tch 兼容的 PyTorch。" >&2
    exit 1
fi

export VIRTUAL_ENV="${venv_dir}"
export PATH="${venv_dir}/bin:${PATH}"
export LIBTORCH_USE_PYTORCH=1
if [[ "$(uname -s)" == "Darwin" ]]; then
    export DYLD_LIBRARY_PATH="${torch_lib}${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
else
    export LD_LIBRARY_PATH="${torch_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# `torch-sys` can be linked with --as-needed on Linux, which may omit
# libtorch_cuda.so from the final Rust executable even when this PyTorch
# installation provides CUDA. Preload it only after PyTorch itself confirms
# CUDA is usable, preserving the CPU fallback for CPU-only or unavailable
# CUDA installations.
runtime_env=()
if [[ "$(uname -s)" == "Linux" && -f "${torch_lib}/libtorch_cuda.so" ]] \
    && "${venv_dir}/bin/python" -c 'import sys, torch; sys.exit(not torch.cuda.is_available())' \
        >/dev/null 2>&1; then
    runtime_env+=("LD_PRELOAD=${torch_lib}/libtorch_cuda.so${LD_PRELOAD:+:${LD_PRELOAD}}")
fi

has_model_path=false
show_help=false
for arg in "$@"; do
    case "${arg}" in
        --model-path) has_model_path=true ;;
        -h|--help) show_help=true ;;
    esac
done

if [[ "${has_model_path}" == false && "${show_help}" == false ]]; then
    model_path="${MINISGL_MODEL_PATH:-${repo_dir}/../mini-sglang/Qwen/Qwen3-0.6B}"
    if [[ ! -f "${model_path}/config.json" ]]; then
        echo "找不到默认模型：${model_path}" >&2
        echo "请使用 --model-path /path/to/model，或设置 MINISGL_MODEL_PATH。" >&2
        exit 1
    fi
    set -- --model-path "${model_path}" "$@"
fi

# Preserve model paths relative to the directory where the script was called.
server_args=("$@")
for ((i = 0; i + 1 < ${#server_args[@]}; i++)); do
    if [[ "${server_args[i]}" == "--model-path" && "${server_args[i + 1]}" != /* ]]; then
        server_args[i + 1]="${PWD}/${server_args[i + 1]}"
    fi
done

attention_backend=""
for ((i = 0; i + 1 < ${#server_args[@]}; i++)); do
    if [[ "${server_args[i]}" == "--attention-backend" ]]; then
        attention_backend="${server_args[i + 1]}"
    fi
done
if [[ "${attention_backend,,}" == "flashinfer" ]]; then
    flashinfer_header=""
    for candidate in "${venv_dir}"/lib/python*/site-packages/flashinfer/data/include/flashinfer/attention/decode.cuh; do
        if [[ -f "${candidate}" ]]; then
            flashinfer_header="${candidate}"
            break
        fi
    done
    if [[ -z "${flashinfer_header}" ]]; then
        echo "FlashInfer 头文件不可用：请在 VENV_DIR 指向的环境中安装 flashinfer-python。" >&2
        exit 1
    fi
fi

cd "${repo_dir}"
echo "正在编译 debug 版本..."
cargo build --bin sglang-rust

target_dir="${CARGO_TARGET_DIR:-${repo_dir}/target}"
if [[ "${target_dir}" != /* ]]; then
    target_dir="${repo_dir}/${target_dir}"
fi

echo "正在启动服务（默认 max-running-req=4、max-seq-len=512；可用命令行参数覆盖）..."
set -- "${target_dir}/debug/sglang-rust" --max-running-req 4 --max-seq-len 512 "${server_args[@]}"
if [[ -n "${runtime_env[0]-}" ]]; then
    exec env "${runtime_env[@]}" "$@"
fi
exec "$@"
