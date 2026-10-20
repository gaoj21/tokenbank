#!/bin/bash
# Local ollama instance for Token Bank experiments (port 11435).
# Why not the system ollama (11434):
#   - we need OLLAMA_CONTEXT_LENGTH=8192 (system one loads 262K ctx / 1 slot)
#   - qwen35moe is single-slot in ollama, so we at least keep ctx small & resident
#   - OLLAMA_LLM_LIBRARY=cuda_v12 forces the CUDA backend (default falls back
#     to Vulkan because driver 535 < ollama's 550 requirement; CUDA minor
#     version compatibility still allows cuda_v12 on this driver)
# Shares the read-only system model store. Requires no sudo.
cd "$(dirname "$0")"
export OLLAMA_HOST=127.0.0.1:11435
export OLLAMA_MODELS=/usr/share/ollama/.ollama/models
export OLLAMA_LLM_LIBRARY="${OLLAMA_LLM_LIBRARY:-cuda_v12}"
export OLLAMA_CONTEXT_LENGTH=8192
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE=24h
exec ollama serve
