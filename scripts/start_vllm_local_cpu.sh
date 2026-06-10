#!/usr/bin/env bash
#
# LOCAL DEV ONLY - tiny model on CPU (e.g. your Mac), for exercising the
# agent / o11y / eval plumbing without a GPU.
#
# This is NOT for any reported numbers. The 30B + real latency/throughput
# come from scripts/start_vllm.sh on the H100. CPU inference here is slow and
# the metrics are unrepresentative - it only proves the wiring works.
#
# Prereqs (one-time, because vLLM 0.10.x is incompatible with transformers 5.x):
#   uv pip install 'transformers>=4.55,<5'
#
# Reference: https://docs.vllm.ai/en/latest/getting_started/installation/cpu.html

set -euo pipefail

# Small stand-in model. Override with: MODEL=... bash scripts/start_vllm_local_cpu.sh
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"

# KV cache size for the CPU backend, in GB. Required for CPU vLLM.
export VLLM_CPU_KVCACHE_SPACE="${VLLM_CPU_KVCACHE_SPACE:-4}"

# Keep the agent's .env in sync with whatever you serve here:
#   VLLM_BASE_URL=http://localhost:8000/v1
#   VLLM_MODEL=Qwen/Qwen3-0.6B

exec uv run --no-sync python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096 \
    --max-num-batched-tokens 4096 \
    --enforce-eager \
    --dtype bfloat16
