# Shared endpoint settings. llm5 serves a pre-quantized FP8 checkpoint
# (qwen3.8-27b-fp8) rather than llm9's runtime-FP8 Qwen3.8-27B. Measured
# throughput is 25 tasks/min at 64 concurrency, against 44-47 at 32 on llm9, so
# schedule accordingly. MODEL_TAG stays Qwen3.8-27B so trace filenames line up
# with the existing table; whether the two endpoints are interchangeable is what
# the _llm5chk run tests.
export OPENAI_BASE_URL=https://llm5.ruoyu-wang.com/v1
export VLLM_API_KEY=EMPTY
export VLLM_MODEL=qwen3.8-27b-fp8
export MODEL_TAG=Qwen3.8-27B
export WORKERS=64
