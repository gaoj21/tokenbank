# Qwen3.8-27B-FP8 on local vLLM — usage reference

Everything below was verified against the running server on 2026-09-02.
Start/stop script: `_start_vllm_qwen38.sh`

## Endpoint

| | |
|---|---|
| Base URL | `http://127.0.0.1:8001/v1` |
| Model ID | `Qwen/Qwen3.8-27B-FP8` (exact string; requests with any other name 404) |
| API key | none required (send anything, e.g. `EMPTY`) |
| Context | `max_model_len = 32768` (model itself supports 262144) |
| Concurrency | `max_num_seqs = 16` |
| KV cache | 152,234 tokens |

## Routes actually served

```
/v1/chat/completions      /v1/completions        /v1/models
/v1/messages              /v1/messages/count_tokens        <- Anthropic-compatible
/v1/responses             /v1/responses/{id}     /v1/chat/completions/batch
/tokenize  /detokenize  /metrics  /health  /ping  /version
```

`/v1/messages` means the **Anthropic SDK can point straight at this server** — the same
client code runs against local weights and against api.anthropic.com with only `base_url`
and `model` changed.

## Verified request parameters

| parameter | status | note |
|---|---|---|
| `temperature`, `top_p`, `seed` | OK | `temperature:0` + `seed` is **byte-reproducible** (3/3 identical md5) |
| `max_tokens`, `min_tokens` | OK | `min_tokens` forces exact-length output (useful for benchmarking) |
| `stop` | OK | |
| `n` | OK | returns n choices |
| `logprobs` + `top_logprobs` | OK | populated |
| `response_format` json_schema | OK | returns schema-valid JSON |
| `chat_template_kwargs` | OK | see thinking control below |
| `tools` / function calling | **400** | needs a restart, see below |
| vision (image/video input) | untested | model is multimodal (`<\|image_pad\|>`, `<\|video_pad\|>`) |

### Thinking control — read this before anything else

The server runs **without** `--reasoning-parser`, so by default the model's chain of thought
is emitted **into `content`**, terminated by a bare `</think>`:

```
'We need to respond ... Need final exactly OK. No extra.\n</think>\n\nOK'
```

There is no `reasoning_content` field to read it from. Two options:

```jsonc
// 1. turn thinking off  (clean output, what you usually want for a harness)
"chat_template_kwargs": {"enable_thinking": false}

// 2. leave it on and split on the last "</think>" yourself
```

Leaving this unset is the single most likely cause of a downstream parser silently
mis-reading model output.

## curl

```bash
# chat
curl -s http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' -d '{
    "model": "Qwen/Qwen3.8-27B-FP8",
    "messages": [{"role":"user","content":"Reply with exactly: OK"}],
    "max_tokens": 64, "temperature": 0, "seed": 1,
    "chat_template_kwargs": {"enable_thinking": false}
  }'

# raw completion (no chat template)
curl -s http://127.0.0.1:8001/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.8-27B-FP8","prompt":"The capital of France is","max_tokens":8,"temperature":0}'

# structured output
curl -s http://127.0.0.1:8001/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"Qwen/Qwen3.8-27B-FP8","max_tokens":60,"temperature":0,
  "messages":[{"role":"user","content":"Give a person: name and age."}],
  "chat_template_kwargs":{"enable_thinking":false},
  "response_format":{"type":"json_schema","json_schema":{"name":"p","schema":{
    "type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},
    "required":["name","age"],"additionalProperties":false}}}}'

# Anthropic-shaped
curl -s http://127.0.0.1:8001/v1/messages \
  -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"Qwen/Qwen3.8-27B-FP8","max_tokens":32,
       "messages":[{"role":"user","content":"Reply with exactly: OK"}]}'

# health / model / cache stats
curl -s http://127.0.0.1:8001/health
curl -s http://127.0.0.1:8001/v1/models | python3 -m json.tool
curl -s http://127.0.0.1:8001/metrics | grep -E 'prefix_cache|kv_cache_usage|num_requests'
```

## Python

```python
# OpenAI SDK
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="EMPTY")
r = c.chat.completions.create(
    model="Qwen/Qwen3.8-27B-FP8",
    messages=[{"role": "user", "content": "Reply with exactly: OK"}],
    max_tokens=64, temperature=0, seed=1,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(r.choices[0].message.content)
```

```python
# Anthropic SDK against the local server — same code shape as the real API.
# Only base_url and model differ; swap them to run the identical harness on
# api.anthropic.com. NOTE: server-side features of the real API (prompt caching
# via cache_control, adaptive thinking, effort) are NOT implemented here.
from anthropic import Anthropic
c = Anthropic(base_url="http://127.0.0.1:8001", api_key="EMPTY")
m = c.messages.create(
    model="Qwen/Qwen3.8-27B-FP8", max_tokens=64,
    messages=[{"role": "user", "content": "Reply with exactly: OK"}],
)
print(m.content[0].text)
```

## Server control

```bash
cd /home/j0018/Desktop/tokenbank

# start (defaults: port 8001, TP=2, 32k ctx, 16 seqs, prefix caching on)
bash _start_vllm_qwen38.sh

# background
nohup bash _start_vllm_qwen38.sh > /dev/null 2>&1 &

# tune via env vars
MAX_MODEL_LEN=8192  MAX_NUM_SEQS=48 bash _start_vllm_qwen38.sh   # short ctx, high concurrency
MAX_MODEL_LEN=100000 MAX_NUM_SEQS=2 bash _start_vllm_qwen38.sh   # long ctx, near single-stream
PORT=8002 bash _start_vllm_qwen38.sh

# stop
pkill -f 'vllm serve' && sleep 5 && pkill -9 -f 'VLLM::'

# roll back to Qwen3.6 (still cached, 29G)
MODEL_DIR=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B-FP8/snapshots/*/ | head -1) \
  bash _start_vllm_qwen38.sh
```

### Constraints on the tuning knobs

- **`TP` must divide 4.** The model has `num_key_value_heads = 4`, so tensor-parallel size
  is restricted to 1, 2, or 4. **`TP=3` is invalid** — the third A5000 cannot be added this
  way, and 29 GB of weights will not fit on one 24 GB card either. GPU 2 is free for other
  work.
- **`MAX_MODEL_LEN` × `MAX_NUM_SEQS` may exceed 152,234** — vLLM oversubscribes and relies
  on prefix reuse plus short sequences. The current 32768×16 = 524k is already 3.4×
  oversubscribed and works fine; push `MAX_NUM_SEQS` up only if episodes stay short.
- Startup takes ~170 s (87 s of that is torch compilation).

## To enable tool calling (requires restart)

Currently `tools` returns HTTP 400:
`"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set`

The model's chat template emits this XML form:

```
<tool_call>
<function=name>
<parameter=key>
value
</parameter>
</function>
</tool_call>
```

Add to the serve command:

```
--enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser deepseek_r1
```

`qwen3_xml` and `qwen3_coder` both target this XML shape — **try `qwen3_xml` first and
verify with a real tool call**, falling back to `qwen3_coder`. `--reasoning-parser` also
moves the `<think>` block out of `content` into `reasoning_content`, which removes the need
for `enable_thinking: false`. Neither has been tested on this deployment yet.

Full tool-parser list: `apertus cohere_command3 cohere_command4 deepseek_v3 deepseek_v31
deepseek_v32 deepseek_v4 ernie45 functiongemma gemma4 gigachat3 glm45 glm47 granite
granite-20b-fc granite4 hermes hunyuan_a13b hy_v3 inkling internlm jamba kimi_k2 kimi_k3
lfm2 llama3_json llama4_json llama4_pythonic longcat mimo minicpm5 minimax_m2 minimax_m3
mistral olmo3 openai phi4_mini_json poolside_v1 pythonic qwen3_coder qwen3_xml seed_oss
step3 step3p5 xlam`

## Measured performance (2026-09-02, TP=2, 32k ctx)

Decode, 256 output tokens per request:

| concurrency | aggregate tok/s | per-request tok/s |
|---|---|---|
| 1 | 41.1 | 41.1 |
| 4 | 145.8 | 36.4 |
| 8 | 265.0 | 33.1 |
| **16** | **459.1** | 28.7 |
| 32 | 459.3 | 14.4 (queued — no gain) |

Peak is 459 tok/s at concurrency 16, exactly `max_num_seqs`. Past that, requests queue with
zero throughput gain and double the latency.

Prefill, cold:

| prompt tokens | latency | prefill tok/s |
|---|---|---|
| 956 | 0.57 s | 1666 |
| 7,451 | 4.23 s | 1762 |
| 26,259 | 15.60 s | 1683 |

**Flat from 1k to 26k.** A dense full-attention model degrades over this range; this one does
not, because only 16 of 64 layers use full attention (the other 48 are linear attention).
Same reason the KV cache holds 152k tokens on 5.36 GiB.

### Fan-out: fire all forks at once

Sixteen continuations sharing one 12k prefix, each variant on its own fresh prefix:

| | wall | aggregate | prefix cache hit |
|---|---|---|---|
| all 16 fired simultaneously | **13.3 s** | 154.1 tok/s | 91.4% |
| warm one first, then 15 | 16.5 s | 116.5 tok/s | 91.5% |

**Do not warm the prefix first on vLLM.** The scheduler already deduplicates identical
prefixes inside a batch, so the hit rate is the same either way, and the warm-up request
just idles the GPU for one round — costing ~24% throughput.

This is the opposite of the rule for the Anthropic API, where a cache entry only becomes
readable once the first response begins streaming, so N simultaneous requests all miss. The
two backends need opposite fan-out strategies; do not share that code path.
