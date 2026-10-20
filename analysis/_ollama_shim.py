"""OpenAI-compatible shim in front of Ollama's native /api/chat.

Why: qwen3.6 via Ollama's /v1/chat/completions ignores every
thinking-disable switch (chat_template_kwargs.enable_thinking,
"think": false, /no_think) — thinking traces land in the `reasoning`
field, eat the small max_tokens budget, and leave `content` empty.
The native /api/chat endpoint honors "think": false, so this shim
translates the subset of the OpenAI chat-completions API that the
experiment harness uses and forces think=false + a sane num_ctx.

Run:  .venv/bin/uvicorn _ollama_shim:app --port 8123
Then: LLM_BACKEND=vllm OPENAI_BASE_URL=http://127.0.0.1:8123/v1 \
      VLLM_API_KEY=ollama VLLM_MODEL=qwen3.6:latest ...
"""
import os
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434')
NUM_CTX = int(os.getenv('SHIM_NUM_CTX', '16384'))
FORCE_NO_THINK = os.getenv('SHIM_THINK', 'false').lower() != 'true'

app = FastAPI()
_client: httpx.AsyncClient | None = None


@app.on_event('startup')
async def _startup():
    global _client
    _client = httpx.AsyncClient(base_url=OLLAMA_URL, timeout=httpx.Timeout(600.0))


@app.on_event('shutdown')
async def _shutdown():
    await _client.aclose()


@app.get('/v1/models')
async def models():
    r = await _client.get('/api/tags')
    r.raise_for_status()
    return {'object': 'list', 'data': [
        {'id': m['name'], 'object': 'model', 'created': 0, 'owned_by': 'ollama'}
        for m in r.json().get('models', [])]}


@app.post('/v1/chat/completions')
async def chat_completions(req: Request):
    body = await req.json()
    options = {
        'num_ctx': NUM_CTX,
        'num_predict': body.get('max_tokens') or body.get('max_completion_tokens') or 1024,
    }
    if body.get('temperature') is not None:
        options['temperature'] = body['temperature']
    if body.get('top_p') is not None:
        options['top_p'] = body['top_p']
    if body.get('seed') is not None:
        options['seed'] = body['seed']
    if body.get('stop') is not None:
        options['stop'] = body['stop']

    payload = {
        'model': body['model'],
        'messages': body['messages'],
        'stream': False,
        'think': not FORCE_NO_THINK and bool(body.get('think', False)),
        'options': options,
    }
    try:
        r = await _client.post('/api/chat', json=payload)
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        return JSONResponse(status_code=e.response.status_code,
                            content={'error': {'message': e.response.text,
                                               'type': 'ollama_error'}})
    except httpx.HTTPError as e:
        return JSONResponse(status_code=502,
                            content={'error': {'message': str(e),
                                               'type': 'ollama_unreachable'}})
    d = r.json()
    msg = d.get('message', {})
    content = msg.get('content') or ''
    # qwen3.6 sometimes leaks its thinking close-tag into content even with
    # think disabled; drop everything up to and including it so downstream
    # answer extraction sees a clean reply.
    if '</think>' in content:
        content = content.split('</think>', 1)[1]
    content = content.lstrip()
    prompt_toks = d.get('prompt_eval_count', 0)
    comp_toks = d.get('eval_count', 0)
    done = d.get('done_reason', 'stop')
    return {
        'id': f'chatcmpl-shim-{int(time.time() * 1000)}',
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': body['model'],
        'choices': [{
            'index': 0,
            'message': {'role': 'assistant', 'content': content},
            'finish_reason': 'stop' if done == 'stop' else 'length',
        }],
        'usage': {'prompt_tokens': prompt_toks, 'completion_tokens': comp_toks,
                  'total_tokens': prompt_toks + comp_toks},
    }
