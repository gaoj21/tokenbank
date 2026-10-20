import matplotlib
matplotlib.use('Agg')


# ============ NOTEBOOK CELL 2 ============
import os
import sys
import json
import time
import random
import re
import math
import ast
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import collections
from collections import defaultdict, deque
from datetime import datetime
import copy
import threading
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import beta as beta_dist
import tiktoken

# Try to import statsmodels for ARIMA; fallback to simple EWMA if unavailable
try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print('statsmodels not available; using EWMA fallback for forecasting')

# ==================== GLOBAL CONFIG ====================

# API Configuration (OpenAI-compatible endpoint)
# Two backends, selected by LLM_BACKEND env var:
#   vllm (default): local vLLM server serving Qwen/Qwen3.6-35B-A3B
#     start with: .venv/bin/vllm serve /root/autodl-tmp/models/Qwen3.6-35B-A3B \
#       --served-model-name Qwen/Qwen3.6-35B-A3B --port 8000
#   deepseek: DeepSeek API (previous default, higher cost)
LLM_BACKEND = os.getenv('LLM_BACKEND', 'vllm')

if LLM_BACKEND == 'deepseek':
    # No default. A literal key here is a key in the git history, which deleting the line
    # later does not undo.
    OPENAI_API_KEY = os.environ['DEEPSEEK_API_KEY']
    OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.deepseek.com')
    PRIMARY_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-v4-pro')
    THROTTLE_MODEL = 'deepseek-v4-flash'   # cheaper tier, used when throttling
    SUMMARIZE_MODEL = 'deepseek-v4-flash'  # for KV-cache summarization
    # Pricing (USD per 1M tokens, DeepSeek V4 list prices, cache-miss input)
    PRICING = {
        'deepseek-v4-pro': {'input': 0.435, 'output': 0.87},
        'deepseek-v4-flash': {'input': 0.14, 'output': 0.28},
    }
    LLM_EXTRA_BODY = {}
else:
    OPENAI_API_KEY = os.getenv('VLLM_API_KEY', 'EMPTY')
    OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'http://127.0.0.1:8000/v1')
    PRIMARY_MODEL = os.getenv('VLLM_MODEL', 'Qwen/Qwen3.6-35B-A3B')
    THROTTLE_MODEL = PRIMARY_MODEL    # single local model: no cheaper tier
    SUMMARIZE_MODEL = PRIMARY_MODEL   # judge/summarization reuse the same model
    # Nominal local pricing for CPS continuity (GPU rental amortized per token,
    # NOT an API bill). Relative cost comparisons across configs stay valid.
    PRICING = {
        PRIMARY_MODEL: {'input': 0.03, 'output': 0.03},
    }
    # Qwen3.6 / Laguna are hybrid-thinking models; thinking traces would eat the
    # small max_tokens budgets used here (same failure mode as DeepSeek V4
    # thinking). Gemma doesn't accept this chat-template kwarg.
    # Mistral Small 4 uses reasoning_effort instead (official per-request switch).
    # REASONING_EFFORT env: 'none' (default, protocol parity) or 'high' (reasoning
    # variant — pair with MAX_TOKENS >= 4096 and larger budgets).
    LLM_EXTRA_BODY = {}
    if 'istral' in PRIMARY_MODEL:
        LLM_EXTRA_BODY = {'reasoning_effort': os.getenv('REASONING_EFFORT', 'none')}
    elif 'qwen' in PRIMARY_MODEL.lower() or 'laguna' in PRIMARY_MODEL.lower():
        LLM_EXTRA_BODY = {'chat_template_kwargs': {'enable_thinking': False}}
    # Router gateways need per-request routing directives that are not a model
    # property, most importantly pinning one upstream provider: OpenRouter
    # otherwise picks a different one per call, and the providers behind a single
    # model id differ in quantization (fp4 / fp8 / bf16) and in whether they
    # return logprobs at all, which the confidence gate depends on. A run whose
    # provider drifts is not a controlled comparison.
    _xb = os.getenv('LLM_EXTRA_BODY_JSON', '')
    if _xb:
        LLM_EXTRA_BODY = {**LLM_EXTRA_BODY, **json.loads(_xb)}

# Per-call output cap. Default 1024 (fast-answer protocol). Reasoning variants
# need more headroom for thinking traces (dispatch estimates scale with this).
MAX_TOKENS = int(os.getenv('MAX_TOKENS', '1024'))
# Sampling temperature. Default 0.7 (legacy); set TEMPERATURE=0 for
# deterministic final-eval runs so SR comparisons are not drowned in noise.
TEMPERATURE = float(os.getenv('TEMPERATURE', '0.7'))
# Stall refinements (OFF by default — A/B at 32K/0.2/temp0 showed no
# reproducible gain over the plain config; keep behind flags):
#   STALL_EXEMPT_TOOL: a round that executed the code tool counts as progress
#     and cannot trigger the runaway counter / stall (answer text may be
#     unchanged while computation is still advancing).
#   GRACE_ANSWER: when a stall fires, make one cheap (max_tokens=256) final
#     answer call before ending the task instead of stopping with the stale
#     draft.
STALL_EXEMPT_TOOL = os.getenv('STALL_EXEMPT_TOOL', '0') == '1'
GRACE_ANSWER = os.getenv('GRACE_ANSWER', '0') == '1'
# Reviewer-score stall signal: reviewer additionally outputs a 0-10 closeness
# score; a round counts as progress when the score improves even if the answer
# text is unchanged. Stall requires answer unchanged AND score not improved.
REVIEWER_SCORE = os.getenv('REVIEWER_SCORE', '0') == '1'

# Experiment Mode
MOCK_MODE = os.getenv('MOCK', '0') == '1'  # True: simulate LLM responses (zero cost)
VERBOSE = True
SAVE_TRAJECTORIES = True

# Model Configuration (set in the backend blocks above)
print(f'LLM Backend: {LLM_BACKEND} | Base URL: {OPENAI_BASE_URL} | Model: {PRIMARY_MODEL}')

# Token encoder for budget accounting: prefer the served model's own tokenizer
# (accurate per-token accounting); fall back to tiktoken gpt-4o when the local
# tokenizer is unavailable (e.g. running against a remote API on a dev laptop).
class _HFEncoder:
    def __init__(self, path):
        from transformers import AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(path)

    def encode(self, text):
        return self._tok.encode(text)

TOKENIZER_PATH = os.getenv(
    'TOKENIZER_PATH', f'/root/autodl-tmp/models/{PRIMARY_MODEL.split("/")[-1]}')
try:
    ENCODER = _HFEncoder(TOKENIZER_PATH) if LLM_BACKEND != 'deepseek' else None
    if ENCODER is None:
        raise ValueError('deepseek backend: use tiktoken')
    print(f'Tokenizer: {TOKENIZER_PATH}')
except Exception as e:
    print(f'Local tokenizer unavailable ({e}); falling back to tiktoken gpt-4o')

    class _PlainTiktoken:
        """tiktoken that counts special-token strings (e.g. a model emitting '<|endoftext|>') as plain
        text instead of raising, which otherwise aborts the whole run."""
        def __init__(self, enc):
            self._enc = enc
        def encode(self, text):
            return self._enc.encode(text, disallowed_special=())
    ENCODER = _PlainTiktoken(tiktoken.encoding_for_model('gpt-4o'))

# Mock-mode simulated per-call token sizes (mean, std) per role.
# Sized so that a single call consumes a meaningful fraction of the per-agent
# quota, so borrowing / truncation / throttling actually get exercised.
MOCK_TOKEN_PROFILES = {
    'Planner': (8000, 2000),
    'Executor': (15000, 4000),
    'Reviewer': (5000, 1500),
    'Debugger': (12000, 3500),
}

# Random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Experiment budgets per benchmark (tokens)
BUDGETS = {
    'GAIA': 100000,
    'SWE': 150000,
    'FINANCIAL': 80000,
}

print(f'Mock Mode: {MOCK_MODE}')
print(f'Primary Model: {PRIMARY_MODEL}')
print(f'Has statsmodels ARIMA: {HAS_STATSMODELS}')


# ============ NOTEBOOK CELL 4 ============
# ==================== DATASET: GAIA ====================
def _extract_attachment_text(path: str, max_chars: int = 3000) -> str:
    """Extract text from a GAIA attachment (pdf/docx/xlsx/ plain text)."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == '.pdf':
            import pypdf
            reader = pypdf.PdfReader(path)
            text = '\n'.join((p.extract_text() or '') for p in reader.pages[:5])
        elif ext == '.docx':
            import docx
            text = '\n'.join(par.text for par in docx.Document(path).paragraphs)
        elif ext == '.xlsx':
            import openpyxl
            wb = openpyxl.load_workbook(path, read_only=True)
            rows = []
            for ws in wb.worksheets:
                for row in ws.iter_rows(values_only=True):
                    rows.append(' '.join('' if c is None else str(c) for c in row))
            text = '\n'.join(rows)
        elif ext == '.pptx':
            import pptx
            prs = pptx.Presentation(path)
            text = '\n'.join(shape.text for slide in prs.slides for shape in slide.shapes
                             if shape.has_text_frame)
        elif ext == '.zip':
            import zipfile, tempfile
            parts = []
            with zipfile.ZipFile(path) as zf, tempfile.TemporaryDirectory() as tmp:
                zf.extractall(tmp)
                for name in zf.namelist()[:10]:
                    p = os.path.join(tmp, name)
                    if os.path.isfile(p) and os.path.getsize(p) < 200_000:
                        try:
                            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                                parts.append(f'--- {name} ---\n{f.read()[:1500]}')
                        except Exception:
                            pass
            text = '\n'.join(parts)
        else:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
    except Exception as e:
        return f'[attachment could not be parsed: {e}]'
    return text[:max_chars]


def _load_gaia_attachment(file_name: str) -> str:
    """Download a GAIA attachment from the (gated) HF dataset repo and extract text."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id='gaia-benchmark/GAIA', repo_type='dataset',
                           filename=f'2023/validation/{file_name}')
    return _extract_attachment_text(path)


def load_gaia_dataset(n_tasks: int = 50, level_filter: List[int] = [1, 2]) -> List[Dict]:
    """Load GAIA benchmark from huggingface or fallback JSON."""
    # Binary media (image/audio) is unanswerable for a text-only model;
    # those tasks are excluded from sampling and counted here.
    MEDIA_EXTS = ('.png', '.jpg', '.jpeg', '.mp3', '.wav')
    try:
        from datasets import load_dataset
        # Gated dataset: requires HF_TOKEN env var with accepted terms
        ds = load_dataset('gaia-benchmark/GAIA', '2023_all', split='validation')
        tasks = []
        n_media_skipped = 0
        for item in ds:
            lvl = item.get('Level', '1')
            try:
                lvl_int = int(lvl)
            except:
                lvl_int = 1
            if lvl_int in level_filter and item.get('Question') and item.get('Final answer'):
                question = item['Question']
                file_name = item.get('file_name') or ''
                if file_name and file_name.lower().endswith(MEDIA_EXTS):
                    n_media_skipped += 1
                    continue
                if file_name:
                    try:
                        attachment = _load_gaia_attachment(file_name)
                        question += f'\n\n[Attached file: {file_name}]\n{attachment}'
                    except Exception as e:
                        print(f'  attachment {file_name} unavailable: {e}')
                tasks.append({
                    'id': f"gaia_{item.get('task_id', len(tasks))}",
                    'question': question,
                    'answer': str(item['Final answer']),
                    'level': lvl_int,
                    'has_attachment': bool(file_name),
                    'expected_agents': ['Planner', 'Executor', 'Reviewer']
                })
        if n_media_skipped:
            print(f'GAIA: skipped {n_media_skipped} image/audio tasks (text-only model)')
        random.shuffle(tasks)
        return tasks[:n_tasks]
    except Exception as e:
        print(f'GAIA huggingface load failed: {e}')
        print('Falling back to synthetic GAIA-like tasks...')
        return load_gaia_fallback(n_tasks)

def load_gaia_fallback(n_tasks: int = 50) -> List[Dict]:
    """Synthetic GAIA-like tasks when huggingface is unavailable."""
    templates = [
        {'q': 'What is the population of {city} (capital of {country})?', 'a': '{pop}'},
        {'q': 'Who composed the score for the {year} film {film}?', 'a': '{composer}'},
        {'q': 'Calculate compound interest on ${p} at {r}% for {t} years, compounded monthly.', 'a': '{result}'},
        {'q': 'Which NASA mission first landed a rover on Mars? Year? Rover name?', 'a': 'Mars Pathfinder; 1997; Sojourner'},
        {'q': 'Sum of all prime numbers between {a} and {b}.', 'a': '{result}'},
    ]
    fillers = [
        {'city': 'Paris', 'country': 'France', 'pop': '2,161,000'},
        {'year': '2023', 'film': 'Oppenheimer', 'composer': 'Ludwig Göransson'},
        {'p': '10000', 'r': '5', 't': '3', 'result': '1614.72'},
        {'a': '100', 'b': '200', 'result': '3167'},
    ]
    # template index -> matching filler index (template 3 needs no filler)
    filler_for_template = {0: 0, 1: 1, 2: 2, 3: 0, 4: 3}
    tasks = []
    for i in range(n_tasks):
        ti = i % len(templates)
        t = templates[ti]
        f = fillers[filler_for_template[ti]]
        q = t['q'].format(**f)
        a = t['a'].format(**f)
        tasks.append({'id': f'gaia_fallback_{i}', 'question': q, 'answer': a, 'level': 1,
                      'expected_agents': ['Planner', 'Executor', 'Reviewer']})
    return tasks

# ==================== DATASET: SWE-bench Lite ====================
def _swe_context_from_gold_patch(patch: str) -> str:
    """Reconstruct the buggy code region from the gold patch's minus/context
    lines — what an agent with repo access would read. Added lines (the fix
    itself) are excluded so the answer is not leaked."""
    lines = []
    for ln in patch.split('\n'):
        if ln.startswith(('+++', '---', '@@')):
            continue
        if ln.startswith('-') or ln.startswith(' '):
            lines.append(ln[1:])
    return '\n'.join(lines)


def load_swebench_lite(n_tasks: int = 30) -> List[Dict]:
    """Load SWE-bench-lite from HuggingFace or fallback."""
    try:
        from datasets import load_dataset
        ds = load_dataset('princeton-nlp/SWE-bench_Lite', split='test')
        tasks = []
        for item in list(ds)[:n_tasks]:
            tasks.append({
                'id': f"swe_{item.get('instance_id', len(tasks))}",
                'issue': item.get('problem_statement', 'Fix the bug'),
                'repo': item.get('repo', ''),
                'code_before': _swe_context_from_gold_patch(item.get('patch', '')),
                'gold_patch': item.get('patch', ''),
                'expected_agents': ['Planner', 'Executor', 'Debugger', 'Reviewer']
            })
        return tasks
    except Exception as e:
        print(f'SWE-bench load failed: {e}')
        return load_swe_fallback(n_tasks)

def load_swe_fallback(n_tasks: int = 30) -> List[Dict]:
    """Synthetic SWE-like tasks."""
    tasks = [
        {'id': 'swe_001', 'issue': 'Fix off-by-one error in list slicing when end index equals list length.',
         'code_before': 'def slice_list(lst, start, end):\n    return lst[start:end+1]',
         'code_after': 'def slice_list(lst, start, end):\n    return lst[start:end]',
         'test': 'assert slice_list([1,2,3], 0, 3) == [1,2,3]'},
        {'id': 'swe_002', 'issue': 'Handle KeyError when dict key might not exist; return default value None.',
         'code_before': 'def get_value(d, key):\n    return d[key]',
         'code_after': 'def get_value(d, key):\n    return d.get(key)',
         'test': 'assert get_value({}, "x") is None'},
        {'id': 'swe_003', 'issue': 'Fix recursive Fibonacci to use memoization to avoid exponential time.',
         'code_before': 'def fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)',
         'code_after': 'def fib(n, memo={}):\n    if n in memo: return memo[n]\n    if n <= 1: return n\n    memo[n] = fib(n-1, memo) + fib(n-2, memo)\n    return memo[n]',
         'test': 'assert fib(30) == 832040'},
    ]
    result = []
    for i in range(n_tasks):
        t = copy.deepcopy(tasks[i % len(tasks)])
        t['id'] = f'swe_fallback_{i}'
        result.append(t)
    return result

# ==================== DATASET: Financial (Synthetic SEC 10-K) ====================
def load_financial_dataset(n_tasks: int = 20) -> List[Dict]:
    """Synthetic financial extraction tasks based on public SEC filing excerpts."""
    docs = [
        {
            'id': 'fin_001',
            'company': 'Apple Inc.',
            'document': """Apple Inc. 10-K Filing (Excerpt)
CEO: Timothy D. Cook
Headquarters: Cupertino, California
Fiscal Year 2024 Revenue: $391.0 billion
Net Income: $101.9 billion
Total Employees: 161,000
Board Members: Arthur Levinson, James Bell, Al Gore, Andrea Jung
Audit Firm: Ernst & Young LLP
Stock Ticker: AAPL
Fiscal Year End: Last Saturday of September""",
            'fields': {
                'ceo': 'Timothy D. Cook', 'headquarters': 'Cupertino, California',
                'revenue': '$391.0 billion', 'net_income': '$101.9 billion',
                'employees': '161,000', 'audit_firm': 'Ernst & Young LLP',
                'ticker': 'AAPL', 'fy_end': 'Last Saturday of September'
            }
        },
        {
            'id': 'fin_002',
            'company': 'Tesla Inc.',
            'document': """Tesla Inc. 10-K Filing (Excerpt)
CEO: Elon Musk
Incorporated: Delaware
Principal Executive Offices: Austin, Texas
2024 Revenue: $97.7 billion
Gross Profit: $17.9 billion
Cash and Cash Equivalents: $14.7 billion
Total Debt: $9.6 billion
Independent Auditor: PricewaterhouseCoopers LLP
Stock Ticker: TSLA
Employees: 125,000""",
            'fields': {
                'ceo': 'Elon Musk', 'incorporated': 'Delaware',
                'headquarters': 'Austin, Texas', 'revenue': '$97.7 billion',
                'gross_profit': '$17.9 billion', 'cash': '$14.7 billion',
                'debt': '$9.6 billion', 'auditor': 'PricewaterhouseCoopers LLP',
                'ticker': 'TSLA', 'employees': '125,000'
            }
        },
        {
            'id': 'fin_003',
            'company': 'Microsoft Corp.',
            'document': """Microsoft Corporation 10-K Filing (Excerpt)
CEO: Satya Nadella
Headquarters: Redmond, Washington
Fiscal Year 2024 Revenue: $245.1 billion
Operating Income: $109.4 billion
Net Income: $88.1 billion
Total Employees: 228,000
Independent Auditor: Deloitte & Touche LLP
Stock Ticker: MSFT
Fiscal Year End: June 30""",
            'fields': {
                'ceo': 'Satya Nadella', 'headquarters': 'Redmond, Washington',
                'revenue': '$245.1 billion', 'operating_income': '$109.4 billion',
                'net_income': '$88.1 billion', 'employees': '228,000',
                'auditor': 'Deloitte & Touche LLP', 'ticker': 'MSFT',
                'fy_end': 'June 30'
            }
        },
    ]
    result = []
    for i in range(n_tasks):
        d = copy.deepcopy(docs[i % len(docs)])
        d['id'] = f"fin_{i:03d}"
        d['expected_agents'] = ['Planner', 'Executor', 'Reviewer', 'Debugger']
        result.append(d)
    return result

# ==================== DATASET: FRAMES ====================
def load_frames_dataset(n_tasks: int = 824) -> List[Dict]:
    """Google FRAMES: multi-step factoid QA (GAIA-style, public, short answers)."""
    from datasets import load_dataset
    ds = load_dataset('google/frames-benchmark', split='test')
    tasks = []
    for item in ds:
        q, a = item.get('Prompt'), item.get('Answer')
        if q and a:
            tasks.append({
                'id': f"frames_{item.get('Unnamed: 0', len(tasks))}",
                'question': q, 'answer': str(a), 'level': 2,
                'has_attachment': False,
                'expected_agents': ['Planner', 'Executor', 'Reviewer']
            })
    random.shuffle(tasks)
    return tasks[:n_tasks]


# ==================== LOAD ALL ====================
GAIA_TASKS = load_gaia_dataset(n_tasks=100)
SWE_TASKS = load_swebench_lite(n_tasks=30)
FINANCIAL_TASKS = load_financial_dataset(n_tasks=20)
FRAMES_TASKS = load_frames_dataset()

print(f'Loaded datasets:')
print(f'  GAIA: {len(GAIA_TASKS)} tasks')
print(f'  SWE-bench Lite: {len(SWE_TASKS)} tasks')
print(f'  Financial: {len(FINANCIAL_TASKS)} tasks')
print(f'  FRAMES: {len(FRAMES_TASKS)} tasks')
print(f'  Total: {len(GAIA_TASKS) + len(SWE_TASKS) + len(FINANCIAL_TASKS) + len(FRAMES_TASKS)} tasks')

# Optional smoke-test subset: SMOKE_N env var truncates each task set and
# skips the ablation / sensitivity sweeps (for quick real-API checks)
SMOKE_N = int(os.getenv('SMOKE_N', '0'))
if SMOKE_N:
    GAIA_TASKS = GAIA_TASKS[:SMOKE_N]
    SWE_TASKS = SWE_TASKS[:SMOKE_N]
    FINANCIAL_TASKS = FINANCIAL_TASKS[:SMOKE_N]
    FRAMES_TASKS = FRAMES_TASKS[:SMOKE_N]
    print(f'SMOKE_N={SMOKE_N}: task sets truncated, ablation/sensitivity skipped')

# Smoke runs write artifacts to *_smoke.* so they don't clobber full-run outputs
OUT_SUFFIX = '_smoke' if SMOKE_N else ''


# ============ NOTEBOOK CELL 6 ============
@dataclass
class AgentAccountV2:
    agent_id: str
    role: str
    quota: int = 0
    used: int = 0
    priority: int = 2
    borrowed: int = 0       # credit line utilized
    lent: int = 0           # credit extended to others
    history: List[Dict] = field(default_factory=list)
    consumption_log: deque = field(default_factory=lambda: deque(maxlen=100))
    kv_cache_tokens: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.quota + self.borrowed - self.used)

    @property
    def utilization(self) -> float:
        denom = self.quota + self.borrowed
        return self.used / denom if denom > 0 else 0.0

    @property
    def effective_tokens(self) -> int:
        """KV-cache aware: prompt + 0.3 * kv_cache"""
        return self.used + int(0.3 * self.kv_cache_tokens)

    @property
    def total_cost(self) -> float:
        return sum(h.get('cost_usd', 0) for h in self.history)


class BudgetAdapter:
    """Wraps pre-execution optimizer outputs into initial quotas."""
    def __init__(self, external_plan: Optional[Dict] = None):
        self.plan = external_plan or {}

    def get_initial_quotas(self, total_budget: int, n_agents: int,
                           role_map: Dict[str, str]) -> Dict[str, int]:
        if not self.plan:
            return {aid: total_budget // n_agents for aid in role_map}
        quotas = {}
        for aid, role in role_map.items():
            q = self.plan.get(aid, self.plan.get(role, total_budget // n_agents))
            quotas[aid] = int(q)
        return quotas

    def compute_deviation(self, planned: int, actual: int) -> float:
        return abs(actual - planned) / max(planned, 1)


class PredictiveRebalancer:
    """ARIMA/EWMA forecast + opportunity cost-driven rebalancing."""
    def __init__(self, horizon: int = 2, min_history: int = 1):
        self.horizon = horizon
        self.min_history = min_history

    def forecast(self, history: deque) -> Tuple[float, float]:
        """Returns (expected_consumption_over_horizon, trend). Agents with no
        history forecast 0, i.e. their unused quota counts as surplus."""
        if len(history) < self.min_history:
            return 0.0, 0.0
        series = list(history)
        if HAS_STATSMODELS and len(series) >= 5:
            try:
                model = ARIMA(series, order=(1,1,1))
                fitted = model.fit()
                fcast = fitted.forecast(steps=self.horizon)
                expected = sum(max(0, x) for x in fcast)
                trend = float(fcast[-1] - fcast[0]) if self.horizon > 1 else 0.0
                return expected, trend
            except:
                pass
        # EWMA fallback
        alpha = 0.3
        ewma = series[0]
        for x in series[1:]:
            ewma = alpha * x + (1 - alpha) * ewma
        expected = ewma * self.horizon
        trend = series[-1] - series[0]
        return expected, trend

    def should_rebalance(self, agents: List[AgentAccountV2],
                         reserve: int, threshold: float = 0.8) -> Tuple[bool, Dict]:
        """Opportunity cost-based rebalancing decision.

        threshold (theta) is the required headroom: an agent is predicted
        deficient when its expected near-term consumption exceeds `threshold`
        of its remaining quota. Lower theta -> triggers earlier."""
        forecasts = {}
        for a in agents:
            exp, trend = self.forecast(a.consumption_log)
            forecasts[a.agent_id] = {'expected': exp, 'trend': trend,
                                      'predicted_surplus': a.remaining - exp}

        deficit_agents = [a for a in agents
                          if forecasts[a.agent_id]['expected'] > threshold * a.remaining]
        surplus_agents = [a for a in agents if forecasts[a.agent_id]['predicted_surplus'] > 0]

        if not deficit_agents or not surplus_agents:
            return False, {}

        cost_miss = sum(a.priority * forecasts[a.agent_id]['expected']
                        for a in deficit_agents)
        reclaimable = sum((a.quota - a.used) * 0.3 for a in surplus_agents if a.utilization < 0.5)
        cost_rebal = reclaimable * 0.05

        return cost_miss > cost_rebal, forecasts


class CreditScorer:
    """Bank-style credit scoring: tracks whether borrowed credit converts to
    task success ('repayment') per role, and sizes/prices credit accordingly.
    Thread-safe so it can be shared across concurrent task banks."""
    def __init__(self, base_rate: float = 0.5, base_interest: float = 0.15):
        self.base_rate = base_rate
        self.base_interest = base_interest
        self.history: Dict[str, Dict[str, float]] = {}
        self._lock = threading.RLock()

    def score(self, role: str) -> float:
        with self._lock:
            h = self.history.get(role, {'alpha': 1.0, 'beta': 1.0})
            return h['alpha'] / (h['alpha'] + h['beta'])  # posterior mean, starts at 0.5

    def credit_rate_for(self, role: str) -> float:
        """score 0 -> 0.5x base_rate, score 1 -> 1.5x base_rate."""
        return self.base_rate * (0.5 + self.score(role))

    def interest_for(self, role: str) -> float:
        """Risk pricing: score 0 -> 2x base interest, score 1 -> 0.5x."""
        return self.base_interest * (1.5 - self.score(role))

    def record(self, role: str, repaid: bool) -> None:
        with self._lock:
            h = self.history.setdefault(role, {'alpha': 1.0, 'beta': 1.0})
            h['alpha' if repaid else 'beta'] += 1

    def summary(self) -> Dict[str, float]:
        with self._lock:
            return {r: round(self.score(r), 3) for r in self.history}


class BudgetBorrowing:
    """Inter-agent credit market with safety guarantees (Theorem 2)."""
    def __init__(self, credit_rate: float = 0.10, interest_rate: float = 0.15,
                 scorer: 'CreditScorer' = None, rogue_cutoff: float = 0.10,
                 role_credit_mult: Dict[str, float] = None):
        self.credit_rate = credit_rate
        self.interest_rate = interest_rate
        self.scorer = scorer
        # Per-role credit-line multipliers (improvement ②): e.g. Executor does
        # the heavy lifting in code tasks, so it gets a bigger line than
        # Planner. None = flat credit_rate for all roles (legacy behavior).
        self.role_credit_mult = role_credit_mult
        # Option C: credit score is used to size/pricing credit lines (shallow)
        # and to hard-cutoff confirmed rogue roles, but does not gate normal
        # peer-to-peer lending decisions.
        self.rogue_cutoff = rogue_cutoff
        self.loans: List[Dict] = []

    def max_credit_line(self, agent: AgentAccountV2) -> int:
        rate = self.scorer.credit_rate_for(agent.role) if self.scorer else self.credit_rate
        if self.role_credit_mult:
            rate *= self.role_credit_mult.get(agent.role, 1.0)
        return int(agent.quota * rate)

    def can_borrow(self, borrower: AgentAccountV2, amount: int,
                   lender: AgentAccountV2, reserve: int) -> bool:
        """Theorem 2 safety check + rogue-only cutoff."""
        if amount <= 0:
            return False
        if self.scorer:
            score = self.scorer.score(borrower.role)
            # Confirmed rogues (very low credit) are cut off entirely.
            if score < self.rogue_cutoff:
                return False
        if borrower.borrowed + amount > self.max_credit_line(borrower):
            return False
        if lender.priority > borrower.priority:
            return False
        if lender.remaining < amount:
            return False
        if reserve <= 0:
            return False
        return True

    def execute_loan(self, borrower: AgentAccountV2, lender: AgentAccountV2,
                     amount: int) -> Dict:
        rate = self.scorer.interest_for(borrower.role) if self.scorer else self.interest_rate
        interest = int(amount * rate)
        borrower.borrowed += amount
        lender.lent += amount + interest
        lender.quota -= amount
        self.loans.append({
            'borrower': borrower.agent_id, 'lender': lender.agent_id,
            'amount': amount, 'interest': interest, 'timestamp': time.time()
        })
        return {'success': True, 'amount': amount, 'interest': interest}


class OnlinePriorityLearner:
    """Thompson Sampling for adaptive role priorities."""
    def __init__(self, roles: List[str], cold_start_tasks: int = 10):
        self.roles = roles
        self.posteriors = {r: {'alpha': 1.0, 'beta': 1.0} for r in roles}
        self.cold_start = cold_start_tasks
        self.task_count = 0
        self.manual_priorities = {'Planner': 3, 'Executor': 2, 'Reviewer': 2, 'Debugger': 1}

    def get_priorities(self) -> Dict[str, int]:
        if self.task_count < self.cold_start:
            return self.manual_priorities.copy()
        priorities = {}
        for r in self.roles:
            p = beta_dist.rvs(self.posteriors[r]['alpha'], self.posteriors[r]['beta'])
            priorities[r] = max(1, min(3, round(1 + 2 * p)))
        return priorities

    def update(self, role_success: Dict[str, bool], role_blocked: Dict[str, bool]) -> None:
        self.task_count += 1
        for role in self.roles:
            if role_success.get(role, False):
                self.posteriors[role]['alpha'] += 1
            elif role_blocked.get(role, False):
                self.posteriors[role]['beta'] += 1

    def get_regret_bound(self, T: int) -> float:
        K = len(self.roles)
        return math.sqrt(K * T * math.log(T + 1))


class KVCacheTracker:
    """Tracks context accumulation and triggers compaction."""
    def __init__(self, compression_factor: float = 0.3, trigger_ratio: float = 0.85,
                 real_summary: bool = False, summary_budget: int = 512,
                 context_window: int = 8192):
        self.compression_factor = compression_factor
        self.trigger_ratio = trigger_ratio
        self.real_summary = real_summary
        self.summary_budget = summary_budget
        self.context_window = context_window
        self.summarization_count = 0

    def effective_tokens(self, prompt_tokens: int, kv_cache: int) -> int:
        return prompt_tokens + int(self.compression_factor * kv_cache)

    def should_summarize(self, prompt_tokens: int) -> bool:
        """Fire when the prompt about to be sent occupies too much of the context
        window. NOTE: this used to compare an agent's *cumulative spend* against its
        budget quota, which is not what compaction reacts to. That made the trigger
        a spend threshold: it fired on exactly the items that were already on track
        to exhaust the budget, so 100% of triggered items ended in truncation and the
        arm could not be read as a compaction baseline. Compaction is a function of
        context occupancy, so that is what is measured here."""
        return prompt_tokens / max(self.context_window, 1) > self.trigger_ratio

    def summarize_context(self, messages: List[Dict], llm_client, bank=None,
                          agent_id='executor') -> Tuple[List[Dict], int, int]:
        """Compaction: rewrite the accumulated trajectory into a bounded summary.

        Two modes. With `real_summary=False` (legacy) the prior turns are simply
        dropped behind a marker -- this is truncation, not compaction, and must
        not be reported as a compaction baseline. With `real_summary=True` an
        LLM call condenses the trajectory into <= `summary_budget` tokens and the
        agent resumes from it, which is what recent agent scaffolds do. The
        summarizer's own prompt and completion are charged to the task budget:
        compaction is not free, and a comparison that hides its cost is unfair to
        the alternative.
        """
        self.summarization_count += 1
        # Preserve the task, not just the system prompt. The first user message
        # carries the question; summarizing it away leaves the model with a
        # digest of its own reasoning and no statement of what it was asked, and
        # every compacted task then scored zero. A compaction scheme condenses
        # the intermediate turns, so that is what `middle` must be.
        head = messages[:1] if messages and messages[0].get('role') == 'system' else []
        task = []
        if len(messages) > len(head) and messages[len(head)].get('role') == 'user':
            task = [messages[len(head)]]
        keep_from = len(head) + len(task)
        kept = messages[-2:] if len(messages) > keep_from + 2 else messages[keep_from:]
        middle = messages[keep_from:len(messages) - len(kept)]

        if not self.real_summary or not middle or llm_client is None:
            summary_msg = {'role': 'system',
                           'content': '[Context summarized due to KV-cache pressure. '
                                      'Prior conversation condensed.]'}
        else:
            transcript = '\n\n'.join(
                f"[{m.get('role')}] {m.get('content','')}" for m in middle)[:12000]
            r = llm_client.call(
                agent_id, bank,
                [{'role': 'system',
                  'content': f'Condense this agent trajectory into at most '
                             f'{self.summary_budget} tokens. Preserve every intermediate '
                             f'result, numeric value and constraint the agent will need to '
                             f'finish the task. Omit restatements and pleasantries.'},
                 {'role': 'user', 'content': transcript}],
                action='summarize', max_tokens=self.summary_budget)
            body = r.get('content') or ''
            summary_msg = {'role': 'system',
                           'content': f'[Compacted trajectory]\n{body}' if body.strip()
                                      else '[Context compacted; prior turns omitted.]'}

        # Merge the summary INTO the existing system message rather than adding a
        # second one. Qwen's chat template rejects a conversation with more than
        # one system message ("System message must be at the beginning", HTTP
        # 400), so the previous `head + [summary_msg] + kept` made every call
        # after a summarization fail. Those failures were silent: the client
        # returns success=False with empty content and the error string is never
        # logged, so a compacted task simply stopped producing answers. Every
        # task that triggered summarization scored zero, on both endpoints, which
        # read as compaction being catastrophically harmful when it was this bug.
        if head:
            new_messages = [{'role': 'system',
                             'content': head[0].get('content', '') + '\n\n'
                                        + summary_msg['content']}] + task + kept
        else:
            new_messages = [summary_msg] + task + kept
        tokens_in = sum(len(ENCODER.encode(m.get('content', ''))) for m in new_messages)
        return new_messages, tokens_in, 0

print('All v2 components defined.')


# ============ NOTEBOOK CELL 7 ============
def _answer_similarity(a: str, b: str) -> float:
    """Token-set Jaccard similarity between two normalized answer strings."""
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ---- v5 Phase 1.1: proper judges (replace substring / first-two-words match) ----
try:
    from math_verify import parse as _mv_parse, verify as _mv_verify
    _HAS_MATH_VERIFY = True
except Exception:
    _HAS_MATH_VERIFY = False


def _norm_basic(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r'[^\w\s\-\.]', ' ', s)
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())


def _norm_number(s: str):
    s = str(s).replace(',', '').replace('$', '').replace('%', '').strip()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except Exception:
        return None


def _extract_answer(text: str) -> str:
    """Pull the final answer out of a chatty completion: ANSWER: marker first,
    then 'the answer is X', else the last non-empty line."""
    m = list(re.finditer(r'answer\s*(?:is|are|was|were)?\s*[:\-]\s*(.+)', text, re.I))
    if m:
        return m[-1].group(1).strip()
    m = list(re.finditer(r'answer\s+(?:is|are|was|were)\s+(.+)', text, re.I))
    if m:
        return m[-1].group(1).strip()
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    return lines[-1] if lines else text.strip()


def judge_gaia(pred: str, gold: str) -> bool:
    """Official-style GAIA scoring: number/list normalization then exact match."""
    def norm(s):
        n = _norm_number(s)
        return n if n is not None else _norm_basic(s)
    cand = _extract_answer(pred)
    gp, pp = re.split(r'[;,]', str(gold)), re.split(r'[;,]', cand)
    if len(gp) > 1 and len(pp) > 1:
        return sorted(norm(x) for x in gp) == sorted(norm(x) for x in pp)
    return norm(cand) == norm(str(gold))


def judge_math(pred: str, gold: str) -> bool:
    # Fixed judge: the previous version parsed answers without math delimiters and rejected
    # equivalent forms such as "16/27" vs "\\frac{16}{27}" (see _math_judge.py).
    from _math_judge import judge_math as _jm
    return _jm(pred, gold)


def em_f1(pred: str, gold: str):
    from collections import Counter
    p, g = _norm_basic(_extract_answer(pred)), _norm_basic(str(gold))
    em = (p == g)
    pt, gt = p.split(), g.split()
    common = sum((Counter(pt) & Counter(gt)).values())
    f1 = 2 * common / (len(pt) + len(gt)) if pt and gt else (1.0 if em else 0.0)
    return em, f1


def judge_swe(pred: str, gold: str) -> Tuple[bool, float]:
    """Proxy judge for SWE tasks: token-level F1 between the model's proposed
    fix and the gold patch's added lines, thresholded.

    This is NOT the official SWE-bench metric, which runs FAIL_TO_PASS and
    PASS_TO_PASS test suites inside the repository. Running those needs the repo
    checked out at base_commit with its dependencies installed, which the
    execution-based variant of this experiment does; results under this proxy
    must be labelled as such and never compared against published SWE-bench
    resolve rates.
    """
    code = _extract_code_block(pred) or pred
    p = set(re.findall(r'[A-Za-z_][A-Za-z0-9_]*|[^\s\w]', code))
    g = set(re.findall(r'[A-Za-z_][A-Za-z0-9_]*|[^\s\w]', gold))
    if not g:
        return False, 0.0
    inter = len(p & g)
    if not inter:
        return False, 0.0
    prec, rec = inter / max(len(p), 1), inter / len(g)
    f1 = 2 * prec * rec / (prec + rec)
    return f1 >= float(os.getenv('SWE_F1_THRESHOLD', '0.75')), f1


def judge_task(task_id: str, pred: str, gold: str):
    """Dataset-aware judge. Returns (correct, f1). No substring / first-two-words."""
    if task_id.startswith('bbh_'):
        from _bbh_judge import judge_bbh     # BBH answer extraction (option letter, first word, ...)
        ok = judge_bbh(pred, gold)
        return ok, (1.0 if ok else 0.0)
    if task_id.startswith('mbpp_'):
        from _mbpp_judge import judge_mbpp
        ok = judge_mbpp(pred, task_id)
        return ok, (1.0 if ok else 0.0)
    if task_id.startswith('lcb_'):
        # LiveCodeBench: `gold` is the test spec; the reply passes when its code passes every test.
        from _lcb_judge import judge_lcb
        ok = judge_lcb(pred, gold)
        return ok, (1.0 if ok else 0.0)
    if task_id.startswith('bcb'):
        # `gold` carries the task's unittest suite; correctness is whether the
        # generated task_func passes it, executed in a sandboxed container.
        from _bcb_judge import judge_bigcodebench
        ok = judge_bigcodebench(pred, gold)
        return ok, (1.0 if ok else 0.0)
    if task_id.startswith('swe'):
        return judge_swe(pred, gold)
    if task_id.startswith('math'):
        ok = judge_math(pred, gold)
        return ok, (1.0 if ok else 0.0)
    if task_id.startswith('bbeh'):
        from _bbeh_judge import judge_bbeh
        ok = judge_bbeh(pred, gold)
        return ok, (1.0 if ok else 0.0)
    if task_id.startswith(('musique', 'frames', 'fin')):
        return em_f1(pred, gold)
    ok = judge_gaia(pred, gold)
    return ok, (1.0 if ok else 0.0)


class TokenBankV2:
    """Adaptive Runtime Budget Governance Middleware."""
    def __init__(self, total_budget: int, policy='hard', external_plan=None,
                 gamma=0.8, rebalance_threshold=0.8, enable_predictive=True,
                 enable_borrowing=True, enable_opl=True, enable_kv=True,
                 min_quota=5000, anomaly_sigma=3.0, shared_opl=None,
                 shared_credit_scorer=None, rogue_cutoff=0.10,
                 credit_rate=0.5, enable_pacing=False,
                 pacing_upfront=0.5, pacing_k=3, pacing_max_rounds=5,
                 adaptive_runaway=False, runaway_token_fraction=0.35,
                 runaway_rounds=1, runaway_min_tokens=0,
                 max_rounds=None, force_final_on_block=False,
                 final_reserve=0, role_credit_mult=None, predictive_est=False,
                 stall_sim=None, runaway_frac_auto=False,
                 predictive_lending=False, kv_trigger=None,
                 exec_compact=False, capture_logprobs=False,
                 real_summary=False, summary_budget=512, accumulate_context=False,
                 context_window=8192):
        self.total_budget = total_budget
        self.reserve = total_budget
        self.policy = policy
        self.gamma = gamma
        self.theta = rebalance_threshold
        self.q_min = min_quota
        self.anomaly_sigma = anomaly_sigma
        self.adapter = BudgetAdapter(external_plan)
        self.predictor = PredictiveRebalancer(horizon=2) if enable_predictive else None
        self.borrowing = BudgetBorrowing(
            credit_rate=credit_rate, scorer=shared_credit_scorer,
            rogue_cutoff=rogue_cutoff, role_credit_mult=role_credit_mult
        ) if enable_borrowing else None
        self.credit_scorer = shared_credit_scorer
        self.rogue_cutoff = rogue_cutoff
        # shared_opl: an OnlinePriorityLearner shared across tasks so priority
        # learning accumulates beyond a single task (else cold-start never ends)
        self.opl = shared_opl
        self.enable_opl = enable_opl
        self.kv_tracker = KVCacheTracker(
            trigger_ratio=kv_trigger,
            real_summary=bool(real_summary),
            summary_budget=int(summary_budget),
            context_window=int(context_window),
        ) if (enable_kv and kv_trigger) else (KVCacheTracker() if enable_kv else None)
        # Budget pacing (soft segmentation): the first `pacing_upfront` fraction
        # of the budget is available upfront; the rest is split into `pacing_k`
        # segments that are released only when (a) >=80% of the released pool
        # is consumed and (b) a Reviewer checkpoint has completed since the
        # last release. Global ceiling independent of per-agent quota/borrowing.
        self.pacing = enable_pacing
        self.pacing_k = pacing_k
        self.pacing_max_rounds = pacing_max_rounds
        # Adaptive runaway guard: in this mode pacing/stall are dormant until
        # the task is both stuck (answer unchanged across rounds) AND has burned
        # a significant fraction of the budget. Normal converging tasks keep the
        # full budget; only runaway tasks get clamped.
        self.adaptive_runaway = adaptive_runaway
        self.runaway_token_fraction = runaway_token_fraction
        self.runaway_rounds = runaway_rounds
        # Absolute floor for the runaway trigger: the guard fires only after
        # max(frac * budget, runaway_min_tokens) tokens are burned, so very
        # tight budgets (where frac * budget is tiny) do not over-kill tasks.
        self.runaway_min_tokens = runaway_min_tokens
        # Baseline hooks: hard round cap (EarlyStop-N) and forced final answer
        # when a call is quota-blocked (BudgetForcing).
        self.max_rounds = max_rounds
        self.force_final_on_block = force_final_on_block
        # --- v3 improvements (all opt-in, default = legacy behavior) ---
        # ① final_reserve: tokens held back from normal dispatch/borrowing so a
        #   task that hits its limits can always afford one closing answer call.
        self.final_reserve = final_reserve
        # ③ predictive_est: LLMClientV2 estimates output length from per-role
        #   history (P80) instead of worst-case max_tokens.
        self.predictive_est = predictive_est
        # ④ stall_sim: token-set Jaccard threshold for "answer unchanged"
        #   (None = exact string match, legacy); runaway_frac_auto scales the
        #   runaway token fraction by budget tier (tighter budget -> earlier).
        self.stall_sim = stall_sim
        if runaway_frac_auto:
            runaway_token_fraction = (0.15 if total_budget <= 8000
                                      else 0.20 if total_budget <= 16000 else 0.25)
        # ⑤ predictive_lending: pick lenders by forecast surplus instead of
        #   static priority ordering.
        self.predictive_lending = predictive_lending
        # ⑥ kv_trigger: earlier KV-compression threshold (default 0.85 of
        #   quota). Lower values compress the prompt before the estimate can
        #   breach quota, instead of rescuing after a block.
        # ⑦ exec_compact: GAIA runner keeps only the latest answer+feedback
        #   pair in the executor context (and truncates tool output) so the
        #   executor prompt stops growing quadratically across rounds.
        self.exec_compact = exec_compact
        self.accumulate_context = accumulate_context
        # v5 M1 signal source: capture per-token logprobs/entropy on real calls
        # (adds no tokens, slight serving overhead; enable for diagnostic runs)
        self.capture_logprobs = capture_logprobs
        self.runaway_active = False
        self._runaway_counter = 0
        self._last_answer = None
        if self.adaptive_runaway:
            # start with full budget; runaway detection will tighten later
            self.released = total_budget
        else:
            self.released = int(total_budget * pacing_upfront)
        # ceil so the final release tops up to exactly total_budget
        self.segment_size = -(-(total_budget - self.released) // self.pacing_k)
        self._reviews_since_release = 0
        self.pacing_releases = 0
        self.ledger = {}
        self.rebalance_count = 0
        self.borrow_count = 0
        self.anomaly_log = []
        self.trajectory = []
        self._velocity_window = defaultdict(list)

    def total_used(self):
        return sum(a.used for a in self.ledger.values())

    def progress_update(self, answer: str, approved: bool, made_progress: bool = False):
        """Runner reports per-round outcome. Adaptive runaway guard activates
        only when the task is stuck (answer unchanged) AND has burned enough
        budget. Approved tasks or tasks that are still changing answers keep
        the full budget. `made_progress=True` (e.g. the round executed a code
        tool) counts as progress even if the textual answer is unchanged."""
        if not self.adaptive_runaway:
            return
        if approved or made_progress:
            self.runaway_active = False if approved else self.runaway_active
            self._runaway_counter = 0
            self._last_answer = None
            return
        norm = ' '.join(answer.lower().split()) if answer else ''
        # ④ similarity-based stuck detection: near-identical answers (token-set
        # Jaccard >= stall_sim) count as "unchanged" even if punctuation or a
        # few words differ. Legacy: exact string match.
        unchanged = False
        if self._last_answer is not None:
            if self.stall_sim is not None:
                unchanged = _answer_similarity(norm, self._last_answer) >= self.stall_sim
            else:
                unchanged = norm == self._last_answer
        if unchanged:
            self._runaway_counter += 1
        else:
            self._runaway_counter = 0
            self._last_answer = norm
        if (self._runaway_counter >= self.runaway_rounds and
                self.total_used() >= max(self.runaway_token_fraction * self.total_budget,
                                         self.runaway_min_tokens)):
            if not self.runaway_active:
                self.runaway_active = True
                # tighten the global ceiling to current burn + 30% of remaining
                self.released = min(
                    self.total_budget,
                    self.total_used() + int((self.total_budget - self.total_used()) * 0.3)
                )
                remaining = self.total_budget - self.released
                self.segment_size = -(-remaining // self.pacing_k) if remaining > 0 else 0
                self._log_event('RUNAWAY_GUARD', {
                    'released': self.released,
                    'counter': self._runaway_counter,
                    'token_fraction': self.total_used() / max(self.total_budget, 1),
                })

    def pacing_release_check(self, action):
        """Called after every successful consume(). Releases the next pacing
        segment when the released pool is >=80% consumed AND a Reviewer
        checkpoint happened since the last release."""
        if not self.pacing or self.released >= self.total_budget:
            return
        if action == 'review':
            self._reviews_since_release += 1
        if (self.total_used() >= 0.8 * self.released
                and self._reviews_since_release >= 1):
            self.released = min(self.total_budget, self.released + self.segment_size)
            self.pacing_releases += 1
            self._reviews_since_release = 0
            self._log_event('PACING_RELEASE', {'released': self.released})

    def register_agents(self, role_map, priorities=None):
        for aid, role in role_map.items():
            p = priorities.get(role, 2) if priorities else 2
            self.ledger[aid] = AgentAccountV2(agent_id=aid, role=role, priority=p)
        if self.enable_opl and self.opl is None:
            self.opl = OnlinePriorityLearner(list(set(role_map.values())))

    def allocate(self, priority_weight=True):
        n = len(self.ledger)
        if n == 0:
            return
        role_map = {aid: acc.role for aid, acc in self.ledger.items()}
        plan_quotas = self.adapter.get_initial_quotas(self.total_budget, n, role_map)
        if self.opl and self.opl.task_count >= self.opl.cold_start:
            learned_p = self.opl.get_priorities()
        else:
            learned_p = {acc.role: acc.priority for acc in self.ledger.values()}

        # Option C: initial allocation uses role priority only, not credit score.
        total_p = sum(learned_p.get(acc.role, 2) for acc in self.ledger.values())
        available = self.total_budget - n * self.q_min
        runtime_quotas = {}
        for aid, acc in self.ledger.items():
            p = learned_p.get(acc.role, 2)
            runtime_quotas[aid] = max(self.q_min, int(p * available / total_p))
        for aid in self.ledger:
            q_plan = plan_quotas.get(aid, self.total_budget // n)
            q_runtime = runtime_quotas.get(aid, self.total_budget // n)
            self.ledger[aid].quota = int(self.gamma * q_plan + (1 - self.gamma) * q_runtime)
        self._log_event('ALLOCATE', {k: self.ledger[k].quota for k in self.ledger})

    def can_dispatch(self, agent_id, estimated_tokens=0, is_final=False):
        if agent_id not in self.ledger:
            return False, 'Agent not registered'
        # ① Final-answer calls (closing summary after a block) bypass quota,
        # pacing and borrowing gates — only the global reserve governs them,
        # and they are the only calls allowed to spend the final_reserve.
        if is_final:
            if estimated_tokens > self.reserve:
                return False, f'Global reserve depleted: {self.reserve}'
            return True, 'OK (final)'
        # Pacing gate: global ceiling on task-level consumption, independent of
        # per-agent quota and borrowing. For adaptive runaway mode the gate is
        # dormant until the task is flagged as runaway; normal converging tasks
        # keep the full budget.
        pacing_active = self.pacing and (not self.adaptive_runaway or self.runaway_active)
        if pacing_active and self.total_used() + estimated_tokens > self.released:
            return False, f'Pacing gate: {self.total_used()}/{self.released} awaiting checkpoint'
        acc = self.ledger[agent_id]
        grace = 1.0
        if self.policy == 'grace':
            grace = {3: 1.10, 2: 1.05, 1: 1.00}.get(acc.priority, 1.0)
        effective_quota = int((acc.quota + acc.borrowed) * grace)
        if acc.used + estimated_tokens > effective_quota:
            if self.borrowing and self._try_borrow(agent_id, estimated_tokens - acc.remaining):
                return True, 'OK (after borrowing)'
            return False, f'Quota exceeded: {acc.used}/{effective_quota}'
        if estimated_tokens > self.reserve - self.final_reserve:
            return False, f'Global reserve depleted: {self.reserve}'
        return True, 'OK'

    def _try_borrow(self, borrower_id, amount):
        if not self.borrowing or amount <= 0:
            return False
        borrower = self.ledger[borrower_id]
        candidates = [a for a in self.ledger.values()
                      if a.agent_id != borrower_id and a.remaining > amount]
        # Option C: normal lending decisions are not gated by credit score.
        # ⑤ predictive lending: order lenders by forecast surplus (remaining
        # quota minus predicted near-term consumption) so credit comes from the
        # agent least likely to need it; legacy: static priority then remaining.
        if self.predictive_lending and self.predictor:
            def _surplus(a):
                exp, _ = self.predictor.forecast(a.consumption_log)
                return a.remaining - exp
            candidates.sort(key=lambda a: (-_surplus(a), a.priority))
        else:
            candidates.sort(key=lambda a: (a.priority, -a.remaining))
        for lender in candidates:
            if self.borrowing.can_borrow(borrower, amount, lender, self.reserve):
                result = self.borrowing.execute_loan(borrower, lender, amount)
                self.borrow_count += 1
                self._log_event('BORROW', result)
                return True
        return False

    def consume(self, agent_id, tokens_in, tokens_out, model=PRIMARY_MODEL,
                action='inference', kv_cache=0):
        if agent_id not in self.ledger:
            return {'allowed': False, 'reason': 'Agent not registered'}
        acc = self.ledger[agent_id]
        acc.kv_cache_tokens += kv_cache
        if self.kv_tracker:
            total_tokens = self.kv_tracker.effective_tokens(tokens_in, acc.kv_cache_tokens) + tokens_out
        else:
            total_tokens = tokens_in + tokens_out
        cost = self._compute_cost(tokens_in, tokens_out, model)
        if self.policy == 'throttle' and acc.utilization > 0.9:
            model = THROTTLE_MODEL
            cost = self._compute_cost(tokens_in, tokens_out, model)
            self._log_event('THROTTLE', {'agent': agent_id, 'new_model': model})
        grace = 1.0
        if self.policy == 'grace':
            grace = {3: 1.10, 2: 1.05, 1: 1.00}.get(acc.priority, 1.0)
        effective_quota = int((acc.quota + acc.borrowed) * grace)
        if acc.used + total_tokens > effective_quota:
            allowed_tokens = max(0, effective_quota - acc.used)
            if self.policy == 'hard' or allowed_tokens == 0:
                return {'allowed': False, 'actual_in': 0, 'actual_out': 0,
                        'reason': f'Quota exceeded: {acc.used}/{effective_quota}'}
            total_tokens = allowed_tokens
            tokens_out = min(tokens_out, allowed_tokens)
            tokens_in = allowed_tokens - tokens_out
        if total_tokens > self.reserve:
            return {'allowed': False, 'actual_in': 0, 'actual_out': 0,
                    'reason': f'Global reserve depleted: {self.reserve}'}
        acc.used += total_tokens
        self.reserve -= total_tokens
        acc.consumption_log.append(total_tokens)
        record = {'timestamp': datetime.now().isoformat(), 'tokens_in': tokens_in,
                  'tokens_out': tokens_out, 'kv_cache': kv_cache, 'model': model,
                  'cost_usd': cost, 'action': action}
        acc.history.append(record)
        self._log_event('CONSUME', {'agent': agent_id, 'tokens': total_tokens, 'cost': cost})
        self._check_anomaly(agent_id, total_tokens)
        return {'allowed': True, 'actual_in': tokens_in, 'actual_out': tokens_out,
                'model': model, 'cost_usd': cost, 'reason': 'OK'}

    def predictive_rebalance(self):
        if not self.predictor:
            return {'performed': False, 'reason': 'Disabled'}
        agents = list(self.ledger.values())
        should, forecasts = self.predictor.should_rebalance(agents, self.reserve, self.theta)
        if not should:
            return {'performed': False, 'reason': 'Opportunity cost too low', 'forecasts': forecasts}
        surplus = [a for a in agents if forecasts[a.agent_id]['predicted_surplus'] > 0 and a.utilization < 0.5]
        deficit = [a for a in agents if forecasts[a.agent_id]['expected'] > self.theta * a.remaining]
        if not surplus or not deficit:
            return {'performed': False, 'reason': 'No matching pairs', 'forecasts': forecasts}
        reclaimable = int(sum((a.quota - a.used) * 0.3 for a in surplus))
        if reclaimable <= 0:
            return {'performed': False, 'reason': 'No reclaimable quota'}
        # Option C: rebalancing uses role priority only, not credit score.
        deficit.sort(key=lambda a: a.priority, reverse=True)
        total_p = sum(a.priority for a in deficit) or 1
        transfers = []
        for a in deficit:
            share = int(reclaimable * (a.priority / total_p))
            if share > 0:
                a.quota += share
                transfers.append({'to': a.agent_id, 'amount': share})
        total_surplus_rem = sum(a.remaining for a in surplus) or 1
        for a in surplus:
            deduction = int(reclaimable * (a.remaining / total_surplus_rem))
            a.quota = max(self.q_min, a.quota - deduction)
        self.rebalance_count += 1
        self._log_event('PREDICTIVE_REBALANCE', {'transfers': transfers, 'total': reclaimable})
        return {'performed': True, 'transfers': transfers, 'total': reclaimable, 'forecasts': forecasts}

    def update_opl(self, task_result):
        if not self.opl:
            return
        role_success, role_blocked = {}, {}
        for call in task_result.get('agent_calls', []):
            aid = call.get('agent_id', '')
            if aid in self.ledger:
                role = self.ledger[aid].role
                role_success[role] = call.get('success', False)
                role_blocked[role] = 'budget' in str(call.get('block_reason', '')).lower()
        self.opl.update(role_success, role_blocked)

    def update_credit(self, task_result):
        """Record credit repayment: an agent that borrowed and saw the task
        succeed gets a positive record; borrowed-but-failed is a default."""
        if not self.credit_scorer:
            return
        for acc in self.ledger.values():
            if acc.borrowed > 0:
                self.credit_scorer.record(acc.role, bool(task_result.get('success')))

    def _check_anomaly(self, agent_id, tokens):
        now = time.time()
        window = self._velocity_window[agent_id]
        window.append((now, tokens))
        window[:] = [(t, tok) for t, tok in window if now - t < 60]
        if len(window) < 3:
            return
        velocities = [tok for _, tok in window]
        mean_v, std_v = np.mean(velocities), np.std(velocities)
        if std_v > 0 and tokens > mean_v + self.anomaly_sigma * std_v:
            self.anomaly_log.append({'agent': agent_id, 'timestamp': now,
                                      'tokens': tokens, 'mean_v': mean_v, 'std_v': std_v})
            self._log_event('ANOMALY', {'agent': agent_id, 'tokens': tokens})

    def _compute_cost(self, tokens_in, tokens_out, model):
        p = PRICING.get(model, PRICING[PRIMARY_MODEL])
        return (tokens_in * p['input'] + tokens_out * p['output']) / 1e6

    def _log_event(self, event_type, data):
        self.trajectory.append({'type': event_type, 'time': time.time(), 'data': data, 'reserve': self.reserve})

    def get_summary(self):
        total_used = sum(a.used for a in self.ledger.values())
        total_cost = sum(a.total_cost for a in self.ledger.values())
        total_borrowed = sum(a.borrowed for a in self.ledger.values())
        return {
            'total_budget': self.total_budget, 'reserve': self.reserve,
            'total_used': total_used, 'total_cost_usd': round(total_cost, 4),
            'total_borrowed': total_borrowed, 'rebalance_count': self.rebalance_count,
            'borrow_count': self.borrow_count, 'anomaly_count': len(self.anomaly_log),
            'summarization_count': self.kv_tracker.summarization_count if self.kv_tracker else 0,
            'pacing_releases': self.pacing_releases,
            'runaway_active': self.runaway_active,
            'agents': {k: {'quota': v.quota, 'used': v.used, 'borrowed': v.borrowed,
                          'utilization': round(v.utilization, 3), 'cost_usd': round(v.total_cost, 4)}
                     for k, v in self.ledger.items()}
        }

print('TokenBankV2 defined successfully.')


# ============ NOTEBOOK CELL 8 ============
from openai import OpenAI


SEGMENT_ACCOUNTING = os.getenv('SEGMENT_ACCOUNTING') == '1'


def _segment_prompt(messages):
    """Attribute every character of a prompt to what it is carrying.

    The paper's motivation rests on how much of each token the pipeline
    processes is re-transmitted coordination state rather than the task. Role
    attribution ('which agent was called') cannot answer that, because the plan
    and the reviewer's critique are re-sent inside the *executor's* own prompt
    every round and are billed to the executor.

    Parsing is by the exact format strings this file constructs, so it is exact
    rather than heuristic. Anything unrecognised lands in 'other' and is
    reported, so a silent mis-attribution shows up as a large 'other'.
    """
    seg = collections.Counter()
    for m in messages:
        c = m.get('content') or ''
        role = m.get('role')
        if role == 'system':
            seg['system'] += len(c); continue
        if role == 'assistant':
            seg['prior_answer'] += len(c); continue
        rest = c
        # "Question: <q>\nPlan: <plan>\nFinal answer."
        if rest.startswith('Question: '):
            body = rest[len('Question: '):]
            if '\nPlan: ' in body:
                q, tail = body.split('\nPlan: ', 1)
                seg['question'] += len(q)
                plan = tail.rsplit('\nFinal answer.', 1)[0]
                seg['plan'] += len(plan)
                seg['scaffold'] += len(rest) - len(q) - len(plan)
            else:
                seg['question'] += len(body); seg['scaffold'] += len('Question: ')
            continue
        if rest.startswith('Reviewer feedback: '):
            crit = rest[len('Reviewer feedback: '):].rsplit('\nRevise', 1)[0]
            seg['critique'] += len(crit); seg['scaffold'] += len(rest) - len(crit)
            continue
        if rest.startswith('[Computation result]') or rest.startswith('[Computation note]'):
            seg['tool_output'] += len(rest)
            continue
        if rest.startswith('Proposed: ') or 'Correct? Explain' in rest:
            seg['prior_answer'] += len(rest); continue
        seg['other'] += len(rest)
    return seg


class LLMClientV2:
    def __init__(self, api_key, base_url):
        if not MOCK_MODE:
            # Long thinking calls can outlast the SDK's 600 s default under load, and the SDK then
            # resends the whole generation; LLM_TIMEOUT / LLM_SDK_RETRIES set both per run.
            # OPENAI_BASE_URLS (comma-separated) spreads one run's calls over several servers, each call
            # going to the one with the fewest calls in flight, so no GPU idles while another still has a
            # queue (a run bound to one server kept its GPU busy long after the others had finished).
            urls = [u.strip() for u in os.getenv('OPENAI_BASE_URLS', '').split(',') if u.strip()] or [base_url]
            self.clients = [OpenAI(api_key=api_key, base_url=u,
                                   timeout=float(os.getenv('LLM_TIMEOUT', '600')),
                                   max_retries=int(os.getenv('LLM_SDK_RETRIES', '2'))) for u in urls]
            self.client = self.clients[0]
            self._inflight = [0] * len(self.clients)
            self._lb_lock = threading.Lock()
        else:
            self.client = None
        self.call_count = 0
        # ③ per-role output-length history for predictive dispatch estimates
        self.out_hist = defaultdict(list)

    def call(self, agent_id, bank, messages, model=PRIMARY_MODEL, max_tokens=MAX_TOKENS,
             temperature=TEMPERATURE, action='inference', kv_cache=0, final=False,
             extra_body=None):
        # KV-cache summarization BEFORE dispatch: compression shrinks the prompt,
        # which may bring the dispatch estimate back under quota. (Previously this
        # ran after can_dispatch, so an over-quota estimate blocked the call
        # before compression ever had a chance.)
        if bank.kv_tracker and agent_id in bank.ledger and bank.kv_tracker.should_summarize(
                sum(len(ENCODER.encode(m.get('content', '') or '')) for m in messages)):
            messages, _, kv_cache = bank.kv_tracker.summarize_context(
                messages, self, bank=bank, agent_id=agent_id)
            if VERBOSE:
                print(f'  [SUMMARIZE] {agent_id}: KV-cache compressed')

        prompt_text = '\n'.join([m.get('content', '') for m in messages])
        role = bank.ledger[agent_id].role if agent_id in bank.ledger else 'Executor'
        # Runtime per-call cap under a fixed total budget. A fixed cap is right for one budget
        # and wrong for another: large caps avoid truncation when the budget is loose but make
        # each round so expensive under a tight one that the task is stopped before it
        # answers. Here every non-final call gets the budget it can afford now: what remains
        # after this call's prompt and the closing reserve, divided by runtime_cap_k (the calls
        # that should still fit, about one round), clipped to [runtime_cap_lo, runtime_cap_hi].
        # Static split (ZEBRA-style baseline): before execution the budget is divided across
        # roles in fixed shares, and each call's cap is its role's share divided by the number
        # of calls that role is expected to make. Shares are set per model from measured role
        # costs, which makes the baseline stronger than an estimated split.
        _ss = getattr(bank, 'static_split', None)
        if _ss and not final and agent_id in _ss:
            _calls = float(getattr(bank, 'static_calls', {}).get(agent_id, 1))
            _cap = _ss[agent_id] * (bank.total_budget - getattr(bank, 'final_reserve', 0)) / _calls
            max_tokens = int(max(256, min(4096, _cap)))
        if getattr(bank, 'runtime_cap', False) and not final:
            _pt = len(ENCODER.encode(prompt_text))
            _avail = bank.reserve - getattr(bank, 'final_reserve', 0) - _pt
            _k = float(getattr(bank, 'runtime_cap_k', 3))
            max_tokens = int(max(getattr(bank, 'runtime_cap_lo', 512),
                                 min(getattr(bank, 'runtime_cap_hi', 4096), _avail / _k)))
        if MOCK_MODE:
            mean, std = MOCK_TOKEN_PROFILES.get(role, (10000, 3000))
            estimated_in = max(50, int(np.random.normal(mean * 0.5, std * 0.15)))
            estimated_out = max(1, int(np.random.normal(mean * 0.9, std * 0.3)))
            # Simulate KV-cache accumulation: the prompt context stays resident
            kv_cache = estimated_in
        else:
            estimated_in = len(ENCODER.encode(prompt_text))
            # ③ predictive estimate: P80 of this role's observed output lengths
            # instead of worst-case max_tokens (cold start falls back to
            # max_tokens). Tighter estimates -> fewer false blocks/borrows.
            hist = self.out_hist.get(role, [])
            if getattr(bank, 'predictive_est', False) and hist:
                estimated_out = max(64, int(np.percentile(hist[-30:], 80)))
            else:
                estimated_out = max_tokens
            # Multi-turn conversations keep prior context resident in KV cache
            if kv_cache == 0 and len(messages) > 2:
                kv_cache = estimated_in
        # Keep the dispatch estimate consistent with consume()'s accounting:
        # when KV tracking is on, consumption is charged as in + 0.3*kv + out.
        if bank.kv_tracker:
            pending_kv = bank.ledger[agent_id].kv_cache_tokens + kv_cache if agent_id in bank.ledger else kv_cache
            estimated_total = bank.kv_tracker.effective_tokens(estimated_in, pending_kv) + estimated_out
        else:
            estimated_total = estimated_in + estimated_out

        can_run, reason = bank.can_dispatch(agent_id, estimated_total, is_final=final)
        if not can_run:
            if VERBOSE:
                print(f'  [BLOCKED] {agent_id}: {reason}')
            return {'success': False, 'content': '', 'tokens_in': 0, 'tokens_out': 0,
                    'block_reason': reason, 'model': model, 'agent_id': agent_id, 'final': final}

        if MOCK_MODE:
            actual_out = estimated_out
            content = f'[MOCK_RESPONSE_{self.call_count}_{role}]'
            time.sleep(0.02)
            lp_stats = {}
        else:
            # A pinned upstream provider rate-limits under load, and with
            # fallbacks off there is nowhere for the call to go. Returning
            # failure on the first 429 silently drops that round from the task,
            # which reads downstream as a model that had nothing to say. These
            # are transient by definition, so retry them with backoff; anything
            # that is not transient still fails on the first attempt.
            _TRANSIENT = ('429', '500', '502', '503', '504', 'timeout',
                          'Timeout', 'timed out', 'temporarily', 'Connection', 'no choices')
            _tries = int(os.getenv('LLM_MAX_RETRIES', '4'))
            for _attempt in range(_tries + 1):
              try:
                xbody = dict(LLM_EXTRA_BODY or {})
                # Per-call overrides. The ladder uses this to buy a cheaper rung
                # on the generation side, which is the term reasoning inflates and
                # the carried-slot bounds do not touch.
                if extra_body:
                    xbody.update(extra_body)
                # LOGPROBS_L01_ONLY=1 requests log-probabilities only on the calls that produce the
                # cheap rungs' scores (marked '_want_lp'), which keeps responses small when the
                # pipeline's long reasoning would otherwise ship top-5 alternatives for every token.
                want_lp = getattr(bank, 'capture_logprobs', False) and (
                    os.getenv('LOGPROBS_L01_ONLY') != '1' or bool(xbody.get('_want_lp')))
                xbody.pop('_want_lp', None)
                _no_tools = bool(xbody.pop('_no_tools', False))
                if want_lp:
                    xbody['logprobs'] = True
                    # TOP_LOGPROBS=0 drops the top-5 alternatives (used only for the entropy gate),
                    # which multiplies the response size on a slow link.
                    _k = int(os.getenv('TOP_LOGPROBS', '5'))
                    if _k > 0:
                        xbody['top_logprobs'] = _k
                # DECLARE_PY_TOOL=1 (gpt-oss): declare a python tool on executor calls. These models
                # route computation through their built-in tool call rather than a ```python block
                # in the reply, and an undeclared call leaves the reply empty; a returned call is
                # turned back into a ```python block below, so the usual execute-and-follow-up runs.
                _tkw = {}
                if os.getenv('DECLARE_PY_TOOL') == '1' and str(agent_id).startswith('executor') and not _no_tools:
                    _tkw['tools'] = [{'type': 'function', 'function': {
                        'name': 'python', 'description': 'Run Python code and return what it prints.',
                        'parameters': {'type': 'object', 'properties': {'code': {'type': 'string'}},
                                       'required': ['code']}}}]
                with self._lb_lock:
                    _ci = min(range(len(self.clients)), key=self._inflight.__getitem__)
                    self._inflight[_ci] += 1
                try:
                    resp = self.clients[_ci].chat.completions.create(model=model, messages=messages,
                                                                max_tokens=max_tokens, temperature=temperature,
                                                                extra_body=xbody or None, **_tkw)
                finally:
                    with self._lb_lock:
                        self._inflight[_ci] -= 1
                # A gateway can answer 200 with choices=null when the upstream
                # hiccups. Indexing it raises a TypeError that reads nothing
                # like a network fault, so name it and let the retry take it.
                if not getattr(resp, 'choices', None):
                    raise RuntimeError('upstream returned no choices')
                content = resp.choices[0].message.content or ''
                _tc = getattr(resp.choices[0].message, 'tool_calls', None)
                if _tc and not content.strip():
                    _codes = []
                    for _c in _tc:
                        _a = getattr(getattr(_c, 'function', None), 'arguments', '') or ''
                        try:
                            _a = json.loads(_a).get('code', _a) if _a.strip().startswith('{') else _a
                        except Exception:
                            pass
                        if str(_a).strip():
                            _codes.append(str(_a))
                    if _codes:
                        content = '\n'.join(f'```python\n{_x}\n```' for _x in _codes)
                actual_out = resp.usage.completion_tokens if resp.usage else len(ENCODER.encode(content))
                self.out_hist[role].append(actual_out)
                lp_stats = {}
                if want_lp:
                    lp = getattr(resp.choices[0], 'logprobs', None)
                    toks = getattr(lp, 'content', None) if lp else None
                    if toks:
                        lps = [t.logprob for t in toks if t.logprob is not None]
                        ents = []
                        for t in toks:
                            tl = getattr(t, 'top_logprobs', None) or []
                            ps = [math.exp(x.logprob) for x in tl if x.logprob is not None]
                            if ps:
                                ents.append(-sum(p * math.log(max(p, 1e-12)) for p in ps))
                        lp_stats = {'mean_logprob': (sum(lps) / len(lps)) if lps else None,
                                    'entropy': (sum(ents) / len(ents)) if ents else None}
                break
              except Exception as e:
                _msg = str(e)
                _last = _attempt >= _tries
                if not _last and any(t in _msg for t in _TRANSIENT):
                    time.sleep(min(2 ** _attempt, 16) * (0.5 + random.random()))
                    continue
                # Log it. This was silent, and a malformed-message 400 therefore
                # looked identical to a model that had nothing to say.
                print(f'  [APIERR] {agent_id}: {_msg[:160]}', flush=True)
                return {'success': False, 'content': '', 'error': _msg, 'agent_id': agent_id,
                        'tokens_in': 0, 'tokens_out': 0, 'model': model, 'final': final}

        actual_in = estimated_in if MOCK_MODE else len(ENCODER.encode(prompt_text))
        if SEGMENT_ACCOUNTING and bank is not None:
            try:
                acc = getattr(bank, 'segment_chars', None)
                if acc is None:
                    acc = bank.segment_chars = collections.Counter()
                acc.update(_segment_prompt(messages))
                acc['_completion'] += len(content or '')
            except Exception:
                pass  # accounting must never take down a run
        result = bank.consume(agent_id, actual_in, actual_out, model=model, action=action, kv_cache=kv_cache)
        if result['allowed']:
            bank.pacing_release_check(action)
        if not result['allowed']:
            content = content[:int(len(content) * result.get('actual_out', 0) / max(actual_out, 1))] if actual_out > 0 else ''
        self.call_count += 1
        if VERBOSE and result['allowed']:
            print(f'  [OK] {agent_id}: {actual_in}+{actual_out} tokens, cost=${result["cost_usd"]:.4f}')
        out = {
            'success': result['allowed'], 'content': content, 'agent_id': agent_id,
            'tokens_in': result.get('actual_in', actual_in), 'tokens_out': result.get('actual_out', actual_out),
            'cost_usd': result.get('cost_usd', 0), 'model': result.get('model', model),
            'block_reason': result['reason'] if not result['allowed'] else None,
            'final': final
        }
        out.update(lp_stats)
        return out

llm = LLMClientV2(OPENAI_API_KEY, OPENAI_BASE_URL)
print(f'LLMClientV2 ready. Mock={MOCK_MODE}')


# ============ NOTEBOOK CELL 9 ============
def _snap(content, limit=int(os.getenv('SNAPSHOT_CHARS', '20000'))):
    """The stored reply that the offline judges read. It used to be the first 2,000 characters,
    which cut the closing ANSWER line off long replies (a quarter to half of Gemma's cheap-rung
    replies); longer replies keep their head and their tail."""
    c = content or ''
    return c if len(c) <= limit else c[:limit // 4] + '\n[...]\n' + c[-(limit - limit // 4):]


_CODE_CLOSE_ASK = 'Give the final answer now: the complete solution code in one ```python block, with no other text.'
_CLOSE_ASK = 'Do not write any more code. Give the final answer now, ending with the line ANSWER: <answer>.'


def _answer_missing(content: str, task_id) -> bool:
    """True when a reply carries no final answer: empty, or code with neither an ANSWER line nor a
    boxed answer. On code-generation tasks the code is the answer, so it never counts as missing."""
    if str(task_id).startswith(('bcb_', 'lcb_', 'mbpp_')):
        return '```python' not in (content or '')    # the code is the answer: missing when no code block
    c = content or ''
    if not c.strip():
        return True
    return '```python' in c and 'ANSWER:' not in c and '\\boxed' not in c


def _tool_loop(r, msgs, call, final_ask, task_id, max_exec=3, shape_out=None, no_tool_body=None):
    """Execute the reply's python block and ask for the final answer, up to max_exec times while the
    model keeps answering with code; then, if the reply still has no answer, one closing call with
    the tool withheld. Models that route computation through a tool often answer the follow-up with
    another tool call, and a single execution left that code as the final answer.
    Returns (reply, last tool output or None, ok)."""
    tool_out = None
    for _ in range(max_exec):
        out = _try_run_code(r['content'])
        if out is None or not _answer_missing(r['content'], task_id) and _ != 0:
            break
        tool_out = shape_out(out) if shape_out else out
        msgs.append({'role': 'assistant', 'content': r['content']})
        msgs.append({'role': 'user', 'content': f'[Computation result]\n{tool_out}\n{final_ask}'})
        r2 = call(msgs, None)
        if not r2['success']:
            return r2, tool_out, False
        r = r2
    if _answer_missing(r['content'], task_id):
        msgs.append({'role': 'assistant', 'content': r['content']})
        code_task = str(task_id).startswith(('bcb_', 'lcb_', 'mbpp_'))
        msgs.append({'role': 'user', 'content': _CODE_CLOSE_ASK if code_task else _CLOSE_ASK})
        # A code task's answer is code, so its closing call keeps the tool declared. The closing call answers
        # without thinking: a thinking model that filled the call cap would otherwise fill it again.
        _nothink = {'chat_template_kwargs': {'thinking': False, 'enable_thinking': False}}
        if os.getenv('OR_REASONING_SWITCH') == '1':
            _nothink['reasoning'] = {'enabled': False}
        r2 = call(msgs, _nothink if code_task else {**(no_tool_body or {}), '_no_tools': True, **_nothink})
        if r2['success'] and (r2['content'] or '').strip():
            r = r2
    return r, tool_out, True


def _try_run_code(content: str, timeout: int = 10) -> Optional[str]:
    """If the Executor emitted a ```python block, run it in a subprocess
    (isolated, time-limited, dangerous operations rejected) and return its
    output. Returns None when there is no code block."""
    m = re.findall(r'```python\s*\n(.*?)```', content, re.DOTALL)
    if not m:
        return None
    code = m[0]
    if re.search(r'(import\s+os|import\s+sys|import\s+subprocess|__import__|open\s*\(|eval\s*\(|exec\s*\()', code):
        return '(blocked: restricted operation)'
    try:
        p = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=timeout)
        out = (p.stdout + p.stderr).strip()
        return out[:1000] if out else '(no output)'
    except Exception as e:
        return f'(execution failed: {e})'


def _forced_answer(bank, llm_client, messages, max_tokens=256):
    """Budget-forcing hook: when a call is quota-blocked, grant a one-time
    quota/release bump so the executor can still emit a final answer (s1-style
    budget forcing adapted to the multi-agent harness)."""
    acc = bank.ledger.get('executor')
    if acc is not None:
        acc.quota += max_tokens + 2048
    if getattr(bank, 'pacing', False):
        bank.released += max_tokens + 2048
    return llm_client.call('executor', bank, messages, action='execute',
                           max_tokens=max_tokens)


def run_gaia_task_v2(task, bank, llm_client):
    results = {'task_id': task['id'], 'agent_calls': [], 'success': False, 'answer': '',
               'stalled': False, 'rounds': [], 'f1': 0.0}
    t0 = time.time()
    stopped_by = 'budget'  # any early return below means the budget ran out
    agents = {'planner': 'Planner', 'executor': 'Executor', 'reviewer': 'Reviewer'}
    bank.register_agents(agents)
    bank.allocate()
    force_final = getattr(bank, 'force_final_on_block', False)
    final_reserve = getattr(bank, 'final_reserve', 0)
    stall_sim = getattr(bank, 'stall_sim', None)
    # ⑦ exec_compact: keep only the latest answer+feedback in executor context
    exec_compact = getattr(bank, 'exec_compact', False)

    def _finalize():
        # v5 Appendix B instrumentation: per-task flags + stop reason + timing
        results['stopped_by'] = stopped_by
        results['wall_clock_ms'] = int((time.time() - t0) * 1000)
        # Per-role token accounting. The `rounds` list only records executor
        # turns, so planner and reviewer spend is invisible there; this is the
        # only place the split that Figure 1 reports can be recovered.
        _rt = {}
        for _c in results['agent_calls']:
            _a = _c.get('agent_id')
            if _a:
                _rt[_a] = _rt.get(_a, 0) + _c.get('tokens_in', 0) + _c.get('tokens_out', 0)
        results['role_tokens'] = _rt
        if SEGMENT_ACCOUNTING:
            results['segment_chars'] = dict(getattr(bank, 'segment_chars', {}) or {})
        if os.getenv('DUMP_RAW_ANSWER') == '1':
            # SWE-oracle needs the WHOLE reply: the answer is a set of code
            # regions, which _extract_answer would reduce to its last line and
            # answer_snapshot would cut at 2000 chars.
            results['raw_answer'] = results.get('answer', '')
        results['flags_fired'] = {
            'borrowing': getattr(bank, 'borrow_count', 0),
            'final_reserve': sum(1 for c in results['agent_calls'] if c.get('final')),
            'stall': 1 if results.get('stalled') else 0,
            'kv_trigger': bank.kv_tracker.summarization_count if bank.kv_tracker else 0,
            'role_credit_mult_effective': (getattr(bank, 'borrow_count', 0)
                                           if getattr(bank.borrowing, 'role_credit_mult', None) else 0),
        }
        return results

    def _record_round(rnd, role, r, tool_out=None):
        # per-round snapshot for D1 (first-correct-round / flip-rate / wasted tokens)
        ok, _f = (False, 0.0) if MOCK_MODE else judge_task(task['id'], r.get('content', ''), task['answer'])
        results['rounds'].append({
            'round': rnd, 'role': role,
            'prompt_tokens': r.get('tokens_in', 0),
            'completion_tokens': r.get('tokens_out', 0),
            'tool_output_tokens': len(ENCODER.encode(tool_out)) if tool_out else 0,
            'answer_snapshot': _snap(r.get('content')),
            'is_correct': ok,
            'mean_logprob': r.get('mean_logprob'),
            'entropy': r.get('entropy'),
        })

    r1 = llm_client.call('planner', bank, [
        {'role': 'system', 'content': 'You are a Planner. Break down the question into sub-tasks.'},
        {'role': 'user', 'content': f"Question: {task['question']}\nProvide a step-by-step plan."}
    ], action='plan')
    results['agent_calls'].append(r1)
    if not r1['success']:
        if force_final:
            g = _forced_answer(bank, llm_client, [
                {'role': 'system', 'content': 'You are an Executor. Answer concisely.'},
                {'role': 'user', 'content': f"Question: {task['question']}\n"
                                           f"Budget exhausted. Give your best final answer now."}
            ])
            results['agent_calls'].append(g)
            if g['success']:
                results['answer'] = g['content']
                if not MOCK_MODE:
                    results['success'], results['f1'] = judge_task(task['id'], g['content'], task['answer'])
        elif final_reserve:
            # ① the reserve pays for one closing answer call (no quota bump
            # needed — final calls bypass quota and draw the held-back tokens)
            g = llm_client.call('executor', bank, [
                {'role': 'system', 'content': 'You are an Executor. Answer concisely.'},
                {'role': 'user', 'content': f"Question: {task['question']}\n"
                                           f"Budget exhausted. Give your best final answer now."}
            ], action='execute', max_tokens=256, final=True)
            results['agent_calls'].append(g)
            if g['success']:
                results['answer'] = g['content']
                if not MOCK_MODE:
                    results['success'], results['f1'] = judge_task(task['id'], g['content'], task['answer'])
        bank.update_opl(results)
        return _finalize()

    # Multi-round execute/review loop (up to 5 rounds) with a code-execution
    # tool: a ```python block in the Executor's reply is run and the result fed
    # back, so arithmetic/multi-hop questions get an extra competent shot.
    exec_messages = [
        {'role': 'system', 'content': 'You are an Executor. Answer concisely. '
         'For ANY computation, put Python code that prints the result in a ```python block; '
         'you will receive the output, then give the final answer.'},
        {'role': 'user', 'content': f"Question: {task['question']}\nPlan: {r1['content']}\nFinal answer."}
    ]
    # Paced runs get a progress gate: an early stop when a revision leaves the
    # answer unchanged (task is stuck, further segments would be wasted). Round
    # count stays configurable — a hard round cap costs SR because some tasks
    # genuinely converge in rounds 4-5; stall detection cuts only tasks that
    # stopped making progress.
    paced = getattr(bank, 'pacing', False)
    max_rounds = getattr(bank, 'pacing_max_rounds', 5) if paced else 5
    # EarlyStop-N baseline: a hard round cap overrides everything else.
    hard_cap = getattr(bank, 'max_rounds', None)
    if hard_cap:
        max_rounds = hard_cap
    r2, r3 = None, None
    prev_answer = None
    prev_score = None
    for rnd in range(max_rounds):
        r2 = llm_client.call('executor', bank, exec_messages, action='execute')
        results['agent_calls'].append(r2)
        if not r2['success']:
            if force_final:
                g = _forced_answer(bank, llm_client, exec_messages + [
                    {'role': 'user', 'content': 'Budget exhausted. Give your best final answer now, concisely.'}
                ])
                results['agent_calls'].append(g)
                if g['success']:
                    r2 = g
            elif final_reserve:
                g = llm_client.call('executor', bank, exec_messages + [
                    {'role': 'user', 'content': 'Budget exhausted. Give your best final answer now, concisely.'}
                ], action='execute', max_tokens=256, final=True)
                results['agent_calls'].append(g)
                if g['success']:
                    r2 = g
            bank.update_opl(results)
            if r2 is None or not r2['success']:
                return _finalize()
            break

        used_tool = False
        tool_out = None
        if not MOCK_MODE:
            def _callA(m, xb):
                rr = llm_client.call('executor', bank, m, action='execute', **({'extra_body': xb} if xb else {}))
                results['agent_calls'].append(rr)
                return rr
            _shape = (lambda o: o[:800] + '\n[... truncated]' if len(o) > 800 else o) if exec_compact else None
            r2, tool_out, _ok = _tool_loop(r2, exec_messages, _callA, 'Now give the final answer.', task['id'], shape_out=_shape)
            used_tool = tool_out is not None
            if not _ok:
                bank.update_opl(results)
                return _finalize()
        _record_round(rnd, 'executor', r2, tool_out if used_tool else None)

        review_user = (f"Question: {task['question']}\nProposed: {r2['content']}\nCorrect? YES/NO")
        if REVIEWER_SCORE:
            review_user += ("\nAlso rate how close the proposed answer is to fully correct "
                            "on a 0-10 scale. Reply exactly: 'Score: N/10' then YES/NO.")
        r3 = llm_client.call('reviewer', bank, [
            {'role': 'system', 'content': 'You are a Reviewer. Validate the answer.'},
            {'role': 'user', 'content': review_user}
        ], action='review')
        results['agent_calls'].append(r3)
        bank.predictive_rebalance()

        if MOCK_MODE:
            approved = bool(np.random.random() < 0.6)
        else:
            approved = r3['success'] and 'yes' in r3['content'].lower()
        score = None
        if REVIEWER_SCORE and r3['success']:
            m = re.search(r'(\d+(?:\.\d+)?)\s*/\s*10', r3['content'])
            if m:
                score = float(m.group(1))
        score_improved = (score is not None and prev_score is not None
                          and score > prev_score)
        cur_answer = ' '.join(r2['content'].lower().split())
        bank.progress_update(r2['content'], approved,
                             made_progress=(used_tool and STALL_EXEMPT_TOOL) or score_improved)
        if approved or rnd == max_rounds - 1:
            stopped_by = 'converged' if approved else 'max_rounds'
            break
        # Stall detection (paced only): in adaptive runaway mode, stop only
        # after the runaway guard has fired (stuck + significant budget burned).
        # In plain paced mode keep the original behavior. A round that used the
        # code tool counts as progress and is exempt from stall. With
        # REVIEWER_SCORE on, an improving closeness score also counts as
        # progress even when the answer text is unchanged.
        adaptive = getattr(bank, 'adaptive_runaway', False)
        stalled = False
        # ④ similarity-based unchanged test (falls back to exact match)
        if prev_answer is not None:
            unchanged = (_answer_similarity(cur_answer, prev_answer) >= stall_sim
                         if stall_sim is not None else cur_answer == prev_answer)
        else:
            unchanged = False
        if (paced and prev_answer is not None and unchanged
                and not (used_tool and STALL_EXEMPT_TOOL)
                and not score_improved):
            stalled = bank.runaway_active if adaptive else True
        if stalled:
            if VERBOSE:
                print('  [PACING] stall detected: answer unchanged, stopping early')
            results['stalled'] = True
            stopped_by = 'stall'
            # Grace answer: one cheap final-answer call before giving up, so a
            # stalled task still submits its best current answer instead of the
            # stale draft it was stuck on.
            if GRACE_ANSWER:
                g = llm_client.call('executor', bank, exec_messages + [
                    {'role': 'assistant', 'content': r2['content']},
                    {'role': 'user', 'content': 'Stop revising. Based on everything above, '
                                                'give your single best final answer now, concisely.'}
                ], action='execute', max_tokens=256)
                results['agent_calls'].append(g)
                if g['success'] and g['content'].strip():
                    r2 = g
                    if VERBOSE:
                        print('  [GRACE] final answer extracted after stall')
            break
        prev_answer = cur_answer
        if score is not None:
            prev_score = score
        if exec_compact:
            # ⑦ drop all prior rounds; keep system + question/plan + the latest
            # answer and reviewer feedback. The executor prompt then stays
            # ~constant across rounds instead of growing by ~2 messages/round.
            exec_messages = exec_messages[:2] + [
                {'role': 'assistant', 'content': r2['content']},
                {'role': 'user', 'content': f"Reviewer feedback: {r3['content']}\nRevise and give the final answer."}
            ]
        else:
            exec_messages.append({'role': 'assistant', 'content': r2['content']})
            exec_messages.append({'role': 'user', 'content': f"Reviewer feedback: {r3['content']}\nRevise and give the final answer."})

    if MOCK_MODE:
        # Mock 模式无法真实作答：按流水线完整性模拟答案正确性，
        # 有调用被预算拦截/截断时准确率显著降低
        blocked = any(not c['success'] for c in results['agent_calls'])
        acc = 0.30 if blocked else 0.85
        results['success'] = bool(np.random.random() < acc)
    else:
        results['success'], results['f1'] = judge_task(task['id'], r2['content'], task['answer'])
    results['answer'] = r2['content']
    bank.update_opl(results)
    return _finalize()


# ==================== v5 Phase 2: mechanism redesign ====================
# M0: dead governance mechanisms (borrowing / role_credit / predictive_est /
# predictive_lending / stall) are simply NOT enabled by v5 configs — D2 showed
# 0% trigger rates. final_reserve is kept (correctness guarantee, not an
# optimization).
# M1 convergence early stop: stability + structured verdict + logprob, with
# entropy override and a hard round-5 cap.
# M2 cascade routing: L0 direct answer -> L1 executor+tool -> L2 full pipeline,
# escalating on low confidence; lower-level output is reused as draft.
# M3 context economics: structured reviewer / tool-output notes / plan freeze /
# ANSWER-line output discipline.
# M5 depth-vs-breadth: strategy in {refine, best_of_n, hybrid}.

def _majority_vote(answers):
    """Self-consistency vote on normalized extracted answers."""
    from collections import Counter as _C
    norm = [_norm_basic(_extract_answer(a)) for a in answers]
    if not any(norm):
        return answers[0]
    best = _C(norm).most_common(1)[0][0]
    for a, n in zip(answers, norm):
        if n == best:
            return a
    return answers[0]


def _v5_verdict(content):
    """M3.1: parse structured reviewer verdict; fall back to legacy YES/NO."""
    m = re.search(r'"verdict"\s*:\s*"?\s*(accept|revise)', content or '', re.I)
    if m:
        return m.group(1).lower() == 'accept'
    return _review_verdict(content)


_REV_NEG = re.compile(r"(?i)\b(incorrect|not correct|isn't correct|is wrong|are wrong|wrong answer|inaccurate|"
                      r"not accurate|invalid|does not answer|doesn't answer|fails to)\b")
_REV_POS = re.compile(r"(?i)\b(correct|right|accurate|valid)\b")


def _review_verdict(content):
    """The reviewer's decision on a plain-text review. It used to be "'yes' in the reply", which read
    "The answer is correct." as a rejection and every empty review as one too, sending correct answers
    back to be rewritten (most of Gemma's reviews). Now: the last "VERDICT: YES/NO" line the prompt asks
    for; failing that, a leading yes/no; failing that, the review's wording (a negative phrase rejects,
    otherwise a positive one accepts). An empty review, usually one whose thinking filled the call cap,
    carries no objection and keeps the answer."""
    c = (content or '').strip()
    if not c:
        return True
    v = re.findall(r'(?i)verdict\W{0,6}(yes|no)\b', c)
    if v:
        return v[-1].lower() == 'yes'
    m = re.match(r'(?i)\W*(yes|no)\b', c)
    if m:
        return m.group(1).lower() == 'yes'
    head = c[:400]
    if _REV_NEG.search(head):
        return False
    return bool(_REV_POS.search(head)) or 'yes' in head.lower()


def run_gaia_task_v5(task, bank, llm_client):
    results = {'task_id': task['id'], 'agent_calls': [], 'success': False, 'answer': '',
               'stalled': False, 'rounds': [], 'f1': 0.0, 'escalation_level': 0}
    t0 = time.time()
    stopped_by = 'budget'
    agents = {'planner': 'Planner', 'executor': 'Executor', 'reviewer': 'Reviewer'}
    bank.register_agents(agents)
    bank.allocate()
    final_reserve = getattr(bank, 'final_reserve', 0)

    # mechanism flags (defaults = all off -> plain fixed pipeline)
    m1 = getattr(bank, 'm1_early_stop', False)
    tau_conf = getattr(bank, 'tau_conf', -0.10)
    tau_high = getattr(bank, 'tau_high', 0.6)
    m2 = getattr(bank, 'm2_cascade', False)
    theta0 = getattr(bank, 'theta0', -0.15)
    theta1 = getattr(bank, 'theta1', -0.10)
    m3_rev = getattr(bank, 'm3_structured_reviewer', False)
    m3_notes = getattr(bank, 'm3_tool_notes', False)
    m3_plan = getattr(bank, 'm3_plan_freeze', False)
    m3_ans = getattr(bank, 'm3_answer_format', False)
    m5 = getattr(bank, 'm5_strategy', 'refine')
    # v5.1 unified cascade: verbalized escalation channel + roomier L0/L1
    m2_verbal = getattr(bank, 'm2_verbal', False)
    l01_max_tokens = int(os.getenv('L01_MAX_TOKENS') or getattr(bank, 'l01_max_tokens', 768))
    # Reasoning as a rung of the ladder. With a thinking model the term that grows
    # is the per-call generation, not the carried prefix, so bounding the slots
    # leaves the binding constraint untouched. Giving the cheap levels a cheaper
    # reasoning setting puts that term under the same admission decision as the
    # scaffolding: a task that the gate keeps at L0 also gets the cheap setting,
    # and a task that escalates gets the default one.
    l01_reasoning = getattr(bank, 'l01_reasoning_effort', None)
    _l01_xb = {'reasoning_effort': l01_reasoning} if l01_reasoning else None
    # Some servers name the graded settings but do not act on them: on the Qwen deployment
    # low, medium and xhigh all produce 524 to 605 reasoning tokens on the same 40 problems,
    # where on GLM low cuts the per-call median from 61 to 15. A hard cap is honoured there
    # instead, so the cheap rung is expressed as a token budget when one is configured. It
    # also fixes a failure the graded setting does not: without a cap, 6 of 40 problems
    # spend the whole output allowance thinking and never write an answer.
    _l01_tbudget = getattr(bank, 'l01_thinking_budget', None)
    if _l01_tbudget:
        _l01_xb = {**(_l01_xb or {}), 'thinking_token_budget': int(_l01_tbudget)}
    # A third form of the same rung, for servers that expose reasoning as a switch on the
    # chat template rather than as a setting. The hard token budget is a mid-thought cut, so
    # what it costs in accuracy is truncation rather than shorter reasoning; turning the
    # thought off means the model never starts one and the answer is written directly. On
    # the DeepSeek deployment the switch is the only knob that binds: a 128-token budget is
    # honoured exactly and the switch yields 0 reasoning tokens, while low, medium and high
    # all sit near the 328-token default.
    #
    # The two must not be sent together. Passing reasoning_effort alongside the switch
    # re-enables the thought on that server (265 tokens, as though the switch were absent),
    # so a configuration that sets the switch drops the graded setting rather than stacking
    # them and silently governing nothing.
    if getattr(bank, 'l01_thinking_off', False):
        _l01_xb = {k: v for k, v in (_l01_xb or {}).items() if k != 'reasoning_effort'}
        # 'thinking' is DeepSeek's template switch, 'enable_thinking' Gemma's and Qwen's; a template
        # ignores keys it does not use, so sending both turns the thought off on either.
        _l01_xb['chat_template_kwargs'] = {'thinking': False, 'enable_thinking': False}
        # Through OpenRouter the template switch is not forwarded; its unified reasoning field is.
        if os.getenv('OR_REASONING_SWITCH') == '1':
            _l01_xb['reasoning'] = {'enabled': False}
    # Reasoning governed by the allowance rather than by the level.
    #
    # With l01_reasoning_effort the reasoning spend is a constant chosen per level, which
    # governs it in the weak sense that a cheap level gets a cheap setting. It still does not
    # respond to how much budget is actually left, so the generation term of the per-round
    # cost stays fixed while every other term is bounded. reason_budget_frac makes it a
    # function of the allowance: each call may spend that fraction of what remains, clamped,
    # so the same policy reasons freely early and tightens as the budget is consumed. The
    # server enforces it as a cap on the trace, so a task that needs less simply uses less.
    _rb_frac = getattr(bank, 'reason_budget_frac', None)
    _rb_lo = int(getattr(bank, 'reason_budget_min', 128))
    _rb_hi = int(getattr(bank, 'reason_budget_max', 4096))

    def _reason_xb(base):
        if not _rb_frac:
            return base
        # bank.reserve is the live remaining allowance and is what can_dispatch checks
        # against, so the reasoning cap tracks the same quantity the admission does rather
        # than a separately reconstructed one.
        remaining = getattr(bank, 'reserve', None)
        if remaining is None:
            return base
        tb = int(max(_rb_lo, min(_rb_hi, remaining * _rb_frac)))
        out = dict(base or {})
        out['thinking_budget'] = tb
        return out
    start_level = task.get('_v5_start_level', 0 if m2 else 2)
    # V5_MAX_LEVEL=1 ends a forced climb after L1: escalation runs the reference from scratch, which is
    # recorded by its own run, so the draft-seeded pipeline (L2) is not needed for the records.
    max_level = int(os.getenv('V5_MAX_LEVEL') or getattr(bank, 'v5_max_level', 2))
    draft = task.get('_v5_draft')  # M4 pass-2 continuation
    early_stopped = [False]

    def _finalize():
        results['stopped_by'] = stopped_by
        results['wall_clock_ms'] = int((time.time() - t0) * 1000)
        # Per-role token accounting. The `rounds` list only records executor
        # turns, so planner and reviewer spend is invisible there; this is the
        # only place the split that Figure 1 reports can be recovered.
        _rt = {}
        for _c in results['agent_calls']:
            _a = _c.get('agent_id')
            if _a:
                _rt[_a] = _rt.get(_a, 0) + _c.get('tokens_in', 0) + _c.get('tokens_out', 0)
        results['role_tokens'] = _rt
        if SEGMENT_ACCOUNTING:
            results['segment_chars'] = dict(getattr(bank, 'segment_chars', {}) or {})
        if os.getenv('DUMP_RAW_ANSWER') == '1':
            # SWE-oracle needs the WHOLE reply: the answer is a set of code
            # regions, which _extract_answer would reduce to its last line and
            # answer_snapshot would cut at 2000 chars.
            results['raw_answer'] = results.get('answer', '')
        results['flags_fired'] = {
            'borrowing': getattr(bank, 'borrow_count', 0),
            'final_reserve': sum(1 for c in results['agent_calls'] if c.get('final')),
            'stall': 1 if results.get('stalled') else 0,
            'kv_trigger': bank.kv_tracker.summarization_count if bank.kv_tracker else 0,
            'm1_early_stop': 1 if early_stopped[0] else 0,
        }
        return results

    def _record_round(rnd, role, r, tool_out=None):
        ok, _f = (False, 0.0) if MOCK_MODE else judge_task(task['id'], r.get('content', ''), task['answer'])
        results['rounds'].append({
            'round': rnd, 'role': role,
            'prompt_tokens': r.get('tokens_in', 0),
            'completion_tokens': r.get('tokens_out', 0),
            'tool_output_tokens': len(ENCODER.encode(tool_out)) if tool_out else 0,
            'answer_snapshot': _snap(r.get('content')),
            'is_correct': ok,
            'mean_logprob': r.get('mean_logprob'),
            'entropy': r.get('entropy'),
            'reserve_after': getattr(bank, 'reserve', None),
        })

    ans_instr = (' End your reply with a single line: "ANSWER: <final answer>".'
                 if m3_ans else '')

    def _confident(r, theta):
        lp, ent = r.get('mean_logprob'), r.get('entropy')
        if lp is None:
            return False  # no signal -> escalate (safe default)
        return lp > theta and (ent is None or ent <= tau_high)

    def _grace_close(msgs):
        """final_reserve: one cheap closing answer when budget is gone."""
        if not final_reserve:
            return None
        g = llm_client.call('executor', bank, msgs + [
            {'role': 'user', 'content': 'Budget exhausted. Give your best final answer now, concisely.'}
        ], action='execute', max_tokens=256, final=True)
        results['agent_calls'].append(g)
        return g if g['success'] else None

    def _exec_call(msgs, temperature=None, max_tokens=None, extra_body=None):
        kw = {'temperature': temperature} if temperature is not None else {}
        if max_tokens is not None:
            kw['max_tokens'] = max_tokens
        if extra_body:
            kw['extra_body'] = extra_body
        r = llm_client.call('executor', bank, msgs, action='execute', **kw)
        results['agent_calls'].append(r)
        return r

    # Optional temperature for the cheap rungs alone (e.g. greedy L0/L1 while the pipeline samples
    # with a model's recommended settings, as Gemma's thinking mode loops under greedy decoding).
    _l01_temp = float(os.environ['L01_TEMPERATURE']) if os.environ.get('L01_TEMPERATURE') else None

    def _direct(draft_ans=None, with_tool=True, temperature=None, reason=False):
        """L0/L1 primitive: one executor call, optionally with code tool. With reason=True (L1_MODE=reason)
        the rung is a single agent at the pipeline's own reasoning setting, sampling and call cap, answering
        from scratch: no draft, no cheap-rung reasoning setting (graded effort or thinking switch)."""
        if reason:
            draft_ans = None
            _rxb, _rtemp, _rmax = {'_want_lp': True}, None, None
        else:
            _rxb, _rtemp, _rmax = {**(_reason_xb(_l01_xb) or {}), '_want_lp': True}, _l01_temp, l01_max_tokens
        if temperature is None:
            temperature = _rtemp
        sys_c = 'You are an Executor. Answer concisely.' + ans_instr
        if with_tool:
            sys_c += (' For ANY computation, put Python code that prints the result '
                      'in a ```python block; you will receive the output, then give the final answer.')
        user_c = f"Question: {task['question']}\n"
        if draft_ans:
            user_c += f"Draft answer: {draft_ans}\nVerify it, fix it if wrong, then answer.\n"
        if m2_verbal and not draft_ans:
            # v5.1: let the model DECLINE to direct-answer. This is the unified
            # routing key — logprob can't tell "knows the answer" from "needs
            # to compute/look up" (MATH failure mode), but the model can say so.
            user_c += ('If this question requires computation, multi-step reasoning, '
                       'or information you are not sure of, reply with exactly '
                       '"ESCALATE" as the first word instead of answering.\n')
        user_c += 'Final answer.'
        msgs = [{'role': 'system', 'content': sys_c}, {'role': 'user', 'content': user_c}]
        r = _exec_call(msgs, temperature=temperature, max_tokens=_rmax, extra_body=dict(_rxb))
        if not r['success']:
            return r, None
        tool_out = None
        # On code-generation tasks the answer is the code itself: after running it, a plain "final
        # answer" request makes some models describe the result in prose, which no test can run.
        _final_ask = ('Now give the final answer: the complete function in one ```python block.'
                      if str(task['id']).startswith('bcb_') else
                      'Now give the final answer: the complete solution in one ```python block.'
                      if str(task['id']).startswith(('lcb_', 'mbpp_')) else 'Now give the final answer.')
        if with_tool and not MOCK_MODE:
            def _shapeC(o):
                if not m3_notes:
                    return o
                nr = _exec_call([
                    {'role': 'system', 'content': 'Summarize this computation result as a note of at most 120 tokens. Facts and numbers only.'},
                    {'role': 'user', 'content': o[:4000]}], temperature=_rtemp, max_tokens=150)
                return nr['content'] if nr['success'] and nr['content'].strip() else o[:800]
            def _callC(m, xb):
                return _exec_call(m, temperature=_rtemp, max_tokens=_rmax, extra_body={**_rxb, **(xb or {})})
            r2, tool_out, _ok = _tool_loop(r, msgs, _callC, _final_ask, task['id'], shape_out=_shapeC)
            if _ok:
                r = r2
        return r, tool_out

    def _best_of_n(draft_ans=None, n=3):
        """M5 breadth: n short samples @ temp 0.7 + majority vote."""
        cands, recs = [], []
        for _ in range(n):
            r, _t = _direct(draft_ans=draft_ans, with_tool=False, temperature=0.7)
            if r['success'] and r['content'].strip():
                cands.append(r['content'])
                recs.append(r)
        if not recs:
            return {'success': False, 'content': ''}
        win = _majority_vote(cands) if cands else recs[-1]['content']
        for r in recs:
            if r['content'] == win:
                return r
        return recs[-1]

    # ---------- L0: direct answer (M5 strategy applies here) ----------
    cur = None
    if start_level <= 0:
        if m5 == 'best_of_n':
            r0 = _best_of_n()
        else:
            # l0_no_tool makes L0 a single call with nothing around it, which is what the
            # single-call baseline needs: with the tool on, L0 can be two calls.
            r0, _ = _direct(with_tool=not getattr(bank, 'l0_no_tool', False))
        if not r0['success']:
            g = _grace_close([{'role': 'system', 'content': 'You are an Executor. Answer concisely.'},
                              {'role': 'user', 'content': f"Question: {task['question']}"}])
            if g:
                results['answer'] = g['content']
                if not MOCK_MODE:
                    results['success'], results['f1'] = judge_task(task['id'], g['content'], task['answer'])
            bank.update_opl(results)
            return _finalize()
        _record_round(0, 'executor', r0)
        cur = r0
        results['escalation_level'] = 0
        # v5.1: verbalized ESCALATE overrides the logprob gate (model says it
        # needs compute/lookup even when it would look confident)
        verbal_esc = m2_verbal and r0['content'].strip().upper().startswith('ESCALATE')
        if verbal_esc:
            cur = None  # draft is a non-answer; don't propagate it
        if max_level <= 0:
            stopped_by = 'max_level'  # M4 probe cap: NOT converged, still pending
        elif _confident(r0, theta0) and not verbal_esc:
            stopped_by = 'converged'
        else:
            stopped_by = ''
        if stopped_by:
            results['answer'] = r0['content']
            if not MOCK_MODE:
                results['success'], results['f1'] = judge_task(task['id'], r0['content'], task['answer'])
            bank.update_opl(results)
            return _finalize()

    # ---------- L1: executor + tool, single round, reuses L0 draft ----------
    if start_level <= 1:
        r1 = None
        if m5 == 'hybrid' and cur is not None and (cur.get('entropy') or 0) > tau_high:
            r1 = _best_of_n(draft_ans=cur['content'])  # high entropy -> breadth
        else:
            r1, _t = _direct(draft_ans=cur['content'] if cur else draft, with_tool=True,
                             reason=os.getenv('L1_MODE') == 'reason')
        if not r1['success']:
            g = _grace_close([{'role': 'system', 'content': 'You are an Executor. Answer concisely.'},
                              {'role': 'user', 'content': f"Question: {task['question']}"}])
            fin = g if g else cur
            if fin:
                results['answer'] = fin['content']
                if not MOCK_MODE:
                    results['success'], results['f1'] = judge_task(task['id'], fin['content'], task['answer'])
            bank.update_opl(results)
            return _finalize()
        _record_round(1, 'executor', r1)
        cur = r1
        results['escalation_level'] = 1
        verbal_esc1 = m2_verbal and r1['content'].strip().upper().startswith('ESCALATE')
        if verbal_esc1:
            cur = None
        if max_level <= 1:
            stopped_by = 'max_level'  # M4 probe cap: NOT converged, still pending
        elif _confident(r1, theta1) and not verbal_esc1:
            stopped_by = 'converged'
        else:
            stopped_by = ''
        if stopped_by:
            results['answer'] = r1['content']
            if not MOCK_MODE:
                results['success'], results['f1'] = judge_task(task['id'], r1['content'], task['answer'])
            bank.update_opl(results)
            return _finalize()

    # ---------- L2, alternative team: multi-agent debate (TEAM=debate) ----------
    # Du et al. (2024): DEBATE_AGENTS agents answer independently at the pipeline's setting, then for
    # DEBATE_ROUNDS rounds each sees the others' latest replies and updates its own; the final answer is
    # the majority of the last replies (ties to the first agent). Every agent call works like the single
    # agent L1: the code tool, up to three executions, and a closing call when a reply has no answer.
    # Each agent's reply in each round is recorded as an executor round (round index = debate round).
    if os.getenv('TEAM') == 'debate':
        results['escalation_level'] = 2
        n_ag = int(os.getenv('DEBATE_AGENTS', '3')); n_rd = int(os.getenv('DEBATE_ROUNDS', '2'))
        sys_d = ('You are an Executor. Answer concisely.' + ans_instr +
                 ' For ANY computation, put Python code that prints the result in a ```python block; '
                 'you will receive the output, then give the final answer.')

        def _agent(user_c):
            msgs = [{'role': 'system', 'content': sys_d}, {'role': 'user', 'content': user_c}]
            r = _exec_call(msgs)
            if r['success'] and not MOCK_MODE:
                def _callD(m, xb):
                    return _exec_call(m, extra_body=xb or None)
                r2, _t, _ok = _tool_loop(r, msgs, _callD, 'Now give the final answer.', task['id'])
                if _ok:
                    r = r2
            return r

        q = f"Question: {task['question']}\n"
        replies = []
        for a in range(n_ag):
            r = _agent(q + 'Final answer.')
            _record_round(0, 'executor', r)
            replies.append(r['content'] if r['success'] else '')
        for rd in range(1, n_rd + 1):
            new = []
            for a in range(n_ag):
                others = ''.join(f"\nAgent {j + 1}'s solution:\n{replies[j][-4000:]}\n"
                                 for j in range(n_ag) if j != a and replies[j])
                r = _agent(q + 'These are solutions to the question from other agents:' + others +
                           '\nUsing their reasoning as additional information, give your updated solution. Final answer.')
                _record_round(rd, 'executor', r)
                new.append(r['content'] if r['success'] else replies[a])
            replies = new
        live = [c for c in replies if (c or '').strip()]
        results['answer'] = _majority_vote(live) if live else ''
        # The voted answer as a last, token-free executor round: the offline judges and the MATH rescorer
        # read the last executor snapshot as the team's final answer.
        _record_round(n_rd + 1, 'executor', {'content': results['answer'], 'tokens_in': 0, 'tokens_out': 0})
        if not MOCK_MODE and results['answer']:
            results['success'], results['f1'] = judge_task(task['id'], results['answer'], task['answer'])
        stopped_by = 'debate'
        bank.update_opl(results)
        return _finalize()

    # ---------- L2: full Plan -> Execute -> Review loop with M1 early stop ----------
    results['escalation_level'] = 2
    plan_txt = ''
    if not (draft and start_level >= 2):
        rp = llm_client.call('planner', bank, [
            {'role': 'system', 'content': 'You are a Planner. Break down the question into sub-tasks.'},
            {'role': 'user', 'content': f"Question: {task['question']}\nProvide a step-by-step plan."}
        ], action='plan')
        results['agent_calls'].append(rp)
        if rp['success']:
            plan_txt = rp['content']
        elif not _grace_close([{'role': 'system', 'content': 'You are an Executor. Answer concisely.'},
                               {'role': 'user', 'content': f"Question: {task['question']}"}]):
            bank.update_opl(results)
            return _finalize()
    draft_txt = (cur['content'] if cur else draft) or ''
    exec_messages = [
        {'role': 'system', 'content': 'You are an Executor. Answer concisely. '
         'For ANY computation, put Python code that prints the result in a ```python block; '
         'you will receive the output, then give the final answer.' + ans_instr},
        {'role': 'user', 'content': f"Question: {task['question']}\n"
         + (f"Plan: {plan_txt}\n" if plan_txt else '')
         + ((f"A quick first attempt by another solver (it may be wrong): {draft_txt}\n"
             "Check it independently. If it is correct, keep it; otherwise solve the problem and "
             "give the corrected answer.\n") if draft_txt and os.getenv('SEED_PROMPT') == 'verify'
            else (f"Draft answer: {draft_txt}\n" if draft_txt else ''))
         + 'Final answer.'}
    ]
    # Drop scaffolding lines so the bounded-plan slot carries a step rather than
    # a heading. Taking plan_lines[1] verbatim kept a bare heading
    # ("**Step-by-Step Plan:**", "Here is the step-by-step plan:") on four of six
    # sampled repair plans against one of six computation plans, because repair
    # plans more often restate the problem before the list. The slot was
    # therefore empty of content on exactly the task type where the plan matters
    # most, which is an indexing defect rather than a property of plans.
    _BOILER = re.compile(r'^(here (is|are) the .{0,30}|step[- ]by[- ]step plan|'
                         r'analysis( of the (bug|issue|problem))?|plan|steps?|'
                         r'proposed (fix|plan)|the plan)$', re.I)
    def _is_scaffold(l):
        t = re.sub(r'^[#>\s]*|[*_`]', '', l).strip().rstrip(':').strip()
        t = re.sub(r'^\d+[.)]\s*', '', t)
        return bool(_BOILER.match(t))
    plan_lines = [l.strip() for l in plan_txt.split('\n') if l.strip()]
    _content = [l for l in plan_lines if not _is_scaffold(l)]
    if _content:
        plan_lines = _content
    # Main experiments use 5. The long-horizon compaction comparison raises
    # this so trajectories reach the scale at which compaction is designed to
    # operate; the cap is a property of the run, not of the method.
    max_rounds = int(getattr(bank, 'max_rounds_v5', 0) or os.getenv('V5_MAX_ROUNDS', '5'))
    prev_norm = None
    r2 = None
    _hist = []
    _summary = ['']
    # tokens of the parts every round carries regardless of state policy (system prompt and
    # question), so that per-round carried state = prompt_tokens - fixed_prompt_tokens
    results['fixed_prompt_tokens'] = len(ENCODER.encode(exec_messages[0]['content'] + f"Question: {task['question']}"))
    for rnd in range(max_rounds):
        r2 = _exec_call(exec_messages)
        if not r2['success']:
            g = _grace_close(exec_messages)
            if g:
                r2 = g
            else:
                bank.update_opl(results)
                return _finalize()
            break
        tool_out = None
        if not MOCK_MODE:
            def _shapeB(o):
                if not m3_notes:
                    return o
                nr = _exec_call([
                    {'role': 'system', 'content': 'Summarize this computation result as a note of at most 120 tokens. Facts and numbers only.'},
                    {'role': 'user', 'content': o[:4000]}], max_tokens=150)
                return nr['content'] if nr['success'] and nr['content'].strip() else o[:800]
            r2, tool_out, _ok = _tool_loop(r2, exec_messages, lambda m, xb: _exec_call(m, extra_body=xb),
                                           'Now give the final answer.', task['id'], shape_out=_shapeB)
            if not _ok:
                bank.update_opl(results)
                return _finalize()
        _record_round(rnd, 'executor', r2, tool_out)

        # Reviewer: M3.1 structured verdict or legacy YES/NO
        if m3_rev:
            r3 = llm_client.call('reviewer', bank, [
                {'role': 'system', 'content': 'You are a Reviewer. Reply ONLY with JSON: '
                 '{"verdict": "accept" or "revise", "issues": ["<=3 items, <=20 words each"], '
                 '"hint": "<=30 words"}. No prose.'},
                {'role': 'user', 'content': f"Question: {task['question']}\nProposed: {r2['content']}"}
            ], action='review', max_tokens=int(getattr(bank, 'review_max_tokens', 150)))
        else:
            r3 = llm_client.call('reviewer', bank, [
                {'role': 'system', 'content': 'You are a Reviewer. Validate the answer.'},
                {'role': 'user', 'content': f"Question: {task['question']}\nProposed: {r2['content']}\n"
                 "Is the proposed answer correct? End your reply with a single line: VERDICT: YES or VERDICT: NO."}
            ], action='review')
            # A review whose reasoning filled the call cap comes back empty. Like the executor's closing
            # call, ask once more for the verdict alone with reasoning at its lowest setting.
            if r3['success'] and not (r3.get('content') or '').strip() and not MOCK_MODE:
                _rv_low = ({'reasoning_effort': 'low'} if 'gpt-oss' in str(os.getenv('VLLM_MODEL', '')) else
                           {'chat_template_kwargs': {'thinking': False, 'enable_thinking': False}})
                if os.getenv('OR_REASONING_SWITCH') == '1':
                    _rv_low['reasoning'] = {'enabled': False}
                r3b = llm_client.call('reviewer', bank, [
                    {'role': 'system', 'content': 'You are a Reviewer. Validate the answer.'},
                    {'role': 'user', 'content': f"Question: {task['question']}\nProposed: {r2['content']}\n"
                     "Is the proposed answer correct? Reply with a single line: VERDICT: YES or VERDICT: NO."}
                ], action='review', extra_body=_rv_low)
                results['agent_calls'].append(r3)
                if r3b['success']:
                    r3 = r3b
        results['agent_calls'].append(r3)
        approved = bool(np.random.random() < 0.6) if MOCK_MODE else (r3['success'] and _v5_verdict(r3['content']))
        # An empty review is not a verdict. On a reasoning model it usually means the review
        # was truncated before any verdict was written, and reading it as "revise" sends correct
        # answers back to be rewritten. With empty_review_keep it is read as "keep the answer".
        if getattr(bank, 'empty_review_keep', False) and r3['success'] and not (r3.get('content') or '').strip():
            approved = True
        # Record the reviewer's decision against the round it judged, so that its
        # discrimination (accepting right answers, rejecting wrong ones) can be measured
        # directly rather than inferred from whether the loop continued.
        if results['rounds']:
            results['rounds'][-1]['review_accept'] = bool(approved)
            results['rounds'][-1]['review_ok'] = bool(r3['success'])
            results['rounds'][-1]['review_snippet'] = _snap(r3.get('content') or '', 1200)

        # M1 three-signal stop: stability / verdict gated by logprob confidence,
        # entropy override, hard cap at round 5. Rollback protection: a
        # low-confidence first round is never early-stopped (rule requires
        # mean_logprob > tau_conf, so low-conf rounds continue automatically).
        lp, ent = r2.get('mean_logprob'), r2.get('entropy')
        cur_norm = _norm_basic(_extract_answer(r2['content']))
        stability = prev_norm is not None and cur_norm == prev_norm
        hi_ent = ent is not None and ent > tau_high
        if m1:
            conf_ok = lp is not None and lp > tau_conf
            if not hi_ent and ((stability and conf_ok) or (approved and conf_ok)):
                stopped_by = 'converged'
                early_stopped[0] = rnd < max_rounds - 1
                break
            if rnd == max_rounds - 1:
                stopped_by = 'max_rounds'
                break
        else:
            if approved:
                stopped_by = 'converged'
                break
            if rnd == max_rounds - 1:
                stopped_by = 'max_rounds'
                break
        prev_norm = cur_norm

        # M3.3 plan freeze: replace the full plan with just the current step
        if m3_plan and plan_lines and rnd == 0:
            step = plan_lines[min(rnd + 1, len(plan_lines) - 1)]
            exec_messages[1] = {'role': 'user', 'content':
                                f"Question: {task['question']}\nCurrent step: {step}\nFinal answer."}
        # Context policy at the round boundary. Default: rebuild from the last
        # answer and the last critique, so size is constant in rounds. With
        # accumulate_context (the compaction baseline) the full trajectory is
        # appended instead, and KVCacheTracker compacts it once it crosses the
        # trigger -- the accumulate-then-compact regime used by agent scaffolds.
        turn = [
            {'role': 'assistant', 'content': r2['content']},
            {'role': 'user', 'content': f"Reviewer feedback: {r3['content']}\nRevise and give the final answer."}
        ]
        # State policies compared in the analysis (all share the stop rule, which is set
        # separately by m1_early_stop):
        #   accumulate : append every turn (full history)
        #   window     : keep the last state_window turns
        #   summary    : replace the history with a generic summary of at most
        #                state_summary_tokens, written by one extra call that is charged
        #   default    : rebuild from the last turn only (window of 1); with the m3 slot
        #                flags this is the replace-only bounded state
        _sp = getattr(bank, 'state_policy', None)
        _hist.extend(turn)
        if getattr(bank, 'accumulate_context', False) or _sp == 'accumulate':
            exec_messages = exec_messages + turn
        elif _sp == 'window':
            k = int(getattr(bank, 'state_window', 2))
            exec_messages = exec_messages[:2] + _hist[-2 * k:]
        elif _sp == 'summary':
            ntok = int(getattr(bank, 'state_summary_tokens', 256))
            rs_ = llm_client.call('executor', bank, [
                {'role': 'system', 'content': f'Summarise the work so far so that it can be continued. '
                 f'Keep the current best answer, key intermediate results, and open issues. '
                 f'At most {ntok} tokens.'},
                {'role': 'user', 'content': f"Question: {task['question']}\n"
                 + (f"Previous summary: {_summary[0]}\n" if _summary[0] else '')
                 + f"Latest answer: {r2['content']}\nLatest review: {r3['content']}"}],
                action='summarize', max_tokens=ntok)
            results['agent_calls'].append(rs_)
            if rs_['success'] and rs_['content'].strip():
                _summary[0] = rs_['content']
            exec_messages = exec_messages[:2] + [
                {'role': 'user', 'content': f"Summary of previous rounds:\n{_summary[0]}\n"
                 "Revise and give the final answer."}]
        else:
            exec_messages = exec_messages[:2] + turn

    if MOCK_MODE:
        blocked = any(not c['success'] for c in results['agent_calls'])
        acc = 0.30 if blocked else 0.85
        results['success'] = bool(np.random.random() < acc)
    else:
        results['success'], results['f1'] = judge_task(task['id'], r2['content'], task['answer'])
    results['answer'] = r2['content']
    bank.update_opl(results)
    return _finalize()


def run_swe_task_v2(task, bank, llm_client):
    results = {'task_id': task['id'], 'agent_calls': [], 'success': False, 'patch': ''}
    agents = {'planner': 'Planner', 'executor': 'Executor', 'debugger': 'Debugger', 'reviewer': 'Reviewer'}
    bank.register_agents(agents)
    bank.allocate()

    r1 = llm_client.call('planner', bank, [
        {'role': 'system', 'content': 'You are a code analyst.'},
        {'role': 'user', 'content': f"Issue: {task['issue']}\nExplain the bug briefly."}
    ], action='plan')
    results['agent_calls'].append(r1)
    if not r1['success']:
        return results

    r2 = llm_client.call('executor', bank, [
        {'role': 'system', 'content': 'You are a programmer. Output ONLY the fixed function.'},
        {'role': 'user', 'content': f"Fix: {task.get('code_before', '')}\nBug: {task['issue']}\nReturn complete fixed function."}
    ], action='execute')
    results['agent_calls'].append(r2)
    if not r2['success']:
        return results

    r3 = llm_client.call('debugger', bank, [
        {'role': 'system', 'content': 'You are a test engineer.'},
        {'role': 'user', 'content': f"Test: {task.get('test', '')}\nPatch: {r2['content']}\nPASS or FAIL?"}
    ], action='debug')
    results['agent_calls'].append(r3)

    r4 = llm_client.call('reviewer', bank, [
        {'role': 'system', 'content': 'You are a code reviewer.'},
        {'role': 'user', 'content': f"Issue: {task['issue']}\nPatch: {r2['content']}\nCorrect? YES/NO"}
    ], action='review')
    results['agent_calls'].append(r4)
    bank.predictive_rebalance()

    if MOCK_MODE:
        blocked = any(not c['success'] for c in results['agent_calls'])
        acc = 0.30 if blocked else 0.80
        results['success'] = bool(np.random.random() < acc)
    elif task.get('test'):
        # Fallback tasks carry unit tests: actually execute the extracted fix
        results['success'] = swe_eval_executable(task, r2['content'])
    else:
        # Real SWE-bench tasks: LLM-judge against the gold patch (reference-
        # based, not the participant's self-report). Judge runs outside bank
        # accounting on the cheap model.
        judge = llm_client.client.chat.completions.create(
            model=SUMMARIZE_MODEL, max_tokens=2 * MAX_TOKENS, temperature=0,
            extra_body=LLM_EXTRA_BODY or None,
            messages=[{'role': 'user', 'content':
                f"Gold fix:\n{task['gold_patch'][:2000]}\n\nProposed fix:\n{r2['content'][:2000]}\n\n"
                'Does the proposed fix implement essentially the same correction as '
                'the gold fix? Answer YES or NO.'}])
        verdict = (judge.choices[0].message.content or '').strip().lower()
        results['success'] = verdict.startswith('yes')
    results['patch'] = r2['content']
    bank.update_opl(results)
    return results


def _extract_code_block(text: str) -> str:
    m = re.findall(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    return m[0] if m else text


def swe_eval_executable(task: Dict, patch_text: str) -> bool:
    """Execute the extracted fix against the task's unit tests (fallback tasks)."""
    code = _extract_code_block(patch_text)
    ns: Dict = {}
    try:
        exec(code, ns)
        exec(task['test'], ns)
        return True
    except Exception:
        return False


def run_financial_task_v2(task, bank, llm_client):
    results = {'task_id': task['id'], 'agent_calls': [], 'success': False, 'extracted': {}}
    agents = {'planner': 'Planner', 'executor': 'Executor', 'reviewer': 'Reviewer', 'debugger': 'Debugger'}
    bank.register_agents(agents)
    bank.allocate()

    r1 = llm_client.call('planner', bank, [
        {'role': 'system', 'content': 'You are a document analyst.'},
        {'role': 'user', 'content': f"Document excerpt: {task['document'][:500]}...\nList key fields to extract."}
    ], action='plan')
    results['agent_calls'].append(r1)
    if not r1['success']:
        return results

    fields_str = '\n'.join([f'- {k}' for k in task['fields'].keys()])
    r2 = llm_client.call('executor', bank, [
        {'role': 'system', 'content': 'You are an extraction engine. Output JSON only.'},
        {'role': 'user', 'content': f"Extract:\n{fields_str}\n\nDocument: {task['document']}\nReturn JSON."}
    ], action='execute')
    results['agent_calls'].append(r2)
    if not r2['success']:
        return results

    r3 = llm_client.call('reviewer', bank, [
        {'role': 'system', 'content': 'You are a data validator.'},
        {'role': 'user', 'content': f"Expected: {list(task['fields'].keys())}\nExtracted: {r2['content']}\nAll present? YES/NO"}
    ], action='review')
    results['agent_calls'].append(r3)

    r4 = llm_client.call('debugger', bank, [
        {'role': 'system', 'content': 'You are a data correction agent.'},
        {'role': 'user', 'content': f"Re-extract missing fields from: {task['document'][:300]}..."}
    ], action='debug')
    results['agent_calls'].append(r4)
    bank.predictive_rebalance()

    if MOCK_MODE:
        blocked = any(not c['success'] for c in results['agent_calls'])
        acc = 0.30 if blocked else 0.85
        results['success'] = bool(np.random.random() < acc)
        correct = int(len(task['fields']) * (0.9 if results['success'] else 0.5))
    else:
        extracted_text = (r2['content'] + ' ' + r4['content']).lower()
        correct = sum(1 for k, v in task['fields'].items() if v.lower() in extracted_text)
        results['success'] = correct >= len(task['fields']) * 0.75
    results['extracted'] = {'correct_fields': correct, 'total_fields': len(task['fields'])}
    bank.update_opl(results)
    return results

print('Task runners v2 defined.')


# ============ NOTEBOOK CELL 10 ============
from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_WORKERS = int(os.getenv('MAX_WORKERS', '16'))  # parallel tasks; V4 Pro limit is 500

def run_experiment_v2(tasks, task_runner, config, llm_client):
    budget = config.get('budget', 100000)
    policy = config.get('policy', 'hard')
    external_plan = config.get('external_plan', None)
    gamma = config.get('gamma', 0.8)
    enable_predictive = config.get('enable_predictive', True)
    enable_borrowing = config.get('enable_borrowing', True)
    enable_opl = config.get('enable_opl', True)
    enable_kv = config.get('enable_kv', True)
    theta = config.get('theta', 0.8)
    min_quota = config.get('min_quota', 5000)
    enable_credit_scoring = config.get('enable_credit_scoring', False)
    rogue_cutoff = config.get('rogue_cutoff', 0.10)
    credit_rate = config.get('credit_rate', 0.5)
    enable_pacing = config.get('enable_pacing', False)
    pacing_upfront = config.get('pacing_upfront', 0.5)
    pacing_k = config.get('pacing_k', 3)
    pacing_max_rounds = config.get('pacing_max_rounds', 5)
    adaptive_runaway = config.get('adaptive_runaway', False)
    runaway_token_fraction = config.get('runaway_token_fraction', 0.35)
    runaway_rounds = config.get('runaway_rounds', 1)
    runaway_min_tokens = config.get('runaway_min_tokens', 0)
    max_rounds_cfg = config.get('max_rounds')
    force_final_on_block = config.get('force_final_on_block', False)

    # One OPL per experiment run, shared across all task banks, so Thompson
    # Sampling accumulates over tasks and exits cold-start after 10 tasks
    shared_opl = (OnlinePriorityLearner(['Planner', 'Executor', 'Reviewer', 'Debugger'])
                  if enable_opl else None)
    # Optional shared credit scorer: sizes/prices credit lines per role from
    # the running history of credit-to-success conversion
    shared_credit_scorer = CreditScorer() if (enable_borrowing and enable_credit_scoring) else None

    def _run_one(task):
        # Each task gets its own bank, so tasks are independent and thread-safe
        bank = TokenBankV2(
            total_budget=budget, policy=policy, external_plan=external_plan, gamma=gamma,
            rebalance_threshold=theta, enable_predictive=enable_predictive,
            enable_borrowing=enable_borrowing, enable_opl=enable_opl, enable_kv=enable_kv,
            min_quota=min_quota, shared_opl=shared_opl,
            shared_credit_scorer=shared_credit_scorer,
            rogue_cutoff=rogue_cutoff, credit_rate=credit_rate,
            enable_pacing=enable_pacing,
            pacing_upfront=pacing_upfront, pacing_k=pacing_k,
            pacing_max_rounds=pacing_max_rounds,
            adaptive_runaway=adaptive_runaway,
            runaway_token_fraction=runaway_token_fraction,
            runaway_rounds=runaway_rounds,
            runaway_min_tokens=runaway_min_tokens,
            max_rounds=max_rounds_cfg,
            force_final_on_block=force_final_on_block,
            final_reserve=config.get('final_reserve', 0),
            role_credit_mult=config.get('role_credit_mult'),
            predictive_est=config.get('predictive_est', False),
            stall_sim=config.get('stall_sim'),
            runaway_frac_auto=config.get('runaway_frac_auto', False),
            predictive_lending=config.get('predictive_lending', False),
            kv_trigger=config.get('kv_trigger'),
            real_summary=config.get('real_summary', False),
            summary_budget=config.get('summary_budget', 512),
            context_window=config.get('context_window', 8192),
            accumulate_context=config.get('accumulate_context', False),
            exec_compact=config.get('exec_compact', False),
            capture_logprobs=config.get('capture_logprobs', False)
        )
        # v5 mechanism flags pass straight through as bank attributes
        for k in ('m1_early_stop', 'tau_conf', 'tau_high', 'm2_cascade',
                  'theta0', 'theta1', 'm3_structured_reviewer', 'm3_tool_notes',
                  'm3_plan_freeze', 'm3_answer_format', 'm5_strategy', 'v5_max_level',
                  'm2_verbal', 'l01_max_tokens', 'l01_reasoning_effort', 'l01_thinking_budget',
                  'l01_thinking_off', 'l0_no_tool', 'state_policy', 'state_window',
                  'state_summary_tokens', 'review_max_tokens', 'runtime_cap', 'runtime_cap_k',
                  'runtime_cap_lo', 'runtime_cap_hi', 'empty_review_keep', 'static_split', 'static_calls',
                  'reason_budget_frac', 'reason_budget_min', 'reason_budget_max'):
            if k in config:
                setattr(bank, k, config[k])
        # M4 pooling may override the per-task budget / start level per task
        # (allocate() runs inside the runner, so overriding here is safe)
        eff_budget = task.get('_v5_budget') or budget
        if task.get('_v5_budget'):
            bank.total_budget = task['_v5_budget']
            bank.reserve = task['_v5_budget']
        if config.get('v5'):
            # v5 (M0): no per-role quota ceilings — role quotas are dead
            # governance (D2: 0% trigger). The global reserve alone bounds the
            # task; allocate() reads q_min lazily inside the runner.
            bank.q_min = eff_budget
        runner = run_gaia_task_v5 if config.get('v5') else task_runner
        task_result = runner(task, bank, llm_client)
        bank.update_credit(task_result)
        task_result['bank_summary'] = bank.get_summary()
        task_result['config'] = config
        task_result['overrun'] = bank.reserve < 0 or any(a.used > a.quota + a.borrowed for a in bank.ledger.values())
        # Recovery: the task needed budget relief (credit or rebalancing) during
        # execution yet still succeeded — it would likely have failed under a
        # rigid quota but recovered through runtime governance.
        task_result['recovered'] = (task_result['success'] and
                                    (bank.borrow_count > 0 or bank.rebalance_count > 0))
        return task_result

    results = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_run_one, t): i for i, t in enumerate(tasks)}
        done = 0
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
            done += 1
            if VERBOSE and done % 10 == 0:
                print(f'  Progress: {done}/{len(tasks)} tasks')

    total_cost = sum(r['bank_summary']['total_cost_usd'] for r in results)

    successes = [r['success'] for r in results]
    tokens_used = [r['bank_summary']['total_used'] for r in results]
    overruns = [r['overrun'] for r in results]
    recovered = [r['recovered'] for r in results]
    borrow_counts = [r['bank_summary']['borrow_count'] for r in results]

    return {
        'config': config, 'results': results, 'success_rate': np.mean(successes),
        'mean_tokens': np.mean(tokens_used), 'std_tokens': np.std(tokens_used),
        'overrun_rate': np.mean(overruns), 'recovery_rate': np.mean(recovered) if any(recovered) else 0.0,
        'total_cost_usd': total_cost, 'cost_per_success': total_cost / (sum(successes) or 1),
        'mean_borrows': np.mean(borrow_counts),
        'opl_tasks': shared_opl.task_count if shared_opl else 0,
        'opl_learned': shared_opl.get_priorities() if shared_opl else {},
        'credit_scores': shared_credit_scorer.summary() if shared_credit_scorer else {},
    }

print('Experiment orchestrator v2 defined.')


# ============ NOTEBOOK CELL 11 ============
# ==================== MAIN EXPERIMENT CONFIGURATIONS ====================
CONFIGS_V2 = [
    {'name': 'Unconstrained', 'budget': 200000, 'policy': 'hard', 'external_plan': None,
     'gamma': 0.0, 'enable_predictive': False, 'enable_borrowing': False, 'enable_opl': False, 'enable_kv': False, 'theta': 1.0},
    {'name': 'Static Quota', 'budget': 4000, 'min_quota': 800, 'policy': 'hard', 'external_plan': None,
     'gamma': 1.0, 'enable_predictive': False, 'enable_borrowing': False, 'enable_opl': False, 'enable_kv': False, 'theta': 1.0},
    {'name': 'TB-v1 (Hard)', 'budget': 4000, 'min_quota': 800, 'policy': 'hard', 'external_plan': None,
     'gamma': 0.0, 'enable_predictive': False, 'enable_borrowing': False, 'enable_opl': False, 'enable_kv': False, 'theta': 0.8},
    {'name': 'TB-v2-Full', 'budget': 4000, 'min_quota': 800, 'policy': 'hard', 'external_plan': None,
     'gamma': 0.8, 'enable_predictive': True, 'enable_borrowing': True, 'enable_opl': True, 'enable_kv': True,
     'theta': 0.8, 'credit_rate': 1.5},
    {'name': 'TB-v2-Grace', 'budget': 4000, 'min_quota': 800, 'policy': 'grace', 'external_plan': None,
     'gamma': 0.8, 'enable_predictive': True, 'enable_borrowing': True, 'enable_opl': True, 'enable_kv': True, 'theta': 0.8},
    {'name': 'TB-v2-Throttle', 'budget': 4000, 'min_quota': 800, 'policy': 'throttle', 'external_plan': None,
     'gamma': 0.8, 'enable_predictive': True, 'enable_borrowing': True, 'enable_opl': True, 'enable_kv': True, 'theta': 0.8},
    {'name': 'TB-v2-Scored', 'budget': 4000, 'min_quota': 800, 'policy': 'hard', 'external_plan': None,
     'gamma': 0.8, 'enable_predictive': True, 'enable_borrowing': True, 'enable_opl': True, 'enable_kv': True,
     'theta': 0.8, 'enable_credit_scoring': True, 'rogue_cutoff': 0.10},
]

# Select benchmark
BENCHMARK = os.getenv('BENCHMARK', 'GAIA')  # 'GAIA', 'SWE', 'FINANCIAL'
if BENCHMARK == 'GAIA':
    TASKS, RUNNER = GAIA_TASKS, run_gaia_task_v2
elif BENCHMARK == 'SWE':
    TASKS, RUNNER = SWE_TASKS, run_swe_task_v2
else:
    TASKS, RUNNER = FINANCIAL_TASKS, run_financial_task_v2

# Scale tight budgets per benchmark so budget pressure is comparable across
# task types (SWE prompts are much longer; Unconstrained stays at 200K)
BENCH_TIGHT = {'GAIA': (4000, 800), 'SWE': (10000, 2000), 'FINANCIAL': (8000, 1600)}
TIGHT_BUDGET, TIGHT_MIN_QUOTA = BENCH_TIGHT[BENCHMARK]
for cfg in CONFIGS_V2:
    if cfg.get('budget') != 200000:
        cfg['budget'] = TIGHT_BUDGET
        cfg['min_quota'] = TIGHT_MIN_QUOTA

# Multi-seed main experiment: SEEDS env var, e.g. SEEDS=42,43,44. Task set is
# fixed across seeds (shuffled once at load with the base SEED); seeds vary
# generation randomness. Results are pooled across seeds for statistics.
SEEDS = [int(s) for s in os.getenv('SEEDS', '42').split(',')]
print(f'Running {len(CONFIGS_V2)} configurations x {len(SEEDS)} seed(s) on {BENCHMARK} ({len(TASKS)} tasks)...')

raw_results_v2 = []
for seed in SEEDS:
    random.seed(seed)
    np.random.seed(seed)
    for cfg in CONFIGS_V2:
        print(f"\n{'='*60}")
        print(f'Running: {cfg["name"]} (seed={seed})')
        print(f"{'='*60}")
        result = run_experiment_v2(TASKS, RUNNER, cfg, llm)
        result['name'] = cfg['name']
        result['seed'] = seed
        raw_results_v2.append(result)
        print(f"  SR: {result['success_rate']:.1%} | Cost: ${result['total_cost_usd']:.4f}")

# Aggregate: pool task results across seeds; report per-seed SR spread
experiment_results_v2 = []
for cfg in CONFIGS_V2:
    runs = [r for r in raw_results_v2 if r['name'] == cfg['name']]
    pooled = [tr for r in runs for tr in r['results']]
    srs = [r['success_rate'] for r in runs]
    total_cost = sum(r['total_cost_usd'] for r in runs)
    agg = {
        'config': cfg, 'name': cfg['name'], 'results': pooled,
        'success_rate': np.mean([t['success'] for t in pooled]),
        'seed_sr_std': float(np.std(srs)) if len(srs) > 1 else 0.0,
        'mean_tokens': np.mean([t['bank_summary']['total_used'] for t in pooled]),
        'std_tokens': np.std([t['bank_summary']['total_used'] for t in pooled]),
        'overrun_rate': np.mean([t['overrun'] for t in pooled]),
        'recovery_rate': np.mean([t['recovered'] for t in pooled]),
        'total_cost_usd': total_cost,
        'cost_per_success': total_cost / (sum(t['success'] for t in pooled) or 1),
        'mean_borrows': np.mean([t['bank_summary']['borrow_count'] for t in pooled]),
        'opl_tasks': runs[-1].get('opl_tasks', 0), 'opl_learned': runs[-1].get('opl_learned', {}),
    }
    experiment_results_v2.append(agg)
    print(f"  {cfg['name']}: SR={agg['success_rate']:.1%}±{agg['seed_sr_std']:.1%} | "
          f"Tokens: {agg['mean_tokens']:.0f}±{agg['std_tokens']:.0f} | "
          f"RR: {agg['recovery_rate']:.1%} | Borrows: {agg['mean_borrows']:.1f} | "
          f"CPS: ${agg['cost_per_success']:.4f}")
    if agg['opl_tasks']:
        print(f"    OPL: {agg['opl_tasks']} tasks, priorities={agg['opl_learned']}")

print('\nAll experiments completed.')


# ============ NOTEBOOK CELL 12 ============
# ==================== ABLATION STUDY ====================
ABLATION_CONFIGS_V2 = [
    {'name': 'Full TB-v2', 'enable_predictive': True, 'enable_borrowing': True, 'enable_opl': True, 'enable_kv': True},
    {'name': 'w/o Predictive', 'enable_predictive': False, 'enable_borrowing': True, 'enable_opl': True, 'enable_kv': True},
    {'name': 'w/o Borrowing', 'enable_predictive': True, 'enable_borrowing': False, 'enable_opl': True, 'enable_kv': True},
    {'name': 'w/o OPL', 'enable_predictive': True, 'enable_borrowing': True, 'enable_opl': False, 'enable_kv': True},
    {'name': 'w/o KV', 'enable_predictive': True, 'enable_borrowing': True, 'enable_opl': True, 'enable_kv': False},
    {'name': 'Static Quota', 'enable_predictive': False, 'enable_borrowing': False, 'enable_opl': False, 'enable_kv': False},
]

base_cfg = {'budget': TIGHT_BUDGET, 'min_quota': TIGHT_MIN_QUOTA, 'policy': 'hard', 'gamma': 0.8, 'theta': 0.8, 'external_plan': None}
ablation_results_v2 = []

print('Running ablation study...')
for abl in ([] if SMOKE_N else ABLATION_CONFIGS_V2):
    cfg = {**base_cfg, **abl}
    result = run_experiment_v2(TASKS, RUNNER, cfg, llm)
    result['name'] = abl['name']
    ablation_results_v2.append(result)
    print(f"  {abl['name']}: SR={result['success_rate']:.1%}, Var={result['std_tokens']:.0f}, OR={result['overrun_rate']:.1%}, RR={result['recovery_rate']:.1%}")

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
names = [r['name'] for r in ablation_results_v2]
sr = [r['success_rate'] * 100 for r in ablation_results_v2]
colors = ['#22c55e', '#fbbf24', '#fb923c', '#f97316', '#a78bfa', '#ef4444']
bars = ax.bar(names, sr, color=colors, edgecolor='black', linewidth=1.2)
ax.set_ylabel('Success Rate (%)')
ax.set_title('Ablation Study: Component Contribution (v2)')
ax.set_ylim(80, 100)
for bar, val in zip(bars, sr):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(f'ablation_v2_{BENCHMARK.lower()}{OUT_SUFFIX}.png', dpi=200, bbox_inches='tight')
plt.show()


# ============ NOTEBOOK CELL 13 ============
# ==================== SENSITIVITY: THETA ====================
THETA_VALUES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
theta_results_v2 = []

print('Running sensitivity analysis on theta...')
for theta in ([] if SMOKE_N else THETA_VALUES):
    cfg = {'budget': TIGHT_BUDGET, 'min_quota': TIGHT_MIN_QUOTA, 'policy': 'hard', 'gamma': 0.8, 'theta': theta, 'external_plan': None,
           'enable_predictive': True, 'enable_borrowing': True, 'enable_opl': True, 'enable_kv': True}
    result = run_experiment_v2(TASKS, RUNNER, cfg, llm)
    result['theta'] = theta
    theta_results_v2.append(result)
    print(f'  theta={theta}: SR={result["success_rate"]:.1%}, OR={result["overrun_rate"]:.1%}, RR={result["recovery_rate"]:.1%}')

fig, ax1 = plt.subplots(figsize=(10, 5))
thetas = [r['theta'] for r in theta_results_v2]
sr_vals = [r['success_rate'] * 100 for r in theta_results_v2]
or_vals = [r['overrun_rate'] * 100 for r in theta_results_v2]
rr_vals = [r['recovery_rate'] * 100 for r in theta_results_v2]
ax2 = ax1.twinx()
l1 = ax1.plot(thetas, sr_vals, 'o-', color='#3b82f6', linewidth=2.5, markersize=10, label='Success Rate')
l2 = ax1.plot(thetas, rr_vals, '^-', color='#22c55e', linewidth=2.5, markersize=10, label='Recovery Rate')
l3 = ax2.plot(thetas, or_vals, 's--', color='#ef4444', linewidth=2.5, markersize=10, label='Overrun Rate')
ax1.set_xlabel('Rebalance Threshold theta')
ax1.set_ylabel('Rate (%)', color='#3b82f6')
ax2.set_ylabel('Overrun Rate (%)', color='#ef4444')
ax1.set_title('Sensitivity: Rebalance Threshold (v2)')
ax1.axvline(x=0.8, color='green', linestyle=':', linewidth=2, alpha=0.7)
ax1.set_ylim(80, 100)
ax2.set_ylim(-2, 15)
ax1.grid(True, alpha=0.3)
lines = l1 + l2 + l3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center left')
plt.tight_layout()
plt.savefig(f'sensitivity_theta_v2_{BENCHMARK.lower()}{OUT_SUFFIX}.png', dpi=200, bbox_inches='tight')
plt.show()


# ============ NOTEBOOK CELL 14 ============
# ==================== FULL VISUALIZATION ====================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f'Token Bank v2 Evaluation: {BENCHMARK}', fontsize=16, fontweight='bold')

names = [r['name'] for r in experiment_results_v2]
success_rates = [r['success_rate'] * 100 for r in experiment_results_v2]
mean_tokens = [r['mean_tokens'] / 1000 for r in experiment_results_v2]
std_tokens = [r['std_tokens'] / 1000 for r in experiment_results_v2]
overrun_rates = [r['overrun_rate'] * 100 for r in experiment_results_v2]
recovery_rates = [r['recovery_rate'] * 100 for r in experiment_results_v2]
costs = [r['total_cost_usd'] for r in experiment_results_v2]
colors = ['#ef4444', '#f97316', '#fbbf24', '#22c55e', '#3b82f6', '#8b5cf6', '#14b8a6']

# (a) Success Rate
ax = axes[0, 0]
bars = ax.bar(names, success_rates, color=colors, edgecolor='black')
ax.set_ylabel('Success Rate (%)')
ax.set_title('(a) Task Success Rate')
ax.set_ylim(80, 100)
ax.tick_params(axis='x', rotation=25)
for bar, val in zip(bars, success_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# (b) Token Consumption
ax = axes[0, 1]
bars = ax.bar(names, mean_tokens, color=colors, edgecolor='black')
ax.set_ylabel('Tokens (K)')
ax.set_title('(b) Mean Token Consumption')
ax.tick_params(axis='x', rotation=25)
for bar, val in zip(bars, mean_tokens):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}K', ha='center', va='bottom', fontsize=9, fontweight='bold')

# (c) Variance
ax = axes[0, 2]
bars = ax.bar(names, std_tokens, color=colors, edgecolor='black')
ax.set_ylabel('Std Dev (K)')
ax.set_title('(c) Consumption Variance')
ax.tick_params(axis='x', rotation=25)
for bar, val in zip(bars, std_tokens):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f'{val:.1f}K', ha='center', va='bottom', fontsize=9, fontweight='bold')

# (d) Overrun Rate
ax = axes[1, 0]
bars = ax.bar(names, overrun_rates, color=colors, edgecolor='black')
ax.set_ylabel('Overrun Rate (%)')
ax.set_title('(d) Budget Overrun Rate')
ax.tick_params(axis='x', rotation=25)
for bar, val in zip(bars, overrun_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# (e) Recovery Rate (NEW METRIC)
ax = axes[1, 1]
bars = ax.bar(names, recovery_rates, color=colors, edgecolor='black')
ax.set_ylabel('Recovery Rate (%)')
ax.set_title('(e) Recovery After Budget Exhaustion')
ax.tick_params(axis='x', rotation=25)
for bar, val in zip(bars, recovery_rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# (f) Pareto Frontier
ax = axes[1, 2]
for i, (name, sr, cost) in enumerate(zip(names, success_rates, costs)):
    ax.scatter(cost, sr, s=300, c=colors[i], edgecolors='black', linewidth=2, zorder=5)
    ax.annotate(name, (cost, sr), textcoords='offset points', xytext=(8, 0), ha='left', fontsize=9, fontweight='bold')
ax.set_xlabel('Total Cost (USD)')
ax.set_ylabel('Success Rate (%)')
ax.set_title('(f) Pareto Frontier')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max(costs) * 1.3)
plt.tight_layout()
plt.savefig(f'evaluation_v2_{BENCHMARK.lower()}{OUT_SUFFIX}.png', dpi=200, bbox_inches='tight')
plt.show()


# ============ NOTEBOOK CELL 15 ============
# ==================== STATISTICAL TESTS ====================
from scipy.stats import mannwhitneyu, chi2_contingency, levene

print('===== Statistical Significance Tests (v2) =====\n')

uc = [r for r in experiment_results_v2 if r['name'] == 'Unconstrained'][0]
tb2 = [r for r in experiment_results_v2 if r['name'] == 'TB-v2-Full'][0]

# 1. Token consumption
uc_tokens = [r['bank_summary']['total_used'] for r in uc['results']]
tb_tokens = [r['bank_summary']['total_used'] for r in tb2['results']]
stat, pval = mannwhitneyu(uc_tokens, tb_tokens, alternative='two-sided')
print(f'1. Token Consumption: Mann-Whitney U={stat:.2f}, p={pval:.4f}')

# 2. Success rate
uc_s = sum(r['success'] for r in uc['results'])
tb_s = sum(r['success'] for r in tb2['results'])
cont = [[uc_s, len(uc['results'])-uc_s], [tb_s, len(tb2['results'])-tb_s]]
if all(col > 0 for col in np.sum(cont, axis=0)):
    chi2, pval, _, _ = chi2_contingency(cont)
    print(f'2. Success Rate: Chi-square={chi2:.4f}, p={pval:.4f}')
else:
    print('2. Success Rate: Chi-square skipped (degenerate table: a row/column is all-zero)')

# 3. Variance
stat, pval = levene(uc_tokens, tb_tokens)
print(f'3. Variance (Levene): stat={stat:.4f}, p={pval:.4f}')

# 4. Recovery rate (unique to v2)
print(f'4. Recovery Rate: TB-v2={tb2["recovery_rate"]:.1%} vs Unconstrained={uc["recovery_rate"]:.1%}')

# 5. Borrow efficiency
borrows = [r['bank_summary']['borrow_count'] for r in tb2['results']]
print(f'5. Mean Borrows per Task: {np.mean(borrows):.2f}')

print('\nTests complete.')


# ============ NOTEBOOK CELL 16 ============
# ==================== PER-AGENT BREAKDOWN ====================
tb2_data = [r for r in experiment_results_v2 if r['name'] == 'TB-v2-Full'][0]
agent_stats = defaultdict(lambda: {'used': 0, 'quota': 0, 'borrowed': 0, 'calls': 0})

for task_result in tb2_data['results']:
    summary = task_result['bank_summary']
    for agent_id, data in summary['agents'].items():
        agent_stats[agent_id]['used'] += data['used']
        agent_stats[agent_id]['quota'] += data['quota']
        agent_stats[agent_id]['borrowed'] += data['borrowed']
        agent_stats[agent_id]['calls'] += len([c for c in task_result['agent_calls'] if c.get('success')])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
agents = list(agent_stats.keys())
used_vals = [agent_stats[a]['used'] / 1000 for a in agents]
quota_vals = [agent_stats[a]['quota'] / 1000 for a in agents]
borrow_vals = [agent_stats[a]['borrowed'] / 1000 for a in agents]
x = np.arange(len(agents))
width = 0.25
ax1.bar(x - width, quota_vals, width, label='Quota', color='#94a3b8', edgecolor='black')
ax1.bar(x, used_vals, width, label='Used', color='#3b82f6', edgecolor='black')
ax1.bar(x + width, borrow_vals, width, label='Borrowed', color='#f59e0b', edgecolor='black')
ax1.set_ylabel('Tokens (K)')
ax1.set_title('Per-Agent Quota vs. Used vs. Borrowed')
ax1.set_xticks(x)
ax1.set_xticklabels(agents)
ax1.legend()

util_rates = [agent_stats[a]['used'] / max(1, agent_stats[a]['quota']) * 100 for a in agents]
bars = ax2.bar(agents, util_rates, color=['#22c55e' if u < 70 else '#fbbf24' if u < 90 else '#ef4444' for u in util_rates], edgecolor='black')
ax2.set_ylabel('Utilization (%)')
ax2.set_title('Per-Agent Utilization Rate')
ax2.axhline(y=80, color='orange', linestyle='--', label='Rebalance Threshold')
ax2.axhline(y=100, color='red', linestyle='--', label='Quota Limit')
ax2.legend()
for bar, val in zip(bars, util_rates):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(f'per_agent_v2_{BENCHMARK.lower()}{OUT_SUFFIX}.png', dpi=200, bbox_inches='tight')
plt.show()


# ============ NOTEBOOK CELL 17 ============
# ==================== EXPORT FINAL REPORT ====================
report = {
    'timestamp': datetime.now().isoformat(),
    'benchmark': BENCHMARK,
    'mock_mode': MOCK_MODE,
    'primary_model': PRIMARY_MODEL,
    'configurations': [
        {'name': r['name'], 'success_rate': round(r['success_rate'], 4),
         'seed_sr_std': round(r.get('seed_sr_std', 0.0), 4),
         'mean_tokens': round(r['mean_tokens'], 1), 'std_tokens': round(r['std_tokens'], 1),
         'overrun_rate': round(r['overrun_rate'], 4), 'recovery_rate': round(r['recovery_rate'], 4),
         'total_cost_usd': round(r['total_cost_usd'], 4), 'cost_per_success': round(r['cost_per_success'], 4),
         'mean_borrows': round(r['mean_borrows'], 2)}
        for r in experiment_results_v2
    ],
    'ablation': [
        {'name': r['name'], 'success_rate': round(r['success_rate'], 4),
         'std_tokens': round(r['std_tokens'], 1), 'overrun_rate': round(r['overrun_rate'], 4),
         'recovery_rate': round(r['recovery_rate'], 4)}
        for r in ablation_results_v2
    ],
    'sensitivity_theta': [
        {'theta': r['theta'], 'success_rate': round(r['success_rate'], 4),
         'overrun_rate': round(r['overrun_rate'], 4), 'recovery_rate': round(r['recovery_rate'], 4),
         'std_tokens': round(r['std_tokens'], 1)}
        for r in theta_results_v2
    ]
}

with open(f'experiment_report_v2_{BENCHMARK.lower()}{OUT_SUFFIX}.json', 'w') as f:
    json.dump(report, f, indent=2)

# Summary DataFrame
df_data = []
for r in experiment_results_v2:
    df_data.append({
        'Method': r['name'], 'Success Rate': f"{r['success_rate']:.1%}",
        'SR±seed': f"{r.get('seed_sr_std', 0):.1%}",
        'Mean Tokens': f"{r['mean_tokens']:.0f}", 'Std Tokens': f"{r['std_tokens']:.0f}",
        'Overrun': f"{r['overrun_rate']:.1%}", 'Recovery': f"{r['recovery_rate']:.1%}",
        'Cost ($)': f"{r['total_cost_usd']:.4f}", 'CPS ($)': f"{r['cost_per_success']:.4f}",
        'Borrows': f"{r['mean_borrows']:.1f}",
        'Rebalances': f"{np.mean([tr['bank_summary']['rebalance_count'] for tr in r['results']]):.2f}",
        'Summarizes': f"{np.mean([tr['bank_summary']['summarization_count'] for tr in r['results']]):.2f}"
    })
df = pd.DataFrame(df_data)
print(df.to_string(index=False))
df.to_csv(f'results_v2_{BENCHMARK.lower()}{OUT_SUFFIX}.csv', index=False)

print('\nExported files:')
print(f'  - experiment_report_v2_{BENCHMARK.lower()}{OUT_SUFFIX}.json')
print(f'  - results_v2_{BENCHMARK.lower()}{OUT_SUFFIX}.csv')
print(f'  - evaluation_v2_{BENCHMARK.lower()}{OUT_SUFFIX}.png')
print(f'  - ablation_v2_{BENCHMARK.lower()}{OUT_SUFFIX}.png')
print(f'  - sensitivity_theta_v2_{BENCHMARK.lower()}{OUT_SUFFIX}.png')
print(f'  - per_agent_v2_{BENCHMARK.lower()}{OUT_SUFFIX}.png')
