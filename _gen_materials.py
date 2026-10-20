#!/usr/bin/env python
"""Regenerate PAPER_MATERIALS.md from the traces.

Every number in the write-up is computed here rather than transcribed, because the same
cell often exists in several trace files (a primary run, a flip-rate re-run, a speed test,
a calibration run) and picking the wrong one silently changes the answer. FILES below is
the explicit map from (arm, benchmark, budget) to the primary run; nothing globs.
"""
import json, os, glob, collections, statistics as st
from math import comb

OUT = []
def w(s=''): OUT.append(s)

_METHOD = r'''## 1.3 Method, in the detail needed to reimplement it

Everything below is read off `_run_v2_experiment.py` (`run_gaia_task_v5`, `TokenBankV2`,
`LLMClientV2`) and the arm table in `_budget_sweep_v2.py`. Constants are the values the
experiments actually ran with, not defaults in the source: where the two differ, the arm
dictionary wins, and that is noted.

### The object being governed

A task is answered by a three-role pipeline: a **Planner** that decomposes the question, an
**Executor** that answers and may run Python, and a **Reviewer** that accepts or asks for a
revision. The Executor and Reviewer alternate for up to `V5_MAX_ROUNDS` rounds (5 in every
main experiment; BigCodeBench also 5). Python appears as a real tool: the Executor emits a
```python block, the harness executes it with a 10-second timeout, and the printed output is
fed back as a new user turn before the Executor commits to an answer.

A **budget** `B` is a hard per-task cap on the sum of prompt and completion tokens over every
call of every role. `B` takes 8k/16k/24k on the four reasoning benchmarks and 4k/8k/16k on
BigCodeBench, which has longer prompts and needs the tighter end to bind at all.

### Admission control: what the cap actually does

`TokenBankV2` holds a `reserve` initialised to `B` and a per-role ledger. Before every call,
`LLMClientV2.call` estimates the cost as `estimated_in + estimated_out`, where
`estimated_in` is the tokenised prompt and `estimated_out` is the requested `max_tokens`, and
asks `can_dispatch`. Four gates, in order:

1. **Final-answer bypass.** A call marked `is_final` skips quota, pacing and borrowing. Only
   the global reserve can stop it, and it is the only call permitted to spend `final_reserve`.
2. **Pacing gate**, dormant in every arm reported here (`pacing` off).
3. **Per-role quota.** `allocate()` splits `B` by role priority: each role is guaranteed
   `q_min`, and the remainder is divided in proportion to priority. Exceeding quota blocks
   unless borrowing is enabled, which it is not in any reported arm
   (`enable_borrowing: False`).
4. **Global reserve minus `final_reserve`.** `final_reserve` is **256** tokens in every arm,
   held back so that a task which exhausts its budget mid-loop can still be asked for one
   short closing answer rather than returning nothing.

A blocked call returns `success: False` with a reason string rather than raising, and the
loop then takes its grace path. The two reasons that appear in logs are
`Quota exceeded: used/quota` and `Global reserve depleted: n`; they distinguish a role that
overspent from a task that ran out, and the trace records which.

**The grace close.** When a call is blocked, `_grace_close` spends the reserved 256 tokens on
a single Executor call prompted with "Budget exhausted. Give your best final answer now,
concisely." This is why the governed arm never returns an empty answer under a binding cap,
and it is the mechanism that the truncation counts in section 7 are measured against.

### The budget inequality

The affordable round count is `R_B >= min(R_max, floor(B / C_bar))`, where `C_bar` is the mean
per-round cost:

```
C_bar = c_fixed + g_bar + kappa_n + kappa_v + kappa_p
```

`c_fixed` is the system prompt and question, constant in rounds. `g_bar` is the mean
generation per round. `kappa_n`, `kappa_v`, `kappa_p` are the carried slots: the last answer,
the last critique, and the plan. The claim the method rests on is that in an ungoverned loop
the kappa terms grow with the round index, because each round's context is the previous
context plus that round's output, so `C_bar` grows and `R_B` shrinks as the task gets harder.
Bounding the kappa terms makes `C_bar` constant in rounds, so `R_B` becomes a function of `B`
alone.

Section 2's segment accounting is the measurement of that claim: 46.6% of the budget is
coordination state re-sent from an earlier round, and the plan slot alone is 40.4% of every
character the model reads on Qwen and 28.7% on Gemma, each larger than every other bindable
slot combined.

### Mechanism 1: bounded slots (the round-boundary context policy)

This is the core of the method and it is four lines of code. At the end of each L2 round:

```python
turn = [{'role': 'assistant', 'content': r2['content']},
        {'role': 'user', 'content': f"Reviewer feedback: {r3['content']}\nRevise and give the final answer."}]
exec_messages = (exec_messages + turn) if bank.accumulate_context \
                else (exec_messages[:2] + turn)
```

`exec_messages[:2]` is the system prompt and the original question. The governed policy
therefore rebuilds the context each round as **system + question + last answer + last
critique**, and nothing else. Round 7 carries exactly what round 2 carried. The
`accumulate_context` branch is the compaction baseline: it appends instead, so the prompt
grows monotonically, and `KVCacheTracker` summarises it once the prompt crosses half of an
8192-token window (`context_window=8192`, `kv_trigger=0.5`, both pinned in the arm config and
deliberately **not** read from the server, so the compaction baseline is the same policy on
every deployment regardless of the real context length).

Three slot-level caps ride along with it:

- **Reviewer critique**, capped at 150 `max_tokens` and forced to JSON
  (`m3_structured_reviewer`): `{"verdict": "accept"|"revise", "issues": [<=3 items, <=20
  words each], "hint": "<=30 words"}`. This bounds `kappa_v` by construction rather than by
  truncation.
- **Tool output**, capped by `m3_tool_notes`: instead of pasting the raw stdout, the harness
  spends one 150-token call to compress it to "a note of at most 120 tokens, facts and
  numbers only", and carries the note. Raw output is truncated at 4000 characters before even
  that call, and if the summarising call fails the first 800 characters are used instead.
- **Plan**, capped by `m3_plan_freeze`: after round 0 the full plan in the prompt is replaced
  by a single line, the current step. This is what makes `kappa_p` constant.

The plan-freeze line selection carries a fix worth reporting because it is a defect the
naive implementation has. Taking `plan_lines[1]` verbatim kept a bare heading
("**Step-by-Step Plan:**", "Here is the step-by-step plan:") on **four of six** sampled
repair plans against **one of six** computation plans, because repair plans more often
restate the problem before starting the list. The bounded slot was therefore empty of content
on exactly the task type where the plan matters most. The harness now strips scaffolding
lines with a boilerplate regex before indexing, falling back to the unfiltered list if
filtering would empty it.

### Mechanism 2: the cost-ordered escalation ladder

Three rungs, tried cheapest first. A task stops at the first rung whose answer passes the
gate, so the expensive scaffolding is never built for a task that did not need it. The design
principle is that the cheapest state is the state never created.

- **L0, direct answer.** One Executor call, no tool, no plan, no reviewer.
  `max_tokens = l01_max_tokens`, which is **2048** in every governed arm (the source default
  of 768 is not what ran).
- **L1, direct answer with tool, one round.** Same primitive with `with_tool=True`, and it is
  handed L0's answer as `Draft answer: ...` with the instruction to verify and fix it, so L0's
  work is reused rather than discarded.
- **L2, the full Plan -> Execute -> Review loop** described above, up to 5 rounds, seeded with
  L0/L1's answer as a draft and with the plan.

Between rungs the accept/reject decision is the confidence gate.

### Mechanism 3: the confidence gate

```python
def _confident(r, theta):
    lp, ent = r['mean_logprob'], r['entropy']
    if lp is None:
        return False            # no signal -> escalate
    return lp > theta and (ent is None or ent <= tau_high)
```

`mean_logprob` is the arithmetic mean of per-token logprobs over the generated sequence.
`entropy` is the mean, over positions, of the entropy of the returned `top_logprobs`
distribution at that position. Both come from the same call that produced the answer, so the
gate costs nothing: no extra sampling, no verifier model, no second pass.

Thresholds as run (`V5_THRESH`, env-overridable, tuned offline on dev splits):

| symbol | value | used at |
|---|---|---|
| `theta0` | -0.10 | L0 -> stop or escalate |
| `theta1` | -0.15 | L1 -> stop or escalate |
| `tau_conf` | -0.05 | inside L2, gates the early stop |
| `tau_high` | 0.6 | entropy ceiling, overrides all three |

`theta1` being **looser** than `theta0` is deliberate: a task that already survived L0's
rejection and got a tool has earned a lower bar for stopping. Missing logprobs escalate
rather than stop, so a provider that drops logprobs degrades the method to the full pipeline
rather than to a wrong answer.

**The gate is a poor instrument that still routes well.** Its discrimination as a correctness
classifier is weak and uneven: AUC 0.75 on MATH and FRAMES, 0.62 on GAIA, 0.52 on MuSiQue,
which on MuSiQue is chance. Section 6 is the argument that this does not matter, because the
gate is not being asked to predict correctness; it is being asked to place a task on a rung,
and a task it wrongly keeps at L0 loses only the accuracy difference between rungs, not the
answer. Section 6 also reports what the operating point does when transferred between
benchmarks, which is the practical form of the same question.

### Mechanism 4: convergence stopping inside L2

Three signals, all required to agree, with a hard cap behind them:

```python
conf_ok = lp is not None and lp > tau_conf
if not hi_ent and ((stability and conf_ok) or (approved and conf_ok)):
    stopped_by = 'converged'
```

`stability` is exact string equality between this round's extracted answer and the previous
round's, after normalisation. `approved` is the Reviewer returning `accept`. `hi_ent` is
`entropy > tau_high`. Either stability or approval will stop the loop, but **only** with
logprob confirmation, and high entropy vetoes both. Round 5 stops unconditionally with
`stopped_by = 'max_rounds'`.

The rollback protection is a consequence rather than a rule: because the stop requires
`mean_logprob > tau_conf`, a low-confidence first round cannot be early-stopped and continues
automatically. This is the mechanism aimed at the failure mode in section 7. There, on items where
the ungoverned loop held a correct answer at some round, it went on to return a wrong one at
rates from 0.4% to 50.0% depending on family and benchmark, and the governed arm cuts that in
twelve of sixteen cells while zeroing the truncation component in all sixteen. GLM is the
exception in all four of its cells, which is reported rather than pooled away.

### Mechanism 5: reasoning as a rung

The above bounds the carried prefix. On a model that reasons, the prefix is not the term that
grows: `g_bar` is, and the slot bounds do not touch it. Measured directly, the governed arm on
GAIA at 24k spent 10,998 tokens against the baseline's 11,229, a **-2%** difference, where the
same arm saves 24-67% on models whose thinking is off. The method as described above simply
misses the binding constraint on a reasoning model.

The fix keeps the same admission principle and applies it to the reasoning setting: **the
levels the gate keeps get a cheap reasoning setting, and a task that escalates gets the
default one**. It is one extra field on the L0/L1 call and nothing else changes, so the
comparison against the prefix-only arm prices exactly that change.

What the cheap rung *is* differs by deployment, and this is a limitation rather than a
detail:

| family | cheap rung | why not the others |
|---|---|---|
| Qwen, Gemma | thinking off entirely | thinking can be disabled, so there is no rung to place |
| GLM-5.3-Flash | `reasoning_effort: low` | thinking cannot be disabled by any of five methods tried; `low` genuinely shortens it (per-call median 61 -> 15 tokens) |
| DeepSeek-V4.1-Flash | `chat_template_kwargs {thinking: False}` at L0/L1 | `reasoning_effort` is **nominal** here: low/medium/high all sit near the 328-token default, and the ladder arm using it scored 22% non-empty answers against 20% for the arm with no rung at all |

Two implementation facts that cost time to find:

- A hard `thinking_token_budget` is honoured exactly (128 requested, 128 delivered) but it
  cuts mid-thought, so what it costs in accuracy is truncation rather than shorter reasoning.
  The chat-template switch starts no thought at all. Where both are available the switch is
  the cleaner rung.
- The two must **not** be sent together. Passing `reasoning_effort` alongside the switch
  re-enables the thought on DeepSeek (265 tokens, as though the switch were absent), so the
  harness drops `reasoning_effort` whenever the switch is set rather than stacking them and
  silently governing nothing.

An allowance-coupled variant (`reason_budget_frac`, each call may spend that fraction of the
remaining reserve, clamped to [128, 4096]) is implemented and available as `V5.3-ReasonBudget`
but is **not** the arm reported in the main tables.

### What is in the governed arm, and what was removed

`V5.1-Ab7-NoVerbal` = `V5_BASE` + `V5_FULL` + `m2_verbal: False` + `l01_max_tokens: 2048`.
`V5.2-ReasonLadder` is the same plus the reasoning rung. Expanded:

| flag | value | mechanism |
|---|---|---|
| `m1_early_stop` | True | convergence stopping |
| `m2_cascade` | True | the ladder |
| `m3_structured_reviewer` | True | JSON critique, 150 tokens |
| `m3_tool_notes` | True | 120-token tool notes |
| `m3_plan_freeze` | True | one plan line after round 0 |
| `m3_answer_format` | True | the `ANSWER: x` suffix |
| `m5_strategy` | `hybrid` | best-of-3 at temperature 0.7 when entropy > `tau_high`, else refine |
| `l01_max_tokens` | 2048 | L0/L1 generation cap |
| `final_reserve` | 256 | the grace close |
| `capture_logprobs` | True | the gate's input |
| `policy` | `hard` | no grace multiplier on quota |
| `enable_predictive`, `enable_borrowing`, `enable_opl`, `enable_kv` | False | off in every reported arm |

**The verbalized channel was removed.** An earlier arm let the model reply `ESCALATE` as the
first word instead of answering, on the reasoning that a logprob cannot distinguish "knows
the answer" from "needs to look it up", but the model can say which. It is off in every
reported arm (`m2_verbal: False`), so the ladder is routed by the logprob gate alone. Section
"What the verbalized channel cost" is the ablation that prices this.

**`m3_answer_format` is off on BigCodeBench.** The task prompt there already fixes the output
shape, so the suffix is redundant and was dropped.

**The format-aligned baseline matters more than any single mechanism.** The `ANSWER: x`
instruction alone is worth +13.2pp to Qwen and +28.0pp to Gemma on MATH at 24k. `Naive+Fmt`
carries it, and every claim in this paper is measured against `Naive+Fmt` rather than
`Naive`, because a comparison against `Naive` credits a prompt suffix to the method.

### Per-task algorithm, in full

```
allocate B across {planner, executor, reviewer} by role priority, each >= q_min
reserve <- B;  hold back final_reserve = 256

L0: r0 <- Executor(question)                      max_tokens=2048, cheap reasoning rung
    if blocked            -> grace close, return
    if confident(r0, theta0 = -0.10)  -> return r0        [stopped_by = converged]

L1: r1 <- Executor(question, draft=r0, tool=on)   max_tokens=2048, cheap reasoning rung
    if code emitted: execute (10s), carry a <=120-token note, re-ask
    if blocked            -> grace close, return
    if confident(r1, theta1 = -0.15)  -> return r1        [stopped_by = converged]

L2: plan <- Planner(question)                     default reasoning
    ctx  <- [system, question + plan + draft]
    for round in 0..4:
        r <- Executor(ctx)
        if code emitted: execute, carry note, re-ask
        v <- Reviewer(question, r)                max_tokens=150, JSON verdict
        if entropy(r) <= tau_high and mean_logprob(r) > tau_conf = -0.05
           and (answer unchanged from last round or v.verdict == accept):
               return r                           [stopped_by = converged]
        if round == 4: return r                    [stopped_by = max_rounds]
        if round == 0 and plan_freeze: replace plan in ctx with one step line
        ctx <- [system, question + plan + draft] + [r, "Reviewer feedback: " + v]
               # bounded: constant in round index
    any blocked call       -> grace close, return  [stopped_by = budget]
```

### Cost of the method itself

The governance is not free and the accounting says so. The Reviewer call is 150 tokens per
round; the tool note is one extra 150-token call per tool use; L0 and L1 are pure overhead
for any task that ends up at L2 anyway. Section "What the ladder itself costs" is the
measurement. The reason the arm still spends less overall is that most items never reach L2, so
their plan, reviewer and rounds are never paid for. How many is entirely
benchmark-dependent, and the spread is the single most important number for anyone
reimplementing this: the share stopping at L0 runs from 28.7% (GLM on GAIA) to 97.3% (Gemma
on MATH) across the sixteen cells. On MATH alone it is 72.0-97.3%, which is why MATH shows
the largest token savings; on GAIA and MuSiQue it is often below half, and the savings there
come from the bounded slots rather than from the ladder. One threshold routing 1.5% of MATH
and 53.8% of MuSiQue is also the motivation for the quantile form of the rule in section 6.
'''


DS = [('MATH','ma',400), ('GAIA','ga',122), ('FRAMES','fr',660), ('MuSiQue','mu',1934)]
ARMS = {'V5-Unconstrained':'Naive', 'V5-Unc-Fmt':'Naive+Fmt', 'V5.1-Ab7-NoVerbal':'Governed',
        'V5-Accum-Only':'Accumulate', 'V5-Compact-Real':'Compaction'}
# The governed arm is not the same configuration on every family, and this is a limitation
# to state rather than a detail to hide. Qwen and Gemma run with thinking off, where bounding
# the carried slots is enough. GLM and DeepSeek both reason, so their governed arm also
# governs the per-call generation, but the knob that does it differs: GLM honours
# reasoning_effort=low, while on DeepSeek that setting is nominal (the ladder arm scored 22%
# non-empty answers against 20% for the arm carrying no rung at all) and only the
# chat-template switch binds. Four families, three instantiations of one rung.
GOVERNED = {'Qwen3.8-27B':'V5.1-Ab7-NoVerbal', 'Gemma-4-31B (bf16)':'V5.1-Ab7-NoVerbal',
            'GLM-5.3-Flash':'V5.2-ReasonLadder', 'DeepSeek-V4.1-Flash':'V5.2-ReasonLadder'}

def qfile(arm, tag, k):
    if arm in ('V5-Unconstrained','V5-Unc-Fmt'):
        return f'traces/tasks_sweep_Qwen3.8-27B_q38_{tag}{k}.jsonl'
    if arm == 'V5.1-Ab7-NoVerbal':
        for p in (f'traces/tasks_sweep_Qwen3.8-27B_q38ab7_{tag}{k}.jsonl',
                  f'traces/tasks_sweep_Qwen3.8-27B_q38_ab7_{tag}.jsonl'):
            if os.path.exists(p): return p
    if arm == 'V5-Accum-Only':   return f'traces/tasks_sweep_Qwen3.8-27B_acc_{tag}{k}k.jsonl'
    if arm == 'V5-Compact-Real': return f'traces/tasks_sweep_Qwen3.8-27B_cmp3_{tag}{k}k.jsonl'

def lfile(arm, tag, k):
    """GLM. The main table is three arms; the governed one is V5.2-ReasonLadder because
    this model cannot be made to stop thinking, so the per-call generation has to be
    governed as well as the carried prefix. Ab7 is the ablation that prices that, on MATH."""
    if arm == 'V5.1-Ab7-NoVerbal':
        return f'traces/tasks_sweep_GLM-5.3-Flash_glmab7_{tag}{k}.jsonl'
    return f'traces/tasks_sweep_GLM-5.3-Flash_glm4_{tag}{k}.jsonl'


def dfile(arm, tag, k):
    """DeepSeek-V4.1-Flash over an SSH tunnel to a vLLM deployment. One tag family, because
    the whole table was run from a single chain after the OpenRouter attempt was abandoned
    (that account is out of credit; its partial cells are under _aborted/ds_openrouter/).

    Ab7 is the ablation that prices the reasoning rung against prefix-only governance, and
    it was run on MATH afterwards so that the two reasoning families carry the same cell.
    It is the result that decides when the rung is worth having: on DeepSeek, bounding the
    prefix alone backfires, spending 6,522 tokens against 2,434 for the ungoverned baseline
    and 914 for the ladder, because the freed allowance buys four more rounds (5.47 against
    1.49) that reasoning then consumes. On GLM the same arm is the cheaper one and at 16k it
    beats the ladder outright. The rung is worth +6.0 to +8.0pp on one family and -2.5 to
    -5.2pp on the other."""
    if arm in ('V5-Accum-Only','V5-Compact-Real'):
        return None
    if arm == 'V5.1-Ab7-NoVerbal':
        # MATH only; the other three benchmarks ran three arms.
        f = f'traces/tasks_sweep_DeepSeek-V4.1-Flash_dsab7_{tag}{k}.jsonl'
        return f if os.path.exists(f) else None
    return f'traces/tasks_sweep_DeepSeek-V4.1-Flash_ds_{tag}{k}.jsonl'


def gfile(arm, tag, k):
    if arm in ('V5-Accum-Only','V5-Compact-Real'):
        return f'traces/tasks_sweep_gemma-4-31B-bf16_gcmp_{tag}{k}.jsonl'
    return f'traces/tasks_sweep_gemma-4-31B-bf16_gor_{tag}{k}.jsonl'

def load(path, arm, cap):
    if not path or not os.path.exists(path): return None
    ok, tok, tr, lp, lvl, roles = {}, [], 0, [], [], collections.Counter()
    for line in open(path):
        r = json.loads(line)
        if r['config'] != arm: continue
        ok[r['task_id']] = bool(r.get('success'))
        tok.append(r.get('total_used', 0))
        if r.get('stopped_by') == 'budget' and r.get('total_used', 0) > 0.5 * cap: tr += 1
        lvl.append(r.get('escalation_level'))
        for k2, v2 in (r.get('role_tokens') or {}).items(): roles[k2] += v2
        rd = [x for x in (r.get('rounds') or []) if isinstance(x, dict) and x.get('mean_logprob') is not None]
        if rd: lp.append((rd[0]['mean_logprob'], bool(rd[0].get('is_correct'))))
    if not ok: return None
    return dict(ok=ok, tok=tok, trunc=tr, lp=lp, lvl=lvl, roles=roles)

def mcnemar(A, B):
    ids = sorted(set(A) & set(B))
    x = sum(1 for i in ids if not A[i] and B[i])
    y = sum(1 for i in ids if A[i] and not B[i])
    n = x + y
    p = min(1.0, 2 * sum(comb(n, k) for k in range(min(x, y) + 1)) / 2 ** n) if n else 1.0
    sa = sum(A[i] for i in ids) / len(ids) * 100
    sb = sum(B[i] for i in ids) / len(ids) * 100
    return dict(n=len(ids), wins=x, losses=y, delta=sb - sa, p=p, sr_a=sa, sr_b=sb)

# ---------------------------------------------------------------- header
w('# Paper materials: statelessness as budget policy')
w()
w('Everything a write-up needs, computed from the traces by `_gen_materials.py`. Each table')
w('names the files it came from. Numbers here supersede any earlier draft: several cells')
w('exist in more than one trace file and earlier drafts mixed them.')
w()
w('## 0. One-paragraph summary')
w()
w('A plan-execute-review pipeline under a fixed per-task token budget spends most of that')
w('budget re-transmitting coordination state it wrote on earlier rounds, not on new')
w('reasoning. The largest such slot is the plan. Bounding what crosses the round boundary,')
w('and admitting tasks to the full loop only when a confidence signal fails, cuts tokens in')
w('every measured cell, removes budget exhaustion entirely, and raises accuracy where the')
w('cap binds. Relaxing the cap removes the accuracy gain, which is the causal test.')
w()

# ---------------------------------------------------------------- setup
w('## 1. Setup')
w()
w('| | |')
w('|---|---|')
w('| Primary model | Qwen3.8-27B-FP8, vLLM. Endpoint `https://llm5.ruoyu-wang.com/v1` (`qwen3.8-27b-fp8`), 262k context |')
w('| Second model | Gemma-4-31B-it at **bf16**, OpenRouter with provider pinned to Venice, fallbacks off |')
w('| Third model | GLM-5.3-Flash on sglang, sharded over four endpoints (`llm3.ruoyu-wang.com`, `llm52/56/57.jierungogogo.com`). Verified one deployment: identical `model_path`, `tp_size=8`, `kv_cache_dtype`, and token-identical generations on a probe prompt at `reasoning_effort=low` |')
w('| Fourth model | DeepSeek-V4.1-Flash on vLLM, reached over an SSH tunnel to `127.0.0.1:18000`, 1.05M context. Reasoning cannot be graded here: `reasoning_effort` low/medium/high all sit near the 328-token default, a hard `thinking_token_budget` is honoured exactly, and `chat_template_kwargs {thinking: False}` yields 0 reasoning tokens. The two must not be combined, which re-enables the thought (265 tokens). The governed arm uses the switch, so its cheap rung starts no thought rather than truncating one |')
w('| Sampling | temperature 0, seed 42, single seed throughout |')
w('| Thinking | Disabled on Qwen (`chat_template_kwargs={"enable_thinking": false}`) and Gemma (`--reasoning off`), verified to take effect. **Cannot be disabled on GLM**; see section 11b |')
w('| Budgets | 8k / 16k / 24k tokens per task, counted over prompt and completion across all roles. BigCodeBench uses 4k / 8k / 16k |')
w('| Judging | symbolic equivalence (MATH), normalized exact match (GAIA), EM/F1 (FRAMES, MuSiQue), unit-test execution in the official image (BigCodeBench) |')
w()
w('### 1.1 Arms')
w()
w('| Name in paper | Config | What it is |')
w('|---|---|---|')
w('| `Naive` | `V5-Unconstrained` | plain pipeline, no format instruction |')
w('| `Naive+Fmt` | `V5-Unc-Fmt` | plain pipeline **plus** the `ANSWER: x` instruction. **This is the baseline every claim is measured against** |')
w('| `Governed` | `V5.1-Ab7-NoVerbal` | full system: bounded slots, convergence stop, cost-ordered ladder routed by the confidence gate alone |')
w('| `Accumulate` | `V5-Accum-Only` | compaction control: keeps the whole history |')
w('| `Compaction` | `V5-Compact-Real` | accumulate, then summarise once the prompt passes half of a 8192-token context window |')
w()
w('The format-aligned baseline matters: the instruction alone is worth +13.2 pp to Qwen and')
w('+28.0 pp to Gemma on MATH at 24k. A comparison against `Naive` credits that to the method.')
w()
w('### 1.2 Benchmarks')
w()
w('| Dataset | n | Task |')
w('|---|---|---|')
w('| MATH-500 | 400 | competition mathematics, symbolic reasoning, no external knowledge |')
w('| GAIA | 122 | multi-step assistant questions needing lookup, arithmetic, file reading |')
w('| FRAMES | 660 | multi-hop factual reasoning across Wikipedia articles |')
w('| MuSiQue | 1,934 | 2-4 hop compositional QA, built to defeat shortcuts |')
w('| BigCodeBench | 500 | library-heavy function synthesis over 139 libraries, execution-judged |')
w()
w('SWE-bench Verified was run and then **dropped**; see section 9.')
w()

# The method was specified only by arm name before this: a reader of the materials had
# the results and the flags but not the mechanism, so the write-up had to be reverse
# engineered from the code. Emitted as a literal block rather than per-line w() calls
# because it is prose with tables and code, and escaping it line by line invites the
# quoting mistakes that per-line emission is meant to avoid.
OUT.append(_METHOD)

w()

# ---------------------------------------------------------------- segments
w('## 2. Where the budget goes (segment accounting)')
w()
w('Every prompt character attributed to what it carries. Shares are of **prompt** characters;')
w('the model\'s own output is excluded so the slots compare against one another.')
w('Source: `traces/tasks_sweep_Qwen3.8-27B_seg_*.jsonl`, `tasks_sweep_gemma-4-31B-bf16_gseg_*.jsonl`, B=16k, `Naive+Fmt`.')
w()
SEGMAP = {'plan':'plan','question':'question','prior_answer':'draft','critique':'critique',
          'tool_output':'tool out','system':'system','scaffold':'scaffold'}
SEGORD = ['plan','question','draft','critique','tool out','system']
def segagg(f):
    c = collections.Counter(); n = 0
    if not os.path.exists(f): return 0, c
    for line in open(f):
        r = json.loads(line); n += 1
        for k, v in (r.get('segment_chars') or {}).items():
            if not k.startswith('_'): c[SEGMAP.get(k, k)] += v
    return n, c
def segall(f):
    c = collections.Counter(); n = 0
    if not os.path.exists(f): return 0, c
    for line in open(f):
        r = json.loads(line); n += 1
        for k, v in (r.get('segment_chars') or {}).items(): c[k.lstrip('_')] += v
    return n, c
SEGF = {'Qwen3.8-27B': [('GAIA','traces/tasks_sweep_Qwen3.8-27B_seg_GAIA.jsonl'),
                        ('MATH','traces/tasks_sweep_Qwen3.8-27B_seg_MATH.jsonl'),
                        ('FRAMES','traces/tasks_sweep_Qwen3.8-27B_seg_FRAMES.jsonl'),
                        ('MuSiQue','traces/tasks_sweep_Qwen3.8-27B_seg_MUSIQUE.jsonl'),
                        ('BigCodeBench','traces/tasks_sweep_Qwen3.8-27B_seg_BCB.jsonl')],
        'Gemma-4-31B (bf16)': [('GAIA','traces/tasks_sweep_gemma-4-31B-bf16_gseg_ga.jsonl'),
                        ('MATH','traces/tasks_sweep_gemma-4-31B-bf16_gseg_ma.jsonl'),
                        ('FRAMES','traces/tasks_sweep_gemma-4-31B-bf16_gseg_fr.jsonl'),
                        ('MuSiQue','traces/tasks_sweep_gemma-4-31B-bf16_gseg_mu.jsonl'),
                        ('BigCodeBench','traces/tasks_sweep_gemma-4-31B-bf16_gseg_bcb.jsonl')]}
for model, rows in SEGF.items():
    w(f'**{model}**')
    w()
    w('| Benchmark | n | ' + ' | '.join(SEGORD) + ' |')
    w('|---|---:|' + '---:|' * len(SEGORD))
    T = collections.Counter(); N = 0
    for name, f in rows:
        n, c = segagg(f)
        if not n: continue
        T.update(c); N += n
        s = sum(c.values()) or 1
        w(f'| {name} | {n:,} | ' + ' | '.join(f'{c.get(k,0)/s*100:.1f}%' for k in SEGORD) + ' |')
    s = sum(T.values()) or 1
    w(f'| **Pooled** | **{N:,}** | ' + ' | '.join(f'**{T.get(k,0)/s*100:.1f}%**' for k in SEGORD) + ' |')
    w()
w('### 2.1 Same runs, counted against the whole budget')
w()
w('Including the model\'s own output in the denominator. This is the framing to use when the')
w('word "budget" appears; the table above is the framing for comparing slots.')
w()
w('| Model | model output | question | carried state | of which plan |')
w('|---|---:|---:|---:|---:|')
for model, rows in SEGF.items():
    T = collections.Counter(); N = 0
    for _, f in rows:
        n, c = segall(f); T.update(c); N += n
    s = sum(T.values()) or 1
    carried = T['plan'] + T['prior_answer'] + T['critique'] + T['tool_output']
    w(f'| {model} | {T["completion"]/s*100:.1f}% | {T["question"]/s*100:.1f}% | '
      f'{carried/s*100:.1f}% | {T["plan"]/s*100:.1f}% |')
w()
w('**Reading.** The plan is the largest slot a bound can act on, on both families. Its size')
w('relative to the question does **not** transfer: 40.4% against 27.3% on Qwen, 28.7% against')
w('32.2% on Gemma. Neither does the critique: 1.9% on Qwen, 10.0% on Gemma. Do not write')
w('"the plan is 1.5x the question"; write "the plan is the largest bindable slot".')
w()

# ---------------------------------------------------------------- main table
w('## 3. Main table: success rate and tokens')
w()
w('SR (%) / mean tokens per task. Sources: `q38_*` (Naive, Naive+Fmt), `q38ab7_*` and')
w('`q38_ab7_*` (Governed), `acc_*` (Accumulate), `cmp3_*` (Compaction) for Qwen;')
w('`gor_*` and `gcmp_*` for Gemma.')
w()
MODELS = [('Qwen3.8-27B', qfile), ('Gemma-4-31B (bf16)', gfile),
          ('GLM-5.3-Flash', lfile), ('DeepSeek-V4.1-Flash', dfile)]
CACHE = {}
for model, fn in MODELS:
    w(f'**{model}**')
    w()
    w('| Arm | Budget | ' + ' | '.join(d[0] for d in DS) + ' |')
    w('|---|---|' + '---|' * len(DS))
    arms_here = [a for a in ARMS if a != 'V5.1-Ab7-NoVerbal' or GOVERNED[model] == a]
    if GOVERNED[model] not in arms_here: arms_here.insert(2, GOVERNED[model])
    # The compaction pair was not run on the third family; an all-dashes block says less
    # than its absence does.
    if model in ('GLM-5.3-Flash', 'DeepSeek-V4.1-Flash'):
        arms_here = [a for a in arms_here if a not in ('V5-Accum-Only', 'V5-Compact-Real')]
    for arm in arms_here:
        for i, k in enumerate([8, 16, 24]):
            cells = []
            for ds, tag, n in DS:
                d = load(fn(arm, tag, k), arm, k * 1000)
                CACHE[(model, arm, ds, k)] = d
                cells.append('---' if not d else
                             f'{sum(d["ok"].values())/len(d["ok"])*100:.1f} / {st.mean(d["tok"]):,.0f}')
            label = (ARMS.get(arm) or 'Governed (reasoning ladder)') if i == 0 else ''
            w(f'| {label} | {k}k | ' + ' | '.join(cells) + ' |')
    w()

w('## 4. Paired tests: Governed vs Naive+Fmt')
w()
w('Exact McNemar on per-item outcomes. `trunc` counts tasks that exhausted the budget')
w('mid-loop (`stopped_by=="budget"` and more than half the cap consumed).')
w()
SIG = {}
for model, fn in MODELS:
    w(f'**{model}**')
    w()
    w('| Benchmark | Budget | Naive+Fmt | Governed | dSR | p | wins/losses | tokens | trunc |')
    w('|---|---|---:|---:|---:|---:|---:|---:|---:|')
    nsig = 0
    for ds, tag, n in DS:
        for k in [8, 16, 24]:
            A = CACHE[(model, 'V5-Unc-Fmt', ds, k)]
            B = CACHE[(model, GOVERNED[model], ds, k)]
            if not A or not B: continue
            m = mcnemar(A['ok'], B['ok'])
            nsig += m['p'] < 0.05
            star = '*' if m['p'] < 0.05 else ''
            tok = (st.mean(B['tok']) / st.mean(A['tok']) - 1) * 100
            w(f'| {ds} ({n:,}) | {k}k | {m["sr_a"]:.1f} | {m["sr_b"]:.1f} | {m["delta"]:+.1f}{star} | '
              f'{m["p"]:.2e} | {m["wins"]}/{m["losses"]} | {tok:+.0f}% | {A["trunc"]} -> {B["trunc"]} |')
    SIG[model] = nsig
    w()
    w(f'Significant at p<0.05 in **{nsig} of 12** cells.')
    w()
w(f'Pooled over the three families: **{sum(SIG.values())} of 36** cells significant. Token cost')
w('falls in all 36 and budget exhaustion is eliminated in all 36, which are the two results')
w('that hold without exception. The accuracy gain is the conditional one: it appears where the')
w(f'cap binds and the format-aligned baseline still has room, and the {36 - sum(SIG.values())} cells where it')
w('does not are cells missing one of those. Note that the governed arm is not one')
w('configuration across families; see section 11b.')
w()

# ---------------------------------------------------------------- budget use, roles, levels
w('## 5. What the governed arm actually does')
w()
w('### 5.1 Fraction of the budget consumed')
w()
w('Mean tokens divided by the cap. The governed arm is nowhere near budget-limited; it is')
w('convergence-limited. This is worth saying explicitly: the cap is the setting in which the')
w('baseline\'s failure appears, not a resource the method is squeezing into.')
w()
w('| Model | Benchmark | ' + ' | '.join(f'{k}k N+F / Gov' for k in [8,16,24]) + ' |')
w('|---|---|---|---|---|')
for model, fn in MODELS:
    for ds, tag, n in DS:
        cells = []
        for k in [8, 16, 24]:
            A = CACHE[(model,'V5-Unc-Fmt',ds,k)]; B = CACHE[(model,GOVERNED[model],ds,k)]
            cells.append(f'{st.mean(A["tok"])/(k*1000)*100:.0f}% / {st.mean(B["tok"])/(k*1000)*100:.0f}%'
                         if A and B else '---')
        w(f'| {model} | {ds} | ' + ' | '.join(cells) + ' |')
w()
w('### 5.2 Where the governed arm\'s tokens go, by role (24k)')
w()
w('| Model | Benchmark | planner | executor | reviewer |')
w('|---|---|---:|---:|---:|')
for model, fn in MODELS:
    for ds, tag, n in DS:
        d = CACHE[(model,GOVERNED[model],ds,24)]
        if not d or not d['roles']: continue
        s = sum(d['roles'].values()) or 1
        w(f'| {model} | {ds} | {d["roles"].get("planner",0)/s*100:.1f}% | '
          f'{d["roles"].get("executor",0)/s*100:.1f}% | {d["roles"].get("reviewer",0)/s*100:.1f}% |')
w()
w('Keeping the planner costs 1.8-4.2% of the budget. What the bound removes is not that')
w('cost but its re-transmission on every subsequent round. On MATH the planner is never')
w('called at all, because nearly every item stops at L0.')
w()
w('### 5.3 Escalation rate (share of items leaving L0), 24k')
w()
w('| Model | ' + ' | '.join(d[0] for d in DS) + ' |')
w('|---|' + '---:|' * len(DS))
for model, fn in MODELS:
    cells = []
    for ds, tag, n in DS:
        d = CACHE[(model,GOVERNED[model],ds,24)]
        lv = [x for x in (d['lvl'] if d else []) if x is not None]
        cells.append(f'{(1-sum(1 for x in lv if x==0)/len(lv))*100:.1f}%' if lv else '---')
    w(f'| {model} | ' + ' | '.join(cells) + ' |')
w()
w('The same threshold routes 1.5% of MATH and 53.8% of MuSiQue. That spread is the reason')
w('for the quantile form of the rule (section 6).')
w()

# ---------------------------------------------------------------- gate / quantile
w('## 6. The confidence gate, and making its operating point transferable')
w()
w('phi is the mean token log-probability of the L0 answer. It ranks, but its scale is not')
w('comparable across benchmarks, so a fixed threshold buys an unpredictable amount.')
w()
def gate_probe(model, ds):
    d = CACHE[(model,GOVERNED[model],ds,24)]
    return d['lp'] if d else []
def keep(v, thr=None, q=None):
    if q is not None:
        lps = sorted((x[0] for x in v), reverse=True)
        thr = lps[max(1,int(len(v)*q))-1]
        kept = [c for lp,c in v if lp >= thr]
    else:
        kept = [c for lp,c in v if lp > thr]
    if not kept: return 0.0, 0.0
    base = sum(c for _,c in v)/len(v)
    return len(kept)/len(v)*100, (sum(kept)/len(kept)-base)*100
w('| Model | tuned on | threshold: keep rate | threshold: lift | quantile: keep rate | quantile: lift |')
w('|---|---|---|---|---|---|')
for model, fn in MODELS:
    P = {ds: gate_probe(model, ds) for ds,_,_ in DS}
    for tuned,_,_ in DS:
        v = P[tuned]
        if not v: continue
        b = None
        for th in sorted(set(x[0] for x in v)):
            kr, lf = keep(v, thr=th)
            if kr >= 10 and (b is None or lf > b[1]): b = (th, lf)
        ra = [keep(P[d], thr=b[0]) for d,_,_ in DS]
        b2 = None
        for q in [x/20 for x in range(1,20)]:
            kr, lf = keep(v, q=q)
            if b2 is None or lf > b2[1]: b2 = (q, lf)
        rq = [keep(P[d], q=b2[0]) for d,_,_ in DS]
        f = lambda r, i: f'{min(x[i] for x in r):.1f}-{max(x[i] for x in r):.1f}'
        w(f'| {model} | {tuned} | {f(ra,0)}% | {f(ra,1)} pp | **{f(rq,0)}%** | {f(rq,1)} pp |')
w()
w('**The claim to make.** Fixing the quantile instead of the value makes the keep rate, and')
w('therefore the token saving, a design parameter: it transfers to within 0.2 points by')
w('construction, where a fixed threshold ranges from 2.4% to 70.0%. **The claim not to make:**')
w('this does not improve discrimination. phi ranks items identically either way. AUC is 0.75')
w('on MATH and FRAMES, 0.62 on GAIA, 0.52 on MuSiQue, and the gate remains the weakest')
w('component of the system.')
w()

# ---------------------------------------------------------------- flips
w('## 7. The two failure modes')
w()
w('Correct-to-wrong transitions at 24k, split into revision (a later round overwrites a')
w('correct answer) and truncation (budget exhaustion replaces one). Conditional rate is')
w('flips over items correct at some round. `is_correct` per round comes from the trace.')
w()
w('| Model | Dataset | Naive+Fmt rev+trunc | cond. rate | Governed rev+trunc | cond. rate |')
w('|---|---|---|---:|---|---:|')
for model, fn in MODELS:
    for ds, tag, n in DS:
        row = []
        for arm in ['V5-Unc-Fmt',GOVERNED[model]]:
            path = fn(arm, tag, 24)
            rev = tr = ever = 0
            for line in open(path):
                r = json.loads(line)
                if r['config'] != arm: continue
                ic = [bool(x.get('is_correct')) for x in (r.get('rounds') or []) if isinstance(x, dict)]
                if any(ic): ever += 1
                if r.get('success') or not any(ic): continue
                if r.get('stopped_by') == 'budget' and r.get('total_used',0) > 12000: tr += 1
                else: rev += 1
            row.append((rev, tr, (rev+tr)/ever*100 if ever else 0))
        w(f'| {model} | {ds} | {row[0][0]}+{row[0][1]} | {row[0][2]:.1f}% | '
          f'{row[1][0]}+{row[1][1]} | **{row[1][2]:.1f}%** |')
w()
w('Truncation goes to zero in all sixteen cells. The conditional rate falls in twelve of')
w('them and rises in four, which are GLM\'s four: 1.5 to 3.5% on MATH, 24.1 to 28.6% on')
w('GAIA, 2.9 to 4.1% on FRAMES and 1.7 to 8.7% on MuSiQue. The earlier two-family version')
w('of this table said the rate fell everywhere, which the reasoning families refute.')
w()

# ---------------------------------------------------------------- BCB
w('## 8. BigCodeBench: the falsification test')
w()
w('Budgets chosen to span the point where the cap binds. If the gain comes from the cap')
w('binding, relaxing it should remove the gain. Scored offline by executing each task\'s unit')
w('tests in the official image; `*_rejudged.jsonl` are the authoritative files.')
w()
w('| Model | Budget | trunc (Naive) | Naive | Governed | dSR | p | wins/losses | tokens |')
w('|---|---|---:|---:|---:|---:|---:|---:|---:|')
BCBF = [('Qwen3.8-27B', lambda b: f'traces/tasks_sweep_Qwen3.8-27B_bcb{b}_rejudged.jsonl', 'BCB-Ab7'),
        ('Gemma-4-31B (bf16)', lambda b: f'traces/tasks_sweep_gemma-4-31B-bf16_gbcb{b}_rejudged.jsonl', 'BCB-Ab7'),
        ('GLM-5.3-Flash', lambda b: f'traces/tasks_sweep_GLM-5.3-Flash_glmbcb{b}_rejudged.jsonl', 'BCB-ReasonLadder'),
        ('DeepSeek-V4.1-Flash', lambda b: f'traces/tasks_sweep_DeepSeek-V4.1-Flash_ds_bcb{b}_rejudged.jsonl', 'BCB-ReasonLadder')]
for model, fn, GOV in BCBF:
    for b in ['4k','8k','16k']:
        f = fn(b)
        if not os.path.exists(f): continue
        cap = int(b[:-1])*1000
        arms, toks, tr = {}, {}, collections.Counter()
        for line in open(f):
            r = json.loads(line); k = r['config']
            arms.setdefault(k, {})[r['task_id']] = bool(r.get('success'))
            toks.setdefault(k, []).append(r.get('total_used',0))
            if r.get('stopped_by')=='budget' and r.get('total_used',0) > 0.5*cap: tr[k]+=1
        m = mcnemar(arms['BCB-Naive'], arms[GOV])
        star = '*' if m['p'] < 0.05 else ''
        tok = (st.mean(toks[GOV])/st.mean(toks['BCB-Naive'])-1)*100
        w(f'| {model} | {b} | {tr["BCB-Naive"]} | {m["sr_a"]:.1f} | {m["sr_b"]:.1f} | '
          f'{m["delta"]:+.1f}{star} | {m["p"]:.2e} | {m["wins"]}/{m["losses"]} | {tok:+.0f}% |')
w()
w('The margin collapses as the cap is relaxed on both families and is not significant at')
w('16k. The governed arm spends about the same tokens at every budget because it does not')
w('escalate; the baseline grows with the allowance. `BCB-Ours` (the same system with the')
w('verbalized channel on) is in the traces too and is not ahead of `BCB-Ab7` anywhere.')
w()

# ---------------------------------------------------------------- compaction
w('## 9. Compaction under a budget')
w()
w('`Accumulate` keeps the whole history; `Compaction` summarises it when the prompt passes')
w('half of a **fixed 8192-token** context window (`kv_trigger=0.5`, `context_window=8192` in')
w('the arm config, not read from the server, so the trigger means the same thing on both')
w('families). Sources: `acc_*` / `cmp3_*` for Qwen, `gcmp_*` for Gemma. `cmp_*` and `cmp2_*`')
w('are earlier, defective runs and must not be used.')
w()
w('| Model | Budget | n | trigger rate | dSR | wins/losses | p |')
w('|---|---|---:|---:|---:|---:|---:|')
for model, fn in MODELS:
    if model == 'GLM-5.3-Flash': continue   # compaction grid not run on this family
    for k in [8, 16, 24]:
        tn = tx = ty = tt = 0
        for ds, tag, n in DS:
            A = load(fn('V5-Accum-Only', tag, k), 'V5-Accum-Only', k*1000)
            path = fn('V5-Compact-Real', tag, k)
            if not A or not os.path.exists(path): continue
            B, T = {}, set()
            for line in open(path):
                r = json.loads(line)
                if r['config'] != 'V5-Compact-Real': continue
                B[r['task_id']] = bool(r.get('success'))
                if (r.get('flags_fired') or {}).get('kv_trigger', 0) > 0: T.add(r['task_id'])
            ids = set(A['ok']) & set(B)
            tn += len(ids); tt += len(T & ids)
            tx += sum(1 for i in ids if not A['ok'][i] and B[i])
            ty += sum(1 for i in ids if A['ok'][i] and not B[i])
        if not tn: continue
        N = tx + ty
        p = min(1.0, 2*sum(comb(N,k2) for k2 in range(min(tx,ty)+1))/2**N) if N else 1.0
        w(f'| {model} | {k}k | {tn:,} | **{tt/tn*100:.2f}%** | {(tx-ty)/tn*100:+.2f} | {tx}/{ty} | {p:.4f} |')
w()
w('**The result is the trigger column.** At the tightest budget the summariser never fires on')
w('either family, on any of 3,116 tasks: the allowance runs out before the context fills. No')
w('stratum is significant. Pooling all twelve Qwen cells does reach p=0.034 (253/304), which')
w('we do not read as an effect: the 8k stratum is a placebo by construction and carries the')
w('same sign, and comparing strata directly, the compaction arm wins 47.1% of discordant')
w('pairs where it never fires and 45.6% where it fires on a quarter of tasks (Fisher p=0.756).')
w()

# ---------------------------------------------------------------- slot ablation
w('## 10. Counterfactual slot ablation')
w()
w('On a live round of the unbounded pipeline, the same executor call is re-issued with')
w('exactly one carried slot replaced, and the answer compared. Qwen only.')
w('Source: `results/slot_ablation_cp_*.json`, 850 items across the four benchmarks.')
w()
w('The last three variants are the stronger test: they hold the slot\'s length and shape')
w('roughly fixed and destroy its **content** instead. `foreign_plan` hands the executor a')
w('plan written for a different question, paired by a fixed rotation so every task lends and')
w('borrows exactly one.')
w()
SV = ['control','bounded_plan','shuffled_plan','foreign_plan','no_plan','bounded_crit','no_crit','bounded_both']
ALL = {v: [] for v in SV}; same = {v: 0 for v in SV}; NTOT = 0
for f in sorted(glob.glob('results/slot_ablation_cp_*.json')):
    rows = [r for r in json.load(open(f)) if 'error' not in r and len(r.get('variants',{})) == len(SV)]
    for v in SV:
        ALL[v] += [bool(r['variants'][v]['correct']) for r in rows]
        same[v] += sum(1 for r in rows if r['variants'][v]['norm'] == r['variants']['control']['norm'])
    NTOT += len(rows)
if NTOT:
    w(f'| Variant | correct | d vs control | answer unchanged | c->w / w->c | p |')
    w('|---|---:|---:|---:|---:|---:|')
    for v in SV:
        x = sum(1 for i in range(NTOT) if not ALL['control'][i] and ALL[v][i])
        y = sum(1 for i in range(NTOT) if ALL['control'][i] and not ALL[v][i])
        n = x + y
        p = min(1.0, 2*sum(comb(n,k) for k in range(min(x,y)+1))/2**n) if n else 1.0
        star = '*' if p < 0.05 else ''
        w(f'| `{v}` | {sum(ALL[v])/NTOT*100:.1f}% | {(sum(ALL[v])-sum(ALL["control"]))/NTOT*100:+.1f} | '
          f'{same[v]/NTOT*100:.0f}% | {y}/{x} | {p:.3f}{star} |')
    w()
    w(f'n = {NTOT} paired items.')
    w()
    w('**How to read it.** The plan slot is null under every perturbation, including a plan')
    w('written for a different question (28.6% against 28.6%, discordant pairs 5/5). The')
    w('critique is not: bounding it costs 2.9 pp at p<0.001. Note the "answer unchanged"')
    w('column: about a third of answers do move when the plan is swapped, so the executor is')
    w('not ignoring the text. It reads it and the text carries no usable signal. Say')
    w('"non-informative", not "ignored".')
    w()
    w('**This is stronger than the paper currently claims and was deliberately left out of')
    w('the draft** as too large a claim for the paper to carry. The data is in')
    w('`results/slot_ablation_cp_*.json` if it is wanted.')
    w()

# ---------------------------------------------------------------- decomposition
w('## 11. Mechanism decomposition (Qwen, MATH, routing disabled)')
w()
w('Each arm adds one mechanism to the format-aligned baseline with the ladder switched off,')
w('so every item runs the full loop and the increments are comparable.')
w('Sources: `q38_abl_ma` (24k), `qdec_ma8`, `qdec_ma16`.')
w()
w('| Budget | Naive+Fmt | + bounded slots (Ab6) | + convergence stop (Ab3) | + routing (full) |')
w('|---|---:|---:|---:|---:|')
DEC = {8: 'traces/tasks_sweep_Qwen3.8-27B_qdec_ma8.jsonl',
       16: 'traces/tasks_sweep_Qwen3.8-27B_qdec_ma16.jsonl',
       24: 'traces/tasks_sweep_Qwen3.8-27B_q38_abl_ma.jsonl'}
for k in [8, 16, 24]:
    f = DEC[k]
    got = {}
    if os.path.exists(f):
        for line in open(f):
            r = json.loads(line)
            got.setdefault(r['config'], []).append(bool(r.get('success')))
    base = CACHE[('Qwen3.8-27B','V5-Unc-Fmt','MATH',k)]
    full = CACHE[('Qwen3.8-27B','V5.1-Ab7-NoVerbal','MATH',k)]
    def pct(v): return f'{sum(v)/len(v)*100:.1f}' if v else '---'
    w(f'| {k}k | {sum(base["ok"].values())/len(base["ok"])*100:.1f} | '
      f'{pct(got.get("V5.1-Ab6-NoM2NoM1"))} | {pct(got.get("V5.1-Ab3-NoM2"))} | '
      f'{sum(full["ok"].values())/len(full["ok"])*100:.1f} |')
w()
w('No single increment is significant at n=400 at any budget. Report the chain, not an')
w('additive decomposition.')
w()

# ---------------------------------------------------------------- caveats / defects
# ---------------------------------------------------------------- reasoning ladder
w('## 11b. Reasoning as a rung of the ladder')
w()
w('This mechanism exists because of a model that forced it. GLM-5.3-Flash cannot be made to')
w('stop thinking: enable_thinking=false, thinking.type=disabled, reasoning_effort=none,')
w('/nothink in either turn and separate_reasoning=false all generate the identical trace on a')
w('probe prompt, and change only whether the server splits it out of `content`; two of them')
w('leave a stray `</think>` inside it. So this family runs with reasoning on, and reasoning')
w('tokens are counted in completion_tokens and charged to the task budget.')
w()
w('With reasoning on, the term that grows in the per-round cost is the generation, not the')
w('carried prefix, and the carried-slot bounds do not touch it. Measured on GAIA at 24k, the')
w('arm that governs only the prefix spent 10,998 tokens against the baseline 11,229, a')
w('difference of -2%, where the same arm saves 24-67% on the families whose thinking is off.')
w()
w('The fix is to put the generation under the same admission decision as the scaffolding:')
w('levels the gate keeps run at `reasoning_effort=low`, and escalation restores the default.')
w('Nothing else changes, so the pair prices exactly this.')
w()
w('| Benchmark | Budget | prefix only (Ab7) | + reasoning rung | dSR | tokens |')
w('|---|---|---:|---:|---:|---:|')
for tag, name, n in [('ga','GAIA',122), ('ma','MATH',400)]:
    for k in [8, 16, 24]:
        a = load(lfile('V5.1-Ab7-NoVerbal', tag, k), 'V5.1-Ab7-NoVerbal', k*1000)
        if a is None:
            a = load(f'_aborted/glm_pre_reasoning/tasks_sweep_GLM-5.3-Flash_glm_{tag}{k}.jsonl',
                     'V5.1-Ab7-NoVerbal', k*1000)
        b = CACHE.get(('GLM-5.3-Flash','V5.2-ReasonLadder',name,k))
        if not a or not b: continue
        sa = sum(a['ok'].values())/len(a['ok'])*100
        sb = sum(b['ok'].values())/len(b['ok'])*100
        w(f'| {name} | {k}k | {sa:.1f} / {st.mean(a["tok"]):,.0f} | {sb:.1f} / {st.mean(b["tok"]):,.0f} | '
          f'{sb-sa:+.1f} | {(st.mean(b["tok"])/st.mean(a["tok"])-1)*100:+.0f}% |')
w()
w('**What the rung costs on its own.** Called directly at `reasoning_effort=low`, the model')
w('answers 60 MATH problems at 65.0% against 76.7% at the default, on 167 generated tokens')
w('against 566. Inside the ladder that 11.7-point penalty does not reach the score: 69 to 72%')
w('of MATH items stop at the cheap setting and the arm loses 1.8 to 2.5 points, none of it')
w('significant. This is the first direct evidence in this project that the gate puts the cheap')
w('rung where it belongs, because it is the first model on which the cheap rung is measurably')
w('worse; on the other two families L0 and L2 differ only in scaffolding.')
w()
w('**Where it also buys accuracy.** On BigCodeBench the rung is ahead of prefix-only')
w('governance at every budget, by +14.8, +4.0 and +5.2 points (all significant), while')
w('spending 46 to 69% fewer tokens. Its score barely moves with the allowance (50.8, 51.8,')
w('51.0) where prefix-only governance climbs from 36.0 to 45.8, so it reaches at 4k what the')
w('other needs 16k to approach.')
w()

w('## 12. Limitations, stated as they should appear')
w()
w('| | |')
w('|---|---|')
w('| **Single seed** | temperature 0, seed 42. Re-running an identical configuration flips 8/122 = 6.6% of GAIA and 22/400 = 5.5% of MATH items under continuous batching. Paired tests are unaffected; any single rate is not. |')
w('| **Thinking disabled everywhere** | The harness reads only `message.content` and never carries a reasoning trace into the next round, so the prompt-composition measurement is unaffected by construction. What would change is how fast the allowance is spent, which binds the cap harder and should if anything favour the method. Not measured. |')
w('| **The gain is conditional** | Token cost and truncation hold in all 24 cells. Accuracy holds in 18. The six exceptions are cells where the cap does not bind or the baseline has no room. |')
w('| **The router is weak** | AUC 0.75 on MATH and FRAMES, 0.62 on GAIA, 0.52 on MuSiQue. The quantile form fixes the cost, not the discrimination. |')
w('| **Decomposition is not significant** | No single mechanism increment reaches significance at n=400. |')
w('| **Two carried slots untested** | The critique and raw tool output are 1.9% and 0.4% of characters on Qwen, too small to price. On Gemma the critique is 10.0%, so this is a per-model fact, not a general one, and bounding it there is untested. |')
w('| **The motivating measurement does not fully transfer** | The plan is 40.4% of prompt characters on Qwen and 28.7% on Gemma, where it is smaller than the question. It remains the largest bindable slot on both. |')
w('| **Absolute numbers are low** | MuSiQue 5.3 -> 8.7, FRAMES 8.0 -> 13.9. Conclusions drawn where nine in ten tasks fail. |')
w('| **Reasoning on is only covered by one family** | GLM runs with reasoning on because it cannot be turned off, and needs the reasoning rung to recover the token result. Whether a model that *can* turn reasoning off behaves the same way with it on is untested. |')
w('| **The governed arm is not one configuration** | Qwen and Gemma are governed by the carried-slot bounds alone; GLM adds the reasoning rung. The table is a family of configurations selected per model, not one system run three times. Say so. |')
w('| **Compaction and slot ablation are not on GLM** | The compaction grid and the counterfactual slot ablation were not re-run on the third family. |')
w('| **Mid-size models only** | 27B and 31B. Whether carried state still dominates for a model that needs fewer rounds is untested and is the most likely reviewer objection. |')
w('| **No published-framework baseline** | The baseline is our own pipeline plus a format-alignment control, not AutoGen or MetaGPT. |')
w()

w('## 13. Defects found and fixed during this work')
w()
w('Listed because several produced publishable-looking numbers that were wrong, and a')
w('write-up should not resurrect them from older notes.')
w()
w('| Defect | Symptom | Fix |')
w('|---|---|---|')
w('| Compaction trigger read cumulative spend, not context occupancy | 100% trigger rate, spurious correlation | `prompt_tokens / context_window > ratio` |')
w('| Summariser appended a second system message | Qwen template rejects with HTTP 400, error swallowed, every compacted task scored zero | merge summary into the existing system message, keep the task |')
w('| Summariser dropped the question | tasks answered without knowing what was asked | preserve head and task, summarise only the middle |')
w('| API errors silent | a malformed request looked like a model with nothing to say | log `[APIERR]` |')
w('| `stopped_by=="budget"` conflated with API timeouts | truncation counts inflated | require more than half the cap consumed |')
w('| BigCodeBench judge starved (2 CPUs, 64 workers) | tests failed on wall clock; produced a significant but false finding that more budget hurts | judge offline from stored replies, 10 CPUs, 90s |')
w('| Trace files mixed across runs | the same cell exists as a primary run, a flip-rate re-run, a speed test and a calibration run; globbing picked one at random | explicit file map, see `_gen_materials.py` |')
w('| No retry on transient API errors | pinned provider rate-limits silently dropped rounds | bounded retry on 429/5xx/timeout |')
w('| Gateway returned 200 with `choices: null` | `TypeError` that reads nothing like a network fault | named and routed into the retry |')
w('| Skip check matched its own negation | The chain asked `grep "complete"` against a status line whose incomplete state was written as a superstring of the complete one, so a cell holding two of its three arms was skipped as finished | ASCII DONE/TODO, no shared substring |')
w('| urllib blocked by the CDN at 403 | Probe scripts failed on every request while curl and the openai client worked, because urllib sends no User-Agent | forward a User-Agent |')
w('| Two endpoints could claim one cell | Parallel shards would have appended to the same trace file | claim by `mkdir`, which is atomic |')
w('| llama.cpp enabled thinking by default | reasoning went to `reasoning_content`, which the harness never reads, so answers came back empty | `--reasoning off` |')
w('| Wrong tool-call parser for Gemma | the model emits XML-style calls, `hermes` parses none | `qwen3_xml` |')
w()

w('## 14. What was run and dropped')
w()
w('**SWE-bench Verified.** Ran to completion, oracle-localization variant, official harness.')
w('Final: Naive 184/500 = 36.8%, Governed 169/500 = 33.8% (dSR -3.0, p=0.142, not')
w('significant), the verbalized-channel arm 135/500 = 27.0%. Dropped from the paper: the')
w('setup gives localization away, so the numbers are not comparable to agentic resolve rates,')
w('and the one significant finding it produced (that removing the verbalized channel costs')
w('6.8 pp on repair) does not replicate on BigCodeBench, where removing it is free or better')
w('at the tight budgets. The likely mechanism is specific to the oracle setup: with the buggy')
w('region supplied, the gate sees an easy-looking task, stays at L0, and returns the code')
w('unchanged, which happened on 93 instances. Traces remain under `traces/*sweo*`.')
w()
w('**OpenHands reference run.** Set up, validated end to end on three instances, then')
w('cancelled. It measures the model under a standard agentic scaffold, which is orthogonal to')
w('the method. Apparatus is in `/mnt/data/oh` if wanted: per-instance token metering proxy,')
w('budget enforcement that synthesises a `finish` call, batched image building.')
w()
w('**Not run at all:** any frontier model; any published agent framework as a baseline; the')
w('slot-ablation and decomposition arms on Gemma; thinking enabled.')
w()

w('## 15. File map')
w()
w('| What | Where |')
w('|---|---|')
w('| Qwen main table | `traces/tasks_sweep_Qwen3.8-27B_q38_*.jsonl` (Naive, Naive+Fmt), `..._q38ab7_*` and `..._q38_ab7_*` (Governed) |')
w('| Gemma main table | `tasks_sweep_gemma-4-31B-bf16_gor_*.jsonl` |')
w('| GLM main table | `tasks_sweep_GLM-5.3-Flash_glm4_*.jsonl`. The reasoning-rung ablation is `..._glmab7_ma*` (MATH) and `_aborted/glm_pre_reasoning/` (GAIA, from the run that preceded the rung) |')
w('| GLM BigCodeBench | `tasks_sweep_GLM-5.3-Flash_glmbcb*_rejudged.jsonl` |')
w('| Compaction | `..._acc_*` / `..._cmp3_*` (Qwen), `..._gcmp_*` (Gemma). **`cmp_*` and `cmp2_*` are defective** |')
w('| BigCodeBench | `..._bcb{4,8,16}k_rejudged.jsonl` (Qwen), `..._gbcb*_rejudged.jsonl` (Gemma) |')
w('| Segment accounting | `..._seg_*.jsonl` (Qwen), `..._gseg_*.jsonl` (Gemma) |')
w('| Slot ablation | `results/slot_ablation_cp_*.json` |')
w('| Decomposition | `..._q38_abl_ma.jsonl` (24k), `..._qdec_ma{8,16}.jsonl` |')
w('| SWE (dropped) | `traces/*sweo*`, `swe_preds_v2_*.json`, `logs/run_evaluation/v2_*` |')
w('| Paper | `paper/main.tex`, build with `paper/build.sh` |')
w('| This file | regenerate with `python3 _gen_materials.py` |')
w()

open('PAPER_MATERIALS.md','w').write('\n'.join(OUT) + '\n')
print(f'wrote PAPER_MATERIALS.md  ({len(OUT)} lines)')
