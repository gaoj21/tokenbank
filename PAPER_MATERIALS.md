# Paper materials: statelessness as budget policy

Everything a write-up needs, computed from the traces by `_gen_materials.py`. Each table
names the files it came from. Numbers here supersede any earlier draft: several cells
exist in more than one trace file and earlier drafts mixed them.

## 0. One-paragraph summary

A plan-execute-review pipeline under a fixed per-task token budget spends most of that
budget re-transmitting coordination state it wrote on earlier rounds, not on new
reasoning. The largest such slot is the plan. Bounding what crosses the round boundary,
and admitting tasks to the full loop only when a confidence signal fails, cuts tokens in
every measured cell, removes budget exhaustion entirely, and raises accuracy where the
cap binds. Relaxing the cap removes the accuracy gain, which is the causal test.

## 1. Setup

| | |
|---|---|
| Primary model | Qwen3.8-27B-FP8, vLLM. Endpoint `https://llm5.ruoyu-wang.com/v1` (`qwen3.8-27b-fp8`), 262k context |
| Second model | Gemma-4-31B-it at **bf16**, OpenRouter with provider pinned to Venice, fallbacks off |
| Third model | GLM-5.3-Flash on sglang, sharded over four endpoints (`llm3.ruoyu-wang.com`, `llm52/56/57.jierungogogo.com`). Verified one deployment: identical `model_path`, `tp_size=8`, `kv_cache_dtype`, and token-identical generations on a probe prompt at `reasoning_effort=low` |
| Fourth model | DeepSeek-V4.1-Flash on vLLM, reached over an SSH tunnel to `127.0.0.1:18000`, 1.05M context. Reasoning cannot be graded here: `reasoning_effort` low/medium/high all sit near the 328-token default, a hard `thinking_token_budget` is honoured exactly, and `chat_template_kwargs {thinking: False}` yields 0 reasoning tokens. The two must not be combined, which re-enables the thought (265 tokens). The governed arm uses the switch, so its cheap rung starts no thought rather than truncating one |
| Sampling | temperature 0, seed 42, single seed throughout |
| Thinking | Disabled on Qwen (`chat_template_kwargs={"enable_thinking": false}`) and Gemma (`--reasoning off`), verified to take effect. **Cannot be disabled on GLM**; see section 11b |
| Budgets | 8k / 16k / 24k tokens per task, counted over prompt and completion across all roles. BigCodeBench uses 4k / 8k / 16k |
| Judging | symbolic equivalence (MATH), normalized exact match (GAIA), EM/F1 (FRAMES, MuSiQue), unit-test execution in the official image (BigCodeBench) |

### 1.1 Arms

| Name in paper | Config | What it is |
|---|---|---|
| `Naive` | `V5-Unconstrained` | plain pipeline, no format instruction |
| `Naive+Fmt` | `V5-Unc-Fmt` | plain pipeline **plus** the `ANSWER: x` instruction. **This is the baseline every claim is measured against** |
| `Governed` | `V5.1-Ab7-NoVerbal` | full system: bounded slots, convergence stop, cost-ordered ladder routed by the confidence gate alone |
| `Accumulate` | `V5-Accum-Only` | compaction control: keeps the whole history |
| `Compaction` | `V5-Compact-Real` | accumulate, then summarise once the prompt passes half of a 8192-token context window |

The format-aligned baseline matters: the instruction alone is worth +13.2 pp to Qwen and
+28.0 pp to Gemma on MATH at 24k. A comparison against `Naive` credits that to the method.

### 1.2 Benchmarks

| Dataset | n | Task |
|---|---|---|
| MATH-500 | 400 | competition mathematics, symbolic reasoning, no external knowledge |
| GAIA | 122 | multi-step assistant questions needing lookup, arithmetic, file reading |
| FRAMES | 660 | multi-hop factual reasoning across Wikipedia articles |
| MuSiQue | 1,934 | 2-4 hop compositional QA, built to defeat shortcuts |
| BigCodeBench | 500 | library-heavy function synthesis over 139 libraries, execution-judged |

SWE-bench Verified was run and then **dropped**; see section 9.

## 1.3 Method, in the detail needed to reimplement it

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


## 2. Where the budget goes (segment accounting)

Every prompt character attributed to what it carries. Shares are of **prompt** characters;
the model's own output is excluded so the slots compare against one another.
Source: `traces/tasks_sweep_Qwen3.8-27B_seg_*.jsonl`, `tasks_sweep_gemma-4-31B-bf16_gseg_*.jsonl`, B=16k, `Naive+Fmt`.

**Qwen3.8-27B**

| Benchmark | n | plan | question | draft | critique | tool out | system |
|---|---:|---:|---:|---:|---:|---:|---:|
| GAIA | 122 | 35.6% | 32.0% | 25.3% | 2.2% | 0.3% | 3.7% |
| MATH | 400 | 50.4% | 23.1% | 14.3% | 1.8% | 1.0% | 7.7% |
| FRAMES | 660 | 38.1% | 26.2% | 26.2% | 2.9% | 0.5% | 4.7% |
| MuSiQue | 1,934 | 40.2% | 26.7% | 25.8% | 1.7% | 0.3% | 4.2% |
| BigCodeBench | 500 | 44.3% | 36.1% | 13.0% | 0.8% | 0.8% | 4.1% |
| **Pooled** | **3,616** | **40.4%** | **27.3%** | **24.4%** | **1.9%** | **0.4%** | **4.4%** |

**Gemma-4-31B (bf16)**

| Benchmark | n | plan | question | draft | critique | tool out | system |
|---|---:|---:|---:|---:|---:|---:|---:|
| GAIA | 122 | 24.9% | 39.3% | 18.2% | 9.3% | 0.4% | 6.3% |
| MATH | 400 | 43.5% | 25.8% | 16.0% | 2.9% | 1.2% | 9.2% |
| FRAMES | 660 | 24.7% | 29.4% | 22.8% | 12.7% | 0.5% | 8.1% |
| MuSiQue | 1,934 | 24.7% | 31.8% | 21.2% | 12.3% | 0.2% | 7.9% |
| BigCodeBench | 400 | 38.3% | 37.7% | 14.5% | 4.0% | 0.8% | 3.9% |
| **Pooled** | **3,516** | **28.7%** | **32.2%** | **19.8%** | **10.0%** | **0.5%** | **7.3%** |

### 2.1 Same runs, counted against the whole budget

Including the model's own output in the denominator. This is the framing to use when the
word "budget" appears; the table above is the framing for comparing slots.

| Model | model output | question | carried state | of which plan |
|---|---:|---:|---:|---:|
| Qwen3.8-27B | 30.5% | 19.0% | 46.6% | 28.0% |
| Gemma-4-31B (bf16) | 34.0% | 21.3% | 38.9% | 18.9% |

**Reading.** The plan is the largest slot a bound can act on, on both families. Its size
relative to the question does **not** transfer: 40.4% against 27.3% on Qwen, 28.7% against
32.2% on Gemma. Neither does the critique: 1.9% on Qwen, 10.0% on Gemma. Do not write
"the plan is 1.5x the question"; write "the plan is the largest bindable slot".

## 3. Main table: success rate and tokens

SR (%) / mean tokens per task. Sources: `q38_*` (Naive, Naive+Fmt), `q38ab7_*` and
`q38_ab7_*` (Governed), `acc_*` (Accumulate), `cmp3_*` (Compaction) for Qwen;
`gor_*` and `gcmp_*` for Gemma.

**Qwen3.8-27B**

| Arm | Budget | MATH | GAIA | FRAMES | MuSiQue |
|---|---|---|---|---|---|
| Naive | 8k | 36.8 / 4,404 | 9.8 / 6,467 | 5.8 / 5,866 | 3.6 / 5,937 |
|  | 16k | 43.5 / 5,819 | 9.8 / 12,653 | 6.8 / 10,335 | 4.0 / 10,692 |
|  | 24k | 44.0 / 6,299 | 13.1 / 16,114 | 7.1 / 12,135 | 4.0 / 12,663 |
| Naive+Fmt | 8k | 49.5 / 4,349 | 7.4 / 6,642 | 7.6 / 6,016 | 4.9 / 6,060 |
|  | 16k | 55.8 / 5,209 | 11.5 / 12,937 | 7.4 / 10,611 | 5.2 / 10,802 |
|  | 24k | 57.2 / 5,635 | 9.8 / 16,309 | 8.0 / 12,574 | 5.3 / 12,926 |
| Governed | 8k | 68.5 / 2,506 | 21.3 / 5,064 | 14.8 / 3,631 | 7.9 / 4,419 |
|  | 16k | 68.5 / 2,588 | 19.7 / 6,705 | 13.3 / 4,734 | 8.5 / 6,599 |
|  | 24k | 68.0 / 2,679 | 21.3 / 7,496 | 13.9 / 5,399 | 8.7 / 7,470 |
| Accumulate | 8k | 52.2 / 4,213 | 9.8 / 6,353 | 6.5 / 5,965 | 4.2 / 6,106 |
|  | 16k | 56.8 / 4,899 | 9.8 / 12,796 | 6.5 / 10,913 | 4.6 / 11,604 |
|  | 24k | 55.8 / 5,382 | 11.5 / 17,419 | 6.8 / 15,609 | 5.0 / 16,196 |
| Compaction | 8k | 50.7 / 4,389 | 7.4 / 6,407 | 6.1 / 6,043 | 4.2 / 6,175 |
|  | 16k | 54.0 / 5,323 | 9.0 / 12,591 | 5.6 / 11,108 | 4.2 / 11,626 |
|  | 24k | 56.5 / 5,981 | 11.5 / 18,579 | 6.4 / 16,211 | 4.2 / 17,252 |

**Gemma-4-31B (bf16)**

| Arm | Budget | MATH | GAIA | FRAMES | MuSiQue |
|---|---|---|---|---|---|
| Naive | 8k | 45.0 / 3,053 | 15.6 / 3,077 | 8.6 / 2,772 | 4.2 / 2,445 |
|  | 16k | 47.5 / 3,345 | 11.5 / 4,284 | 8.0 / 3,458 | 3.9 / 2,822 |
|  | 24k | 45.8 / 3,594 | 12.3 / 4,542 | 7.9 / 4,119 | 4.2 / 2,929 |
| Naive+Fmt | 8k | 76.2 / 3,223 | 19.7 / 3,304 | 17.0 / 2,593 | 7.8 / 2,393 |
|  | 16k | 76.8 / 3,488 | 23.0 / 4,591 | 17.7 / 3,481 | 8.1 / 2,943 |
|  | 24k | 73.8 / 3,724 | 19.7 / 4,834 | 17.7 / 3,765 | 8.4 / 2,921 |
| Governed | 8k | 79.5 / 1,191 | 23.8 / 2,091 | 16.2 / 1,270 | 9.6 / 1,554 |
|  | 16k | 79.5 / 1,200 | 21.3 / 2,409 | 16.5 / 1,360 | 9.6 / 1,726 |
|  | 24k | 79.2 / 1,227 | 23.0 / 2,363 | 16.8 / 1,379 | 9.8 / 1,761 |
| Accumulate | 8k | 75.8 / 3,214 | 21.3 / 3,258 | 16.2 / 2,629 | 7.4 / 2,363 |
|  | 16k | 77.5 / 3,480 | 19.7 / 4,567 | 16.2 / 3,913 | 7.5 / 3,658 |
|  | 24k | 77.0 / 3,574 | 21.3 / 5,306 | 17.0 / 4,453 | 8.3 / 3,806 |
| Compaction | 8k | 75.2 / 3,467 | 19.7 / 3,329 | 15.9 / 2,709 | 7.8 / 2,516 |
|  | 16k | 77.2 / 3,786 | 22.1 / 5,119 | 17.3 / 4,056 | 7.9 / 3,453 |
|  | 24k | 77.0 / 3,921 | 18.9 / 6,151 | 15.5 / 5,051 | 8.6 / 4,586 |

**GLM-5.3-Flash**

| Arm | Budget | MATH | GAIA | FRAMES | MuSiQue |
|---|---|---|---|---|---|
| Naive | 8k | 20.8 / 3,705 | 6.6 / 6,559 | 9.5 / 4,495 | 7.9 / 4,206 |
|  | 16k | 23.5 / 4,649 | 11.5 / 10,701 | 11.5 / 5,926 | 8.9 / 5,569 |
|  | 24k | 24.2 / 4,783 | 10.7 / 11,303 | 11.2 / 5,896 | 8.8 / 5,438 |
| Naive+Fmt | 8k | 63.7 / 3,631 | 18.0 / 6,539 | 25.2 / 4,408 | 16.8 / 4,090 |
|  | 16k | 64.5 / 4,496 | 22.1 / 10,545 | 25.5 / 5,807 | 18.3 / 5,326 |
|  | 24k | 64.2 / 4,404 | 18.0 / 11,100 | 25.5 / 5,710 | 17.7 / 5,238 |
| Governed (reasoning ladder) | 8k | 61.3 / 1,592 | 24.6 / 3,965 | 25.5 / 1,976 | 18.4 / 2,093 |
|  | 16k | 62.0 / 2,032 | 22.1 / 5,155 | 28.2 / 2,040 | 18.4 / 2,219 |
|  | 24k | 62.5 / 1,933 | 20.5 / 5,764 | 28.0 / 2,233 | 19.6 / 2,317 |

**DeepSeek-V4.1-Flash**

| Arm | Budget | MATH | GAIA | FRAMES | MuSiQue |
|---|---|---|---|---|---|
| Naive | 8k | 42.5 / 3,090 | 9.8 / 6,610 | 6.4 / 4,639 | 3.2 / 4,982 |
|  | 16k | 43.0 / 3,652 | 8.2 / 10,964 | 6.2 / 6,266 | 3.1 / 6,677 |
|  | 24k | 43.0 / 3,777 | 10.7 / 11,046 | 6.5 / 6,279 | 3.4 / 6,760 |
| Naive+Fmt | 8k | 56.2 / 3,018 | 17.2 / 6,645 | 23.9 / 4,568 | 15.9 / 5,012 |
|  | 16k | 55.8 / 3,729 | 18.9 / 10,932 | 27.4 / 6,403 | 17.6 / 6,915 |
|  | 24k | 57.8 / 3,738 | 18.9 / 11,213 | 27.4 / 6,416 | 17.4 / 6,884 |
| Governed (reasoning ladder) | 8k | 62.7 / 1,592 | 24.6 / 2,663 | 28.9 / 1,082 | 17.4 / 1,181 |
|  | 16k | 61.0 / 1,702 | 30.3 / 3,466 | 26.8 / 1,138 | 17.2 / 1,250 |
|  | 24k | 63.2 / 1,705 | 30.3 / 3,844 | 27.1 / 1,186 | 17.0 / 1,313 |

## 4. Paired tests: Governed vs Naive+Fmt

Exact McNemar on per-item outcomes. `trunc` counts tasks that exhausted the budget
mid-loop (`stopped_by=="budget"` and more than half the cap consumed).

**Qwen3.8-27B**

| Benchmark | Budget | Naive+Fmt | Governed | dSR | p | wins/losses | tokens | trunc |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MATH (400) | 8k | 49.5 | 68.5 | +19.0* | 4.14e-17 | 84/8 | -42% | 89 -> 0 |
| MATH (400) | 16k | 55.8 | 68.5 | +12.8* | 4.27e-11 | 58/7 | -50% | 23 -> 0 |
| MATH (400) | 24k | 57.2 | 68.0 | +10.8* | 3.38e-08 | 53/10 | -52% | 6 -> 0 |
| GAIA (122) | 8k | 7.4 | 21.3 | +13.9* | 1.53e-05 | 17/0 | -24% | 105 -> 0 |
| GAIA (122) | 16k | 11.5 | 19.7 | +8.2* | 6.35e-03 | 11/1 | -48% | 80 -> 0 |
| GAIA (122) | 24k | 9.8 | 21.3 | +11.5* | 1.31e-03 | 16/2 | -54% | 16 -> 0 |
| FRAMES (660) | 8k | 7.6 | 14.8 | +7.3* | 5.56e-10 | 56/8 | -40% | 457 -> 0 |
| FRAMES (660) | 16k | 7.4 | 13.3 | +5.9* | 4.59e-07 | 50/11 | -55% | 255 -> 0 |
| FRAMES (660) | 24k | 8.0 | 13.9 | +5.9* | 8.07e-08 | 47/8 | -57% | 14 -> 0 |
| MuSiQue (1,934) | 8k | 4.9 | 7.9 | +3.0* | 3.58e-08 | 85/27 | -27% | 1399 -> 0 |
| MuSiQue (1,934) | 16k | 5.2 | 8.5 | +3.4* | 2.44e-09 | 93/28 | -39% | 840 -> 0 |
| MuSiQue (1,934) | 24k | 5.3 | 8.7 | +3.4* | 3.04e-09 | 96/30 | -42% | 26 -> 0 |

Significant at p<0.05 in **12 of 12** cells.

**Gemma-4-31B (bf16)**

| Benchmark | Budget | Naive+Fmt | Governed | dSR | p | wins/losses | tokens | trunc |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MATH (400) | 8k | 76.2 | 79.5 | +3.2* | 9.77e-04 | 14/1 | -63% | 18 -> 0 |
| MATH (400) | 16k | 76.8 | 79.5 | +2.8* | 3.42e-03 | 12/1 | -66% | 9 -> 0 |
| MATH (400) | 24k | 73.8 | 79.2 | +5.5* | 2.98e-06 | 23/1 | -67% | 6 -> 0 |
| GAIA (122) | 8k | 19.7 | 23.8 | +4.1 | 1.80e-01 | 7/2 | -37% | 27 -> 0 |
| GAIA (122) | 16k | 23.0 | 21.3 | -1.6 | 7.74e-01 | 5/7 | -48% | 10 -> 0 |
| GAIA (122) | 24k | 19.7 | 23.0 | +3.3 | 3.44e-01 | 7/3 | -51% | 4 -> 0 |
| FRAMES (660) | 8k | 17.0 | 16.2 | -0.8 | 5.51e-01 | 20/25 | -51% | 108 -> 0 |
| FRAMES (660) | 16k | 17.7 | 16.5 | -1.2 | 3.12e-01 | 20/28 | -61% | 26 -> 0 |
| FRAMES (660) | 24k | 17.7 | 16.8 | -0.9 | 4.80e-01 | 22/28 | -63% | 1 -> 0 |
| MuSiQue (1,934) | 8k | 7.8 | 9.6 | +1.9* | 5.34e-04 | 70/34 | -35% | 291 -> 0 |
| MuSiQue (1,934) | 16k | 8.1 | 9.6 | +1.4* | 7.80e-03 | 66/38 | -41% | 14 -> 0 |
| MuSiQue (1,934) | 24k | 8.4 | 9.8 | +1.4* | 9.34e-03 | 64/37 | -40% | 0 -> 0 |

Significant at p<0.05 in **6 of 12** cells.

**GLM-5.3-Flash**

| Benchmark | Budget | Naive+Fmt | Governed | dSR | p | wins/losses | tokens | trunc |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MATH (400) | 8k | 63.7 | 61.3 | -2.5 | 2.53e-01 | 26/36 | -56% | 60 -> 0 |
| MATH (400) | 16k | 64.5 | 62.0 | -2.5 | 2.20e-01 | 22/32 | -55% | 5 -> 0 |
| MATH (400) | 24k | 64.2 | 62.5 | -1.8 | 4.43e-01 | 27/34 | -56% | 0 -> 0 |
| GAIA (122) | 8k | 18.0 | 24.6 | +6.6 | 5.74e-02 | 11/3 | -39% | 89 -> 0 |
| GAIA (122) | 16k | 22.1 | 22.1 | +0.0 | 1.00e+00 | 7/7 | -51% | 12 -> 0 |
| GAIA (122) | 24k | 18.0 | 20.5 | +2.5 | 6.29e-01 | 10/7 | -48% | 1 -> 0 |
| FRAMES (660) | 8k | 25.2 | 25.5 | +0.3 | 9.11e-01 | 41/39 | -55% | 207 -> 0 |
| FRAMES (660) | 16k | 25.5 | 28.2 | +2.7 | 5.05e-02 | 47/29 | -65% | 3 -> 0 |
| FRAMES (660) | 24k | 25.5 | 28.0 | +2.6 | 7.84e-02 | 50/33 | -61% | 0 -> 0 |
| MuSiQue (1,934) | 8k | 16.8 | 18.4 | +1.7* | 2.28e-02 | 109/77 | -49% | 561 -> 0 |
| MuSiQue (1,934) | 16k | 18.3 | 18.4 | +0.1 | 9.43e-01 | 100/98 | -58% | 1 -> 0 |
| MuSiQue (1,934) | 24k | 17.7 | 19.6 | +1.9* | 7.28e-03 | 109/72 | -56% | 0 -> 0 |

Significant at p<0.05 in **2 of 12** cells.

**DeepSeek-V4.1-Flash**

| Benchmark | Budget | Naive+Fmt | Governed | dSR | p | wins/losses | tokens | trunc |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MATH (400) | 8k | 56.2 | 62.7 | +6.5* | 1.30e-03 | 44/18 | -47% | 47 -> 0 |
| MATH (400) | 16k | 55.8 | 61.0 | +5.2* | 4.60e-03 | 36/15 | -54% | 1 -> 0 |
| MATH (400) | 24k | 57.8 | 63.2 | +5.5* | 3.84e-03 | 38/16 | -54% | 0 -> 0 |
| GAIA (122) | 8k | 17.2 | 24.6 | +7.4* | 2.25e-02 | 11/2 | -60% | 97 -> 0 |
| GAIA (122) | 16k | 18.9 | 30.3 | +11.5* | 2.58e-03 | 17/3 | -68% | 4 -> 0 |
| GAIA (122) | 24k | 18.9 | 30.3 | +11.5* | 1.31e-03 | 16/2 | -66% | 3 -> 0 |
| FRAMES (660) | 8k | 23.9 | 28.9 | +5.0* | 1.33e-03 | 67/34 | -76% | 270 -> 0 |
| FRAMES (660) | 16k | 27.4 | 26.8 | -0.6 | 7.55e-01 | 44/48 | -82% | 0 -> 0 |
| FRAMES (660) | 24k | 27.4 | 27.1 | -0.3 | 9.17e-01 | 45/47 | -82% | 0 -> 0 |
| MuSiQue (1,934) | 8k | 15.9 | 17.4 | +1.4 | 6.47e-02 | 121/93 | -76% | 941 -> 0 |
| MuSiQue (1,934) | 16k | 17.6 | 17.2 | -0.5 | 5.73e-01 | 96/105 | -82% | 0 -> 0 |
| MuSiQue (1,934) | 24k | 17.4 | 17.0 | -0.5 | 5.75e-01 | 97/106 | -81% | 0 -> 0 |

Significant at p<0.05 in **7 of 12** cells.

Pooled over the three families: **27 of 36** cells significant. Token cost
falls in all 36 and budget exhaustion is eliminated in all 36, which are the two results
that hold without exception. The accuracy gain is the conditional one: it appears where the
cap binds and the format-aligned baseline still has room, and the 9 cells where it
does not are cells missing one of those. Note that the governed arm is not one
configuration across families; see section 11b.

## 5. What the governed arm actually does

### 5.1 Fraction of the budget consumed

Mean tokens divided by the cap. The governed arm is nowhere near budget-limited; it is
convergence-limited. This is worth saying explicitly: the cap is the setting in which the
baseline's failure appears, not a resource the method is squeezing into.

| Model | Benchmark | 8k N+F / Gov | 16k N+F / Gov | 24k N+F / Gov |
|---|---|---|---|---|
| Qwen3.8-27B | MATH | 54% / 31% | 33% / 16% | 23% / 11% |
| Qwen3.8-27B | GAIA | 83% / 63% | 81% / 42% | 68% / 31% |
| Qwen3.8-27B | FRAMES | 75% / 45% | 66% / 30% | 52% / 22% |
| Qwen3.8-27B | MuSiQue | 76% / 55% | 68% / 41% | 54% / 31% |
| Gemma-4-31B (bf16) | MATH | 40% / 15% | 22% / 7% | 16% / 5% |
| Gemma-4-31B (bf16) | GAIA | 41% / 26% | 29% / 15% | 20% / 10% |
| Gemma-4-31B (bf16) | FRAMES | 32% / 16% | 22% / 9% | 16% / 6% |
| Gemma-4-31B (bf16) | MuSiQue | 30% / 19% | 18% / 11% | 12% / 7% |
| GLM-5.3-Flash | MATH | 45% / 20% | 28% / 13% | 18% / 8% |
| GLM-5.3-Flash | GAIA | 82% / 50% | 66% / 32% | 46% / 24% |
| GLM-5.3-Flash | FRAMES | 55% / 25% | 36% / 13% | 24% / 9% |
| GLM-5.3-Flash | MuSiQue | 51% / 26% | 33% / 14% | 22% / 10% |
| DeepSeek-V4.1-Flash | MATH | 38% / 20% | 23% / 11% | 16% / 7% |
| DeepSeek-V4.1-Flash | GAIA | 83% / 33% | 68% / 22% | 47% / 16% |
| DeepSeek-V4.1-Flash | FRAMES | 57% / 14% | 40% / 7% | 27% / 5% |
| DeepSeek-V4.1-Flash | MuSiQue | 63% / 15% | 43% / 8% | 29% / 5% |

### 5.2 Where the governed arm's tokens go, by role (24k)

| Model | Benchmark | planner | executor | reviewer |
|---|---|---:|---:|---:|
| Qwen3.8-27B | GAIA | 2.3% | 91.0% | 6.8% |
| Qwen3.8-27B | FRAMES | 2.4% | 90.8% | 6.8% |
| Qwen3.8-27B | MuSiQue | 3.7% | 87.4% | 8.8% |
| Gemma-4-31B (bf16) | MATH | 0.0% | 100.0% | 0.0% |
| Gemma-4-31B (bf16) | GAIA | 4.2% | 86.1% | 9.7% |
| Gemma-4-31B (bf16) | FRAMES | 1.8% | 89.5% | 8.7% |
| Gemma-4-31B (bf16) | MuSiQue | 4.1% | 79.7% | 16.3% |
| GLM-5.3-Flash | MATH | 5.2% | 86.6% | 8.2% |
| GLM-5.3-Flash | GAIA | 8.3% | 79.7% | 12.0% |
| GLM-5.3-Flash | FRAMES | 6.9% | 82.3% | 10.7% |
| GLM-5.3-Flash | MuSiQue | 8.4% | 79.2% | 12.4% |
| DeepSeek-V4.1-Flash | MATH | 0.8% | 97.9% | 1.3% |
| DeepSeek-V4.1-Flash | GAIA | 4.1% | 90.7% | 5.2% |
| DeepSeek-V4.1-Flash | FRAMES | 1.5% | 96.8% | 1.6% |
| DeepSeek-V4.1-Flash | MuSiQue | 3.2% | 93.4% | 3.5% |

Keeping the planner costs 1.8-4.2% of the budget. What the bound removes is not that
cost but its re-transmission on every subsequent round. On MATH the planner is never
called at all, because nearly every item stops at L0.

### 5.3 Escalation rate (share of items leaving L0), 24k

| Model | MATH | GAIA | FRAMES | MuSiQue |
|---|---:|---:|---:|---:|
| Qwen3.8-27B | 23.3% | 63.9% | 53.8% | 69.8% |
| Gemma-4-31B (bf16) | 2.7% | 50.0% | 27.9% | 53.7% |
| GLM-5.3-Flash | 28.0% | 71.3% | 49.2% | 63.9% |
| DeepSeek-V4.1-Flash | 22.5% | 45.9% | 19.4% | 29.9% |

The same threshold routes 1.5% of MATH and 53.8% of MuSiQue. That spread is the reason
for the quantile form of the rule (section 6).

## 6. The confidence gate, and making its operating point transferable

phi is the mean token log-probability of the L0 answer. It ranks, but its scale is not
comparable across benchmarks, so a fixed threshold buys an unpredictable amount.

| Model | tuned on | threshold: keep rate | threshold: lift | quantile: keep rate | quantile: lift |
|---|---|---|---|---|---|
| Qwen3.8-27B | MATH | 2.4-11.0% | 3.0-36.7 pp | **4.9-5.0%** | -0.8-46.2 pp |
| Qwen3.8-27B | GAIA | 7.4-32.8% | 1.7-46.2 pp | **4.9-5.0%** | -0.8-46.2 pp |
| Qwen3.8-27B | FRAMES | 17.0-59.0% | -0.4-20.9 pp | **29.5-30.0%** | 1.8-15.6 pp |
| Qwen3.8-27B | MuSiQue | 12.3-48.0% | -1.3-23.5 pp | **19.7-20.0%** | -0.2-25.3 pp |
| Gemma-4-31B (bf16) | MATH | 5.7-12.4% | -6.2-7.7 pp | **4.9-5.0%** | -20.5-11.0 pp |
| Gemma-4-31B (bf16) | GAIA | 19.7-70.0% | 1.0-21.2 pp | **19.7-20.0%** | 3.5-21.2 pp |
| Gemma-4-31B (bf16) | FRAMES | 5.7-12.8% | -6.2-9.4 pp | **14.8-15.0%** | 4.3-9.0 pp |
| Gemma-4-31B (bf16) | MuSiQue | 13.2-45.0% | -0.1-9.9 pp | **14.8-15.0%** | 4.3-9.0 pp |
| GLM-5.3-Flash | MATH | 0.5-17.5% | 3.3-75.4 pp | **4.9-5.0%** | 6.7-25.4 pp |
| GLM-5.3-Flash | GAIA | 5.4-46.5% | 2.8-29.3 pp | **9.8-10.0%** | 3.6-33.7 pp |
| GLM-5.3-Flash | FRAMES | 36.1-80.2% | 4.2-16.3 pp | **4.9-5.0%** | 6.7-25.4 pp |
| GLM-5.3-Flash | MuSiQue | 11.7-56.5% | 0.1-15.4 pp | **14.8-15.0%** | 0.6-19.9 pp |
| DeepSeek-V4.1-Flash | MATH | 7.9-14.8% | 8.5-26.9 pp | **4.9-5.0%** | 12.9-38.0 pp |
| DeepSeek-V4.1-Flash | GAIA | 5.3-11.5% | 11.8-35.6 pp | **4.9-5.0%** | 12.9-38.0 pp |
| DeepSeek-V4.1-Flash | FRAMES | 10.3-16.4% | 10.4-21.3 pp | **9.8-10.0%** | 8.7-38.0 pp |
| DeepSeek-V4.1-Flash | MuSiQue | 26.5-34.2% | 7.6-22.7 pp | **4.9-5.0%** | 12.9-38.0 pp |

**The claim to make.** Fixing the quantile instead of the value makes the keep rate, and
therefore the token saving, a design parameter: it transfers to within 0.2 points by
construction, where a fixed threshold ranges from 2.4% to 70.0%. **The claim not to make:**
this does not improve discrimination. phi ranks items identically either way. AUC is 0.75
on MATH and FRAMES, 0.62 on GAIA, 0.52 on MuSiQue, and the gate remains the weakest
component of the system.

## 7. The two failure modes

Correct-to-wrong transitions at 24k, split into revision (a later round overwrites a
correct answer) and truncation (budget exhaustion replaces one). Conditional rate is
flips over items correct at some round. `is_correct` per round comes from the trace.

| Model | Dataset | Naive+Fmt rev+trunc | cond. rate | Governed rev+trunc | cond. rate |
|---|---|---|---:|---|---:|
| Qwen3.8-27B | MATH | 13+2 | 6.1% | 5+0 | **1.8%** |
| Qwen3.8-27B | GAIA | 11+1 | 50.0% | 2+0 | **7.1%** |
| Qwen3.8-27B | FRAMES | 38+0 | 41.8% | 8+0 | **8.0%** |
| Qwen3.8-27B | MuSiQue | 64+3 | 39.6% | 20+0 | **10.6%** |
| Gemma-4-31B (bf16) | MATH | 2+1 | 1.0% | 0+0 | **0.0%** |
| Gemma-4-31B (bf16) | GAIA | 5+3 | 25.0% | 0+0 | **0.0%** |
| Gemma-4-31B (bf16) | FRAMES | 12+0 | 9.3% | 2+0 | **1.8%** |
| Gemma-4-31B (bf16) | MuSiQue | 21+0 | 11.5% | 7+0 | **3.6%** |
| GLM-5.3-Flash | MATH | 4+0 | 1.5% | 9+0 | **3.5%** |
| GLM-5.3-Flash | GAIA | 7+0 | 24.1% | 10+0 | **28.6%** |
| GLM-5.3-Flash | FRAMES | 5+0 | 2.9% | 8+0 | **4.1%** |
| GLM-5.3-Flash | MuSiQue | 6+0 | 1.7% | 36+0 | **8.7%** |
| DeepSeek-V4.1-Flash | MATH | 1+0 | 0.4% | 3+0 | **1.2%** |
| DeepSeek-V4.1-Flash | GAIA | 0+0 | 0.0% | 0+0 | **0.0%** |
| DeepSeek-V4.1-Flash | FRAMES | 10+0 | 5.2% | 2+0 | **1.1%** |
| DeepSeek-V4.1-Flash | MuSiQue | 15+0 | 4.3% | 8+0 | **2.4%** |

Truncation goes to zero in all sixteen cells. The conditional rate falls in twelve of
them and rises in four, which are GLM's four: 1.5 to 3.5% on MATH, 24.1 to 28.6% on
GAIA, 2.9 to 4.1% on FRAMES and 1.7 to 8.7% on MuSiQue. The earlier two-family version
of this table said the rate fell everywhere, which the reasoning families refute.

## 8. BigCodeBench: the falsification test

Budgets chosen to span the point where the cap binds. If the gain comes from the cap
binding, relaxing it should remove the gain. Scored offline by executing each task's unit
tests in the official image; `*_rejudged.jsonl` are the authoritative files.

| Model | Budget | trunc (Naive) | Naive | Governed | dSR | p | wins/losses | tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B | 4k | 282 | 26.0 | 47.8 | +21.8* | 1.59e-22 | 124/15 | -53% |
| Qwen3.8-27B | 8k | 157 | 40.4 | 48.6 | +8.2* | 3.80e-05 | 69/28 | -72% |
| Qwen3.8-27B | 16k | 94 | 46.2 | 48.8 | +2.6 | 1.65e-01 | 44/31 | -81% |
| Gemma-4-31B (bf16) | 4k | 339 | 14.6 | 47.2 | +32.6* | 2.10e-43 | 168/5 | -51% |
| Gemma-4-31B (bf16) | 8k | 86 | 42.0 | 48.6 | +6.6* | 2.17e-04 | 55/22 | -69% |
| Gemma-4-31B (bf16) | 16k | 12 | 47.8 | 48.2 | +0.4 | 8.92e-01 | 28/26 | -71% |
| GLM-5.3-Flash | 4k | 490 | 6.4 | 50.8 | +44.4* | 1.67e-65 | 223/1 | -64% |
| GLM-5.3-Flash | 8k | 259 | 29.0 | 51.8 | +22.8* | 1.22e-28 | 120/6 | -81% |
| GLM-5.3-Flash | 16k | 51 | 42.6 | 51.0 | +8.4* | 3.71e-06 | 62/20 | -87% |
| DeepSeek-V4.1-Flash | 4k | 440 | 10.2 | 52.4 | +42.2* | 1.62e-59 | 214/3 | -68% |
| DeepSeek-V4.1-Flash | 8k | 334 | 23.0 | 52.2 | +29.2* | 1.72e-35 | 155/9 | -82% |
| DeepSeek-V4.1-Flash | 16k | 9 | 29.8 | 53.0 | +23.2* | 3.46e-25 | 129/13 | -89% |

The margin collapses as the cap is relaxed on both families and is not significant at
16k. The governed arm spends about the same tokens at every budget because it does not
escalate; the baseline grows with the allowance. `BCB-Ours` (the same system with the
verbalized channel on) is in the traces too and is not ahead of `BCB-Ab7` anywhere.

## 9. Compaction under a budget

`Accumulate` keeps the whole history; `Compaction` summarises it when the prompt passes
half of a **fixed 8192-token** context window (`kv_trigger=0.5`, `context_window=8192` in
the arm config, not read from the server, so the trigger means the same thing on both
families). Sources: `acc_*` / `cmp3_*` for Qwen, `gcmp_*` for Gemma. `cmp_*` and `cmp2_*`
are earlier, defective runs and must not be used.

| Model | Budget | n | trigger rate | dSR | wins/losses | p |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.8-27B | 8k | 3,116 | **0.00%** | -0.35 | 89/100 | 0.4671 |
| Qwen3.8-27B | 16k | 3,116 | **2.79%** | -0.77 | 81/105 | 0.0914 |
| Qwen3.8-27B | 24k | 3,116 | **26.16%** | -0.51 | 83/99 | 0.2661 |
| Gemma-4-31B (bf16) | 8k | 3,116 | **0.00%** | +0.06 | 42/40 | 0.9122 |
| Gemma-4-31B (bf16) | 16k | 3,116 | **0.90%** | +0.51 | 45/29 | 0.0805 |
| Gemma-4-31B (bf16) | 24k | 3,116 | **2.79%** | -0.22 | 41/48 | 0.5250 |

**The result is the trigger column.** At the tightest budget the summariser never fires on
either family, on any of 3,116 tasks: the allowance runs out before the context fills. No
stratum is significant. Pooling all twelve Qwen cells does reach p=0.034 (253/304), which
we do not read as an effect: the 8k stratum is a placebo by construction and carries the
same sign, and comparing strata directly, the compaction arm wins 47.1% of discordant
pairs where it never fires and 45.6% where it fires on a quarter of tasks (Fisher p=0.756).

## 10. Counterfactual slot ablation

On a live round of the unbounded pipeline, the same executor call is re-issued with
exactly one carried slot replaced, and the answer compared. Qwen only.
Source: `results/slot_ablation_cp_*.json`, 850 items across the four benchmarks.

The last three variants are the stronger test: they hold the slot's length and shape
roughly fixed and destroy its **content** instead. `foreign_plan` hands the executor a
plan written for a different question, paired by a fixed rotation so every task lends and
borrows exactly one.

| Variant | correct | d vs control | answer unchanged | c->w / w->c | p |
|---|---:|---:|---:|---:|---:|
| `control` | 28.6% | +0.0 | 100% | 0/0 | 1.000 |
| `bounded_plan` | 29.2% | +0.6 | 65% | 2/7 | 0.180 |
| `shuffled_plan` | 29.2% | +0.6 | 67% | 3/8 | 0.227 |
| `foreign_plan` | 28.6% | +0.0 | 66% | 5/5 | 1.000 |
| `no_plan` | 29.2% | +0.6 | 66% | 4/9 | 0.267 |
| `bounded_crit` | 25.6% | -2.9 | 48% | 34/9 | 0.000* |
| `no_crit` | 27.9% | -0.7 | 54% | 18/12 | 0.362 |
| `bounded_both` | 25.9% | -2.7 | 47% | 37/14 | 0.002* |

n = 850 paired items.

**How to read it.** The plan slot is null under every perturbation, including a plan
written for a different question (28.6% against 28.6%, discordant pairs 5/5). The
critique is not: bounding it costs 2.9 pp at p<0.001. Note the "answer unchanged"
column: about a third of answers do move when the plan is swapped, so the executor is
not ignoring the text. It reads it and the text carries no usable signal. Say
"non-informative", not "ignored".

**This is stronger than the paper currently claims and was deliberately left out of
the draft** as too large a claim for the paper to carry. The data is in
`results/slot_ablation_cp_*.json` if it is wanted.

## 11. Mechanism decomposition (Qwen, MATH, routing disabled)

Each arm adds one mechanism to the format-aligned baseline with the ladder switched off,
so every item runs the full loop and the increments are comparable.
Sources: `q38_abl_ma` (24k), `qdec_ma8`, `qdec_ma16`.

| Budget | Naive+Fmt | + bounded slots (Ab6) | + convergence stop (Ab3) | + routing (full) |
|---|---:|---:|---:|---:|
| 8k | 49.5 | 49.0 | 44.8 | 68.5 |
| 16k | 55.8 | 55.8 | 58.5 | 68.5 |
| 24k | 57.2 | 58.2 | 60.8 | 68.0 |

No single increment is significant at n=400 at any budget. Report the chain, not an
additive decomposition.

## 11b. Reasoning as a rung of the ladder

This mechanism exists because of a model that forced it. GLM-5.3-Flash cannot be made to
stop thinking: enable_thinking=false, thinking.type=disabled, reasoning_effort=none,
/nothink in either turn and separate_reasoning=false all generate the identical trace on a
probe prompt, and change only whether the server splits it out of `content`; two of them
leave a stray `</think>` inside it. So this family runs with reasoning on, and reasoning
tokens are counted in completion_tokens and charged to the task budget.

With reasoning on, the term that grows in the per-round cost is the generation, not the
carried prefix, and the carried-slot bounds do not touch it. Measured on GAIA at 24k, the
arm that governs only the prefix spent 10,998 tokens against the baseline 11,229, a
difference of -2%, where the same arm saves 24-67% on the families whose thinking is off.

The fix is to put the generation under the same admission decision as the scaffolding:
levels the gate keeps run at `reasoning_effort=low`, and escalation restores the default.
Nothing else changes, so the pair prices exactly this.

| Benchmark | Budget | prefix only (Ab7) | + reasoning rung | dSR | tokens |
|---|---|---:|---:|---:|---:|
| GAIA | 8k | 12.3 / 6,349 | 24.6 / 3,965 | +12.3 | -38% |
| GAIA | 16k | 16.4 / 10,710 | 22.1 / 5,155 | +5.7 | -52% |
| GAIA | 24k | 21.3 / 10,998 | 20.5 / 5,764 | -0.8 | -48% |
| MATH | 8k | 62.0 / 2,887 | 61.3 / 1,592 | -0.7 | -45% |
| MATH | 16k | 67.2 / 3,610 | 62.0 / 2,032 | -5.2 | -44% |
| MATH | 24k | 65.0 / 3,468 | 62.5 / 1,933 | -2.5 | -44% |

**What the rung costs on its own.** Called directly at `reasoning_effort=low`, the model
answers 60 MATH problems at 65.0% against 76.7% at the default, on 167 generated tokens
against 566. Inside the ladder that 11.7-point penalty does not reach the score: 69 to 72%
of MATH items stop at the cheap setting and the arm loses 1.8 to 2.5 points, none of it
significant. This is the first direct evidence in this project that the gate puts the cheap
rung where it belongs, because it is the first model on which the cheap rung is measurably
worse; on the other two families L0 and L2 differ only in scaffolding.

**Where it also buys accuracy.** On BigCodeBench the rung is ahead of prefix-only
governance at every budget, by +14.8, +4.0 and +5.2 points (all significant), while
spending 46 to 69% fewer tokens. Its score barely moves with the allowance (50.8, 51.8,
51.0) where prefix-only governance climbs from 36.0 to 45.8, so it reaches at 4k what the
other needs 16k to approach.

## 12. Limitations, stated as they should appear

| | |
|---|---|
| **Single seed** | temperature 0, seed 42. Re-running an identical configuration flips 8/122 = 6.6% of GAIA and 22/400 = 5.5% of MATH items under continuous batching. Paired tests are unaffected; any single rate is not. |
| **Thinking disabled everywhere** | The harness reads only `message.content` and never carries a reasoning trace into the next round, so the prompt-composition measurement is unaffected by construction. What would change is how fast the allowance is spent, which binds the cap harder and should if anything favour the method. Not measured. |
| **The gain is conditional** | Token cost and truncation hold in all 24 cells. Accuracy holds in 18. The six exceptions are cells where the cap does not bind or the baseline has no room. |
| **The router is weak** | AUC 0.75 on MATH and FRAMES, 0.62 on GAIA, 0.52 on MuSiQue. The quantile form fixes the cost, not the discrimination. |
| **Decomposition is not significant** | No single mechanism increment reaches significance at n=400. |
| **Two carried slots untested** | The critique and raw tool output are 1.9% and 0.4% of characters on Qwen, too small to price. On Gemma the critique is 10.0%, so this is a per-model fact, not a general one, and bounding it there is untested. |
| **The motivating measurement does not fully transfer** | The plan is 40.4% of prompt characters on Qwen and 28.7% on Gemma, where it is smaller than the question. It remains the largest bindable slot on both. |
| **Absolute numbers are low** | MuSiQue 5.3 -> 8.7, FRAMES 8.0 -> 13.9. Conclusions drawn where nine in ten tasks fail. |
| **Reasoning on is only covered by one family** | GLM runs with reasoning on because it cannot be turned off, and needs the reasoning rung to recover the token result. Whether a model that *can* turn reasoning off behaves the same way with it on is untested. |
| **The governed arm is not one configuration** | Qwen and Gemma are governed by the carried-slot bounds alone; GLM adds the reasoning rung. The table is a family of configurations selected per model, not one system run three times. Say so. |
| **Compaction and slot ablation are not on GLM** | The compaction grid and the counterfactual slot ablation were not re-run on the third family. |
| **Mid-size models only** | 27B and 31B. Whether carried state still dominates for a model that needs fewer rounds is untested and is the most likely reviewer objection. |
| **No published-framework baseline** | The baseline is our own pipeline plus a format-alignment control, not AutoGen or MetaGPT. |

## 13. Defects found and fixed during this work

Listed because several produced publishable-looking numbers that were wrong, and a
write-up should not resurrect them from older notes.

| Defect | Symptom | Fix |
|---|---|---|
| Compaction trigger read cumulative spend, not context occupancy | 100% trigger rate, spurious correlation | `prompt_tokens / context_window > ratio` |
| Summariser appended a second system message | Qwen template rejects with HTTP 400, error swallowed, every compacted task scored zero | merge summary into the existing system message, keep the task |
| Summariser dropped the question | tasks answered without knowing what was asked | preserve head and task, summarise only the middle |
| API errors silent | a malformed request looked like a model with nothing to say | log `[APIERR]` |
| `stopped_by=="budget"` conflated with API timeouts | truncation counts inflated | require more than half the cap consumed |
| BigCodeBench judge starved (2 CPUs, 64 workers) | tests failed on wall clock; produced a significant but false finding that more budget hurts | judge offline from stored replies, 10 CPUs, 90s |
| Trace files mixed across runs | the same cell exists as a primary run, a flip-rate re-run, a speed test and a calibration run; globbing picked one at random | explicit file map, see `_gen_materials.py` |
| No retry on transient API errors | pinned provider rate-limits silently dropped rounds | bounded retry on 429/5xx/timeout |
| Gateway returned 200 with `choices: null` | `TypeError` that reads nothing like a network fault | named and routed into the retry |
| Skip check matched its own negation | The chain asked `grep "complete"` against a status line whose incomplete state was written as a superstring of the complete one, so a cell holding two of its three arms was skipped as finished | ASCII DONE/TODO, no shared substring |
| urllib blocked by the CDN at 403 | Probe scripts failed on every request while curl and the openai client worked, because urllib sends no User-Agent | forward a User-Agent |
| Two endpoints could claim one cell | Parallel shards would have appended to the same trace file | claim by `mkdir`, which is atomic |
| llama.cpp enabled thinking by default | reasoning went to `reasoning_content`, which the harness never reads, so answers came back empty | `--reasoning off` |
| Wrong tool-call parser for Gemma | the model emits XML-style calls, `hermes` parses none | `qwen3_xml` |

## 14. What was run and dropped

**SWE-bench Verified.** Ran to completion, oracle-localization variant, official harness.
Final: Naive 184/500 = 36.8%, Governed 169/500 = 33.8% (dSR -3.0, p=0.142, not
significant), the verbalized-channel arm 135/500 = 27.0%. Dropped from the paper: the
setup gives localization away, so the numbers are not comparable to agentic resolve rates,
and the one significant finding it produced (that removing the verbalized channel costs
6.8 pp on repair) does not replicate on BigCodeBench, where removing it is free or better
at the tight budgets. The likely mechanism is specific to the oracle setup: with the buggy
region supplied, the gate sees an easy-looking task, stays at L0, and returns the code
unchanged, which happened on 93 instances. Traces remain under `traces/*sweo*`.

**OpenHands reference run.** Set up, validated end to end on three instances, then
cancelled. It measures the model under a standard agentic scaffold, which is orthogonal to
the method. Apparatus is in `/mnt/data/oh` if wanted: per-instance token metering proxy,
budget enforcement that synthesises a `finish` call, batched image building.

**Not run at all:** any frontier model; any published agent framework as a baseline; the
slot-ablation and decomposition arms on Gemma; thinking enabled.

## 15. File map

| What | Where |
|---|---|
| Qwen main table | `traces/tasks_sweep_Qwen3.8-27B_q38_*.jsonl` (Naive, Naive+Fmt), `..._q38ab7_*` and `..._q38_ab7_*` (Governed) |
| Gemma main table | `tasks_sweep_gemma-4-31B-bf16_gor_*.jsonl` |
| GLM main table | `tasks_sweep_GLM-5.3-Flash_glm4_*.jsonl`. The reasoning-rung ablation is `..._glmab7_ma*` (MATH) and `_aborted/glm_pre_reasoning/` (GAIA, from the run that preceded the rung) |
| GLM BigCodeBench | `tasks_sweep_GLM-5.3-Flash_glmbcb*_rejudged.jsonl` |
| Compaction | `..._acc_*` / `..._cmp3_*` (Qwen), `..._gcmp_*` (Gemma). **`cmp_*` and `cmp2_*` are defective** |
| BigCodeBench | `..._bcb{4,8,16}k_rejudged.jsonl` (Qwen), `..._gbcb*_rejudged.jsonl` (Gemma) |
| Segment accounting | `..._seg_*.jsonl` (Qwen), `..._gseg_*.jsonl` (Gemma) |
| Slot ablation | `results/slot_ablation_cp_*.json` |
| Decomposition | `..._q38_abl_ma.jsonl` (24k), `..._qdec_ma{8,16}.jsonl` |
| SWE (dropped) | `traces/*sweo*`, `swe_preds_v2_*.json`, `logs/run_evaluation/v2_*` |
| Paper | `paper/main.tex`, build with `paper/build.sh` |
| This file | regenerate with `python3 _gen_materials.py` |

