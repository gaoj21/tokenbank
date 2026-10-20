# Statelessness as Budget Policy: Governing Multi-Agent Test-Time Compute

## Abstract

A multi-agent pipeline under a fixed token budget appears to spend that budget on reasoning.
It does not. Attributing every prompt character to what it carries, across five benchmarks
and four model families, we find that 46.6% of the budget is coordination state the loop
wrote on an earlier round and re-sends on this one, against 30.5% that is the model producing
anything new. One slot dominates the portion a loop can act on: the plan alone accounts for
40.4% of every character the model reads on one family and 28.7% on another, in both cases
exceeding every other bindable slot combined. The asymmetry is structural rather than
incidental. A loop that accumulates its history pays a per-round cost that rises by roughly
670 tokens a round, making total cost quadratic in rounds; a loop that rebuilds its context
each round avoids the growth but inherits the verbosity of the round before. Under a hard cap it converts into a specific
failure: the accumulated state exhausts the budget mid-loop and the pipeline returns a
truncated answer worse than one it was already holding.

We treat statelessness as a budget policy. Each round forwards only the last answer and the
last critique, each capped at constant size, so that the affordable round count ceases to
depend on how much the previous round wrote. Above the loop, a cost-ordered ladder admits a
task to the full pipeline only when a confidence gate on the model's own log-probabilities
fails, on the principle that the cheapest state is the state never created. On models that
reason, bounding the prefix is not enough, because the term that grows is the per-call
generation rather than the carried state; the same arm that saves 24 to 67% elsewhere saved
2% on one such model. We therefore add reasoning as a rung of the same ladder, giving the
levels the gate keeps a cheap reasoning setting while leaving everything else identical, so
that one admission rule governs scaffolding, carried state and generation together. Whether
the reasoning rung helps is predictable in advance: the share of items whose generation hits
its cap without the rung separates the sign of its effect in all twelve cells where we
measure it.

Across four benchmarks, three budgets and four model families, the governed system spends 24
to 82% fewer tokens than a format-aligned baseline in all 48 cells and exhausts the budget in
none, against a baseline that truncates on up to 1,399 tasks of 1,934. Accuracy improves
significantly in 27 of the 48 cells and falls significantly in none. On BigCodeBench, where
execution judging removes scoring ambiguity, the governed arm wins 10 of 12 cells, by 42.2
points at the tightest budget on the family whose reasoning is longest. The result is
deliberately asymmetric: the token saving is unconditional and free, while the accuracy gain
is contingent on whether a task needs many rounds and on how much of the generation allowance
reasoning consumes. Relaxing the cap removes the gain, which is the test we use to argue that
the budget produced it.

## 1. Introduction

Test-time compute is now a design variable. Given a fixed token allowance per task, a system
designer chooses how to spend it: more samples, more rounds of self-critique, more agents,
longer reasoning traces. The literature on scaling test-time compute has largely asked how
much to spend and on what. This paper asks a prior question, which is what a multi-agent
pipeline actually spends its allowance on once it has one, and finds that the answer is
mostly not what the designer intended.

The finding that motivates everything else is a measurement rather than a method. We
instrument a plan-execute-review pipeline to attribute every prompt character to the slot it
occupies: the plan, the question, the draft answer under revision, the reviewer's critique,
tool output, or the system prompt. Pooled over five benchmarks and four model families, under
a 16k budget, the largest single category is not the model writing anything. It is
coordination state that a previous round produced and that this round re-sends verbatim. The
plan slot alone is larger than every other bindable slot put together.

Where this state lives depends on how the loop builds its context, and there are three
regimes rather than two. An *accumulating* loop, the design most agent scaffolds use, appends
each round's answer and critique to the previous context, so the carried prefix grows linearly
in the round index and the total cost of $R$ rounds grows quadratically. A *rebuilding* loop
reconstructs its context each round from the question, the plan, the last answer and the last
critique; its per-round cost is constant in the round index but is set by how much the
previous round wrote, which is unbounded. A *bounded* loop rebuilds and also caps each carried
slot. Measured on the same pipeline, per-round prompt cost rises by roughly 670 tokens per
round under accumulation on both non-reasoning families, is flat but large under rebuilding,
and is flat and 2.5 to 2.9 times smaller under bounding. Write the affordable round count under a budget $B$ as
$R_B \ge \min(R_{\max}, \lfloor B / \bar{C} \rfloor)$ with mean per-round cost
$\bar{C} = c_{\text{fixed}} + \bar{g} + \kappa_n + \kappa_v + \kappa_p$, where $c_{\text{fixed}}$
is the system prompt and question, $\bar{g}$ is mean generation, and the $\kappa$ terms are
the carried answer, critique and plan. Under accumulation the $\kappa$ terms grow with the
round index, so $\bar{C}$ grows and $R_B$ scales as $\sqrt{B}$ rather than $B$; under
rebuilding they are constant but inherit the verbosity of the round before. In both cases the
budget is consumed fastest by the tasks that need the most rounds.

Our intervention is to make the $\kappa$ terms constant. At each round boundary the loop
rebuilds its context as the system prompt, the original question, the last answer and the
last critique, and nothing else, with the critique and any tool output capped at fixed size
and the plan reduced to a single step line. Round seven then carries exactly what round two
carried. This is the sense in which statelessness is a budget policy rather than an
architectural preference: it converts $R_B$ from a function of the loop's own verbosity into
a function of the budget alone.

We organise the method around one principle applied to three resources. Every term of
$\bar{C}$ is something the system can choose to spend or not: whether to build the loop at
all, how much state each round carries, and how long each call reasons. A single admission
rule, spend the cheaper option unless a free confidence signal says it is insufficient,
governs all three.

Bounding what a loop carries does nothing for a task that never needed the loop. We therefore
place the pipeline at the top of a cost-ordered ladder whose cheaper rungs are a single direct
answer and a single tool-assisted answer, and admit a task upward only when a confidence gate
fails. The gate reads the mean log-probability and the mean top-$k$ entropy of the answer the
model has already produced, so routing costs nothing beyond what the answering call already
returned. No verifier model, no additional sample, no second pass.

A late finding reshaped the paper. On a model whose thinking cannot be disabled, the governed
arm saved 2% of tokens where it saved 24 to 67% on models with thinking off. The diagnosis is
that our bound applies to $\kappa$ while the term that grows on a reasoning model is
$\bar{g}$. The fix preserves the ladder's admission principle and extends it to the reasoning
setting: the levels the gate keeps get a cheap reasoning setting, and a task that escalates
gets the default one. This restored the token advantage to 39 to 65% on that model and, on a
second reasoning family added to test the fix, produced the largest accuracy gains in the
paper. Whether the reasoning rung helps or hurts turns out to be predictable before it is
switched on: across twelve cells on two reasoning families, the share of items whose
generation hits its cap under prefix-only governance has a rank correlation of 0.78 with the
rung's accuracy effect, and separates its sign in every cell.

We report a result that is uneven by design. Four families, four benchmarks and three budgets
give 48 main cells. The governed arm spends fewer tokens in all 48. It improves accuracy
significantly in 27 and degrades it significantly in none. The distribution of those 27 is
not uniform: one family gains in all twelve of its cells, another in two. We take the
unconditional claim to be the token claim, and we are explicit throughout about which
benchmark and which family each accuracy claim rests on.

**Contributions.** (i) A segment-level accounting of where a budgeted agent loop's tokens go,
which locates the plan slot as the dominant bindable term, and a measured distinction between
accumulating, rebuilding and bounded loops: quadratic total cost, constant but
verbosity-dependent cost, and constant bounded cost. (ii) Test-time compute governance stated
as one admission rule applied to three resources, whether to build the loop, how much state to
carry, and how long to reason, each a term of the per-round cost. (iii) A zero-cost confidence
gate built from log-probabilities the answering call already produced, which routes usefully
although it is a poor correctness classifier, for a reason we make explicit. (iv) Reasoning as
a rung of the same ladder, with the measurement showing prefix-only governance fails on
reasoning models, and a criterion measurable in advance that predicts the sign of the rung's
effect. (v) An evaluation across 48 main and 12 execution-judged cells against a
format-aligned baseline, with the token saving holding in every cell and the accuracy gain
reported cell by cell.

## 2. Related Work

**Test-time compute scaling.** Work on allocating more computation at inference has
established that additional sampling, search and self-revision buy accuracy, and has studied
how that trade curves. That literature takes the spend as the independent variable. We take
it as a constraint and ask what the system does with a fixed one, which makes the object of
study the composition of the spend rather than its size.

**Multi-agent and self-refinement pipelines.** Plan-execute-review and debate-style
architectures improve over single calls on many tasks, and their cost is usually reported as
a multiple of a single call. We measure that multiple at the level of what each token
carries, which is what exposes the prefix as the dominant term and makes it addressable.

**Bounded-state reasoning.** The Markovian Thinker (Delethink) trains a reasoning policy to
continue from a constant-size textual state at fixed chunk boundaries, making the cost of long
reasoning linear rather than quadratic in its length; InftyThink takes a related iterative
summarisation approach. We make the analogous move for a multi-role pipeline rather than a
single chain, and without training: the round boundary is the chunk boundary, and the capped
answer, critique and plan line are the carried state.

**Cost attribution in agent workflows.** Total Cost of Agency decomposes multi-agent workflow
cost into base prompt, inference, memory injection and context accumulation, and finds
injection growing linearly with workflow depth to 27.6% of variable cost. Our accounting is at
the granularity of pipeline slots rather than memory retrieval, finds a larger share of
re-sent coordination state (46.6%) dominated by a single slot, the plan, and is paired with an
intervention whose effect is measured.

**Budget-aware agents.** BATS gives a tool-using agent a running view of its remaining budget
and lets it adapt its search, and BAMAS structures a multi-agent system under a cost budget.
These make the agent aware of the budget; we instead enforce it outside the model, by
admission before each call, so that compliance does not depend on the model acting on what it
is told.

**Context compression and memory.** Summarisation, retrieval over past turns, and
compaction-on-threshold all reduce a growing context after the fact. Our policy differs in
kind rather than degree: it never lets the context grow, so there is nothing to compress. We
include accumulate-then-compact as an explicit baseline, pinned to a fixed window so that it
is the same policy on every deployment, and report where it stands relative to bounded state.

**Cascades and routing.** Recent work characterises the cost-quality frontier of LLM cascades
decision-theoretically and gives conformal guarantees on cascade accuracy or cost overrun. Our
ladder is a cascade over configurations of one model rather than over models, and its three
rungs differ in scaffolding and reasoning length rather than in parameter count.

**Confidence estimation and routing.** Log-probability confidence, entropy and verifier-based
selection have been used to decide when to sample again or escalate to a larger model. Our
use is unusual in two respects: the signal is free, taken from the answering call rather than
a separate pass; and we argue that its poor calibration as a correctness classifier does not
disqualify it for routing, because the cost of a routing error is bounded by the accuracy
difference between adjacent rungs rather than by returning a wrong answer.

**Budgeted and constrained decoding.** Systems that enforce token or latency budgets typically
do so by truncation or by early exit. We enforce the budget by admission control before each
call, which lets a blocked task fall back to a reserved closing answer instead of returning a
truncated one, and we measure that this removes budget-exhaustion truncation entirely.

## 3. What a Budgeted Agent Loop Spends

### 3.1 Segment accounting

We instrument the pipeline so that every prompt character sent to the model is attributed to
one of six slots: the plan, the question, the draft answer under revision, the reviewer's
critique, tool output, and the system prompt. Table 1 pools this attribution over five
benchmarks at a 16k budget under the format-aligned baseline.

**Table 1.** Share of all prompt characters by slot, pooled over five benchmarks, `Naive+Fmt`
at 16k.

| Model | plan | question | draft | critique | tool | system |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B | **40.4%** | 27.3% | 24.4% | 1.9% | 0.4% | 4.4% |
| Gemma-4-31B (bf16) | **28.7%** | 32.2% | 19.8% | 10.0% | 0.5% | 7.3% |

Counted against the whole budget rather than the prompt alone, 30.5% of a Qwen task's tokens
are the model generating, 19.0% are the question, and 46.6% are coordination state written on
an earlier round and re-sent on this one. The plan is the largest bindable slot on both
families, and on Qwen it is larger than the draft, critique, tool output and system prompt
combined.

### 3.2 Three regimes of carried state

How the carried state behaves over rounds depends on how the loop builds each round's context.
We distinguish three regimes and measure all three on the same pipeline. An *accumulating*
loop appends each round's answer and critique to the previous context. A *rebuilding* loop,
which is what our `Naive` and `Naive+Fmt` baselines do, reconstructs the context each round
from the question, the full plan, the last answer and the last critique, all at whatever
length the previous round produced. A *bounded* loop rebuilds and caps every carried slot.

**Table 2.** Mean executor prompt tokens at each round of the full loop, 24k, pooled over
four benchmarks. The governed arm's first round carries the full plan, which is then frozen to
one line.

| Model | Regime | $t=1$ | $t=2$ | $t=3$ | $t=4$ | $t=5$ | slope / round |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B | accumulating | 966 | 1,496 | 2,461 | 3,217 | 3,435 | **+666** |
| Qwen3.8-27B | rebuilding (`Naive+Fmt`) | 975 | 1,388 | 1,633 | 1,607 | 1,589 | +145 |
| Qwen3.8-27B | bounded (Governed) | 2,137 | 625 | 678 | 605 | 547 | flat |
| Gemma-4-31B (bf16) | accumulating | 436 | 1,141 | 1,882 | 2,572 | 3,082 | **+672** |
| Gemma-4-31B (bf16) | rebuilding (`Naive+Fmt`) | 433 | 1,031 | 1,058 | 1,055 | 1,108 | +137 |
| Gemma-4-31B (bf16) | bounded (Governed) | 599 | 354 | 367 | 386 | 386 | flat |

Under accumulation the per-round cost rises by about 670 tokens per round on both families,
so the total cost of $R$ rounds is $R c_0 + a R^2 / 2$ and the affordable round count grows
only as $\sqrt{2B/a}$. Rebuilding removes the growth after the first round, which is why it
is the stronger baseline, but its plateau is set by the length of the previous answer and
critique, which nothing bounds. Bounding cuts that plateau by a factor of 2.5 on Qwen and 2.9
on Gemma. The consequence shows up as the share of items the budget stops:

**Table 3.** Share of items stopped by budget exhaustion, pooled over four benchmarks.

| Model | Regime | 8k | 16k | 24k |
|---|---|---:|---:|---:|
| Qwen3.8-27B | accumulating | 70.7% | 59.5% | 27.1% |
| Qwen3.8-27B | rebuilding | 65.8% | 38.4% | 2.0% |
| Qwen3.8-27B | bounded | **0.0%** | **0.0%** | **0.7%** |
| Gemma-4-31B (bf16) | accumulating | 16.6% | 8.3% | 2.2% |
| Gemma-4-31B (bf16) | rebuilding | 14.2% | 1.9% | 0.4% |
| Gemma-4-31B (bf16) | bounded | **0.0%** | **0.0%** | **0.0%** |

The bounded loop is stopped by the budget on 0.7% of Qwen items at 24k and on none in any
other cell, against up to 70.7% under accumulation. The
argument of this section can be stated through the per-round cost. With mean per-round cost
$\bar{C} = c_{\text{fixed}} + \bar{g} + \kappa_n + \kappa_v + \kappa_p$ and a hard budget $B$,
the affordable round count is $R_B \ge \min(R_{\max}, \lfloor B / \bar{C} \rfloor)$. The
$\kappa$ terms are the carried answer, critique and plan. When they grow with the round
index, $\bar{C}$ grows; when they are rebuilt but uncapped, $\bar{C}$ is constant but large
and varies with the verbosity of the previous round. Only when every $\kappa$ is capped is
$R_B$ a function of $B$ alone. This mirrors the argument made for single reasoning chains by
the Markovian Thinker, which bounds the state a policy reads in order to make reasoning cost
linear rather than quadratic in its length; we make the same move at the level of a
multi-role pipeline, without training.

### 3.3 The failure this produces

Under a binding cap, the growth converts into a concrete failure mode. A loop holding a
correct answer at some round continues, the accumulated state exhausts the budget, and the
pipeline returns whatever it has at that moment, which is often worse. Table 4 separates
correct-to-wrong transitions into revision, where a later round overwrites a correct answer,
and truncation, where budget exhaustion replaces it. Rates are conditional on the item having
been correct at some round.

**Table 4.** Correct-to-wrong transitions at 24k, conditional rate.

| Model | Dataset | `Naive+Fmt` | Governed |
|---|---|---:|---:|
| Qwen3.8-27B | GAIA | 50.0% | **7.1%** |
| Qwen3.8-27B | FRAMES | 41.8% | **8.0%** |
| Qwen3.8-27B | MuSiQue | 39.6% | **10.6%** |
| Gemma-4-31B (bf16) | GAIA | 25.0% | **0.0%** |
| Gemma-4-31B (bf16) | MuSiQue | 11.5% | **3.6%** |
| GLM-5.3-Flash | GAIA | 24.1% | 28.6% |
| GLM-5.3-Flash | MuSiQue | 1.7% | 8.7% |
| DeepSeek-V4.1-Flash | FRAMES | 5.2% | **1.1%** |

Truncation falls to zero in all sixteen cells, since a blocked call falls back to a reserved
closing answer rather than returning a fragment. The conditional rate falls in twelve cells
and rises in four, and the four are all GLM's. We report this rather than pooling it away: on
the family where the ladder escalates most aggressively, the governed arm revises more often
and sometimes revises away a correct answer.

## 4. Method

**One rule, three resources.** Each term of the per-round cost $\bar{C}$ is a resource the
system can choose to spend. The method applies one admission rule to all of them: take the
cheaper option, and pay for the expensive one only when a confidence signal that costs nothing
says the cheap one is insufficient.

| Resource | Term in $\bar{C}$ | Cheap option | Expensive option | Decided by |
|---|---|---|---|---|
| Scaffolding | whether rounds occur at all | direct answer ($L_0$, $L_1$) | full plan-execute-review loop ($L_2$) | gate $\phi_\ell$ at each rung |
| Carried state | $\kappa_n + \kappa_v + \kappa_p$ | last answer, capped critique, one plan line | full history | fixed policy, every round |
| Generation | $\bar{g}$ | cheap reasoning setting | default reasoning | the same gate $\phi_\ell$ |

The three are not independent mechanisms that happen to coexist. The gate that decides whether
a task is admitted to the loop also decides how long it may reason, so a task is cheap or
expensive along all three axes together, and the expected cost of a workload is set by one
quantity, the share of items the gate keeps at the cheap rungs.

We describe the system in the order in which a task passes through it: the budget model and
the admission rule that enforces it (§4.1), the bounded-state policy that makes per-round
cost constant (§4.2), the cost-ordered ladder that decides whether the loop is built at all
(§4.3), the confidence gate that routes between rungs (§4.4), the stopping rule inside the
loop (§4.5), and the extension that makes the whole thing apply to models that reason (§4.6).
§4.7 states the full configuration and what we removed from it; §4.8 accounts for what the
governance itself costs.

### 4.1 Budget model and admission control

**The pipeline.** A task $x$ is answered by three roles. A Planner decomposes the question
into steps. An Executor answers, and may emit a Python block which the harness executes with
a ten-second timeout, returning the printed output as a new turn so that the Executor can
revise before committing. A Reviewer inspects the answer and either accepts it or requests a
revision. Executor and Reviewer alternate for at most $R_{\max} = 5$ rounds.

**The budget.** A budget $B$ is a hard per-task cap on $\sum_{\text{calls}} (\text{prompt} +
\text{completion})$ tokens, summed over every call of every role. It is not a per-call limit,
a context-window limit, or a latency target. It is the quantity a deployment actually pays,
and the whole point of the paper is what a pipeline does when that quantity is fixed in
advance rather than discovered afterwards.

**Admission rather than truncation.** The budget is enforced before each call rather than by
cutting a call short. The harness maintains a reserve $\rho$ initialised to $B$ and a per-role
ledger with quotas $q_r$. Before a call it estimates the cost as
$\hat{c} = |\text{prompt}|_{\text{tok}} + m$, where $m$ is the requested generation cap, and
admits the call only if

$$u_r + \hat{c} \le q_r \quad \wedge \quad \hat{c} \le \rho - \rho_{\text{final}},$$

with $u_r$ the role's consumption so far. Quotas are assigned at task start by splitting $B$
across roles in proportion to a fixed role priority, each role guaranteed a floor. A rejected
call returns a failure with a reason string rather than raising, and the two reasons that
appear, quota exhaustion and reserve exhaustion, distinguish a role that overspent from a task
that ran out. Both are recorded per item.

**The closing reserve.** $\rho_{\text{final}} = 256$ tokens are withheld from every ordinary
call. When a call is refused, the loop spends this reserve on a single Executor call prompted
to give its best final answer concisely, marked so that it bypasses quota and pacing and is
checked only against the raw reserve. This is a small mechanism with a disproportionate
effect: it is the reason budget-exhaustion truncation falls to zero in all sixteen cells of
Table 4. An ungoverned loop that runs out of budget returns whatever fragment it is holding;
this one always returns a deliberate answer. The 256 tokens are a fixed cost of roughly 1% of
the tightest budget we run, paid on every task whether or not it is used.

**The inequality that motivates the rest.** Let $\bar{C}$ be the mean per-round cost of the
loop. The number of rounds a task can afford is

$$R_B \;\ge\; \min\!\left(R_{\max},\ \left\lfloor B / \bar{C} \right\rfloor\right), \qquad
\bar{C} \;=\; c_{\text{fixed}} + \bar{g} + \kappa_n + \kappa_v + \kappa_p ,$$

where $c_{\text{fixed}}$ is the system prompt and question, $\bar{g}$ the mean generation per
round, and $\kappa_n$, $\kappa_v$, $\kappa_p$ the carried answer, critique and plan. In an
ungoverned loop the $\kappa$ terms are functions of the round index, because round $t$'s
context contains rounds $1{:}t{-}1$. Then $\bar{C}$ grows in $t$, so $R_B$ falls, and it falls
fastest on the tasks that need the most rounds. Everything in §4.2 exists to make the $\kappa$
terms constants, which turns $R_B$ into a function of $B$ alone. §4.6 exists because on a
reasoning model the growing term is $\bar{g}$ instead, and bounding $\kappa$ leaves it
untouched.

### 4.2 Bounded state

**The policy.** At the end of each round the loop discards its history and rebuilds the
context from four parts: the system prompt, the original question, the last answer, and the
last critique. Formally, writing $s$ for the system prompt, $q$ for the question, $a_t$ for
the answer at round $t$ and $v_t$ for the critique,

$$\text{ctx}_{t+1} \;=\; [\,s,\; q \oplus \pi_t,\; a_t,\; v_t\,]
\qquad\text{rather than}\qquad
\text{ctx}_{t+1} = \text{ctx}_t \oplus [a_t, v_t],$$

where $\pi_t$ is the plan slot described below. Round seven therefore carries exactly what
round two carried. The contrast on the right is the accumulating policy, which is what the
compaction baseline in §6.4 runs.

This is a small change in code and a large one in the cost model. Under accumulation,
$\kappa_n$ and $\kappa_v$ are sums over all previous rounds; under the bounded policy they are
single terms whose size is capped. The loop's per-round cost stops depending on how verbose it
has been.

**Caps on the three carried slots.** Bounding what is carried is not enough if what is carried
is itself unbounded, so each slot has a construction that fixes its size.

*The critique.* The Reviewer is constrained to emit JSON with at most three issues of twenty
words each and a hint of thirty, under a 150-token generation cap. This bounds $\kappa_v$ by
construction rather than by truncation, which matters because a truncated critique is often a
critique whose conclusion has been cut off, and is worse than a short one.

*Tool output.* Raw stdout from an executed block is not carried. The harness spends one short
call compressing it to a note of at most 120 tokens containing facts and numbers only, and
carries the note. Raw output is capped at 4,000 characters before even that call, and if the
compressing call fails the first 800 characters are used. This is the one place where the
method spends tokens to save tokens, and §4.8 accounts for it.

*The plan.* After the first round, the full plan in the prompt is replaced by a single line,
the current step. This is what makes $\kappa_p$ constant, and given that the plan is the
largest slot in Table 1, it is the single largest reduction the policy makes.

**A correction in the plan slot worth stating.** The obvious implementation of plan-line
selection, taking the second line of the plan text, is wrong in a way that no aggregate metric
reveals. Plans frequently open with a heading, and repair-style plans more often than
computation-style ones restate the problem before starting the list. Sampling six plans of
each type, taking the second line verbatim kept a bare heading such as "Step-by-Step Plan:" on
four of six repair plans against one of six computation plans. The bounded slot was therefore
empty of content precisely on the task type where the plan matters most, while the token
accounting looked correct because a heading costs tokens like any other line. We strip
scaffolding lines with a boilerplate pattern before indexing, and fall back to the unfiltered
list if filtering would empty it. We report this because a bounded-state policy is exactly the
setting in which such a defect hides: the slot is small by design, so its being empty is not
visible in the size statistics.

### 4.3 The cost-ordered ladder

Bounding what a loop carries helps a task that enters the loop. It does nothing for a task
that never needed one. The cheapest state is the state never created, so we place the pipeline
at the top of three rungs and enter at the bottom.

**$L_0$, direct answer.** One Executor call with no plan, no tool and no Reviewer, capped at
2,048 generated tokens.

**$L_1$, tool-assisted answer.** The same primitive with the Python tool enabled, and seeded
with $L_0$'s answer as a draft together with an instruction to verify and correct it. $L_0$'s
work is reused rather than discarded, so the ladder is incremental in cost rather than
restarting.

**$L_2$, the full loop.** Planner, then the bounded Executor-Reviewer alternation of §4.2 for
up to five rounds, seeded with the plan and with whatever answer $L_1$ produced.

A task stops at the first rung whose answer passes the gate of §4.4 and otherwise escalates.
The ordering is by cost rather than by capability, which is what makes the arrangement a
budget mechanism: the expected spend of a population of tasks is the spend of the cheapest
rung plus the escalation rate times the marginal cost of the next, recursively. Table 10 shows
that the escalation rate varies from 2.7% to 71.3% across benchmarks under one fixed
threshold, which is the quantity that determines how much the ladder saves on a given
workload.

### 4.4 The confidence gate

**Definition.** Let $r$ be a completion with per-token log-probabilities $\{\log p_i\}$ and,
at each position, a returned top-$k$ distribution. Write
$\overline{\log p}(r) = \frac{1}{n}\sum_i \log p_i$ and
$\bar{H}(r) = \frac{1}{n} \sum_i H(\text{top-}k_i)$. The gate at rung $\ell$ is

$$\phi_\ell(r) \;=\; \mathbb{1}\!\left[\,\overline{\log p}(r) > \theta_\ell \;\wedge\;
\bar{H}(r) \le \tau_H \,\right].$$

Both statistics come from the call that produced $r$. There is no verifier model, no
additional sample, and no second pass, so routing is free in tokens and free in latency. This
is the property that makes a ladder practical under a token budget: a routing mechanism that
itself costs tokens competes with the thing it is routing.

**Thresholds.** $\theta_0 = -0.10$ at $L_0$, $\theta_1 = -0.15$ at $L_1$, $\tau_H = 0.6$
throughout, and $\tau_{\text{conf}} = -0.05$ for the early stop inside $L_2$. These were tuned
once on development splits and held fixed across every benchmark, budget and model family
reported in the paper.

**Why $\theta_1$ is looser than $\theta_0$.** A task reaching $L_1$ has already been rejected
once and has since been given a tool. The evidence available at $L_1$ is strictly better, and
the alternative to stopping is the full loop, which is expensive. Requiring more confidence at
a later rung would escalate tasks whose remaining uncertainty the loop cannot resolve either.
The asymmetry is therefore deliberate, not a tuning artefact.

**The entropy veto.** $\bar{H} > \tau_H$ blocks stopping regardless of log-probability. Mean
log-probability is high both when the model is right and when it is confidently repeating a
malformed pattern; entropy separates those cases better than the mean does, and using it as a
veto rather than as a second scored term keeps the rule interpretable.

**Failure direction.** A missing log-probability is treated as failing the gate, so the task
escalates. A deployment that does not return log-probabilities therefore degrades this system
to the ungoverned full pipeline rather than to a wrong answer. Given how unevenly providers
support the feature, this is a deployment property we wanted to be safe by construction rather
than by configuration.

### 4.5 Stopping inside the loop

Within $L_2$ the loop stops on the conjunction of a stability or approval signal with a
confidence confirmation, subject to the entropy veto:

$$\text{stop}_t \;=\; \left[\bar{H}(a_t) \le \tau_H\right] \wedge
\left[\overline{\log p}(a_t) > \tau_{\text{conf}}\right] \wedge
\left[\,\underbrace{\mathcal{N}(a_t) = \mathcal{N}(a_{t-1})}_{\text{stability}} \;\vee\;
\underbrace{v_t = \texttt{accept}}_{\text{approval}}\,\right],$$

where $\mathcal{N}$ is answer normalisation and extraction. Round $R_{\max}$ stops
unconditionally.

Two properties follow from the shape of this rule rather than from an added safeguard. First,
neither stability nor approval alone can stop the loop, so a loop that has settled into
repeating a wrong answer with low confidence continues rather than terminating early on its
own persistence. Second, because stopping requires $\overline{\log p} > \tau_{\text{conf}}$, a
low-confidence first round cannot be early-stopped at all. This is the mechanism aimed at the
revision failure in Table 4, where an ungoverned loop overwrites a correct answer it was
already holding, and it is a consequence of the conjunction rather than a rollback rule bolted
on afterwards.

### 4.6 Reasoning as a rung of the same ladder

**The problem.** Sections 4.2 through 4.5 bound $\kappa$. On a model whose thinking cannot be
disabled, $\kappa$ is not the term that grows. Measured on GAIA at 24k, the arm that governs
only the prefix spent 10,998 tokens against the baseline's 11,229, a difference of 2%, where
the same arm saves 24 to 67% on models whose thinking is off. The method as stated to this
point simply misses the binding constraint on such a model, and no amount of tightening the
slot caps would recover it, because the slots are not where the tokens are going.

**The extension.** The ladder already expresses a policy about how much machinery a task
deserves. We extend the same policy to the reasoning setting: the levels the gate keeps get a
cheap reasoning setting, and a task that escalates gets the default one. In the implementation
this is one additional field on the $L_0$ and $L_1$ calls and nothing else, so a comparison
against the prefix-only arm prices exactly this one change and nothing besides it.

The conceptual point is that reasoning length is an allocation decision of the same kind as
whether to build a plan, and it should be made by the same admission rule. A system that
bounds its scaffolding while leaving generation unbounded has governed the part of the cost it
can see rather than the part that binds.

**What the cheap setting is, per deployment.** This is where the method meets an unpleasant
practical reality, and we state it as a limitation rather than hiding it in a footnote.

| Deployment | Cheap rung | Why not the alternatives |
|---|---|---|
| Thinking can be disabled (Qwen, Gemma) | not applicable | there is no rung to place; the slot bounds suffice |
| GLM-5.3-Flash | `reasoning_effort: low` | thinking cannot be disabled by any of five methods tried, all returning an identical 80-token trace; `low` genuinely shortens it, from a per-call median of 61 tokens to 15 |
| DeepSeek-V4.1-Flash | chat-template switch | `reasoning_effort` is nominal here: low, medium and high all sit near the 328-token default, and a ladder arm built on it produced non-empty answers on 22% of BigCodeBench items against 20% for the arm carrying no rung at all |

Two implementation facts cost us time and are worth recording. A hard reasoning-token budget
is honoured exactly where it is supported, but it cuts mid-thought, so what it costs in
accuracy is truncation rather than shorter reasoning; the chat-template switch starts no
thought at all, and where both are available the switch is the cleaner rung. And the two must
not be sent together: passing a graded effort alongside the switch re-enables the thought,
yielding 265 tokens as though the switch were absent, so an implementation that sets both
silently governs nothing. Our harness drops the graded setting whenever the switch is set.

**That the rung is placed correctly, not merely applied.** Applying low reasoning to every
call costs 11.7 points on MATH. Inside the ladder the same setting costs 1.8 to 2.5 points,
because the items that would have been harmed are the ones the gate escalates to the default
setting. This is the cleanest evidence in the paper that the gate, poor as it is at predicting
correctness, is placing tasks usefully.

### 4.7 The configuration, and what we removed

The governed arm is the following, held fixed across every benchmark, budget and family:

| Component | Setting |
|---|---|
| Round-boundary context | bounded: system, question, last answer, last critique |
| Reviewer | JSON verdict, 150-token cap |
| Tool output | compressed to a 120-token note |
| Plan | one step line after round 0 |
| Answer format | `ANSWER: x` suffix (disabled on BigCodeBench, where the task prompt fixes the shape) |
| Ladder | three rungs, $L_0$ and $L_1$ capped at 2,048 generated tokens |
| Gate | $\theta_0 = -0.10$, $\theta_1 = -0.15$, $\tau_H = 0.6$, $\tau_{\text{conf}} = -0.05$ |
| Loop | $R_{\max} = 5$, three-signal stop |
| Closing reserve | 256 tokens |
| Breadth | best-of-3 at temperature 0.7 when entropy exceeds $\tau_H$, otherwise single refinement |
| Reasoning rung | per deployment, per the table in §4.6 |

**The verbalized escalation channel, removed.** An earlier version let the model reply with
`ESCALATE` as the first word instead of answering, on the reasoning that a log-probability
cannot distinguish "knows the answer" from "needs to compute or look something up", whereas
the model can say which. It is disabled in every arm reported here, so the ladder is routed by
the log-probability gate alone. We removed it because it introduces a second routing signal
whose behaviour is model-dependent in a way the log-probability gate is not, and because the
ablation showed the gate alone was sufficient. Reporting the system without it makes the
routing claim cleaner: nothing in the ladder depends on the model's willingness to follow a
meta-instruction.

**What we do not enable.** Predictive rebalancing, inter-role borrowing, online priority
learning and KV-cache accounting all exist in the harness and are switched off in every
reported arm. They belong to an earlier design in which the budget was managed dynamically
across roles. We report the simpler system because the simpler system is what the results are
about, and because each additional adaptive component would need its own ablation to be
defensible.

### 4.8 What the governance costs

The method is not free, and the accounting should say so rather than reporting only the net.
Three components are pure overhead. The Reviewer costs up to 150 tokens per round on every
task that reaches $L_2$. Tool-output compression costs one 150-token call per tool use, paid
to avoid carrying a larger note. And for any task that ends up at $L_2$ anyway, $L_0$ and $L_1$
were wasted, costing up to 2,048 generated tokens each plus their prompts.

That the arm nonetheless spends 24 to 90% less is a statement about the distribution of tasks
rather than about the per-task cost. Most items never reach $L_2$, and for those the plan, the
Reviewer and the rounds are never paid for at all. How many is entirely workload-dependent:
Table 10 shows the share stopping at $L_0$ ranging from 28.7% to 97.3% across sixteen cells. On
MATH, where that share is 72 to 97%, the ladder is doing the saving; on GAIA and MuSiQue, where
it is often below half, the saving comes from bounded state instead. A practitioner can
estimate the ladder's contribution on a new workload from the escalation rate alone, which is
observable after a single pass at the cheapest rung.

### 4.9 Algorithm

```
Input: task x, budget B
Allocate B across {planner, executor, reviewer} by role priority, each above a floor
reserve <- B;  withhold rho_final = 256

L0:  r0 <- Executor(x)                       cap 2048, cheap reasoning rung
     if refused: return GraceClose()
     if phi(r0, theta0 = -0.10): return r0                        [converged]

L1:  r1 <- Executor(x, draft = r0, tool = on) cap 2048, cheap reasoning rung
     if code emitted: execute (10s), compress output to <=120 tokens, re-ask
     if refused: return GraceClose()
     if phi(r1, theta1 = -0.15): return r1                        [converged]

L2:  plan <- Planner(x)                       default reasoning setting
     ctx  <- [system, x + plan + draft]
     for t in 1..5:
         a_t <- Executor(ctx)
         if code emitted: execute, compress, re-ask
         v_t <- Reviewer(x, a_t)              cap 150, JSON verdict
         if H(a_t) <= 0.6 and logp(a_t) > -0.05
            and (N(a_t) = N(a_{t-1}) or v_t = accept):
                return a_t                                        [converged]
         if t = 5: return a_t                                     [max_rounds]
         if t = 1: replace plan in ctx with one step line
         ctx <- [system, x + plan + draft] + [a_t, v_t]      # bounded, constant in t
     any refused call: return GraceClose()                        [budget]

GraceClose(): one Executor call against rho_final, bypassing quota, asking for
              the best final answer concisely
```

## 5. Experimental Setup

**Models.** Four families, selected so that two reason and two do not, because that split
turned out to determine whether prefix governance suffices. Qwen3.8-27B-FP8 runs on vLLM with
thinking disabled and verified off. Gemma-4-31B-it runs at bf16 on OpenRouter with the
provider pinned and fallbacks disabled, since sixteen providers sit behind one model
identifier and differ in quantisation and log-probability support. GLM-5.3-Flash runs on
sglang across four endpoints we verified to be one deployment, and its thinking cannot be
disabled. DeepSeek-V4.1-Flash runs on vLLM reached through an SSH tunnel, and its thinking can
be disabled per call. All sampling is at temperature 0 with seed 42.

**Benchmarks.** MATH-500 (400 items, symbolic equivalence), GAIA (122, normalised exact
match), FRAMES (660, EM and F1), MuSiQue (1,934, EM and F1) and BigCodeBench (500, judged by
executing each task's unit tests and re-judged offline). Budgets are 8k, 16k and 24k on the
first four, and 4k, 8k and 16k on BigCodeBench, whose prompts are long enough that the looser
end never binds.

**Arms.** `Naive` is the plain pipeline. `Naive+Fmt` adds a single instruction to end the
reply with `ANSWER: x`, and is the baseline every claim is measured against. `Governed` is
the full system. Measuring against `Naive+Fmt` rather than `Naive` is not a formality: the
format instruction alone is worth 13.2 points to Qwen and 28.0 points to Gemma on MATH at
24k, and a comparison against `Naive` credits a prompt suffix to the method. Two further arms,
`Accumulate` and `Compaction`, serve the compaction comparison in Section 6.5.

**Statistics.** Exact McNemar on paired per-item outcomes throughout, with $b$ and $c$ the
discordant counts. We use a single seed and substitute paired tests together with a measured
run-to-run flip rate, which is 6.6% on GAIA and 5.5% on MATH. That rate is the noise floor for
any individual number in these tables; the paired tests are unaffected by it, since they
compare per item.

## 6. Results

### 6.1 Main table

Table 5 gives the governed arm against `Naive+Fmt` over four benchmarks, three budgets and
four model families. Across all 48 cells the governed arm spends between 24% and 82% fewer
tokens, with no cell in which it spends more, and exhausts the budget in none. Accuracy
improves significantly in 27 cells and falls significantly in none, with differences ranging
from $-2.5$ to $+19.0$ points.

**Table 5.** Governed against `Naive+Fmt`, summary by family. Full per-cell results in
Appendix A.

| Family | cells | significant gains | ΔSR range | token saving |
|---|---:|---:|---|---|
| Qwen3.8-27B | 12 | **12** | +3.0 to +19.0 | 24–57% |
| Gemma-4-31B (bf16) | 12 | 6 | −1.6 to +5.5 | 35–67% |
| GLM-5.3-Flash | 12 | 2 | −2.5 to +6.6 | 39–65% |
| DeepSeek-V4.1-Flash | 12 | 7 | −0.6 to +11.5 | 47–82% |

The distribution across families is the result rather than a defect in it. Qwen improves in
every cell. GLM improves significantly in two of twelve while saving 39 to 65% of the tokens
at accuracy that is statistically indistinguishable elsewhere. The claim that holds across all
four families is the token claim; accuracy gains are contingent, and Section 7 argues what
they are contingent on.

### 6.2 BigCodeBench

Execution judging removes the scoring ambiguity that string-matched benchmarks carry, and the
tighter budgets make the cap bind. For the two reasoning families we compare against `Ab7`,
the arm that bounds the prefix but leaves reasoning untouched, so that the comparison isolates
the reasoning rung.

**Table 6.** BigCodeBench, 500 tasks per cell, execution-judged.

| Family | B | baseline | base SR | Governed SR | ΔSR | p | tokens |
|---|---|---|---:|---:|---:|---:|---:|
| Qwen3.8-27B | 16k | Naive | 46.2% | 48.8% | +2.6 | 1.7e-01 | −81% |
| Qwen3.8-27B | 8k | Naive | 40.4% | 48.6% | **+8.2** | 3.8e-05 | −72% |
| Qwen3.8-27B | 4k | Naive | 26.0% | 47.8% | **+21.8** | 1.6e-22 | −53% |
| Gemma-4-31B | 16k | Naive | 47.8% | 48.2% | +0.4 | 8.9e-01 | −71% |
| Gemma-4-31B | 8k | Naive | 42.0% | 48.6% | **+6.6** | 2.2e-04 | −69% |
| Gemma-4-31B | 4k | Naive | 14.6% | 47.2% | **+32.6** | 2.1e-43 | −51% |
| GLM-5.3-Flash | 16k | Ab7 | 45.8% | 51.0% | **+5.2** | 3.4e-03 | −69% |
| GLM-5.3-Flash | 8k | Ab7 | 47.8% | 51.8% | **+4.0** | 3.3e-02 | −65% |
| GLM-5.3-Flash | 4k | Ab7 | 36.0% | 50.8% | **+14.8** | 1.2e-12 | −46% |
| DeepSeek-V4.1 | 16k | Ab7 | 25.6% | 53.0% | **+27.4** | 3.4e-30 | −90% |
| DeepSeek-V4.1 | 8k | Ab7 | 18.6% | 52.2% | **+33.6** | 1.3e-39 | −83% |
| DeepSeek-V4.1 | 4k | Ab7 | 10.2% | 52.4% | **+42.2** | 2.0e-55 | −57% |

Ten of twelve cells are significant, with 46 to 90% fewer tokens. The two that are not are
both at the loosest budget, where the cap does not bind, which is the expected shape if the
budget is what the method addresses.

Two features of Table 6 deserve separate statement. First, the governed arm is close to
budget-independent: on DeepSeek it scores 52.4, 52.2 and 53.0% at 4k, 8k and 16k while
spending 1,020, 1,020 and 1,018 tokens. Tightening the budget fourfold changes nothing,
because with the cheap rung the task converges in a single round and the budget is never
reached, whereas the baselines move by fifteen to twenty points over the same range. Second,
prefix-only governance can be worse than no governance at all. On DeepSeek at 8k, `Ab7` spends
7,372 tokens to `Naive`'s 7,318 and scores 18.6% against 23.0%; at 16k it spends 14,052 against
11,834 and scores 25.6% against 29.8%. Freeing prefix budget buys more rounds, 5.11 against
3.51, and on a model whose reasoning is long each additional round is consumed by reasoning
before it produces an answer.

The mechanism is visible directly. At 4k, DeepSeek's `Naive` and `Ab7` return a non-empty
answer on 18% of items and hit the generation cap on 61 to 65% of them; the governed arm
returns one on 100% and hits the cap on none.

### 6.3 Pricing the reasoning rung

The difference between `Ab7` and the reasoning ladder is what the rung is worth. Table 7 shows
that this quantity has opposite signs on the two reasoning families.

**Table 7.** MATH, prefix-only governance against the reasoning ladder.

| B | DeepSeek `Ab7` | DeepSeek ladder | Δ | GLM `Ab7` | GLM ladder | Δ |
|---|---|---|---|---|---|---|
| 24k | 6,522 tok / 55.2% | 914 / 63.2% | **+8.0** (p=5.6e-06) | 2,107 / 65.0% | 647 / 62.5% | −2.5 (p=0.25) |
| 16k | 7,002 / 55.0% | 934 / 61.0% | **+6.0** (p=7.2e-04) | 2,015 / 67.2% | 631 / 62.0% | **−5.2** (p=0.013) |

On DeepSeek the rung is worth six to eight points; on GLM it costs two and a half to five.
The determining quantity is how much of the generation allowance reasoning consumes. GLM's
trace is short enough that prefix governance alone is the better trade, and DeepSeek's is not.
A practitioner choosing between the two arms can make that determination from a single
measurement of trace length under the intended budget.

Table 7 also provides the first direct evidence that the gate places the cheap rung correctly.
Applying low reasoning unconditionally costs 11.7 points on MATH. Inside the ladder it costs
1.8 to 2.5, because the items that would have been harmed are the ones the gate escalates.

### 6.4 Predicting when the reasoning rung helps

Table 7 shows the rung helping on one reasoning family and hurting on the other. That is a
weak result if the practitioner must run both arms to find out which applies, and a useful one
if the sign can be read off a quantity measured beforehand. We test three candidates, each
measured on the prefix-only arm, that is, before the rung is switched on: the share of items
whose generation hits its cap in some round, the share returning an empty answer, and the
share of total tokens that are generation. Table 8 gives every cell in which both arms exist.

**Table 8.** The reasoning rung's effect against a quantity measured without it. "cap" is the
share of items whose generation reaches the cap in at least one round under prefix-only
governance.

| Family | Benchmark | B | cap | generation share | ΔSR (rung − prefix-only) | p |
|---|---|---|---:|---:|---:|---:|
| GLM-5.3-Flash | MATH | 24k | 16% | 16% | −2.5 | 0.25 |
| GLM-5.3-Flash | MATH | 16k | 14% | 15% | −5.2 | 0.013 |
| GLM-5.3-Flash | MATH | 8k | 14% | 15% | −0.8 | 0.82 |
| DeepSeek-V4.1-Flash | MATH | 8k | 18% | 20% | **+21.0** | 3.2e-19 |
| DeepSeek-V4.1-Flash | MATH | 24k | 21% | 23% | **+8.0** | 5.6e-06 |
| DeepSeek-V4.1-Flash | MATH | 16k | 24% | 22% | **+6.0** | 7.2e-04 |
| GLM-5.3-Flash | BCB | 8k | 27% | 30% | **+4.0** | 0.033 |
| GLM-5.3-Flash | BCB | 16k | 29% | 31% | **+5.2** | 3.4e-03 |
| DeepSeek-V4.1-Flash | BCB | 8k | 53% | 32% | **+33.6** | 1.3e-39 |
| GLM-5.3-Flash | BCB | 4k | 58% | 56% | **+14.8** | 1.2e-12 |
| DeepSeek-V4.1-Flash | BCB | 4k | 66% | 52% | **+42.2** | 2.0e-55 |
| DeepSeek-V4.1-Flash | BCB | 16k | 70% | 36% | **+27.4** | 3.4e-30 |

The cap-hit rate has a Spearman correlation of 0.78 with the rung's effect ($p = 0.003$,
$n = 12$); generation share gives 0.76. More usefully for a practitioner, the cap-hit rate
separates the sign: the three cells at or below 16% are the only ones where the rung does not
help, and every cell at 18% or above improves, nine of nine significantly. The separation is
not simply a family effect, since GLM itself crosses the threshold, losing on MATH at 14 to 16%
and gaining on BigCodeBench at 27 to 58%.

The criterion has an obvious reading. The cheap rung buys answers on items whose reasoning
would otherwise exhaust the generation allowance before producing one. Where few items do,
there is little to buy, and what the cheap setting costs in reasoning quality dominates. The
quantity is observable from a single pass of the prefix-only arm on a sample of the workload,
so the decision to enable the rung can be made before deployment. We caution that twelve
cells on two families locate a threshold rather than establish one, and that the three
negative cells share a family and a benchmark.

### 6.5 Compaction

Accumulate-then-compact is the policy most agent scaffolds implement, and it is the natural
competitor to bounded state. We run it with the trigger and window pinned in the arm
configuration rather than read from the server, so that it is the same policy on every
deployment regardless of the deployed context length.

**Table 9.** Compaction against accumulation, pooled over four benchmarks, 3,116 paired items
per cell.

| Model | B | summariser fired on | ΔSR | wins/losses | p |
|---|---|---:|---:|---:|---:|
| Qwen3.8-27B | 8k | **0.00%** | −0.35 | 89/100 | 0.47 |
| Qwen3.8-27B | 16k | 2.79% | −0.77 | 81/105 | 0.09 |
| Qwen3.8-27B | 24k | 26.16% | −0.51 | 83/99 | 0.27 |
| Gemma-4-31B (bf16) | 8k | **0.00%** | +0.06 | 42/40 | 0.91 |
| Gemma-4-31B (bf16) | 16k | 0.90% | +0.51 | 45/29 | 0.08 |
| Gemma-4-31B (bf16) | 24k | 2.79% | −0.22 | 41/48 | 0.53 |

The result is the third column rather than the fourth. At the tightest budget the summariser
never fires on either family on any of 3,116 tasks, because the allowance runs out before the
context fills. Compaction is a policy for long contexts, and a hard token budget is a
different constraint that binds first. No stratum is significant. Pooling all twelve Qwen
cells does reach $p = 0.034$, which we do not read as an effect: the 8k stratum is a placebo
by construction and carries the same sign, and comparing strata directly, the compaction arm
wins 47.1% of discordant pairs where it never fires against 45.6% where it fires on a quarter
of tasks (Fisher $p = 0.756$).

This is the sense in which bounded state and compaction are not competing solutions to one
problem. Compaction reduces a context after it has already been paid for once; bounded state
never lets it grow. Under a budget tight enough to bind, only the second is available.

## 7. Analysis

### 7.1 A poor classifier that routes well

The confidence gate is not a good predictor of correctness. Treated as a binary classifier on
per-item outcomes, $\phi$ reaches an AUC of 0.75 on MATH and FRAMES, 0.62 on GAIA, and 0.52 on
MuSiQue, which is chance. A reader is entitled to ask why a system built on it works.

The answer is that the gate is not asked to predict correctness. It is asked to place a task
on a rung. The cost of keeping a task at a cheap rung when it should have escalated is the
accuracy difference between the rungs, not the loss of the answer, because the cheap rung has
already produced one. The cost of escalating a task that did not need it is tokens. Both
errors are bounded, and neither returns nothing. A classifier at chance still routes usefully
when the penalty for misrouting is a bounded quality difference rather than a failure.

This also explains the design choice to escalate on missing log-probabilities. A deployment
that omits them collapses the system to the full pipeline, which is the ungoverned baseline,
rather than to a wrong answer.

### 7.2 Where the savings come from, and why one threshold is not enough

A fixed threshold routes very differently across benchmarks. Table 10 gives the share of items
that stop at the first rung.

**Table 10.** Share of items stopping at the cheapest rung, 24k.

| Family | MATH | GAIA | FRAMES | MuSiQue |
|---|---:|---:|---:|---:|
| Qwen3.8-27B | 76.7% | 36.1% | 46.2% | 30.2% |
| Gemma-4-31B (bf16) | 97.3% | 50.0% | 72.1% | 46.3% |
| GLM-5.3-Flash | 72.0% | 28.7% | 50.8% | 36.1% |
| DeepSeek-V4.1-Flash | 77.5% | 54.1% | 80.6% | 70.1% |

The spread is 28.7% to 97.3% under one fixed rule. This is the practical objection to
absolute thresholds, and it motivates specifying the operating point as a keep-rate quantile
instead: fixing the fraction of items retained at each rung makes the token saving a design
parameter chosen by the practitioner rather than an outcome that varies by benchmark.

Table 10 also decomposes where the savings originate. On MATH, where 72 to 97% of items stop
at the first rung, the ladder does the work and the loop is rarely built at all. On GAIA and
MuSiQue, where the share is often below half, the savings come from bounded state instead. The
two mechanisms are not redundant, and neither dominates across benchmarks.

### 7.3 When the accuracy gain appears

The 27 significant cells of 48 are not randomly distributed, and the pattern is interpretable.
Gains are largest where the budget binds hardest and where the task needs several rounds.
BigCodeBench at 4k, the tightest cell in the paper, produces the largest effects on every
family. At the loosest budgets on benchmarks that converge in one or two rounds, the governed
arm matches the baseline while spending a fraction of the tokens, which is the outcome we
expect when the constraint the method addresses is not active.

On reasoning models a second condition applies. Where reasoning consumes enough of the
generation allowance that answers are truncated, the rung is decisive; DeepSeek at 4k
BigCodeBench moves from 18% non-empty answers to 100%. Where the trace is short, prefix
governance alone is already the better trade and the rung costs a little accuracy. We regard
the ability to state this condition, and to test it with one measurement, as more useful than
a claim of uniform improvement.

## 8. Limitations

**Single seed.** All runs use temperature 0 and seed 42. Paired tests and a measured
run-to-run flip rate of 6.6% on GAIA and 5.5% on MATH substitute for multi-seed error bars,
which is weaker. Individual table entries should be read against that floor; the paired tests
are not affected, since they compare per item.

**The governed arm is not one configuration across families.** Four families require three
instantiations of the cheap rung: thinking disabled, `reasoning_effort: low`, and a
chat-template switch. Each is the cheapest rung that binds on that deployment, which is a
property of the deployment rather than a choice we controlled. The cross-family comparison is
therefore of a method rather than of a fixed configuration.

**The prefix-only ablation is not symmetric.** It exists on MATH and BigCodeBench for the two
reasoning families, and not on GAIA, FRAMES or MuSiQue.

**The gate's input is not identical across deployments.** On the tunnelled DeepSeek and on
GLM, log-probabilities cover the reasoning trace as well as the answer; on other providers
they covered the answer only. Arms are internally consistent, but $\phi$ is not the same
quantity everywhere.

**MuSiQue should be read on both metrics.** Exact match is flat while F1 rises by roughly 1.8
points on both reasoning families at two budgets. Reporting either alone misrepresents it.

**Accuracy gains do not generalise as the token savings do.** Twenty-seven of 48 main cells,
ten of twelve on BigCodeBench, and two of twelve on one family. The unconditional claim is the
token claim, and we have stated it that way throughout.

**No agentic software engineering.** SWE-bench Verified was run and dropped, because without
an agentic scaffold the pipeline patches from an oracle-localised context rather than
navigating a repository, which makes the budget question uninteresting there.

## 9. Conclusion

A multi-agent pipeline under a fixed token budget spends most of that budget re-reading state
it wrote earlier, and the largest single bindable component is the plan. Bounding what each
round may carry makes the per-round cost constant in the round index, which converts the
affordable round count into a function of the budget rather than of the loop's own verbosity.
Placing the loop behind a cost-ordered ladder, admitted by a confidence gate that reuses
log-probabilities the answering call already produced, avoids building the loop for tasks that
never needed it. On models that reason, the same admission principle has to be extended to the
reasoning setting, because there the growing term is generation rather than carried state.

Across 48 main cells and 12 execution-judged cells, the resulting system never spends more
than the baseline, saves 24 to 90%, eliminates budget-exhaustion truncation entirely, and
improves accuracy in rather more than half of the cells while degrading it significantly in
none. The saving is unconditional; the accuracy gain is not, and we have tried to be precise
about what it depends on. The condition is testable in advance from a single measurement of
how many rounds a task needs and how long the model's reasoning trace runs under the intended
budget, which we think is the more useful form for a result of this kind to take.

## Appendix A. Per-cell results

Full 48-cell table, governed arm against `Naive+Fmt`. Bold marks $p < 0.05$ in the governed
arm's favour.

| Family | Benchmark | B | n | Naive+Fmt SR | Governed SR | dSR | p | tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B | MATH | 24k | 400 | 57.2% | 68.0% | **+10.8** | 3.4e-08 | -52% |
| Qwen3.8-27B | MATH | 16k | 400 | 55.8% | 68.5% | **+12.8** | 4.3e-11 | -50% |
| Qwen3.8-27B | MATH | 8k | 400 | 49.5% | 68.5% | **+19.0** | 4.1e-17 | -42% |
| Qwen3.8-27B | GAIA | 24k | 122 | 9.8% | 21.3% | **+11.5** | 1.3e-03 | -54% |
| Qwen3.8-27B | GAIA | 16k | 122 | 11.5% | 19.7% | **+8.2** | 6.3e-03 | -48% |
| Qwen3.8-27B | GAIA | 8k | 122 | 7.4% | 21.3% | **+13.9** | 1.5e-05 | -24% |
| Qwen3.8-27B | FRAMES | 24k | 660 | 8.0% | 13.9% | **+5.9** | 8.1e-08 | -57% |
| Qwen3.8-27B | FRAMES | 16k | 660 | 7.4% | 13.3% | **+5.9** | 4.6e-07 | -55% |
| Qwen3.8-27B | FRAMES | 8k | 660 | 7.6% | 14.8% | **+7.3** | 5.6e-10 | -40% |
| Qwen3.8-27B | MuSiQue | 24k | 1934 | 5.3% | 8.7% | **+3.4** | 3.0e-09 | -42% |
| Qwen3.8-27B | MuSiQue | 16k | 1934 | 5.2% | 8.5% | **+3.4** | 2.4e-09 | -39% |
| Qwen3.8-27B | MuSiQue | 8k | 1934 | 4.9% | 7.9% | **+3.0** | 3.6e-08 | -27% |
| Gemma-4-31B (bf16) | MATH | 24k | 400 | 73.8% | 79.2% | **+5.5** | 3.0e-06 | -67% |
| Gemma-4-31B (bf16) | MATH | 16k | 400 | 76.8% | 79.5% | **+2.8** | 3.4e-03 | -66% |
| Gemma-4-31B (bf16) | MATH | 8k | 400 | 76.2% | 79.5% | **+3.2** | 9.8e-04 | -63% |
| Gemma-4-31B (bf16) | GAIA | 24k | 122 | 19.7% | 23.0% | +3.3 | 3.4e-01 | -51% |
| Gemma-4-31B (bf16) | GAIA | 16k | 122 | 23.0% | 21.3% | -1.6 | 7.7e-01 | -48% |
| Gemma-4-31B (bf16) | GAIA | 8k | 122 | 19.7% | 23.8% | +4.1 | 1.8e-01 | -37% |
| Gemma-4-31B (bf16) | FRAMES | 24k | 660 | 17.7% | 16.8% | -0.9 | 4.8e-01 | -63% |
| Gemma-4-31B (bf16) | FRAMES | 16k | 660 | 17.7% | 16.5% | -1.2 | 3.1e-01 | -61% |
| Gemma-4-31B (bf16) | FRAMES | 8k | 660 | 17.0% | 16.2% | -0.8 | 5.5e-01 | -51% |
| Gemma-4-31B (bf16) | MuSiQue | 24k | 1934 | 8.4% | 9.8% | **+1.4** | 9.3e-03 | -40% |
| Gemma-4-31B (bf16) | MuSiQue | 16k | 1934 | 8.1% | 9.6% | **+1.4** | 7.8e-03 | -41% |
| Gemma-4-31B (bf16) | MuSiQue | 8k | 1934 | 7.8% | 9.6% | **+1.9** | 5.3e-04 | -35% |
| GLM-5.3-Flash | MATH | 24k | 400 | 64.2% | 62.5% | -1.8 | 4.4e-01 | -56% |
| GLM-5.3-Flash | MATH | 16k | 400 | 64.5% | 62.0% | -2.5 | 2.2e-01 | -55% |
| GLM-5.3-Flash | MATH | 8k | 400 | 63.7% | 61.3% | -2.5 | 2.5e-01 | -56% |
| GLM-5.3-Flash | GAIA | 24k | 122 | 18.0% | 20.5% | +2.5 | 6.3e-01 | -48% |
| GLM-5.3-Flash | GAIA | 16k | 122 | 22.1% | 22.1% | +0.0 | 1.0e+00 | -51% |
| GLM-5.3-Flash | GAIA | 8k | 122 | 18.0% | 24.6% | +6.6 | 5.7e-02 | -39% |
| GLM-5.3-Flash | FRAMES | 24k | 660 | 25.5% | 28.0% | +2.6 | 7.8e-02 | -61% |
| GLM-5.3-Flash | FRAMES | 16k | 660 | 25.5% | 28.2% | +2.7 | 5.0e-02 | -65% |
| GLM-5.3-Flash | FRAMES | 8k | 660 | 25.2% | 25.5% | +0.3 | 9.1e-01 | -55% |
| GLM-5.3-Flash | MuSiQue | 24k | 1934 | 17.7% | 19.6% | **+1.9** | 7.3e-03 | -56% |
| GLM-5.3-Flash | MuSiQue | 16k | 1934 | 18.3% | 18.4% | +0.1 | 9.4e-01 | -58% |
| GLM-5.3-Flash | MuSiQue | 8k | 1934 | 16.8% | 18.4% | **+1.7** | 2.3e-02 | -49% |
| DeepSeek-V4.1-Flash | MATH | 24k | 400 | 57.8% | 63.2% | **+5.5** | 3.8e-03 | -54% |
| DeepSeek-V4.1-Flash | MATH | 16k | 400 | 55.8% | 61.0% | **+5.2** | 4.6e-03 | -54% |
| DeepSeek-V4.1-Flash | MATH | 8k | 400 | 56.2% | 62.7% | **+6.5** | 1.3e-03 | -47% |
| DeepSeek-V4.1-Flash | GAIA | 24k | 122 | 18.9% | 30.3% | **+11.5** | 1.3e-03 | -66% |
| DeepSeek-V4.1-Flash | GAIA | 16k | 122 | 18.9% | 30.3% | **+11.5** | 2.6e-03 | -68% |
| DeepSeek-V4.1-Flash | GAIA | 8k | 122 | 17.2% | 24.6% | **+7.4** | 2.2e-02 | -60% |
| DeepSeek-V4.1-Flash | FRAMES | 24k | 660 | 27.4% | 27.1% | -0.3 | 9.2e-01 | -82% |
| DeepSeek-V4.1-Flash | FRAMES | 16k | 660 | 27.4% | 26.8% | -0.6 | 7.5e-01 | -82% |
| DeepSeek-V4.1-Flash | FRAMES | 8k | 660 | 23.9% | 28.9% | **+5.0** | 1.3e-03 | -76% |
| DeepSeek-V4.1-Flash | MuSiQue | 24k | 1934 | 17.4% | 17.0% | -0.5 | 5.7e-01 | -81% |
| DeepSeek-V4.1-Flash | MuSiQue | 16k | 1934 | 17.6% | 17.2% | -0.5 | 5.7e-01 | -82% |
| DeepSeek-V4.1-Flash | MuSiQue | 8k | 1934 | 15.9% | 17.4% | +1.4 | 6.5e-02 | -76% |

## Appendix B. Reproduction

```bash
# one cell of the main table
env OPENAI_BASE_URL=<endpoint> VLLM_MODEL=<model> MODEL_TAG=<tag> \
    BENCHMARK=MATH MATH_N=500 BUDGETS=24000 SPLIT=test \
    CONFIG_ONLY=V5-Unconstrained,V5-Unc-Fmt,V5.2-ReasonLadder \
    OUT_TAG=_x SEEDS=42 MAX_WORKERS=64 TEMPERATURE=0 V5=1 \
    L01_THINKING_OFF=1 .venv/bin/python -u _budget_sweep_v2.py

# BigCodeBench requires offline execution judging
DUMP_RAW_ANSWER=1 ...
REJUDGE_GLOB='traces/tasks_sweep_<tag>_bcb*.jsonl' .venv/bin/python _bcb_rejudge.py

# recompute every number in this document
.venv/bin/python _gen_materials.py
```

Traces are not distributed with the code (927 MB). `_gen_materials.py` reads them from
`traces/` and recomputes every figure reported here, so no number in this document is
transcribed by hand.
