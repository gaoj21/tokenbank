# What Does a Multi-Agent Pipeline Buy Under a Token Budget?

*Analysis paper draft. Numbers are from completed runs unless marked **[PENDING: run]**, which
names the experiment that fills them (ETA 2026-10-02 evening). Every comparison is made at an
equal, stated per-call generation cap.*

---

## Abstract

Multi-agent LLM pipelines are routinely compared with single calls under a fixed token budget,
with each system summarised by one accuracy-cost point. We show that such comparisons are
dominated by a setting that is rarely reported: the per-call generation cap. Holding the
budget fixed, raising the cap from 1,024 to 4,096 tokens moves a plan-execute-review pipeline
from 57.2% to 75.0% on MATH, for a model with thinking disabled, because 31% of its items had
a call truncated. The effect is asymmetric by construction: a pipeline spends 46.6% of its
budget re-sending coordination state and needs longer calls than a single answer does, so a
cap that is harmless for one call silently cripples the other. Across three model families,
five benchmarks and three budgets, and with the cap controlled, a pipeline is significantly
better than a direct answer in only two of twelve cells, both on mathematical reasoning, at 6
to 26 times the tokens, and indistinguishable in nine. Compressing its carried state recovers
the tokens and, where the task needs the plan, gives back the accuracy. We also ask whether the cases where escalation helps can be identified in advance. They cannot
be read from the model's confidence, which predicts rescue poorly (median AUC 0.53); whether
extra reasoning helps, however, can be predicted from how often generation hits its cap. An audit of
21 budgeted-agent papers finds that 7 of the 9 comparing multi-agent systems do not state the
per-call cap of the systems they compare. We release our traces and propose reporting per-call
caps alongside budgets.

---

## 1. Introduction

Test-time compute has become a design variable. Under a fixed token allowance per task, a
system can answer in one call, sample several answers, reason longer, or run a pipeline in
which a planner decomposes the task, an executor answers and a reviewer critiques, for several
rounds. Pipelines of the last kind are now common, and they are increasingly evaluated under
explicit budgets, with each system reduced to a point in the plane of accuracy against tokens
spent. A pipeline that sits above and to the left of another is preferred.

This paper argues that the position of that point depends, to a degree that changes
conclusions, on a parameter that is usually set once in a configuration file and never
reported: the maximum number of tokens any single call may generate. A token budget limits the
sum over all calls. A per-call cap limits each call. The two interact differently with
different architectures. A single call spends its whole allowance on one generation and is
rarely near a modest cap. A pipeline spends a large share of its allowance re-sending
coordination state, and each of its calls must carry a plan step, a revision and often
explicit reasoning, so its calls are long. A cap that a single call never touches can truncate
a third of a pipeline's answers.

We measure this and what follows from it across three model families, one with thinking
disabled and two reasoning models, five benchmarks spanning mathematical reasoning, multi-hop
retrieval, assistant tasks and code, and three budgets. Our findings are five.

1. **Most of a pipeline's budget is re-sent state.** 46.6% of a pipeline's budget is
   coordination state written on an earlier round, against 30.5% that is the model producing
   anything new. The plan slot alone is 40.4% of every character the model reads. How that
   state is carried matters: an accumulating context grows by roughly 670 tokens per round
   and is stopped by an 8k budget on 70.7% of tasks; a context rebuilt each round is flat but
   large; a context with bounded slots is flat and 2.5 to 2.9 times smaller.
2. **The per-call cap is the largest single confound.** Raising it from 1,024 to 4,096 with the
   budget fixed moves the same pipeline from 57.2% to 75.0% on MATH for a model with thinking
   off, and from 18.9% to 32.8% on GAIA for a reasoning model. The effect is not specific to
   reasoning models.
3. **With the cap controlled, a pipeline rarely beats a direct answer.** It is significantly
   better in two of twelve cells, both mathematical reasoning, significantly worse in one, and
   indistinguishable in nine, at 6 to 26 times the tokens. **[PENDING: single-call 4k baseline.]**
4. **Compressing a pipeline's state trades its accuracy back for tokens.** Bounding the carried
   slots recovers most of the tokens; on mathematical reasoning and assistant tasks it also
   gives back most of the accuracy, while on multi-hop retrieval it does not.
5. **The model's confidence does not say where escalation helps.** Of items sent through every
   stage of the pipeline at an adequate cap, 1.6 to 8.0% are rescued and 4.7 to 14.8% are
   harmed, and the answer's log-probability predicts rescue poorly (AUC 0.33 to 0.67). Whether extra reasoning helps
   is predictable, from the share of calls that hit the cap without it.

An audit of 21 recent papers that evaluate agents under a token or cost criterion finds that 7
of the 9 comparing multi-agent systems do not state the per-call cap of the systems they
compare, and only one fixes it as a shared protocol. We close with reporting recommendations.

**Contributions.** (i) A controlled measurement of what a multi-agent pipeline spends and buys
under a hard token budget, over three families, five benchmarks and three budgets, with every
comparison at a stated, equal per-call cap. (ii) Identification of the per-call cap as a
confound large enough to reverse conclusions, and an account of why it is asymmetric between
pipelines and single calls. (iii) A per-item counterfactual, in which every item is run through
every stage, that measures rescue and harm directly rather than inferring them. (iv) A
criterion, measurable before deployment, for whether additional reasoning will help. (v) A
literature audit and a reporting standard.

---

## 2. Related Work

**Scaling and controlling test-time compute.** A large body of work shows that more inference
compute, through sampling, search, revision or longer reasoning, buys accuracy, and a second
body controls how much is spent: budget forcing, length-controlled policy optimisation, control
tokens (BudgetThinker), and decomposition with per-step budgets (Plan-and-Budget). These works
treat generation length as the object of study and, in our audit, uniformly report and equalise
it. Our point is that work treating the *architecture* as the object of study often does not,
and that the omission matters.

**Multi-agent pipelines and their efficiency.** Plan-execute-review and debate-style systems
improve over single calls on many tasks. Efficiency methods prune agents or messages
(AgentPrune, AgentDropout), train for concise communication (Optima, Agent-GSPO), learn routing
and memory gates under a cost penalty (Gated-Memory Routing), structure systems under a budget
(BAMAS) or train a controller to respect one (CoRL). Total Cost of Agency decomposes workflow
cost and finds injected memory growing linearly with depth. Rethinking the Evaluation of
Efficiency Methods for MAS argues that reported gains often come from method-specific setups,
and is the one paper in our audit that fixes the per-call cap (2,048) as part of a shared
protocol. We share its concern and isolate one specific setup variable whose effect we
quantify.

**Bounded and compressed agent state.** The Markovian Thinker bounds the state a reasoning
policy reads to make long reasoning linear in cost; MEM1 and Context-Folding train agents to
consolidate or fold their context; ACON optimises compression guidelines. These are training
methods; we measure, without training, what bounding a pipeline's carried state costs and buys.

**Cascades and routing.** Model cascades defer from a cheap model to an expensive one on
low-confidence inputs. Signed Rescue Routing observes that uncertainty is the wrong target:
escalation helps only when the expensive model repairs an error, and hurts when it overturns a
correct answer. We find the same in a different setting, one model at increasing levels of
scaffolding, with free-form outputs and heterogeneous per-item costs, and measure rescue and
harm directly through a full counterfactual.

**Budget-aware agents.** BATS gives an agent a running view of its remaining tool budget;
Budget-Aware Agency trains agents to estimate remaining cost and stop early; BAGEN asks whether
agents are budget-aware at all. These concern agent behaviour under a budget. We concern the
evaluation of agents under a budget.

---

## 3. Experimental Framework

### 3.1 Systems compared

All systems share one harness, one prompt set and one judge per benchmark. They differ only in
how a task is turned into calls.

**Single call.** One Executor call with the instruction to end with a line `ANSWER: x`, no
plan, no tool, no reviewer, the model's default reasoning setting.

**Plain pipeline.** A Planner writes a step-by-step plan. An Executor answers, and may emit a
Python block that the harness runs with a ten-second timeout and returns as a new turn. A
Reviewer judges the answer. Executor and Reviewer alternate for up to five rounds. At each round
boundary the context is *rebuilt*: system prompt, question with the full plan, the last answer
and the last critique, each at whatever length the previous round produced. The Executor carries
the same `ANSWER: x` instruction as the single call.

**Accumulating pipeline.** As the plain pipeline, but each round's answer and critique are
appended to the previous context instead of replacing it. This is the regime most agent
scaffolds implement.

**Compressed pipeline.** As the plain pipeline, but every carried slot is capped: after the
first round the plan is reduced to the current step line, the Reviewer emits a JSON verdict of
at most 150 tokens, and tool output is replaced by a note of at most 120 tokens. The compressed
pipeline is reached through a cost-ordered cascade: a direct answer (L0), then a tool-assisted
answer seeded with L0's (L1), then the compressed loop (L2). A task stops at the first level
whose answer passes a confidence gate on the answer's own mean log-probability and token
entropy. On reasoning models the two cheap levels use a cheaper reasoning setting
(`reasoning_effort: low` on GLM; thinking disabled through the chat template on DeepSeek).

**Full counterfactual.** The compressed pipeline with gates that never pass, so that every item
runs L0, L1 and L2 and its trace records the outcome at every level. This is an instrument, not
a system: it gives, per item, what each level would have produced, from which rescue, harm and
any gating policy can be computed exactly.

### 3.2 Budget and per-call cap

A budget $B$ is a hard cap on the sum of prompt and completion tokens over all calls of all
roles for one task. It is enforced by admission: before each call the harness estimates its
cost as the tokenised prompt plus the requested generation cap and refuses the call if it does
not fit. A refused call ends the task with whatever answer the system holds.

The **per-call cap** $m$ is the generation limit passed with each call. It is distinct from $B$:
$B$ bounds the total, $m$ bounds each call. We run every system at $m = 4{,}096$ unless the
experiment varies $m$, in which case both values are stated. The cascade's cheap levels use
$m = 2{,}048$, which is part of that system's definition.

### 3.3 Models, benchmarks, budgets

| Model | Serving | Thinking |
|---|---|---|
| Qwen3.8-27B (FP8) | vLLM, 8 x H20, one replica per GPU | disabled |
| GLM-5.3-Flash (FP8) | SGLang, tensor-parallel 8 | cannot be disabled |
| DeepSeek-V4.1-Flash | vLLM, 8 x H20, tensor and expert parallel | default on, disableable per call |

Gemma-4-31B (bf16) appears only in the budget accounting of Section 4, which does not depend on
the per-call cap.

| Benchmark | Items | Task | Judge |
|---|---:|---|---|
| MATH-500 | 400 | competition mathematics | symbolic equivalence |
| GAIA | 122 | multi-step assistant tasks | normalised exact match |
| FRAMES | 660 | multi-hop factual reasoning | exact match and F1 |
| MuSiQue | 1,934 | 2 to 4 hop compositional QA | exact match and F1 |
| BigCodeBench | 500 | library-heavy code synthesis | unit-test execution |

Budgets are 8k, 16k and 24k tokens, and 4k, 8k and 16k on BigCodeBench, whose prompts are long
enough that the looser end does not bind. Decoding is at temperature 0 with one seed.

### 3.4 Measurement instruments

**Segment accounting.** Every prompt character is attributed to the slot it occupies: plan,
question, draft answer, critique, tool output, or system prompt.

**Truncation.** A call is counted as truncated when its completion reaches the per-call cap.

**Rescue and harm.** From the full counterfactual, an item is *rescued* if L0 is wrong and L2 is
right, and *harmed* if L0 is right and L2 is wrong.

**Statistics.** Paired comparisons use the exact McNemar test on per-item outcomes. With a
single seed, we substitute paired tests and a measured run-to-run flip rate (6.6% of items on
GAIA, 5.5% on MATH) for multi-seed intervals.

---

## 4. Where a Pipeline's Budget Goes

We first ask what a pipeline spends its budget on, independently of what that spending buys.

**Most of the budget is re-sent state.** Table 1 attributes every prompt character of the plain
pipeline to the slot it occupies, pooled over five benchmarks at a 16k budget.

**Table 1.** Share of prompt characters by slot, plain pipeline, 16k, pooled over five benchmarks.

| Model | plan | question | draft | critique | tool | system |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B | **40.4%** | 27.3% | 24.4% | 1.9% | 0.4% | 4.4% |
| Gemma-4-31B | **28.7%** | 32.2% | 19.8% | 10.0% | 0.5% | 7.3% |

Counted against the whole budget, 30.5% of a Qwen task's tokens are the model generating, 19.0%
are the question, and 46.6% are coordination state written on an earlier round and re-sent on
this one. The plan, written once and read every round, is the largest single component on both
families.

**How the state is carried determines how fast it grows.** Table 2 measures the executor's
prompt at each round under three ways of building the context, at an equal per-call cap.

**Table 2.** Executor prompt tokens per round of the loop, 24k, pooled over four benchmarks.

| Model | Context | $t=1$ | $t=2$ | $t=3$ | $t=4$ | $t=5$ | per round |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B | accumulating | 966 | 1,496 | 2,461 | 3,217 | 3,435 | **+666** |
| Qwen3.8-27B | rebuilt | 975 | 1,388 | 1,633 | 1,607 | 1,589 | flat |
| Qwen3.8-27B | bounded | 2,137 | 625 | 678 | 605 | 547 | flat |
| Gemma-4-31B | accumulating | 436 | 1,141 | 1,882 | 2,572 | 3,082 | **+672** |
| Gemma-4-31B | rebuilt | 433 | 1,031 | 1,058 | 1,055 | 1,108 | flat |
| Gemma-4-31B | bounded | 599 | 354 | 367 | 386 | 386 | flat |

Accumulation adds about 670 tokens per round on both families, so the total cost of $R$ rounds
is quadratic in $R$ and the affordable round count grows only with $\sqrt{B}$. Rebuilding removes
the growth but its plateau is set by how much the previous round wrote. Bounding every slot
lowers the plateau by a factor of 2.5 (Qwen) and 2.9 (Gemma). At an 8k budget the three regimes
are stopped by the budget on 70.7%, 65.8% and 0.0% of Qwen tasks.

---

## 5. The Per-Call Cap

**Raising the cap changes the pipeline's accuracy at a fixed budget.** Table 3 runs the plain
pipeline at a 24k budget with the per-call cap at 1,024 and at 4,096 tokens. Nothing else
changes.

**Table 3.** Plain pipeline, 24k budget, per-call cap 1,024 vs 4,096. Paired exact McNemar.
"Truncated" is the share of items with at least one call reaching the 1,024 cap.

| Model | Benchmark | cap 1,024 | cap 4,096 | Δ | p | truncated at 1,024 |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.8-27B | MATH | 57.2 | 75.0 | **+17.8** | 8e-16 | 31.0% |
| | GAIA | 9.8 | 17.2 | **+7.4** | 0.035 | 80.3% |
| | FRAMES | 8.0 | 12.0 | **+3.9** | 2e-4 | 68.0% |
| | MuSiQue | 5.3 | 7.4 | **+2.2** | 1e-4 | 69.6% |
| GLM-5.3-Flash | MATH | 64.2 | 69.2 | **+5.0** | 0.006 | 15.2% |
| | GAIA | 18.0 | 27.0 | **+9.0** | 0.007 | 78.7% |
| | FRAMES | 25.5 | 28.8 | **+3.3** | 0.008 | 37.0% |
| | MuSiQue | 17.7 | 18.7 | +1.0 | 0.11 | 41.4% |
| DeepSeek-V4.1-Flash | MATH | 57.8 | 62.0 | **+4.2** | 0.014 | 14.5% |
| | GAIA | 18.9 | 32.8 | **+13.9** | 8e-5 | 84.4% |
| | FRAMES | 27.4 | 28.9 | +1.5 | 0.32 | 49.4% |
| | MuSiQue | 17.4 | 17.3 | −0.1 | 0.94 | 58.3% |

The cap moves accuracy significantly in ten of twelve cells, by up to 17.8 points, and never
significantly in the other direction. It is not a property of reasoning models: the largest
effect is on Qwen, which runs with thinking disabled. **[PENDING: Qwen deployment control, 1,024
on the H20 deployment, to rule out the change of serving hardware for Qwen's cap-1,024 cell.]**

**Why the cap is asymmetric.** A single call answers once; its generation is whatever the answer
needs. A pipeline's calls each carry a plan step, a revision against a critique, and, on
reasoning models, a reasoning trace, and the pipeline issues many of them, so each is longer and
there are more chances for one to hit the cap. The same cap therefore truncates a far larger
share of a pipeline's work than of a single call's. On DeepSeek, the single call loses 5.8 points
on GAIA and 5.2 on FRAMES when its own cap falls from 2,048 to 1,024, a smaller absolute effect
than the 13.9 the pipeline loses on GAIA over a larger cap change. **[PENDING: single-call 4k, to
state the single call's own cap sensitivity on equal terms.]**

The practical consequence is that a pipeline-versus-single-call comparison run at a small
per-call cap measures, in part, how often each is truncated.

---

## 6. What the Pipeline Buys

With the cap controlled, we compare the plain pipeline against a direct answer.

**Table 4.** Plain pipeline (cap 4,096) against a direct answer, 24k. The direct answer is the
cascade's first level: one call (plus one tool round if the model emits code), cap 2,048, with
the cheap reasoning setting on reasoning models. Accuracy % / mean tokens. Paired exact McNemar.

| Model | Benchmark | direct answer | pipeline | Δ | p | token ratio |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.8-27B | MATH | 67.2 / 1.2k | 75.0 / 7.1k | **+7.8** | 2e-4 | 6x |
| | GAIA | 22.1 / 1.8k | 17.2 / 17.4k | **−4.9** | 0.03 | 10x |
| | FRAMES | 11.7 / 1.4k | 12.0 / 14.6k | +0.3 | 0.9 | 10x |
| | MuSiQue | 8.4 / 1.4k | 7.4 / 15.1k | −1.0 | 0.1 | 11x |
| GLM-5.3-Flash | MATH | 63.0 / 0.3k | 69.2 / 5.1k | **+6.2** | 1e-3 | 17x |
| | GAIA | 26.2 / 0.6k | 27.0 / 15.6k | +0.8 | 1.0 | 26x |
| | FRAMES | 28.2 / 0.3k | 28.8 / 6.8k | +0.6 | 0.7 | 23x |
| | MuSiQue | 18.4 / 0.3k | 18.7 / 7.0k | +0.3 | 0.7 | 23x |
| DeepSeek-V4.1-Flash | MATH | 63.2 / 0.6k | 62.0 / 3.5k | −1.2 | 0.6 | 6x |
| | GAIA | 27.0 / 0.9k | 32.8 / 13.6k | +5.7 | 0.2 | 15x |
| | FRAMES | 27.3 / 0.4k | 28.9 / 6.4k | +1.7 | 0.3 | 16x |
| | MuSiQue | 16.8 / 0.4k | 17.3 / 6.9k | +0.5 | 0.5 | 17x |

**The pipeline is significantly better than a direct answer in two of twelve cells, both
mathematical reasoning, and significantly worse in one, at 6 to 26 times the tokens.** In the
remaining nine cells the two are indistinguishable. The two gains are on the two families whose
direct answer does not already reason (Qwen with thinking off; GLM at low effort); on DeepSeek the
direct answer, with thinking disabled, already matches the pipeline on MATH. Point estimates on
GAIA favour the pipeline on two families, but with 122 items the test has little power.
**[PENDING: single-call 4k, which replaces the direct answer with a plain single call at the
pipeline's own cap and default reasoning, the cleanest single-call baseline.]**

---

## 7. What Compressing the State Costs

Bounding the carried slots is the obvious response to Section 4. Table 5 asks what it costs, at
the same 4,096 cap.

**Table 5.** Plain pipeline against the compressed pipeline, 24k, cap 4,096. "Always loop" sends
every item through the compressed loop (seeded with the direct answer and the tool-assisted
answer). "Gated" replays the confidence gate on the same per-item records, so that only items the
gate rejects reach the loop.

| Model | Benchmark | plain | compressed, always loop | compressed, gated | gated reaches loop |
|---|---|---:|---:|---:|---:|
| GLM-5.3-Flash | MATH | 69.2 / 5.1k | 59.8 / 12.1k | 64.8 / 1.4k | 7.2% |
| | GAIA | 27.0 / 15.6k | 13.3 / 19.3k | 23.3 / 8.8k | 43.3% |
| | FRAMES | 28.8 / 6.8k | 27.0 / 12.3k | 28.3 / 2.9k | 17.9% |
| | MuSiQue | 18.7 / 7.0k | 17.6 / 10.8k | 19.6 / 3.5k | 24.2% |
| DeepSeek-V4.1-Flash | MATH | 62.0 / 3.5k | 60.0 / 10.2k | 64.5 / 1.1k | 2.2% |
| | GAIA | 32.8 / 13.6k | 21.5 / 19.8k | 28.1 / 4.2k | 14.0% |
| | FRAMES | 28.9 / 6.4k | 30.2 / 13.6k | 27.4 / 0.8k | 1.1% |
| | MuSiQue | 17.3 / 6.9k | 18.7 / 14.1k | 17.7 / 1.3k | 3.7% |

**[PENDING: Qwen rows, from the full counterfactual at cap 4,096.]**

Two observations. First, the compressed loop is worse than the plain loop where the task needs
the plan: on GLM MATH it scores 59.8 against 69.2, on GAIA 13.3 against 27.0 (GLM) and 21.5
against 32.8 (DeepSeek). On multi-hop retrieval it is not worse. Reducing the plan to one line
and the critique to 150 tokens removes what a mathematical or assistant task needs to revise
productively. Second, the gated system mostly avoids the loop, so it inherits the direct
answer's accuracy at a fraction of the plain pipeline's tokens: 1.1k against 3.5k on DeepSeek
MATH at +2.5 points, 4.2k against 13.6k on DeepSeek GAIA at −4.7 points. It is a cheap point on
the frontier, not one that dominates the plain pipeline. Token counts for gated early stops omit
the reviewer and tool-note calls of the levels not reached and slightly understate cost.

---

## 8. Can You Tell in Advance Where Extra Compute Helps?

**Rescue is rare and harm is common.** From the full counterfactual at cap 4,096, Table 6 gives,
per item, how often the compressed loop repairs a wrong direct answer and how often it overturns
a right one.

**Table 6.** Rescue and harm of the loop over the direct answer, and the ceiling of a perfect
router (stop at the cheapest level that is correct), 24k, cap 4,096. AUC is the direct answer's
mean log-probability used as a score for the event.

| Model | Benchmark | rescue | harm | direct | perfect router | AUC (correct) | AUC (rescue) |
|---|---|---:|---:|---:|---:|---:|---:|
| GLM-5.3-Flash | MATH | 5.0 | 8.2 | 63.0 | 69.5 | 0.61 | 0.67 |
| | GAIA | 1.6 | 14.8 | 26.7 | 30.0 | 0.71 | 0.33 |
| | FRAMES | 4.8 | 6.1 | 28.2 | 33.6 | 0.59 | 0.52 |
| | MuSiQue | 4.6 | 5.4 | 18.4 | 23.5 | 0.66 | 0.49 |
| DeepSeek-V4.1-Flash | MATH | 2.8 | 6.0 | 63.2 | 67.2 | 0.57 | 0.58 |
| | GAIA | 4.1 | 9.8 | 27.3 | 32.2 | 0.65 | 0.57 |
| | FRAMES | 8.0 | 5.2 | 27.3 | 35.9 | 0.59 | 0.58 |
| | MuSiQue | 6.6 | 4.7 | 16.8 | 24.3 | 0.69 | 0.48 |

**[PENDING: Qwen rows.]**

Harm exceeds rescue in six of eight cells. A perfect router would gain 3.3 to 8.6 points over the
direct answer, which bounds what any escalation policy can buy here. The model's confidence
predicts whether its direct answer is correct moderately (AUC 0.57 to 0.71) but predicts rescue
poorly (0.33 to 0.67, median 0.53): the items a confident-looking gate escalates are mostly items
the loop cannot repair either. This is the observation Signed Rescue Routing makes for two-model
cascades on multiple-choice tasks; it holds as well for one model at increasing levels of
scaffolding with free-form answers.

**Whether extra reasoning helps is predictable.** On reasoning models the cascade gives its cheap
levels a cheaper reasoning setting. Across twelve cells on the two reasoning families, the effect
of that setting against leaving reasoning untouched has opposite signs: positive on DeepSeek and
on code, negative on GLM MATH. Table 7 relates it to a quantity measured without the setting.

**Table 7.** Effect of the cheap reasoning setting against none, and the share of items whose
generation reaches the cap in some call without it (measured at the cap in use).

| Model | Benchmark | B | cap-hit share | ΔSR |
|---|---|---|---:|---:|
| GLM | MATH | 24k | 16% | −2.5 |
| GLM | MATH | 16k | 14% | −5.2 |
| GLM | MATH | 8k | 14% | −0.8 |
| DeepSeek | MATH | 8k | 18% | **+21.0** |
| DeepSeek | MATH | 24k | 21% | **+8.0** |
| DeepSeek | MATH | 16k | 24% | **+6.0** |
| GLM | BigCodeBench | 8k | 27% | **+4.0** |
| GLM | BigCodeBench | 16k | 29% | **+5.2** |
| DeepSeek | BigCodeBench | 8k | 53% | **+33.6** |
| GLM | BigCodeBench | 4k | 58% | **+14.8** |
| DeepSeek | BigCodeBench | 4k | 66% | **+42.2** |
| DeepSeek | BigCodeBench | 16k | 70% | **+27.4** |

The cap-hit share has a rank correlation of 0.78 with the effect ($p = 0.003$) and separates its
sign: every cell at 18% or above improves, nine of nine significantly; the three at or below 16%
do not. GLM crosses the threshold within one family. The reading is direct: the cheaper setting
buys answers where reasoning would otherwise exhaust the call. Because the quantity is measured at
the cap in use, the criterion is cap-relative, which is consistent with Section 5. Twelve cells on
two families locate a threshold rather than establish one, and the three negative cells share a
family and a benchmark.

---

## 9. How the Literature Reports the Cap

We audited 21 recent papers that evaluate agent or multi-agent systems under a token or cost
criterion, drawn from the works cited above. For each we extracted the full text, searched it for
every statement of a generation limit, and read every match and every statement of decoding
settings by hand.

**Table 8.** Does the paper state the per-call generation cap used at evaluation (Q1), and that
compared systems share it (Q2)?

| | all (21) | multi-agent (9) |
|---|---:|---:|
| Q1: cap stated | 8 | 2 |
| of which for the authors' own system only | 1 | 1 |
| training-time cap only | 2 | 1 |
| no cap stated | 11 | 6 |
| Q2: compared systems stated to share it | 7 | 1 |

Seven of the nine papers comparing multi-agent systems do not state the evaluation cap of the
systems they compare. One sets per-role caps of 384 to 1,024 tokens as part of its own system's
configuration without stating the baselines'. One fixes a shared cap of 2,048 as part of an
evaluation protocol. Papers whose object of study is reasoning length (budget forcing,
length-controlled training, per-step budgets) all report and equalise the cap. The pattern is
that communities which study generation length control it and communities which study
architecture mostly do not. The audit is a convenience sample, and "not stated" means not stated
in the paper; released code may fix a cap that the paper does not report. The full table with
line-level evidence is in Appendix C.

---

## 10. Recommendations

1. **Report the per-call generation cap with every budget**, for every compared system, and
   compare systems at an equal cap unless the cap is the variable under study.
2. **Report the truncation rate.** The share of calls that reach the cap is cheap to compute and
   is the quantity that determines whether a comparison is about the systems or about their caps.
3. **Include a direct-answer baseline at the same cap.** In nine of twelve of our cells it is
   indistinguishable from a five-round pipeline at a fraction of the tokens.
4. **State how the context is built across rounds.** Accumulating, rebuilt and bounded contexts
   differ by a factor of several in per-round cost and in how often the budget stops the task.
5. **Do not use the model's confidence as a proxy for where extra compute helps.** It predicts
   correctness moderately and rescue poorly.
6. **Before enabling a cheaper reasoning setting, measure the cap-hit share.** It predicted the
   sign of the setting's effect in every cell we tested.

---

## Limitations

**Single seed.** All runs are at temperature 0 with one seed. Paired tests are robust to this, and
we report a measured flip rate (5.5% on MATH, 6.6% on GAIA), but individual percentages carry
that noise.

**Three families.** One runs with thinking disabled and two are reasoning models. Gemma is used
only for the budget accounting of Section 4, which does not depend on the per-call cap; it was
not re-run at the larger cap.

**Serving changes.** Qwen's cap-4,096 runs were served on a different deployment from its
cap-1,024 runs, with identical weights; a control at cap 1,024 on the new deployment isolates the
cap from the deployment. **[PENDING: result of that control.]**

**Two cap values.** We compare 1,024 and 4,096. The relationship between cap and accuracy is not
characterised between or beyond them.

**GAIA is small.** With 122 items, several GAIA differences of four to six points are not
significant.

**One pipeline.** We study one plan-execute-review pipeline. Other topologies may distribute
generation across calls differently and be more or less sensitive to the cap.

**Audit scope.** A convenience sample of 21 papers, not a systematic review.

---

## Conclusion

Under a token budget, a multi-agent pipeline spends nearly half of its allowance re-sending its
own state, and the per-call generation cap, a setting seldom reported, moves its accuracy by up to
18 points with the budget unchanged. Once that cap is controlled, a pipeline buys accuracy over a
direct answer on mathematical reasoning and little elsewhere, at an order of magnitude more
tokens; compressing its state recovers the tokens and gives back the accuracy; and the model's
confidence does not identify the cases where extra compute helps, although the cap-hit share
predicts when extra reasoning will. Most papers that compare multi-agent systems under a budget do
not report the cap. We propose that they should.

---

## Appendix A. Further measurements

- **Format instruction.** A single instruction to end with `ANSWER: x` is worth +13.2 points
  (Qwen) and +28.0 (Gemma) on MATH at 24k, cap 1,024. Every system in this paper carries it.
- **Compaction.** Accumulate-then-summarise with a fixed 8,192-token window never triggers on any
  of 3,116 tasks at an 8k budget on either Qwen or Gemma: the budget runs out before the context
  fills. No stratum shows a significant difference against accumulation.
- **Prefix-only governance on a reasoning model.** Bounding the carried slots without changing
  the reasoning setting saved 2% of tokens on GLM GAIA at 24k (10,998 against 11,229), against 24
  to 67% on models with thinking off.

## Appendix B. Runs filling the [PENDING] markers

| Marker | Run | Chain |
|---|---|---|
| single-call 4k (Sections 5, 6, abstract) | `V5-Single-Fmt-4k`, `BCB-Single-4k`, three families | `_chain_single4k.sh` |
| Qwen deployment control (Section 5, Limitations) | `_qx1k_ma24` | `_chain_analysis_qwen2.sh` |
| Qwen rows of Tables 5 and 6 | `_qx4ka_*24` | `_chain_analysis_qwen2.sh` |
| 8k and 16k cells, BigCodeBench at cap 4,096 | `_qr4k_*`, `_glmr4k_*`, `_dsr4k_*` | `_chain_rerun4k.sh` |

## Appendix C. Audit table

See `results/lit_audit/audit.md`: 21 rows, each with the quoted line and its location in the
extracted text.
