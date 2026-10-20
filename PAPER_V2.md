# The Price and the Value of Coordination: What a Multi-Agent Pipeline Buys Under a Token Budget

*Draft v2, 2026-10-02. Two reasoning models (GLM-5.3-Flash, DeepSeek-V4.1-Flash). Every number is
from a completed run and names the script that recomputes it. **[PENDING]** marks results of the
state-policy experiment (ETA: GLM tonight, DeepSeek early tomorrow).*

---

## Abstract

Multi-agent pipelines are evaluated against single calls under a fixed token budget, but the
comparison conflates what the pipeline pays with what it buys. We separate the two with three
controlled measurements on two reasoning models across four benchmarks. **(1) A cap
intervention.** With the budget fixed, choosing the per-call generation caps of the two systems
alone moves the pipeline-minus-single-call difference by up to 23.8 points and flips its sign in
6 of 8 cells; at equal caps the comparison is stable in 6 of 8. Both architectures are
cap-sensitive, so the confound is whether they are truncated equally, and 7 of 9 recent
multi-agent comparisons in our audit do not report the cap. **(2) A role-level cost
decomposition.** Planner and reviewer calls take 52 to 58% of the budget. The plan is the largest
slot in every prompt, 41 to 49% of prompt characters, yet prompts are a small share of tokens:
on reasoning models the price of coordination is mostly the coordinators' own reasoning, not the
state they re-read. **(3) The interaction between carried state and the value of revision.**
Whether another round helps depends on what the state preserves. Carrying the full plan and
critique, revisions repair more answers than they break; compressing them to fixed slots reverses
the sign. **[PENDING: with the stop rule held fixed, how much of the reversal is information loss
and how much is over-revision.]** At an equal cap, these account for where a pipeline is worth its
price: it beats a single call by 4 to 8 points on mathematical reasoning and loses by up to 3.6
on multi-hop retrieval, at four to seven times the tokens.

---

## 1. Introduction

Under a fixed token allowance per task, a system may answer once, reason longer, or run a
pipeline in which a planner decomposes the task, an executor answers and a reviewer critiques,
over several rounds. Such pipelines are now routinely compared with single calls under explicit
budgets, each reduced to one point of accuracy against tokens.

That point conflates two quantities. One is the **price** of coordination: what the pipeline
spends that a single call does not, on planner and reviewer calls and on state carried between
rounds. The other is its **value**: what those extra calls and that state buy. A pipeline can sit
above a single call because coordination is valuable, or because the comparison happened to
truncate one system more than the other; it can sit below because coordination is worthless, or
because its state is too compressed to revise from. Telling these apart requires measuring price
and value separately, under conditions that do not themselves decide the outcome.

We do so with three measurements.

**A cap intervention.** A token budget limits the sum over calls; a per-call generation cap limits
each call. On reasoning models, where any call can be consumed by its own reasoning, the cap
changes both a pipeline's and a single call's accuracy by several points. We show that choosing
the two systems' caps independently is enough to make the same pipeline look 9 points worse or
15 points better than the same single call on the same items, and that most papers comparing
multi-agent systems do not report it.

**A role-level cost decomposition.** We attribute every token to the role that spent it and every
prompt character to the slot it came from. Coordination roles take more than half of the budget.
The re-sent plan dominates every prompt, but prompts are a small share of tokens. On reasoning
models, coordination is expensive because coordinators reason, not mainly because they remember.

**State and the value of revision.** Judging the answer after every round, we count, for each
consecutive pair of rounds, whether the revision repaired a wrong answer or broke a right one.
With the full plan and critique carried forward, revisions are net positive. With both compressed
to fixed-size slots, the most aggressive way to cut the state's price, they are net negative.
Holding the stop rule fixed, we separate how much of this is information the slots discard and
how much is revision the stop rule permits. **[PENDING]**

At an equal cap, the result is task-dependent and interpretable: the pipeline buys accuracy on
mathematical reasoning, where a full plan and critique can be revised against, and loses on
multi-hop retrieval, where the single call's direct reasoning already does what revision would.

**Contributions.**

1. **A cap intervention that isolates truncation from architecture.** With the budget fixed, the
   systems' per-call caps alone swing the pipeline-minus-single-call difference by up to 23.8
   points and flip its sign in 6 of 8 cells; at equal caps it is stable in 6 of 8. An audit of 21
   budgeted-agent papers finds the cap unstated in 7 of the 9 that compare multi-agent systems.
2. **A role-level decomposition of a pipeline's budget.** Planner and reviewer take 52 to 58% of
   tokens; the plan is 41 to 49% of prompt characters; prompts are a small share of tokens.
3. **A measurement of how carried state changes the value of an extra round**, from per-round
   judged answers under five state policies and two stop rules.

---

## 2. Related Work

**Controlling test-time compute.** Budget forcing, length-controlled training (LCPO),
control-token methods (BudgetThinker) and per-step budgets (Plan-and-Budget) study how long a
model should reason. They treat generation length as the object of study and, in our audit,
report and equalise it. We show that work treating the architecture as the object of study mostly
does not, and that this matters.

**Multi-agent efficiency.** AgentPrune and AgentDropout prune agents and messages; Optima and
Agent-GSPO train for concise communication; Gated-Memory Routing learns memory gates under a cost
penalty; BAMAS structures systems under a budget; CoRL trains a budget-respecting controller.
Total Cost of Agency decomposes workflow cost into base prompt, inference, injected memory and
accumulation. Rethinking the Evaluation of Efficiency Methods for MAS argues that reported gains
often come from method-specific setups, and fixes a per-call cap of 2,048 as part of its protocol;
we isolate and quantify that one variable. Our role decomposition differs from Total Cost of
Agency in attributing cost to coordination roles and prompt slots and in finding, on reasoning
models, that coordinators' generation rather than injected state dominates.

**Bounded and compressed agent state.** The Markovian Thinker bounds the state a reasoning policy
reads to make long reasoning linear in cost; MEM1, Context-Folding and ACON train or optimise
agents to consolidate their context. These change what state costs. We measure, without training,
what the choice of state policy does to the value of the next revision.

**Cascades and the value of escalation.** Signed Rescue Routing shows that escalating on
uncertainty targets the wrong quantity, because extra compute helps only when it repairs an error
and hurts when it overturns a correct answer. Our repair and breakage counts are the same
quantity, measured per round inside a pipeline rather than across a model pair.

---

## 3. Setup

### 3.1 Systems

All systems share one harness, one prompt set and one judge per benchmark.

**Single call.** One executor call instructed to end with `ANSWER: x`; no plan, no tool, no
reviewer; default reasoning.

**Pipeline.** A planner writes a step-by-step plan. An executor answers, optionally emitting
Python that the harness runs and returns. A reviewer judges the answer. Executor and reviewer
alternate for up to five rounds; the loop stops when the reviewer accepts or at round five. What
the executor sees in each round is set by the **state policy**:

| State policy | Carried into the next round |
|---|---|
| Rebuild | system, question with the full plan, the last answer, the last critique |
| Accumulate | everything from every previous round |
| Window-2 | system, question with plan, the last two answer-critique pairs |
| Summary | system, question with plan, a summary of at most 256 tokens of all previous rounds, written by one extra call that is charged to the budget |
| Slots | system, question with only the current plan step, the last answer, a JSON critique of at most 150 tokens, tool output as a note of at most 120 tokens |

Rebuild is the default pipeline in all comparisons with the single call.

**Stop rules.** *Accept*: stop when the reviewer accepts. *Strict*: stop only when the answer is
unchanged from the previous round or the reviewer accepts, and its mean log-probability exceeds
−0.05 and its token entropy is at most 0.6. The state-policy comparison uses Accept throughout;
Strict is applied to Rebuild and Slots to measure over-revision.

### 3.2 Budget and per-call cap

The budget $B$ caps the sum of prompt and completion tokens over all calls for one task, enforced
by admission: a call whose estimated cost does not fit is refused and the task ends with the
answer it holds. The per-call cap $m$ is the generation limit passed with each call. Unless an
experiment varies it, $m = 4{,}096$ for every call of every system.

### 3.3 Models and benchmarks

| Model | Serving | Reasoning |
|---|---|---|
| GLM-5.3-Flash (FP8) | SGLang, tensor-parallel 8 | cannot be disabled |
| DeepSeek-V4.1-Flash | vLLM, 8 x H20 | on by default |

| Benchmark | Items | Judge |
|---|---:|---|
| MATH-500 | 400 | symbolic equivalence |
| GAIA | 122 | normalised exact match |
| FRAMES | 660 | exact match and F1 |
| MuSiQue | 1,934 (600 in the state-policy study) | exact match and F1 |

Budgets are 8k, 16k and 24k tokens. Decoding is at temperature 0 with one seed.

### 3.4 Instruments

**Role accounting.** Every call's prompt and completion tokens, including reasoning tokens, are
charged to the role that made it.

**Slot accounting.** Every prompt character is attributed to the slot it comes from: plan,
question, prior answer, critique, tool output, system, scaffolding. Characters, not tokens; and
reasoning text is not part of any prompt, so slot shares describe prompts only.

**Per-round judging.** The executor's answer is judged after every round. Between consecutive
rounds a revision is a *repair* (wrong to right), a *breakage* (right to wrong), or neutral.

**Statistics.** Paired comparisons use the exact McNemar test. With one seed, we report a measured
run-to-run flip rate (5.5% of items on MATH, 6.6% on GAIA) as the noise floor for unpaired
percentages.

---

## 4. Contribution 1: The Cap Intervention

**Both architectures are cap-sensitive.** Table 1 changes only the per-call cap, from 1,024 to
4,096, for the single call and for the pipeline separately, at a 24k budget.

**Table 1.** Effect of raising the per-call cap from 1,024 to 4,096, budget fixed at 24k. Paired
exact McNemar. "Truncated" is the share of pipeline items with at least one call reaching 1,024.
(`analysis/` inline scripts; `results/cap_effect.json`)

| Model | Benchmark | single call Δ | p | pipeline Δ | p | pipeline truncated at 1,024 |
|---|---|---:|---:|---:|---:|---:|
| GLM | MATH | +5.5 | 0.005 | +5.0 | 0.006 | 15.2% |
| | GAIA | +4.1 | 0.23 | +9.0 | 0.007 | 78.7% |
| | FRAMES | +3.6 | 9e-4 | +3.3 | 0.008 | 37.0% |
| | MuSiQue | +1.6 | 0.013 | +1.0 | 0.11 | 41.4% |
| DeepSeek | MATH | +5.0 | 3e-4 | +4.2 | 0.014 | 14.5% |
| | GAIA | +9.8 | 5e-4 | +13.9 | 8e-5 | 84.4% |
| | FRAMES | +6.7 | 1e-8 | +1.5 | 0.32 | 49.4% |
| | MuSiQue | +3.9 | 2e-10 | −0.1 | 0.94 | 58.3% |

The cap moves the single call significantly in 7 of 8 cells and the pipeline in 5 of 8, by up to
13.9 points, never significantly downward. On reasoning models a call of either kind can be
consumed by its own reasoning, so neither architecture is protected. The effects differ in size
and differ by task, which is what makes the cap a confound for the comparison between them.

**Choosing caps can decide the comparison.** Table 2 computes the pipeline-minus-single-call
difference on the same items under all four combinations of caps.

**Table 2.** Pipeline minus single call, accuracy points, 24k, by the caps given to each.

| Model | Benchmark | both 1k | both 4k | pipeline 1k, single 4k | pipeline 4k, single 1k | swing |
|---|---|---:|---:|---:|---:|---:|
| GLM | MATH | +8.5 | +8.0 | +3.0 | +13.5 | 10.5 |
| | GAIA | +1.6 | +6.6 | −2.5 | +10.7 | 13.1 |
| | FRAMES | −1.1 | −1.4 | −4.7 | +2.3 | 7.0 |
| | MuSiQue | −1.4 | −1.9 | −2.9 | −0.4 | 2.6 |
| DeepSeek | MATH | +4.8 | +4.0 | −0.2 | +9.0 | 9.2 |
| | GAIA | +0.8 | +4.9 | **−9.0** | **+14.8** | **23.8** |
| | FRAMES | +1.5 | −3.6 | −5.2 | +3.0 | 8.2 |
| | MuSiQue | +0.4 | −3.6 | −3.5 | +0.3 | 4.0 |

Across cap choices the difference swings by 2.6 to 23.8 points and changes sign in 6 of 8 cells.
At equal caps its sign is the same at 1k and 4k in 6 of 8 cells; the two exceptions, DeepSeek
FRAMES and MuSiQue, move from a non-significant pipeline advantage at 1k to a significant
pipeline deficit at 4k, because the single call gains far more from the larger cap there. A
pipeline-versus-single-call comparison is therefore interpretable only at equal, stated caps, and
only above the point where the caps truncate.

---

## 5. Contribution 2: Where the Budget Goes

**Coordination roles take more than half of the budget.** Table 3 charges every token, prompt and
completion including reasoning, to the role whose call spent it.

**Table 3.** Share of total tokens by role, rebuild pipeline, cap 4,096, 24k.
(`analysis/role_decomp.py`)

| Model | Benchmark | planner | executor | reviewer | planner + reviewer |
|---|---|---:|---:|---:|---:|
| GLM | MATH | 30.4% | 46.5% | 23.2% | 53.6% |
| | GAIA | 15.1% | 56.4% | 28.6% | 43.7% |
| | FRAMES | 20.4% | 48.0% | 31.6% | 52.0% |
| | MuSiQue | 20.8% | 46.4% | 32.8% | 53.6% |
| | **pooled** | 21.1% | 47.6% | 31.3% | **52.4%** |
| DeepSeek | MATH | 37.5% | 45.3% | 17.1% | 54.6% |
| | GAIA | 17.9% | 51.2% | 30.9% | 48.8% |
| | FRAMES | 28.8% | 42.5% | 28.7% | 57.5% |
| | MuSiQue | 31.6% | 40.7% | 27.7% | 59.3% |
| | **pooled** | 30.3% | 42.2% | 27.5% | **57.8%** |

**The plan dominates every prompt, but prompts are a small share of tokens.** Table 4 attributes
prompt characters, over all calls of all roles, to their slots.

**Table 4.** Share of prompt characters by slot, rebuild pipeline, cap 4,096, 16k.
(`analysis/role_decomp.py`)

| Model | plan | question | prior answer | system | scaffold | critique | tool output |
|---|---:|---:|---:|---:|---:|---:|---:|
| GLM (5 benchmarks, 3,616 items) | **41.2%** | 30.0% | 13.4% | 10.4% | 1.8% | 1.7% | 1.4% |
| DeepSeek **[PENDING: final; 3 benchmarks so far]** | **48.7%** | 30.3% | 3.4% | 14.0% | 2.2% | 0.0% | 1.2% |

The plan, written once and re-sent in every executor call, is the largest slot, larger than the
question. Yet executor prompts, recorded per round, account for 6.6% (DeepSeek) to 8.9% (GLM) of
total tokens, against about 30% for executor completions; calls not recorded per round (code
re-asks) bound the executor's prompt share above by 12 to 17%. On these models the price of
coordination is paid mainly in the coordinators' and executor's own generation. Re-sent state is
the largest part of what the model reads and a minor part of what the budget buys.

This also bounds what compressing the state can save on reasoning models: at most the carried
part of the executor's prompt, roughly a tenth of the budget. **[PENDING: the state-policy study
reports the carried tokens per round and total spend under each policy, including accumulation,
where the carried share grows with the round index.]**

---

## 6. Contribution 3: State and the Value of Revision

**Current evidence.** Two existing runs judge the answer after every round at cap 4,096 and 24k:
the rebuild pipeline with the Accept stop rule, and the slot-compressed loop with the Strict stop
rule (reached after a direct answer and a tool-assisted answer, so seeded with a draft).

**Table 5.** Revisions between consecutive rounds, pooled over four benchmarks and both models.
(`analysis/value_of_round.py` and inline script)

| Pipeline | State | Stop rule | revisions | repairs | breakages | net per 100 | cells net positive |
|---|---|---|---:|---:|---:|---:|---:|
| plain | rebuild (full plan, free critique) | Accept | 1,986 | 89 | 22 | **+3.37** | 8 of 8 |
| compressed | slots (one plan line, 150-token critique) | Strict | 22,840 | 581 | 675 | **−0.41** | 0 of 8 |

In the compressed loop most revisions happen with more than 16k of the 24k budget still unspent,
so its negative value is not budget exhaustion. Two other differences remain confounded: what the
slots discard, and the Strict rule, which rarely lets the loop stop and so permits more
revisions. Table 5 is therefore not yet evidence about state alone.

**Controlled comparison.** **[PENDING: Table 6, from `analysis/state_policy.py`: for each model,
benchmark and budget (24k, 8k), and for each of Rebuild, Accumulate, Window-2, Summary and Slots
under the Accept rule, plus Rebuild and Slots under the Strict rule: accuracy and its paired
difference from Rebuild, total tokens, rounds run, share stopped by budget, carried tokens per
round, budget remaining before each revision, reviewer accept rate, repairs, breakages and net
value per 100 revisions.]**

The comparison is designed to answer three questions. Does carried state raise the price of later
calls enough to matter on reasoning models (Accumulate against Rebuild, carried tokens and
budget-stop share)? Is the value of revision a function of compression as such, or of which
information is kept (Summary and Window-2 against Slots)? And how much of a negative revision
value comes from a stop rule that permits more revision rather than from the state (each state
under Accept against Strict)?

---

## 7. Where the Pipeline Is Worth Its Price

At an equal cap of 4,096, Table 7 compares the rebuild pipeline with the single call.

**Table 7.** Pipeline against single call, both at cap 4,096, 24k. Accuracy % / mean tokens.
Paired exact McNemar.

| Model | Benchmark | single call | pipeline | Δ | p | token ratio |
|---|---|---:|---:|---:|---:|---:|
| GLM | MATH | 61.3 / 0.7k | 69.2 / 5.1k | **+8.0** | 1e-4 | 7x |
| | GAIA | 20.5 / 2.8k | 27.0 / 15.6k | +6.6 | 0.057 | 6x |
| | FRAMES | 30.2 / 1.4k | 28.8 / 6.8k | −1.4 | 0.34 | 5x |
| | MuSiQue | 20.6 / 1.6k | 18.7 / 7.0k | **−1.9** | 0.009 | 4x |
| DeepSeek | MATH | 58.0 / 0.7k | 62.0 / 3.5k | **+4.0** | 0.017 | 5x |
| | GAIA | 27.9 / 3.1k | 32.8 / 13.6k | +4.9 | 0.15 | 4x |
| | FRAMES | 32.6 / 1.5k | 28.9 / 6.4k | **−3.6** | 0.004 | 4x |
| | MuSiQue | 20.9 / 1.9k | 17.3 / 6.9k | **−3.6** | 2e-7 | 4x |

The pipeline is significantly better on mathematical reasoning for both models, significantly
worse on multi-hop retrieval in three of four cells, and not distinguishable on GAIA, where point
estimates favour it but 122 items give little power. It costs four to seven times the tokens
throughout. This is the pattern Section 6 predicts if revision has value where the carried plan
and critique give the executor something to revise against, as in a multi-step derivation, and
little where the answer depends on retrieving and composing facts the model either has or does
not.

---

## 8. Predicting Where Extra Compute Helps

Two further measurements bear on whether a system could decide, per item, to spend more.

**Confidence does not locate repairable items.** Running every item through a direct answer, a
tool-assisted answer and the compressed loop at cap 4,096 gives, per item, whether the loop
repairs or breaks the direct answer. Harm exceeds rescue in six of eight cells; a perfect router
would gain 3.3 to 8.6 points over the direct answer; and the direct answer's mean log-probability
predicts its correctness moderately (AUC 0.57 to 0.71) but rescue poorly (0.33 to 0.67, median
0.53). (`results/gate_auc_4k.json`)

**The cap-hit rate predicts whether cheaper reasoning helps.** Giving a cascade's cheap levels a
cheaper reasoning setting has opposite effects across twelve cells on the two models, from −5.2
to +42.2 points. The share of items whose generation reaches the cap without the setting, measured
beforehand, has a rank correlation of 0.78 with the effect ($p = 0.003$) and separates its sign in
all twelve: every cell at or above 18% improves, every cell at or below 16% does not. The
criterion is relative to the cap in use, consistent with Section 4.
(`analysis/c_rung_predictor.py`)

---

## 9. How the Literature Reports the Cap

We audited 21 recent papers evaluating agent or multi-agent systems under a token or cost
criterion. For each, the full text was extracted and searched for every statement of a generation
limit, and every match and every statement of decoding settings was read.

**Table 8.** (`results/lit_audit/audit.md`)

| | all (21) | multi-agent (9) |
|---|---:|---:|
| evaluation cap stated | 8 | 2 |
| of which only for the authors' own system | 1 | 1 |
| training cap only | 2 | 1 |
| no cap stated | 11 | 6 |
| compared systems stated to share it | 7 | 1 |

Seven of nine multi-agent comparisons do not state the evaluation cap of the systems compared. One
sets per-role caps (384 to 1,024) for its own system without stating the baselines'. One fixes a
shared cap of 2,048 as an evaluation protocol. All papers whose object of study is reasoning
length report and equalise the cap. The sample is drawn from our own related work, and "not
stated" means not stated in the paper.

---

## 10. Recommendations

1. Report the per-call generation cap with every budget, for every compared system, and compare
   at equal caps above the point where they truncate.
2. Report the truncation rate of every system.
3. Include a single-call baseline at the same cap.
4. Report cost by role. On reasoning models most of a pipeline's price is its coordinators'
   generation.
5. State the context policy across rounds and the stop rule; both change the value of revision.

---

## Limitations

**Two reasoning models.** Both models reason by default and GLM cannot be made not to. Whether
the role decomposition and the state-revision interaction hold for models with reasoning disabled
is not tested here; we expect re-sent state to be a larger share of tokens there.

**Single seed.** Paired tests are robust to it; individual percentages carry a measured flip rate
of 5.5 to 6.6%.

**Two cap values.** We compare 1,024 and 4,096 and do not characterise the curve between or
beyond.

**Slot accounting is in characters** and covers prompts only. Token-level role shares are exact;
token-level slot shares are not measured.

**Small GAIA.** With 122 items, GAIA differences of four to seven points are not significant.

**One pipeline topology.** Other topologies distribute generation across calls differently.

**MuSiQue subsample** of 600 items in the state-policy study, paired by item.

**Audit scope.** A convenience sample of 21 papers.

---

## Conclusion

A pipeline's position on the accuracy-token plane mixes what coordination costs with what it buys
and with how the comparison was configured. Holding the configuration fixed exposes each part.
The per-call cap alone can reverse a pipeline-versus-single-call comparison, and most multi-agent
comparisons do not report it. On reasoning models, more than half of a pipeline's budget is its
coordinators' own reasoning, while the state it re-sends is the bulk of what it reads and a small
part of what it pays. Whether another round of that coordination is worth its price depends on
what the carried state lets the executor revise against. **[PENDING: final statement of the
state-policy result.]** At equal caps, the pipeline is worth its price on mathematical reasoning
and not on multi-hop retrieval.

---

## Appendix A. Pending results

| Marker | Source | Chain | ETA |
|---|---|---|---|
| Section 6 Table 6, abstract (3), conclusion | state-policy runs `_glmst_*`, `_dsst_*` | `_chain_state_glm.sh`, `_chain_state_ds.sh` | GLM tonight, DeepSeek early tomorrow |
| Table 4 DeepSeek row | `_dsr4k_*16` with segment accounting | `_chain_ds_final.sh` | this evening |

## Appendix B. Scripts

| Result | Script |
|---|---|
| Tables 1, 2, 7 | inline paired-test scripts, `results/cap_effect.json` |
| Tables 3, 4 | `analysis/role_decomp.py` |
| Table 5 | `analysis/value_of_round.py` |
| Table 6 | `analysis/state_policy.py` |
| Section 8 | `results/gate_auc_4k.json`, `analysis/c_rung_predictor.py` |
| Table 8 | `results/lit_audit/audit.md` |
