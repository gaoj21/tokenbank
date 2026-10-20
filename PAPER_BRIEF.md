# Writing brief: analysis paper

Status as of 2026-10-02 noon. Every number below is from a finished run and names the script
that recomputes it. Numbers marked **[PENDING: run]** are waiting on the 4096-cap reruns (ETA
tonight); write the sentence now and fill the number later.

---

## 1. What the paper now argues

Under a hard per-task token budget, the cost and benefit of running a multi-agent pipeline
over a single call depend on three things that prior budgeted comparisons do not control:

1. **The per-call generation cap.** It is the largest confound we found. Raising it from 1024
   to 4096 moved a plain plan-execute-review baseline from 57.2% to 75.0% on Qwen MATH. A
   comparison between a pipeline and anything else, under a budget, is mostly a comparison of
   how often each gets truncated unless this is controlled.
2. **The task type.** With adequate per-call room, the pipeline buys accuracy on mathematical
   reasoning (+7.8 pp Qwen, +6.2 pp GLM over one call, both significant) at 5 to 6 times the
   tokens. On multi-hop retrieval and assistant tasks it buys little or nothing, and on Qwen
   GAIA it loses 4.9 pp.
3. **What the loop carries.** Bounding the carried state saves tokens but costs accuracy: at
   the same 4096 cap, the bounded loop scores 59.8% on GLM MATH against 69.2% for the plain
   loop, and 21.3% against 32.8% on DeepSeek GAIA.

Framing decision (2026-10-02): **analysis paper, no development narrative.** The paper does not
tell the story of a method that was later found to win only against a truncated baseline. The
governed configuration appears only as "a pipeline with compressed state", one point on the
accuracy-token frontier next to the plain pipeline and the single call. Every comparison in the
paper uses the controlled cap and states it; the 1,024-cap results appear only as the
demonstration of the cap's effect on the plain pipeline itself.

Suggested title direction: *What Does a Multi-Agent Pipeline Buy Under a Token Budget?*

## 1b. Abstract (current)

Multi-agent LLM pipelines are routinely compared with single calls under a fixed token budget,
with each system summarised by one accuracy-cost point. We show that such comparisons are
dominated by a setting that is rarely reported: the per-call generation cap. Holding the budget
fixed, raising the cap from 1,024 to 4,096 tokens moves a plan-execute-review pipeline from
57.2% to 75.0% on MATH, for a model with thinking disabled, because 31% of its items had a call
truncated. The effect is asymmetric by construction: a pipeline spends 46.6% of its budget
re-sending coordination state and needs longer calls than a single answer does, so a cap that
is harmless for one call silently cripples the other. Across three model families, five
benchmarks and three budgets, and with the cap controlled, pipelines improve over a single call
by 6 to 8 points on mathematical reasoning at five to six times the tokens, and give no gain on
multi-hop retrieval. Compressing the state they carry recovers the tokens and gives back the
accuracy. Where escalation helps cannot be read from the model's confidence, which predicts
rescue at or below chance, but whether extra reasoning helps can be predicted from how often
generation hits its cap. An audit of 21 budgeted-agent papers finds that 7 of the 9 comparing
multi-agent systems do not state the per-call cap of the systems they compare. We release our
traces and propose reporting per-call caps alongside budgets.

Pending checks before submission: "6 to 8 points" holds for Qwen and GLM, not DeepSeek (-1.2,
n.s.); state "on two of three families" in the body. "No gain on multi-hop retrieval" awaits
GLM's 4096 retrieval cells. The 57.2% to 75.0% gap awaits the Qwen deployment control.

---

## 2. Claims from earlier drafts that must not reappear

| Old claim | Why it goes |
|---|---|
| "Governed improves accuracy significantly in 27 of 48 cells, loses in none" | Baseline was truncated at 1024. Against a 4096 baseline, Governed **loses** 7.0 pp on Qwen MATH and 6.7 pp on GLM MATH at 24k |
| "30/30 significant against Naive" | Same artefact, and Naive is not the baseline any claim should rest on |
| "Token saving is unconditional and free" | It is unconditional; it is not free. It is paid for in accuracy on reasoning tasks |
| "+42.2 pp on BigCodeBench" | Baseline Ab7 ran at 1024 with reasoning on; most of the gap is empty answers. **[PENDING: BCB 4096 rerun]** |
| "The multi-agent loop buys nothing" (from the 2026-10-01 discussion) | False on MATH once the cap is lifted |
| "The cap mainly affects reasoning models" | False: Qwen has thinking off and 31% of its MATH items were truncated |

---

## 3. Findings that are settled

### F1. Where the budget goes (RQ1)
- 46.6% of a Qwen task's budget is coordination state re-sent from earlier rounds; the plan
  slot alone is 40.4% of characters read (Qwen) and 28.7% (Gemma). `_gen_materials.py` §2.
- Three context regimes, per-round executor prompt tokens, 24k (`analysis/a_round_cost.py`):
  accumulating +666/round (Qwen), +672 (Gemma); rebuilding flat at ~1,550 / ~1,060; bounded
  flat at ~610 / ~370.
- Budget-stopped share at 8k (`results/a_budget_stop.json`): accumulating 70.7% (Qwen),
  rebuilding 65.8%, bounded 0.0%.

### F2. The per-call cap is the dominant confound
Naive+Fmt, 24k, MATH (`analysis/rq_crossfam.py`):

| | cap 1024 | cap 4096 | items with a call truncated at 1024 |
|---|---|---|---|
| Qwen | 57.2% | **75.0%** | 31.0% |
| GLM | 64.2% | **69.2%** | 15.2% |
| DeepSeek | 57.8% | 62.0% | (11% empty answers → 2%) |

DeepSeek GAIA: 18.9% → 32.8% (+13.9, p=8e-05). DeepSeek MuSiQue: no change (−0.1).
**[PENDING: deployment control]** Qwen 1024 rerun on the H20 deployment, to show the 18-point
gap is the cap and not the change of serving hardware.

### F3. What the pipeline buys over one call (RQ2)
L0 = one call (the governed arm's cheap rung), against Naive+Fmt at 4096, 24k, paired:

| | MATH | GAIA | FRAMES | MuSiQue |
|---|---|---|---|---|
| Qwen | **+7.8** (p=2e-4) | **−4.9** (p=0.03) | +0.3 | [PENDING] |
| GLM | **+6.2** (p=1e-3) | [PENDING] | [PENDING] | [PENDING] |
| DeepSeek | −1.2 | +5.7 (n.s., n=122) | +1.7 | +0.5 |

Token ratio pipeline / one call: roughly 5× to 30×.

### F4. Escalation outcomes, per item (RQ2/RQ3)
From the all-escalate runs (every item runs all three rungs; `analysis/rq_allesc.py`,
`analysis/rq_crossfam.py`). Of items, the share where L2 rescues a wrong L0 answer vs harms a
right one:

| | MATH | GAIA | FRAMES | MuSiQue |
|---|---|---|---|---|
| DeepSeek (4096) | 2.8 / 6.0 | 4.1 / 9.8 | 8.0 / 5.2 | 6.6 / 4.7 |
| GLM (4096) | 5.0 / 8.2 | [PENDING] | [PENDING] | [PENDING] |
| Qwen | [PENDING: 4096 all-escalate] | | | |

The perfect router's ceiling over L0 is 3.5 to 4.4 pp (DeepSeek, 1024 run).

### F5. Uncertainty does not predict where escalation helps (RQ3)
AUC of the L0 log-probability gate (DeepSeek all-escalate):
- predicting L0 correctness: 0.55 to 0.71
- predicting rescue: 0.25 to 0.55, at or below chance on every benchmark.

This is the same observation as Signed Rescue Routing (arXiv 2609.07786) in a different
setting: same model, free-form generation, three sequential rungs, heterogeneous cost. Cite
it and state the difference; do not claim the idea.

### F6. Reasoning (RQ4)
- Prefix-only governance on a reasoning model saves 2% (GAIA 24k, 10,998 vs 11,229 tokens).
- The cheap reasoning rung's effect has opposite signs on GLM and DeepSeek; the share of
  generation calls hitting the cap without the rung separates the sign in 12 of 12 cells,
  Spearman 0.78, p=0.003 (`analysis/c_rung_predictor.py`, Table 8 of the old draft).
- One call with thinking on (2k) vs off (L0), DeepSeek: MuSiQue 19.2% vs 16.8%, FRAMES 31.1%
  vs 27.3%, MATH 55.2% vs 63.2%. Thinking helps retrieval and hurts MATH at this cap.

### F7. Other confounds worth one paragraph (RQ5)
- The `ANSWER: x` format instruction alone is worth +13.2 pp (Qwen) and +28.0 pp (Gemma) on
  MATH at 24k.
- Compaction never fires at the tightest budget on any of 3,116 tasks (Table 6 old draft).

---

## 4. Section plan

```
1 Introduction          question; 3 findings above with numbers; contribution = measurement
2 Setup                 pipeline, 3 families (Gemma: see §5), 5 benchmarks, budgets,
                        per-call cap stated explicitly as a controlled variable
3 Where the budget goes F1
4 The per-call cap      F2  (lead with this; it reframes everything after)
5 What the pipeline buys F3, F4
6 Can you tell in advance F5, F6 predictor
7 Reasoning             F6
8 Implications          single call as default; pipeline only on reasoning-type tasks;
                        always report per-call caps; bounded state is a cost/accuracy knob
9 Related work          SRR, MEM1, Context-Folding, CoRL, Budget-Aware Agents, BATS,
                        Total Cost of Agency, Markovian Thinker; Rethinking MAS Efficiency
                        (the one prior protocol that fixes the cap, 2,048)
10 Audit                21 papers, results/lit_audit/audit.md
Limitations             single seed; Qwen redeployed on H20; Gemma used for F1 only; GAIA n=122
```

---

## 5. Open decisions

- **Gemma: decided, not rerun (2026-10-02).** New experiments cover three families: Qwen
  (thinking off), GLM and DeepSeek (reasoning). Gemma's existing runs are used only for F1
  (budget accounting and context regimes), which does not depend on the per-call cap. Every
  cap-sensitive claim (F2 onward) is stated for the three rerun families, and the Limitations
  section says Gemma was not rerun at 4096.
- **Old main table (48 + 12 cells).** Becomes an appendix table with both baselines (1024 and
  4096) side by side once the reruns finish, so the confound is visible rather than hidden.

---

## 6. Reruns in flight

| Chain | Covers | ETA |
|---|---|---|
| `_chain_analysis_qwen.sh`, `_qwen2.sh` | Qwen MuSiQue 4096, deployment control, all-escalate at 4096 | ~2 h |
| `_chain_analysis_glm.sh` | GLM 4096 GAIA / FRAMES / MuSiQue, all-escalate | ~2 h |
| `_chain_rerun4k.sh` | Naive+Fmt 4096 at 8k and 16k, BCB baselines at 4096, all three families | tonight ~21:00 |
