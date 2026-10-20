# ICLR paper — complete writing brief

Everything needed to write this paper is in this file. Companion data file:
`ALL_RESULTS.json` (machine-readable dump of all 32 runs).

**The one rule:** every number in the paper must trace to §5 of this document. Nothing is
estimated, rounded from memory, or inferred. If a number is not in §5, the experiment was
not run — say so or leave a marked TODO. Do not fill it in.

---

## 1. Current state

| Item | Status |
|---|---|
| `main.tex` | Complete 8-page draft, compiles clean, 0 unresolved citations |
| `references.bib` | 17 entries, all cited, all real papers |
| Template | `iclr2026_conference.{sty,bst}` + `fancyhdr.sty` + `math_commands.tex` |
| Main text length | 6 pages of 9 allowed → **~3 pages free** (see §7) |
| Appendix B | Red `[PENDING]` placeholder; experiment still running (see §8) |
| `main_v1_draft.tex.bak` | Superseded weaker draft. Ignore; kept only for diffing |

The draft is structurally sound. The job is to (a) strengthen the writing, (b) fill the
3 free pages, (c) drop in the pending result when it lands.

### Build

```bash
TECTONIC=/tmp/claude-1001/-home-j0018-Desktop-tokenbank/f14fef7e-adf9-443b-9307-5c07475168b4/scratchpad/texenv/bin/tectonic
$TECTONIC -X compile main.tex --outdir build      # -> build/main.pdf
```

No system LaTeX is installed (host glibc is too old for the official tectonic binary); the
working tectonic came from conda-forge and lives in the scratchpad. If it is gone:
`conda create -y -p ./texenv -c conda-forge tectonic`.

### Checks before calling a draft done

```bash
# citation integrity
python3 -c "
import re
t=open('main.tex').read(); b=open('references.bib').read()
c=set(k.strip() for m in re.findall(r'\\\\cite[a-z]*\{([^}]*)\}',t) for k in m.split(','))
d=set(re.findall(r'^@\w+\{([^,]+),',b,re.M))
print('undefined:',sorted(c-d) or 'none'); print('uncited:',sorted(d-c) or 'none')"

# page count and unresolved refs
../.venv/bin/python -c "
from pypdf import PdfReader
r=PdfReader('build/main.pdf'); t=''.join(p.extract_text() or '' for p in r.pages)
print('pages:',len(r.pages),'| [?]:',t.count('[?]'))"
```

---

## 2. The thesis — do not drift from this

> Multi-agent LLM pipelines are reported to improve accuracy under a fixed token budget.
> Most of that reported improvement is an **answer-extraction artifact**: the governed
> system is instructed to emit `ANSWER: x`, the naive baseline is not, and the grader
> silently charges the baseline for every correct-but-unparseable response. Once the
> baseline is given that one line, three of four benchmarks show **no gain at all**. On the
> fourth, a real gain survives — but its mechanism is that the multi-round loop **actively
> damages** computational tasks, and the system wins mainly by declining to run it.

This is a **measurement / negative-result paper with one positive finding**. It is NOT a
systems paper proposing a method. Do not reframe it as "we propose TokenBank."

**Positioning (settled).** Title: *Statelessness as Budget Policy: Governing
Multi-Agent Test-Time Compute*. Abstract is 235 words, three parts: problem / method /
results.

### The thesis, in one sentence

> Under a fixed token budget, what consumes the budget is the **coordination state** a
> multi-agent loop accumulates, not the reasoning it does. Every mechanism that works is a
> different way of **not retaining that state**.

Supporting measurement: the executor accounts for only 50--56% of tokens; the rest is plan
text, reviewer prose and raw tool output carried round to round. When the budget runs out
mid-loop the pipeline emits a truncated answer worse than one it already held.

### The four mechanisms are one principle

Do not present them as a feature list. Each deletes a different kind of state:

| Mechanism | State it removes |
|---|---|
| Escalation ladder | Never creates it -- no plan/review context is built for items that do not need it |
| Per-round context rebuild | Destroys it every round: the loop is rebuilt from question + last answer + last critique, nothing else survives |
| Tool notes / JSON verdict / plan freeze | Caps each state class at constant size: raw tool output to a <=120-token note, reviewer prose to a bounded verdict, full plan to its current step |
| Discarding an ESCALATE reply | Refuses to propagate a non-answer downstream as a draft |

### Implementation facts (verified by reading `run_gaia_task_v5`)

These correct errors made in earlier drafts. Check against the code, not the summary docs.

1. **L0 already has the code tool.** `_direct()` defaults to `with_tool=True` and L0 calls
   it with no arguments. L0 and L1 are the *same primitive*; they differ only in whether a
   draft is passed in (`"Verify it, fix it if wrong, then answer"`). Do **not** write "a
   direct call, then a tool-using call" -- that is wrong.
2. **The escalation-signal argument is threshold transfer, NOT lack of signal.** An earlier
   draft claimed log-probability carries no information about admission, citing -0.070 vs
   -0.077. That was Phase 0 numbers from GAIA/MuSiQue applied to MATH, and it is wrong.
   Measured on MATH (n=500, A0 arm), phi separates L0-correct from L0-incorrect at
   **AUC 0.783, Cohen d 0.77**. The gate fails because phi has no distribution-independent
   scale: theta0 = -0.10, tuned on GAIA/MuSiQue dev (mean phi -0.17 to -0.19), admits 27%
   and 13% of their items but **97% of MATH items** (mean phi -0.035), so admitted accuracy
   (50%) collapses to the base rate (49%). Per-task recalibration would fix it and is what a
   deployed router cannot do. A declaration is scale-free. Never write "confidence carries
   no signal". The code
   comment states it: log-probability "can't tell 'knows the answer' from 'needs to
   compute/look up'". Confidence measures generation fluency; escalation needs to know
   whether the item requires external computation. These are orthogonal, so no threshold on
   log-probability can work. Write it that way, not as "the gate is poorly calibrated".
3. **An ESCALATE reply sets `cur = None`** -- the output is discarded, not passed down as a
   draft. Without this it would anchor the next level, which is the same failure that sank
   M4 pooling.
4. **Context is rebuilt, not appended** (`exec_messages[:2] + [last answer, last feedback]`)
   -- but this line sits at loop-body indentation, **outside every `if`**, so the baseline
   gets it too. It is the harness, NOT a contribution. Do not claim it. What we control is
   the *size* of the two carried elements: measured executor prompt at round 1 is 1,897
   tokens (naive) vs 660 (ours), a 2.9x reduction at equal pipeline depth.
5. **M1's stop rule requires `mean_logprob > tau_conf`**, so a low-confidence round can
   never be early-stopped; it continues automatically. This is deliberate rollback
   protection.
6. `_confident()` returns False when logprob is missing, i.e. escalate is the safe default.
7. **The allocation mechanisms did not all read zero.** Transfer mechanisms (lending,
   role-credit, predictive rebalancing, stall) recorded 0% trigger rates. But the per-role
   quota floor **did** bind and was harmful: min_quota = B/5 left the executor 960 tokens of
   a 4,800 budget and blocked its tool calls, which is why v5 sets `bank.q_min = budget` to
   remove per-role ceilings. Write "inert or harmful", never "all recorded zero".


---

## 3. Hard constraints — violating any of these makes the paper wrong

Not stylistic preferences. Each corresponds to an error actually made and retracted during
the analysis.

1. **The baseline is `Naive+Fmt`, never `Naive`.** Any Δ quoted against `Naive` must be
   explicitly labelled as the uncontrolled comparison. The paper's own argument is that
   this comparison is misleading.
2. **No SR claim on GAIA, at all.** n=122 with a 6.6% run-to-run flip rate ⇒ noise floor
   ±3.3pp, larger than any difference observed. GAIA is used only for the
   budget-does-not-bind result and for token discussion.
3. **No token-efficiency claim on GAIA or FRAMES.** Three budget points show the same
   frontier; at 8k on FRAMES the *baseline* is more efficient.
4. **MuSiQue has no SR claim.** Every mechanism ablation is null and `Naive+Fmt` alone
   matches the full system. Its only surviving claim is −20% tokens at equal accuracy.
5. **M1's accuracy effect is NOT significant** (+2.2pp, p=0.175). Permitted: "M1 reduces
   measured over-revision (25→17 mid-loop flips)". Forbidden: "M1 contributes +2.2pp".
6. **The +5.2pp M3 increment is confounded with M5.** `V5_FULL` sets
   `m5_strategy=hybrid`; `V5-Unc-Fmt` has no such key. Always write "context economics
   (bundled with a depth-vs-breadth setting)". Never "M3 contributes +5.2pp" unqualified.
7. **Cross-model gain is validated** (this constraint was reversed on 2026-08-31, when
   the Gemma format-aligned baseline finished). Gemma `Naive+Fmt` = 53.8%, system = 64.8%:
   **+11.0pp, McNemar 49/5, p=3.9e-10**. The format share is **57% on both model families**
   (Qwen 10.4/18.4, Gemma 14.7/25.7). Cite the format-aligned deltas (+8.0 Qwen, +11.0
   Gemma); never quote the raw +25.8pp against unformatted `Naive`.
8. **Every causal sentence must point to an arm.** If you cannot name the two runs whose
   difference supports a sentence, rewrite the sentence as description.
9. **Single seed.** Never write "significant" without noting that significance is per-item
   paired (McNemar), not across seeds, and that run-to-run drift is measured separately.

---

## 4. Section-by-section spec

Current structure, with the job of each section. Keep the order.

| § | Title | Job |
|---|---|---|
| 1 | Introduction | Set up the popular claim, state our headline number *as measured the usual way*, then reveal the confound. End with contributions: one claim, three tests, one warning. |
| 2 | An Alternative Explanation: The Output Contract | The mechanism, with the extraction-gap decomposition. Derives the prediction that the artifact is bounded by `SR_true × f`, i.e. largest when the baseline is often right but hard to parse. **No experiments here.** |
| 3 | Setup | Pipeline, arms, models, data, judging, measured noise floor. |
| 4 | Test 1: A Format-Aligned Baseline Removes the Gain on Three of Four Benchmarks | Table 1. Plus two negative controls: same-frontier (3 budget points) and budget-does-not-bind (GAIA 16k vs 24k). |
| 5 | Test 2: The Artifact Can Be Manufactured | Table 2. The Mirage-style move: one line of prompt moves results 2.8–10.4pp, spanning typical published effect sizes. |
| 6 | Test 3: Where a Gain Survives, the Loop Is Doing Harm | Flip audit; then Table 3 (additive decomposition with routing disabled); then the cross-model backfire result. |
| 7 | A Warning: Ablation Under an Active Router Is Uninformative | Table 4. Generalize to any system with a bypass (escalation ladder, difficulty-conditioned policy, cache, early exit). |
| 8 | Related Work | Self-critique failure literature; test-time compute; evaluation artifacts. |
| 9 | Discussion | Three recommended practices, then what the positive finding implies for design. |
| 10 | Limitations | The constraints from §3, stated plainly. |

**Tone.** The draft was already revised once for this: the first version was full of "we
caution" / "we make no claim" and read as a failed-project postmortem. Be assertive about
what was established; put caveats in Limitations where they belong. One line worth
preserving verbatim from §5: *"In our own case it did exactly that, and we believed it."*
— the paper's credibility rests on owning the error.

### What must NOT be added

- Any claim that budget allocation across agents helps. It does not, in this data.
- The deleted mechanisms (inter-agent lending, per-role quotas, predictive rebalancing,
  stall detection, credit scoring) as contributions. One appendix paragraph, no more.
- Cross-task budget pooling (`V5.1-Unified`). Rejected; it loses accuracy on retrieval
  tasks via draft-anchored premature convergence. At most one appendix sentence.
- Multi-seed claims. One seed was run.

---

## 5. Every number, with provenance

All runs: temperature 0, seed 42, **test splits**, Qwen3.6-27B-FP8 via vLLM unless marked
Gemma. `Governed` = `V5.1-NoPool` in result files. Budgets are per-task token caps over all
roles (prompt + completion). All values below were machine-verified against the source
files (47/47 checks passed).

### 5.0 Three transcription traps

- **MATH `Naive` is 52.8%, not 53.2%.** 53.2% is the full 500-item set; 52.8% is the
  400-item test split, which is what every other number uses. `ALL_RESULTS.json` under
  `v5al_ma` stores the *uncorrected* 53.2% because it is the raw run file. Use 52.8%.
- **FRAMES test n is 660, not 824.** 824 was a pre-split count in older notes.
- **`V5.1-Unified` is a different, rejected configuration** (cross-task pooling). The
  system in the paper is `V5.1-NoPool`. Do not mix them.

### 5.1 Main table (24k) — Table 1

| Dataset | n | Governed SR | tok | Naive+Fmt SR | tok | Naive SR | tok |
|---|---|---|---|---|---|---|---|
| MATH | 400 | **71.2%** | 3,283 | 63.2% | 6,489 | 52.8% | 6,333 |
| MuSiQue | 1934 | 9.7% | 6,582 | 9.8% | 8,220 | 7.0% | 7,665 |
| GAIA | 122 | 20.5% | 10,259 | 19.7% | 13,264 | 15.6% | 13,084 |
| FRAMES | 660 | 15.2% | 7,237 | 14.8% | 8,201 | 10.6% | 8,251 |

Governed vs Naive+Fmt, exact McNemar (discordant b/c):

| Dataset | Δ | b/c | p | source |
|---|---|---|---|---|
| MATH | +8.0pp | 34/2 | 1.94e-08 | `v53_phasez.json` → `z1b….ablations["NoPool(full)"].vs_unc_fmt` |
| MuSiQue | −0.2pp | — | 0.842 | `experiment_findings_v5.md` §8.4 E2 |
| GAIA | +0.8pp | 7/6 | 1.000 | `z2_unc_fmt_stats["ga\|24"]` |
| FRAMES | +0.3pp | 29/27 | 0.894 | `z2_unc_fmt_stats["fr\|24"]` |

Bootstrap 95% CI (B=1e4): GAIA Naive+Fmt 19.7% [13.1, 27.0], Governed 20.5% [13.9, 27.9];
FRAMES Naive+Fmt 14.8% [12.3, 17.6], Governed 15.2% [12.4, 17.9].

### 5.2 The manufactured artifact — Table 2

| Dataset | Naive | Naive+Fmt | format effect | gain vs Naive | artifact share |
|---|---|---|---|---|---|
| MATH | 52.8% | 63.2% | **+10.4pp** | +18.4pp | 57% |
| GAIA | 15.6% | 19.7% | +4.1pp | +4.9pp | 84% |
| FRAMES | 10.6% | 14.8% | +4.2pp | +4.6pp | 91% |
| MuSiQue | 7.0% | 9.8% | +2.8pp | +2.7pp | 104% |

The two arms differ by **one line of prompt** (`m3_answer_format`) and nothing else — same
model, decoding, budget, tools, reserve, max_rounds. Configs: `V5-Unc-Fmt` vs
`V5-Unconstrained`.

### 5.3 Additive decomposition, MATH 24k, routing DISABLED — Table 3

| Arm | config name | SR | tok | flips (strict + budget-final) | Δ vs row above | p |
|---|---|---|---|---|---|---|
| Naive+Fmt | `V5-Unc-Fmt` | 63.2% | 6,489 | 26 (17+9) | — | — |
| + context economics (+M5) | `V5.1-Ab6-NoM2NoM1` | 68.5% | 5,750 | 25 (25+0) | **+5.2pp** | **0.0046** (36/15) |
| + M1 early stop | `V5.1-Ab3-NoM2` | 70.8% | 5,648 | 17 (17+0) | +2.2pp | 0.175 (22/13) |
| + M2 routing | `V5.1-NoPool` | 71.2% | 3,283 | 9 (9+0) | +0.4pp | — |

M1 effect 95% CI: [−0.5, +5.2]pp. Sum 5.2+2.2+0.4 = 7.8 ≈ 8.0pp total.
Source: `z1c_gap_closing.ab6_m1_isolation`, `z1b….per_budget["24"]`.

Conditional flip rates (flips / items ever correct): Naive 13.5% (244 ever-correct),
Naive+Fmt 9.3% (279), Ab6 8.4%, Ab3 5.7%, Governed 3.1% (294).

### 5.4 Router-shadow ablations, MATH 24k, routing ACTIVE — Table 4

| Arm | SR | tok | routing L0/L1/L2 | Δ vs Naive+Fmt | b/c | p |
|---|---|---|---|---|---|---|
| Full system | 71.2% | 3,283 | 309/11/80 | +8.0 | 34/2 | 1.94e-08 |
| −M1 | 70.2% | 3,281 | 305/18/77 | +7.0 | 32/4 | 1.94e-06 |
| −M2 (forces all L2) | 70.8% | 5,648 | 0/0/400 | +7.5 | 34/4 | 6.04e-07 |
| −M3.2 notes | 70.8% | 3,139 | 310/12/78 | +7.5 | 34/4 | 6.04e-07 |

The point: every arm keeps ≥7.0pp; removing any one mechanism costs ≤1pp; the routing
column is the explanation.

### 5.5 Flip audit (correct→wrong), 24k

An item flips if some round was judged correct and the final answer was not. `strict` = a
later round turned it wrong. `budget-final` = correct through the last logged round but the
budget-exhaustion reserve call emitted a wrong final answer.

| Arm | MATH flips | ever-correct | cond. rate |
|---|---|---|---|
| Naive | 33 (16 strict + 17 budget) | 244 | 13.5% |
| Naive+Fmt | 26 (17 + 9) | 279 | 9.3% |
| Governed | 9 (9 + 0) | 294 | 3.1% |

Item-wise causal split, MATH 24k: of the **34** items Governed wins and Naive+Fmt loses,
**16 (47%)** are flips; the other **18** the naive pipeline never solved at any round. At
16k: 21 of 40 (52%). Aggregate net-flip share: 53% (24k), 72% (16k).

**Sub-split of those 16 (verified from traces):**

| Baseline's failure on the 34 decisive items | count |
|---|---|
| revised a correct answer into an error (strict) | **8** |
| held a correct answer, lost it to budget-exhaustion truncation | **8** |
| never correct at any round | 18 |

> **Do not write "16 of 34 wins arise because the baseline revises a correct answer into an
> error."** That is wrong by 2×: only **8** are revision; the other 8 are truncation, a
> different failure mode. The 8/8 symmetry is worth stating — it maps onto the two
> mechanisms (M3 removes truncation 9→0, M1 removes revision 25→17).

> Explain the non-monotonicity: context economics *raises* strict flips 17→25 because it
> rescues items that previously died of budget exhaustion into the loop, where they become
> exposed to revision. Then M1 cuts them back to 17.

### 5.6 Route-level split of the +8.0pp, MATH 24k

| Router level | n | Governed | Naive+Fmt | disc (b/c) | net | share |
|---|---|---|---|---|---|---|
| L0 (single call) | 309 | 79.0% | 73.1% | 20/2 | +18 | **+4.5pp** |
| L1 | 11 | 63.6% | 36.4% | 3/0 | +3 | +0.8pp |
| L2 (full loop) | 80 | 42.5% | 28.7% | 11/0 | +11 | +2.8pp |

Interpretation constraint: this measures **damage the naive pipeline inflicts** on those
items, not accuracy created by answering in one call. §5.3 shows the same damage is
repairable in-loop.

### 5.7 Three budget points (frontier argument)

| Dataset | Arm | 8k | 16k | 24k |
|---|---|---|---|---|
| GAIA | Governed | 17.2% / 5,459 | 17.2% / 9,398 | 20.5% / 10,259 |
| GAIA | Naive+Fmt | 15.6% / 5,919 | 19.7% / 10,986 | 19.7% / 13,264 |
| FRAMES | Governed | 14.1% / 5,047 | 15.3% / 6,981 | 15.2% / 7,237 |
| FRAMES | Naive+Fmt | 13.2% / 4,726 | 14.2% / 7,648 | 14.8% / 8,201 |
| MATH | Governed | — | 70.5% / 2,965 | 71.2% / 3,283 |
| MATH | Naive+Fmt | — | 62.5% / 5,158 | 63.2% / 6,489 |
| MuSiQue | Governed | — | 10.1% / 6,489 | 9.7% / 6,582 |

**At 8k on FRAMES the baseline is more token-efficient** (13.2% @ 4,726 vs 14.1% @ 5,047).
This is why constraint #3 forbids a token-efficiency claim on FRAMES/GAIA.

**Budget-does-not-bind (GAIA):** Naive+Fmt scores 19.7% (24/122) at *both* 16k and 24k.
Per-item correct sets: intersection 20, union 28, Jaccard 0.714, symmetric difference
8/122 = 6.6% — exactly the measured run-to-run flip rate. Source: `z4_gaia_unc_fmt_overlap`.

### 5.8 Noise floor

Re-running an identical configuration:

| Dataset | flips | rate | SR shift |
|---|---|---|---|
| GAIA (Governed, 24k) | 8/122 | 6.6% | 20.5% → 17.2% |
| MATH (Governed, 24k) | 22/400 | 5.5% | 71.2% → 70.8% |

Drift SD = √(f/n): GAIA 2.33pp, MATH 1.28pp, FRAMES 1.00pp, MuSiQue 0.58pp.
Combined SE (sampling ⊕ drift): GAIA 3.78, MATH 2.59, FRAMES 1.52, MuSiQue 0.84 pp.
Key deltas in combined SE: MATH 7.14×, MuSiQue 3.13×, GAIA 0.22×, FRAMES 0.20×.
Source: `z5_noise_combined`.

### 5.9 MuSiQue — all null (24k, n=1934)

| Arm | SR | tok | vs Governed |
|---|---|---|---|
| Governed | 9.7% | 6,582 | — |
| −M1 | 9.8% | 6,564 | null |
| −M3 context | 10.0% | 8,451 | p=0.61 |
| −M2 | 9.2% | 6,350 | p=0.362 |
| **Naive+Fmt** | **9.8%** | 8,220 | **p=0.842** |
| Naive | 7.0% | 7,665 | — |

Format alone (+2.8pp) exceeds the entire gain originally attributed to the method (+2.7pp).

### 5.10 Cross-model (Gemma-4-31B-it, q4_0 GGUF via llama.cpp, 24k)

| Arm | MATH | tok | GAIA | tok |
|---|---|---|---|---|
| Governed (`V5.1-NoPool`) | 64.8% | 2,035 | 15.6% | 9,569 |
| **Naive+Fmt** | **53.8%** | **9,137** | not run | |
| Naive | 39.0% | 9,112 | 12.3% | 12,337 |
| v5 A0 (log-prob-only router) | 36.0% | 2,223 | 13.1% | 11,488 |

Gemma MATH paired tests (n=400): system vs `Naive+Fmt` **+11.0pp, 49/5, p=3.9e-10**;
`Naive+Fmt` vs `Naive` +14.7pp, 84/25, p=1.2e-08; system vs `Naive` +25.7pp, 118/15,
p=5.1e-21. Flips: `Naive` 15 (15 revision + 0 truncation), `Naive+Fmt` 20 (18+2), system
**0**. Source: `results/budget_sweep_v2_gemma-4-31B-it_v54b_ma_fmt.json`.

**Both cross-model claims now hold.** (a) The log-prob-only router *backfires* on
computational tasks — Qwen 49.2% vs Naive 52.8%, Gemma 36.0% vs Naive 39.0% — and
verbalized escalation repairs it; both arms carry the format line, so no format confound.
(b) The system's SR gain survives format alignment on both families: +8.0pp (Qwen),
+11.0pp (Gemma). **Still invalid:** quoting the raw +25.8pp against unformatted `Naive`.

### 5.11 Arm → config-name map (`_budget_sweep_v2.py`, `CONFIGS_V5`)

| Paper name | Config name | Key settings |
|---|---|---|
| Naive | `V5-Unconstrained` | `V5_BASE` only |
| Naive+Fmt | `V5-Unc-Fmt` | `V5_BASE` + `m3_answer_format` |
| Governed | `V5.1-NoPool` | `V5_BASE`+`V5_FULL`, `m2_verbal`, `l01_max_tokens=2048` |
| −M1 | `V5.1-Ab1-NoM1` | `m1_early_stop=False` |
| −M2 | `V5.1-Ab3-NoM2` | `m2_cascade=False` (forces all L2) |
| −M3.2 | `V5.1-Ab2-NoNotes` | `m3_tool_notes=False` |
| −M3ctx | `V5.1-Ab5-NoM3ctx` | reviewer/notes/plan-freeze off, format kept |
| −M2−M1 | `V5.1-Ab6-NoM2NoM1` | both off; the M1-isolation arm |
| (rejected) pooling | `V5.1-Unified` | `m4_pool=True` — **not the system in the paper** |

---

## 6. Source-file map

| Content | Path |
|---|---|
| All main SRs and token counts | `paper/ALL_RESULTS.json` and §5 above |
| McNemar / bootstrap CI / flip audits / route split | `results/v53_phasez.json`, `v53_phasez_report.md` |
| Earlier statistics, Pareto reanalysis | `results/v52_reanalysis.json`, `v52_reanalysis_report.md` |
| Full experiment history and every verdict | `experiment_findings_v5.md` §8.3, §8.4, §8.4b, §8.4c |
| Condensed conclusions | `tokenbank_v51_summary.md` |
| Config definitions for every arm | `_budget_sweep_v2.py`, `CONFIGS_V5` block |
| Per-item traces (for worked examples) | `traces/tasks_sweep_<model>_<tag>.jsonl` |
| Pareto figures | `figures/v53_pareto_{ga,ma,fr,mu}.png` |

`experiment_findings_v5.md` §8.4b and §8.4c document two conclusions that had to be
**retracted** mid-analysis. Read them before writing paper §6 and §7 — they are the reason
those sections are worded the way they are.

---

## 7. The 3 free pages — priority order

1. **Pareto figure** (highest value). `figures/v53_pareto_ga.png` and
   `figures/v53_pareto_fr.png` exist and show 16k/24k. The "same frontier" claim is
   currently text-only, and it is the load-bearing evidence for constraint #3.
   *Caveat:* those PNGs are labelled "8k pending"; the 8k numbers now exist (§5.7), so
   re-plot from `ALL_RESULTS.json` or add the 8k points to `_v53_phasez.py` and rerun it.
2. **MuSiQue all-null ablation table** (§5.9) — currently buried in Appendix C. Strong
   negative evidence (five arms, all null, baseline matches system); belongs in paper §4.
3. **Full three-budget table** (§5.7) — supports both the frontier and binding-budget
   arguments.
4. **A worked example** of a correct→wrong flip: one MATH item, correct at round 2, wrong
   at the end. Traces in `traces/tasks_sweep_Qwen3.6-27B-FP8_v53e1_ma24.jsonl`
   (`rounds[].answer_snapshot`, `rounds[].is_correct`). Would make paper §6 concrete.

---

## 8. Open items

| Item | Blocking? | Note |
|---|---|---|
| ~~Appendix B: Gemma x `Naive+Fmt`~~ | **Done 2026-08-31** | 53.8%; cross-model gain survives (+11.0pp, p=3.9e-10). Constraint #7 reversed. The red `[PENDING]` block in `main.tex` Appendix B must be replaced with these numbers. |
| Template year | **Decide before submission** | Currently `iclr2026_conference.sty`; header prints "ICLR 2026". Given the date, the next deadline is likely ICLR 2027. Swap `.sty`/`.bst` if so. |
| M3/M5 separation | Not blocking | One 400-item arm (Ab6 minus M5) would clear constraint #6. Not run. |

---

## 9. Regenerating `ALL_RESULTS.json`

```python
import json, glob, os
PRE = 'budget_sweep_v2_'
MODELS = ('Qwen3.6-27B-FP8', 'gemma-4-31B-it')

def cells(d):
    res = {}
    def walk(o):
        if isinstance(o, dict):
            if 'name' in o and 'success_rate' in o:
                res[o['name']] = {'sr': round(o['success_rate']*100, 1),
                                  'tok': round(o.get('mean_tokens', 0))}
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
    walk(d); return res

out = {}
for p in glob.glob(f'results/{PRE}*.json'):
    rest = os.path.basename(p)[len(PRE):-5]   # <model>_<tag> -- basename has no dir prefix
    model = next((m for m in MODELS if rest.startswith(m + '_')), None)
    if model is None:
        continue
    out[rest[len(model)+1:]] = {'model': model, 'cells': cells(json.load(open(p)))}
json.dump(out, open('paper/ALL_RESULTS.json', 'w'), indent=1, ensure_ascii=False)
print(len(out), 'runs')   # 69 as of 2026-08-31
```

Note the sweep writes result JSON and traces to the **project root**; they are moved to
`results/` and `traces/` by hand afterwards. Check the root for stray
`budget_sweep_v2_*.json` / `tasks_sweep_*.jsonl` before regenerating.
