# TokenBank results — 2026-09-02 00:31

Model: Qwen3.8-27B (FP8), temperature 0, seed 42. All comparisons are paired per item with an exact McNemar test. "Truncated" counts runs that spent more than half the budget and stopped on exhaustion, which excludes API timeouts.

> **Two sections are being re-measured (2026-09-02).** The `Compact` column in §1 and
> everything derived from it are withdrawn: the summarizer appended a second system
> message, which Qwen's template rejects with HTTP 400, and the client swallowed the
> error, so every compacted task scored zero. A second defect summarized away the task
> statement itself. Both are fixed; on the two cells re-run so far compaction is *not*
> significantly different from the untreated baseline (GAIA 16k p=0.63, MATH 16k p=0.42),
> against the "significantly worse" this file previously reported. §6 (SWE-bench) is also
> being re-scored after a constructor fix described in §7.

Arms: **Naive** = unbounded loop. **Naive+Fmt** = same, plus the answer-format instruction (the control every gain is measured against). **Compact** = accumulate the trajectory and LLM-summarize past a context threshold. **Accum** = accumulate without summarizing. **Ours-v1** = bounded slots plus a verbalized escalation channel. **Ours** = bounded slots, routed by the log-probability gate alone.

## 1. Main matrix

| set | B | n | Naive | Naive+Fmt | Compact | Accum | Ours-v1 | Ours |
|---|---|---|---|---|---|---|---|---|
| MATH | 8k | 400 | 36.8\*<br><sub>4,404 tok · 119 tr</sub> | 49.5<br><sub>4,349 tok · 89 tr</sub> | 50.2<br><sub>4,393 tok · 85 tr</sub> | 52.2<br><sub>4,213 tok · 64 tr</sub> | 60.2\*<br><sub>3,715 tok · 0 tr</sub> | 68.5\*<br><sub>2,506 tok · 0 tr</sub> |
| MATH | 16k | 400 | 43.5\*<br><sub>5,819 tok · 46 tr</sub> | 55.8<br><sub>5,209 tok · 23 tr</sub> | 51.5\*<br><sub>5,705 tok · 46 tr</sub> | 56.8<br><sub>4,899 tok · 23 tr</sub> | 62.5\*<br><sub>4,838 tok · 0 tr</sub> | 68.5\*<br><sub>2,588 tok · 0 tr</sub> |
| MATH | 24k | 400 | 44.0\*<br><sub>6,299 tok · 18 tr</sub> | 57.2<br><sub>5,635 tok · 6 tr</sub> | 55.8<br><sub>6,205 tok · 25 tr</sub> | 55.8<br><sub>5,382 tok · 15 tr</sub> | 64.2\*<br><sub>5,044 tok · 0 tr</sub> | 68.0\*<br><sub>2,679 tok · 0 tr</sub> |
| GAIA | 8k | 122 | 9.8<br><sub>6,467 tok · 108 tr</sub> | 7.4<br><sub>6,642 tok · 105 tr</sub> | 8.2<br><sub>6,393 tok · 108 tr</sub> | 9.8<br><sub>6,353 tok · 105 tr</sub> | 14.8\*<br><sub>6,003 tok · 0 tr</sub> | 21.3\*<br><sub>5,064 tok · 0 tr</sub> |
| GAIA | 16k | 122 | 9.8<br><sub>12,653 tok · 69 tr</sub> | 11.5<br><sub>12,937 tok · 80 tr</sub> | 7.4<br><sub>12,381 tok · 99 tr</sub> | 9.8<br><sub>12,796 tok · 93 tr</sub> | 18.9\*<br><sub>10,750 tok · 0 tr</sub> | 19.7\*<br><sub>6,705 tok · 0 tr</sub> |
| GAIA | 24k | 122 | 13.1<br><sub>16,114 tok · 17 tr</sub> | 9.8<br><sub>16,309 tok · 16 tr</sub> | 11.5<br><sub>17,529 tok · 81 tr</sub> | 11.5<br><sub>17,419 tok · 56 tr</sub> | 20.5\*<br><sub>12,785 tok · 0 tr</sub> | 21.3\*<br><sub>7,496 tok · 0 tr</sub> |
| FRAMES | 8k | 660 | 5.8<br><sub>5,866 tok · 454 tr</sub> | 7.6<br><sub>6,016 tok · 457 tr</sub> | 6.2<br><sub>6,015 tok · 499 tr</sub> | 6.5<br><sub>5,965 tok · 484 tr</sub> | 12.3\*<br><sub>5,351 tok · 0 tr</sub> | 14.8\*<br><sub>3,631 tok · 0 tr</sub> |
| FRAMES | 16k | 660 | 6.8<br><sub>10,335 tok · 230 tr</sub> | 7.4<br><sub>10,611 tok · 255 tr</sub> | 5.5\*<br><sub>11,130 tok · 455 tr</sub> | 6.5<br><sub>10,913 tok · 398 tr</sub> | 12.7\*<br><sub>8,521 tok · 0 tr</sub> | 13.3\*<br><sub>4,734 tok · 0 tr</sub> |
| FRAMES | 24k | 660 | 7.1<br><sub>12,135 tok · 16 tr</sub> | 8.0<br><sub>12,574 tok · 14 tr</sub> | 5.8\*<br><sub>16,181 tok · 392 tr</sub> | 6.8<br><sub>15,609 tok · 178 tr</sub> | 13.0\*<br><sub>8,608 tok · 0 tr</sub> | 13.9\*<br><sub>5,399 tok · 0 tr</sub> |
| MuSiQue | 8k | 1934 | 3.6\*<br><sub>5,937 tok · 1432 tr</sub> | 4.9<br><sub>6,060 tok · 1399 tr</sub> | 3.9<br><sub>6,093 tok · 1594 tr</sub> | 4.2<br><sub>6,106 tok · 1549 tr</sub> | 7.9\*<br><sub>5,397 tok · 0 tr</sub> | 7.9\*<br><sub>4,419 tok · 0 tr</sub> |
| MuSiQue | 16k | 1934 | 4.0\*<br><sub>10,692 tok · 771 tr</sub> | 5.2<br><sub>10,802 tok · 840 tr</sub> | 4.5<br><sub>11,629 tok · 1469 tr</sub> | 4.6<br><sub>11,604 tok · 1339 tr</sub> | 7.5\*<br><sub>8,422 tok · 0 tr</sub> | 8.5\*<br><sub>6,599 tok · 0 tr</sub> |
| MuSiQue | 24k | 1934 | 4.0\*<br><sub>12,663 tok · 26 tr</sub> | 5.3<br><sub>12,926 tok · 26 tr</sub> | 4.4<br><sub>16,997 tok · 1263 tr</sub> | 5.0<br><sub>16,196 tok · 595 tr</sub> | 8.7\*<br><sub>8,868 tok · 0 tr</sub> | 8.7\*<br><sub>7,470 tok · 0 tr</sub> |

SR% with `*` for p<0.05 vs Naive+Fmt; second line is mean tokens and real truncations.

## 2. Ours vs the format-aligned baseline

| set | B | n | Naive+Fmt | Ours | Δ pp | win/lose | p | tokens | truncations |
|---|---|---|---|---|---|---|---|---|---|
| MATH | 8k | 400 | 49.5 | **68.5** | **+19.0** | 84/8 | 4.1e-17 | -42% | 89 → **0** |
| MATH | 16k | 400 | 55.8 | **68.5** | **+12.8** | 58/7 | 4.3e-11 | -50% | 23 → **0** |
| MATH | 24k | 400 | 57.2 | **68.0** | **+10.8** | 53/10 | 3.4e-08 | -52% | 6 → **0** |
| GAIA | 8k | 122 | 7.4 | **21.3** | **+13.9** | 17/0 | 1.5e-05 | -24% | 105 → **0** |
| GAIA | 16k | 122 | 11.5 | **19.7** | **+8.2** | 11/1 | 6.3e-03 | -48% | 80 → **0** |
| GAIA | 24k | 122 | 9.8 | **21.3** | **+11.5** | 16/2 | 1.3e-03 | -54% | 16 → **0** |
| FRAMES | 8k | 660 | 7.6 | **14.8** | **+7.3** | 56/8 | 5.6e-10 | -40% | 457 → **0** |
| FRAMES | 16k | 660 | 7.4 | **13.3** | **+5.9** | 50/11 | 4.6e-07 | -55% | 255 → **0** |
| FRAMES | 24k | 660 | 8.0 | **13.9** | **+5.9** | 47/8 | 8.1e-08 | -57% | 14 → **0** |
| MuSiQue | 8k | 1934 | 4.9 | **7.9** | **+3.0** | 85/27 | 3.6e-08 | -27% | 1399 → **0** |
| MuSiQue | 16k | 1934 | 5.2 | **8.5** | **+3.4** | 93/28 | 2.4e-09 | -39% | 840 → **0** |
| MuSiQue | 24k | 1934 | 5.3 | **8.7** | **+3.4** | 96/30 | 3.0e-09 | -42% | 26 → **0** |

## 3. What the pipeline's tokens carry

Every character of every prompt in the untreated pipeline, attributed to what it carries.

| benchmark | n | plan | question | draft | critique | tool out | system |
|---|---|---|---|---|---|---|---|
| GAIA | 122 | **35.6%** | 32.0% | 25.3% | 2.2% | 0.3% | 3.7% |
| MATH | 400 | **50.4%** | 23.1% | 14.3% | 1.8% | 1.0% | 7.7% |
| FRAMES | 660 | **38.1%** | 26.2% | 26.2% | 2.9% | 0.5% | 4.7% |
| MuSiQue | 1934 | **40.2%** | 26.7% | 25.8% | 1.7% | 0.3% | 4.2% |
| BigCodeBench | 500 | **44.3%** | 36.1% | 13.0% | 0.8% | 0.8% | 4.1% |
| SWE-bench | 500 | **37.5%** | 48.3% | 10.5% | 0.1% | 0.8% | 2.2% |
| **pooled** | **4116** | **39.9%** | 30.4% | 22.3% | 1.7% | 0.5% | 4.1% |

## 4. What bounding each slot costs

Counterfactual: on a live round, re-issue the identical executor call with one slot replaced. Flips are correct→wrong / wrong→correct against the unbounded control.

| benchmark | n | plan → step | p | critique → verdict | p | critique removed | p |
|---|---|---|---|---|---|---|---|
| MATH | 250 | 5/3 | 0.727 | 17/6 | 0.035 | 15/1 | 0.001 |
| GAIA | 100 | 1/2 | 1.000 | 5/2 | 0.453 | 2/3 | 1.000 |
| FRAMES | 250 | 1/0 | 1.000 | 9/1 | 0.021 | 6/1 | 0.125 |
| MuSiQue | 250 | 1/0 | 1.000 | 0/4 | 0.125 | 0/4 | 0.125 |
| **pooled** | **850** | **8/5** | **0.581** | **31/13** | **0.010** | **23/9** | **0.020** |

The plan is 40% of the budget and free to remove; the critique is 2% and is not.
## 5. BigCodeBench: turning the cap off

500 tasks, scored by executing each task's unit tests. Budgets span the point where
the cap starts to bind. Scored offline from stored replies; the judge is validated
against the benchmark's canonical solutions at 118/120.

| budget | over cap | Naive | Ours-v1 | Ours | Δ vs Naive | p | tokens |
|---|---|---|---|---|---|---|---|
| 4k | 35% | 26.0% | 44.2% | **47.8%** | **+21.8** | 1.6e-22 | −53% |
| 8k | 8.6% | 40.4% | 48.4% | **48.6%** | **+8.2** | 3.8e-05 | −72% |
| 16k | 0% | 46.2% | 51.0% | 48.8% | +2.6 | 0.17 | −81% |

The advantage shrinks monotonically as the cap is relaxed and is not significant once
nothing truncates. This is what the thesis predicts; a competing explanation of the
gains would not produce it.

## 6. SWE-bench Verified: where this fails

500 instances, oracle localization, scored by executing FAIL_TO_PASS and
PASS_TO_PASS. No patch counts as unresolved. The diff constructor is validated by
routing the benchmark's own gold fixes through it: 496/500 resolve.

| arm | patches | resolved | rate | of submitted | vs Naive | p |
|---|---|---|---|---|---|---|
| Naive | 343 | **183** | **36.6%** | **53.4%** | — | — |
| Ours-v1 | 412 | 150 | 30.0% | 36.4% | −6.6 | 6.1e-04 |
| Ours | 428 | 130 | **26.0%** | 30.4% | **−10.6** | **4.3e-08** |

The budget mechanism works here as everywhere else: the governed arms submit *more*
patches than the baseline, 428 against 343, because they do not run out of budget.
Of what they submit, 30.4% resolve against the baseline's 53.4%. The patches are
worse.

The two arms fail differently, which their totals hide:

| | escalation rate | of its losses, had escalated |
|---|---|---|
| Ours-v1 | 99% | 98% |
| Ours | 18% | 14% |

Ours-v1 ran the loop and still lost, implicating the bounded slots. Ours mostly lost
without escalating, implicating the router: on repair the gate judges 82% of first
answers confident enough, and repair is the task type that most needs another round.

### 6b. The plan slot is not free on repair

Same counterfactual as §4, with execution-based scoring: 250 instances run with the
full plan, then with the plan replaced by its first line (1,366 chars → 122 chars).

| | n | correct→wrong / wrong→correct | Δ | p | flip rate |
|---|---|---|---|---|---|
| question answering (4 sets) | 850 | 8 / 5 | ~0 | **0.581** | 1.5% |
| code repair (SWE) | 250 | **15 / 5** | **−4.0pp** | **0.041** | **8.0%** |

Not a power difference: five times as many items move on repair, and the asymmetry
is three to one rather than even.

**Whether a plan can be bounded depends on what it holds.** On question answering the
model writes a decomposition and its first line loses an ordering the executor can
reconstruct. On repair it writes the diagnosis and the first line discards the
analysis. The distinction is not the domain — BigCodeBench is also code, is synthesis
rather than repair, and there the same bounding is free — but whether the plan holds
steps or conclusions, which is visible in the planner's output before any evaluation.

That accounts for roughly half the deficit. The other half is the router, for which
we have no fix: a gate that admits on low confidence cannot work where the model is
confident and wrong.

## 7. Corrections made during this work

Each produced an internally consistent, plausible, wrong result. None was caught by a
self-consistency check; all were caught against an external reference.

| what | how it presented | what caught it |
|---|---|---|
| Compaction trigger read cumulative spend, not context occupancy | 100% of triggered items truncated — a perfect spurious correlation | the correlation looked too clean, so we read the trigger's definition |
| `pdflatex` absent; page count read off a stale PDF | three "successful" compiles | file mtime |
| `stopped_by == 'budget'` also catches API timeouts | truncation counts inflated | timeouts spend ~0 tokens |
| Judge container starved (2 CPUs, 64 concurrent) | a significant, mechanistic finding that more budget hurts the baseline | re-judging offline without contention |
| Patch constructor added a phantom line to every final hunk | passed two self-consistency checks | 5 of 6 gold patches failed to apply |
| Undefined-reference check grepped a log tectonic never writes | "no undefined refs" while the PDF rendered `??` | reading the PDF |
| Chain script passed arguments in the wrong order | every step "completed" in 20 seconds having run zero tasks | a step named `x=1` in the log |
| One SWE arm reused a general config and inherited an answer-format instruction the others lacked | 421 of 500 replies carried `ANSWER:`, none in the other arms | the user asked whether the result was a bug |
| Reply parser required `### REGION` headers, discarding bare code blocks on multi-hunk instances | the single-call arm looked like it failed to follow the format | it produced a code block on all 500 instances and labelled 118 |
