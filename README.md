# tokenbank

Layout after the 2026-09-21 tidy. 433 files sat at the top level before it; 16 do now, and
nothing was deleted except three byte-identical duplicate images.

## Top level: only what something else calls

| File | Role |
|---|---|
| `_run_v2_experiment.py` | the harness. Plan/execute/review loop, budget accounting, the escalation ladder, the reasoning rung |
| `_budget_sweep_v2.py` | arm definitions and the sweep driver. Writes traces to `traces/` and summaries to `summaries/` |
| `_gen_materials.py` | regenerates `PAPER_MATERIALS.md` from traces. Every number in the write-up is computed here, never transcribed |
| `_bcb_rejudge.py` | offline BigCodeBench judging by executing each task's tests |
| `_slot_ablation.py` | counterfactual slot replacement |
| `_bcb_judge.py`, `_bcb_validate.py`, `_budget_sweep_vllm.py`, `_glm_status.py`, `_informed_static_baseline.py`, `_swe_slot_ablation.py` | called by scripts in `chains/`; kept here because those call them by bare name after cd-ing to the project root |
| `lahman.sqlite`, `sales_dataS.csv` | BigCodeBench fixtures. No script in this repo references them: task code reads them by relative path at judging time, so moving them breaks BCB scoring silently |

## Directories

| Directory | Contents |
|---|---|
| `traces/` | every per-item trace, all four model families. This is the only place traces live |
| `summaries/` | per-sweep aggregate JSON |
| `chains/` | the 100 run scripts. Each cd-s to the project root first, so paths inside are root-relative |
| `analysis/` | one-off analysis and table scripts, kept for provenance |
| `results/` | ablation and slot-ablation outputs |
| `paper/` | LaTeX. ACL/ARR format, `build.sh` is the only correct way to build it |
| `logs/` | run logs |
| `docs/` | historical plans and reports, superseded |
| `figures/` | plots. The paper carries none of them |
| `_aborted/` | runs that were stopped or invalidated, each with a README saying why. Read these before trusting a number that looks anomalous |
| `_archive/code_backups/` | `.bak_*` copies made before each risky edit |

## Two conventions worth keeping

**Traces are written to `traces/`, never beside the code.** The harness used to write to the
working directory, which is how one family's runs ended up at the top level while another's
sat in `traces/`. `TRACE_DIR` overrides it.

**`_gen_materials.py` maps files explicitly and never globs.** The same cell often exists in
several trace files: a primary run, a flip-rate re-run, a speed test, a segment-accounting
run. Globbing picked one at random and once put a wrong number in the paper (MATH 24k read
+12.2pp when the primary run says +10.8). Add a family by extending the map, not the pattern.
