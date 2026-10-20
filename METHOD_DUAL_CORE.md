# TokenBank: Budget-Constrained Runtime Governance for Multi-Agent LLM Systems

## Problem

A multi-agent LLM pipeline (Planner → Executor → Reviewer, ≤ R rounds) must
complete each task under a hard per-task token budget B. Naive static
allocation (B/3 per role) starves the compute-heavy role and collapses as B
tightens. Goal: maximize success rate under a given B; secondarily, spend
fewer tokens when the task allows it.

## Method (dual core)

### Core 1 — Adaptive allocation (keeps SR under tight B)

Per task, each agent i gets quota q_i (floor q_min = B/5). Two mechanisms
move budget to where it is actually consumed:

1. **Predictive rebalancing.** After every Reviewer checkpoint, forecast each
   agent's remaining need from its recent consumption velocity; unused quota
   is moved from agents with surplus to agents forecast to run out.
2. **Peer borrowing.** If a call exceeds q_i at dispatch, the agent borrows
   the shortfall from a peer with spare quota, subject to safety conditions:
   lender priority ≤ borrower priority, amount ≤ lender remaining, global
   reserve > 0, and cumulative debt ≤ 1.5 × q_i. (Ablation at B=4K: removing
   borrowing drops SR from 11.8% to 2.6% — it is the critical rescue valve
   in the binding regime.)

### Core 2 — Adaptive runaway stall (saves tokens)

Dormant for converging tasks. The executor–reviewer loop stops early only
when a task is provably stuck:

- **Runaway guard** fires when the answer is unchanged for r=1 consecutive
  rounds AND tokens burned ≥ max(0.2·B, M) (M=2000 absolute floor protects
  the tight-budget regime). On firing, the global release ceiling tightens
  to used + 0.3·(B − used).
- **Stall**: if the answer is still unchanged at the next checkpoint after
  the guard fired, the task terminates with its current best answer.

Normal tasks never see the constraint; only stuck tasks are cut. This is the
only module that reduces tokens (26–41% across benchmarks); its SR cost is
≤ noise on FRAMES/MATH and ~3pp on GAIA at B=32K.

### Algorithm (per task)

```
1:  q_i ← max(B/5, B/n) for each agent;  released ← B
2:  for round = 1..R (R ≤ 5):
3:      plan/act calls proceed if dispatch(q_i, est) or borrow(peer, est)
4:      after each Reviewer: rebalance(forecasts)
5:      if answer unchanged and used ≥ max(0.2B, M):
6:          released ← used + 0.3(B − used)        # runaway guard
7:          if answer unchanged at next checkpoint: break   # stall
8:  return best answer
```

## Evaluated and rejected

- **Credit scoring** (role-priced credit lines): SR 11.8% → 7.9% at GAIA-4K.
  Cold-start score instability chokes the borrowing valve under tight B.
- OPL / predictive-only / KV-tracking: individual ablations show ≤ noise
  (±2–3pp) at the tested operating points; retained in the framework but not
  claimed as contributors.

## Notes

- Minimum viable budget for the 3-agent pipeline is ~3–4K tokens (at B=2K all
  methods score 0% across three benchmarks).
- All results: deepseek-v4-flash, temp=0, seed 42. Single-seed noise band:
  ±3pp (GAIA n=152), ±2pp (FRAMES n=824), ±4pp (MATH n=500).
