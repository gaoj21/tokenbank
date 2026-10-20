# Method figure: design spec

One figure, method only. Full width (`figure*` in the ACL two-column template). It carries a
single idea: **one zero-cost gate decides, at every rung, both whether to build more
scaffolding and how long to reason, while every round inside the loop carries a fixed-size
state.**

---

## 1. Layout

```
                       ┌─────────────── gate φ ───────────────┐
                       │  mean logprob > θℓ   and  entropy ≤ τ │
                       └──┬───────────────┬───────────────┬────┘
              escalate?   │   reasoning?  │   (one decision)
                          ▼               ▼               ▼
                                                   ┌─────────────────────────────┐
                                                   │ L2  Plan → Execute ⇄ Review  │
                                                   │     ≤5 rounds, default ▬▬▬▬▬▬│
                                    ┌──────────────┤                             │
                                    │ L1 + tool     │  context of every round:    │
                                    │ reuses L0     │  [sys][q][plan:1 line][ans] │
                     ┌──────────────┤ draft         │  [critique≤150][note≤120]   │
     task ──────────►│ L0 direct    │ short ▬       │   t=1 ▇▇▇▇▇                 │
                     │ answer       │               │   t=2 ▇▇▇▇▇                 │
                     │ short ▬      │               │   t=3 ▇▇▇▇▇   same width    │
                     └──────┬───────┴──────┬────────┴─────────────┬───────────────┘
                       φ0 ✔ │         φ1 ✔ │           converged / round 5
                            ▼              ▼                      ▼
                          answer         answer                 answer

  budget (admission before every call)
  [██████████████████░░░░░░░░░░░░░|▒▒256▒▒]   refused call → answer from reserve
        used               remaining   closing reserve
```

---

## 2. Elements

### 2.1 The ladder (scaffolding, blue)

- Three boxes left to right, **each taller and wider than the last**. Height encodes cost,
  so "ordered by cost" is read from the shape rather than from a label.
- L0: *direct answer*. L1: *+ tool*. L2: *Plan → Execute ⇄ Review, ≤5 rounds*.
- A curved arrow from L0's output into L1, labelled *draft reused*. Without it the reader
  assumes every rung starts from scratch.
- Each rung has an exit arrow downward to *answer*, labelled φ0 ✔, φ1 ✔, and
  *converged / round 5*.

### 2.2 The gate φ (grey, drawn once)

- A single rounded bar or diamond across the top, containing
  `mean logprob > θℓ  ∧  entropy ≤ τ`.
- Dashed lines from it down to **both** the escalation arrows **and** the reasoning bars in
  each rung. This is the most important stroke in the figure: one decision, two effects.
- Optional small note beside it: *free: read from the answering call*.

### 2.3 Reasoning length (generation, purple)

- **Not a fourth rung.** It is an attribute of each rung.
- A short purple bar inside L0 and L1 (*cheap reasoning*), a long one inside L2
  (*default reasoning*).
- Message: a task the gate keeps low is also kept short.

### 2.4 Bounded state (carried state, orange)

- Inside the L2 box, the round context drawn as a row of slot blocks with their caps:
  `sys | question | plan: 1 line | last answer | critique ≤150 | tool note ≤120`.
- Under it, three stacked rows labelled t=1, t=2, t=3, **all the same width**.
- That equal width is the only message of this block. Do not add a growing stack for
  contrast; the figure is method only.

### 2.5 Budget bar (grey)

- One horizontal bar along the bottom: *used* (filled), *remaining* (light), *closing
  reserve 256* (hatched, at the right end).
- Caption beside it: *every call admitted against remaining budget; a refused call is
  answered from the reserve.*
- Small, but it is what makes truncation vanish, so it must be there.

---

## 3. Visual encoding

| Meaning | Colour | Where |
|---|---|---|
| Scaffolding (whether to build the loop) | blue | rung boxes |
| Carried state (κ) | orange | slot blocks and round rows in L2 |
| Generation (ḡ, reasoning length) | purple | reasoning bars in each rung |
| Control (gate, budget) | grey | gate bar, budget bar, dashed lines |
| Output | black | answer arrows |

- Three semantic colours only. Keep them identical if any later figure reuses them.
- Pick colours that stay distinct in greyscale: e.g. blue `#3B6FB6`, orange `#E08A2E`,
  purple `#8A5CC2`, grey `#8C8C8C`.
- L2 occupies about half the width, since the bounded-state detail lives inside it.
- No text smaller than 80% of body size after scaling to full width; in practice nothing
  below 7 pt.

---

## 4. Common mistakes

1. Drawing L0 → L1 → L2 as a plain flowchart of equal boxes. The cost ordering disappears.
2. Omitting the *draft reused* arrow, which hides that the ladder is incremental.
3. Drawing reasoning as a separate rung or a separate module.
4. Drawing the gate three times. Once, with dashed lines fanning out.
5. Leaving out the 256-token reserve.
6. Putting numbers from the results (savings, accuracy) in the method figure.

---

## 5. Caption

> **Figure 1.** One admission rule over three resources. A task enters at the cheapest rung.
> At each rung a zero-cost gate φ, computed from the answer's own log-probabilities and
> entropy, decides whether to stop or escalate (blue), and the same decision fixes the
> reasoning length: rungs the gate keeps run with a short reasoning setting, the full loop
> with the default (purple). Inside the loop every round rebuilds its context from
> fixed-size slots, so round *t* carries exactly what round 2 carried (orange). Every call is
> admitted against the remaining budget, and a refused call is answered from a 256-token
> reserve.

---

## 6. Correspondence with the text

| Figure element | Section |
|---|---|
| Ladder L0/L1/L2 | §4.3 |
| Gate φ, thresholds θ0 = −0.10, θ1 = −0.15, τ = 0.6 | §4.4 |
| Stop rule inside L2 | §4.5 |
| Reasoning bars | §4.6 |
| Slot caps (plan 1 line, critique ≤150, note ≤120) | §4.2 |
| Budget bar, 256 reserve | §4.1 |
| Colour legend = three resources | §4 opening table |
