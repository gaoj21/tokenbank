# When to Call the Team: Risk-Controlled Escalation for Multi-Agent LLM Pipelines

*Draft v3, 2026-10-03. Method paper. Two reasoning models (GLM-5.3-Flash, DeepSeek-V4.1-Flash),
four benchmarks. All calls in all systems use the same per-call generation cap of 4,096 tokens.*

---

## Abstract

Multi-agent LLM pipelines that plan, execute and review improve accuracy but multiply inference
cost; on reasoning models, more than half of that cost goes to coordination rather than answering.
Many tasks need only one cheap attempt, so the pipeline should be invoked selectively, yet the
confidence signals available at inference time are weak, and hand-tuned escalation rules lose
accuracy without warning. We formulate escalation as a constrained sequential stopping problem:
minimize expected cost subject to a user-specified bound on accuracy loss relative to the full
pipeline. We characterize its Lagrangian solution, showing that under a monotone-gain condition the
optimal stopping rule reduces to a confidence threshold at each stage, and we calibrate the
resulting policy family through multiple hypothesis testing, obtaining a distribution-free, finite-
sample guarantee on accuracy loss with a fallback to the full pipeline when no policy is safe.
Across two reasoning models and four benchmarks, our method, CLIMB, stays within a 5% tolerance
while using 25 to 76% of the pipeline's tokens in seven of eight settings. Deployed live, it saves
20 to 80% on one model, but at most 60% and sometimes nothing on the other. More expressive policies
save less under the same guarantee.

---

## 1. Introduction

<!-- Background -->
Multi-agent systems built from large language models divide a task among specialised roles: a
planner decomposes it, an executor solves it, and a reviewer critiques the result over several
rounds (Du et al., 2023; Wu et al., 2023; Hong et al., 2024). Such pipelines are a common way to
spend more inference compute on a task, and they often outperform a single call to the same
model. They are also expensive. On the two reasoning models we study, a plan-execute-review
pipeline spends 3.5 to 15.5 thousand tokens per task, several times a single call, and more than
half of those tokens (52% and 58%) are the planner's and the reviewer's own reasoning rather than
the executor's answer. The cost of a multi-agent system is, to a large extent, the cost of
coordination.

<!-- Motivation -->
Most tasks do not need that coordination. Of the tasks the full pipeline answers correctly, 62 to
91% are already answered correctly by a single direct attempt of the executor. A cascade that
answers cheaply first and calls the team only when the cheap answer is doubtful could therefore
save most of the cost. What stands in the way is the decision of when to escalate. The signals
available at inference time without extra calls, such as the answer's token log-probability and
entropy, are weak predictors of correctness, with an area under the ROC curve between 0.57 and
0.71 on our tasks. An escalation rule built on such a signal must be tuned, and a tuned rule fails
silently: a threshold that is safe on one benchmark loses 4.1 points of accuracy on another, and
nothing in the system's behaviour reveals it. Practitioners are left choosing between a cheaper
system with an unknown accuracy cost and the full pipeline at full price.

<!-- Objective -->
We ask whether the decision of when to call the team can be made with a guarantee. Concretely,
the user states how much accuracy they are willing to give up relative to always running the
full pipeline, and the system should return the cheapest escalation policy that provably stays
within that tolerance, or recognise that no such policy exists and run the pipeline.

<!-- Method -->
We formulate escalation as a constrained sequential stopping problem: after each rung of a ladder
of increasingly expensive configurations, the system observes a confidence score and decides
whether to stop, minimising expected cost subject to a bound on accuracy loss. Analysing the
Lagrangian relaxation, we show that when the gain from escalating decreases with confidence, the
optimal policy stops at the first rung whose confidence exceeds a threshold, so the search reduces
to a low-dimensional family of threshold policies. We then select a member of that family by
multiple hypothesis testing in the Learn-then-Test framework (Angelopoulos et al., 2021): each
candidate threshold is tested, in order from conservative to aggressive, against the hypothesis
that its accuracy loss exceeds the tolerance, using an exact binomial test on a calibration set.
The result is a distribution-free, finite-sample guarantee on accuracy loss that requires no
training and no assumption on the confidence signal; a weak signal only makes the selected policy
more conservative, and when no threshold passes, the method falls back to the full pipeline. We
call the method CLIMB.

<!-- Experiments -->
We evaluate CLIMB on two open reasoning models, GLM-5.3-Flash and DeepSeek-V4.1-Flash, and four
benchmarks covering mathematical reasoning (MATH-500), tool-assisted question answering (GAIA) and
multi-hop retrieval (FRAMES, MuSiQue). At a 5% tolerance, CLIMB uses 25 to 76% of the full
pipeline's tokens in seven of eight settings, with accuracy within 1.5 points of the pipeline and
higher in five; in the eighth it detects that its signal is uninformative and runs the pipeline.
Deployed live on held-out tasks, it stays within the tolerance in every setting and saves 20 to 80%
of tokens on one model, while on the other the savings shrink because the pipeline becomes more
expensive when seeded with the cheap draft. Four richer escalation policies, including learned
estimates of the value of escalating, save less under the same guarantee, and the monotone-gain
condition behind our analysis holds in 11 of 16 (benchmark, rung) pairs, failing precisely where
CLIMB declines to save.

<!-- Contributions -->
Our contributions are as follows.
1. We formulate the decision of when to invoke a multi-agent pipeline as constrained sequential
   stopping, and prove that under a testable monotone-gain condition its optimal solution is a
   confidence-threshold policy (Section 3, Section 4.1).
2. We propose CLIMB, a training-free escalation method that calibrates the threshold by
   multiple hypothesis testing and guarantees, with high probability, that accuracy loss relative
   to the full pipeline stays within a user-specified tolerance (Section 4.2).
3. We evaluate CLIMB in replay and live deployment on two reasoning models and four benchmarks,
   showing when it saves, when and why it declines to, and that richer policies do not save more
   under the same guarantee (Section 6).
4. We measure where a multi-agent pipeline's budget goes and which parts of its state can be
   compressed without loss, which informs the design of the escalation ladder (Section 7).

---

## 2. Related Work

**Cascades and routing.** Model cascades defer from a cheap model to an expensive one on
uncertain inputs (FrugalGPT, AutoMix, RouteLLM). Signed Rescue Routing shows that uncertainty is
an imperfect target, since escalation helps only when it repairs an error, and learns to predict
rescue and harm. Conformal Cascade gives distribution-free accuracy guarantees for multi-tier
model cascades. Our cascade is over levels of scaffolding of one model rather than over models,
our outputs are free-form, and our guarantee is relative to the most expensive level. We find, in
contrast to learned routers, that under a weak confidence signal a calibrated threshold saves more
than learned value estimates, and we give a condition under which thresholds are optimal.

**Budget allocation in multi-agent systems.** ZEBRA allocates a fixed budget across pipeline
phases by estimating per-phase utility curves and solving a continuous knapsack. BAMAS structures
a multi-agent system under a cost budget; CoRL trains a controller to respect one; BATS makes an
agent aware of its remaining budget. These decide how much to spend within a pipeline that always
runs. CLIMB decides whether to run it, and gives a guarantee on the consequence.

**Multi-agent efficiency.** AgentPrune and AgentDropout prune communication; Optima and Agent-GSPO
train for concise messages. These reduce the cost of a pipeline that runs on every task and are
complementary to CLIMB, which can sit in front of any of them.

**Risk control.** Learn-then-Test and conformal risk control give finite-sample guarantees for
selecting hyperparameters of a predictor. We use Learn-then-Test with a fixed-sequence procedure
and an exact binomial test on an upper bound of the loss.

---

## 3. Escalation as Constrained Sequential Stopping

**Rungs.** A task $x$ can be answered at rungs $k = 0, 1, \dots, K$ of increasing cost, where rung
$K$ is the full multi-agent pipeline. In our system $K = 2$: a direct answer ($L_0$), a
tool-assisted answer seeded with $L_0$'s draft ($L_1$), and the plan-execute-review pipeline
($L_2$). Rung $k$ yields an answer with correctness $Y_k \in \{0, 1\}$; stopping at rung $k$ incurs
the cumulative cost $C_k$ of all rungs run so far. After each rung $k < K$, the system observes a
confidence score $S_k$ computed from that rung's own output and decides whether to stop or to
continue. Escalation is therefore an optimal stopping problem with a terminal reward (whether the
answer is correct) and a running cost (tokens).

**Objective.** A stopping policy $\pi$ maps the scores observed so far to a stopping rung
$\tau_\pi \in \{0, \dots, K\}$, possibly at random. We measure its risk relative to the full
pipeline,

$$R(\pi) \;=\; \mathbb{E}\big[\,Y_K - Y_{\tau_\pi}\,\big],$$

the accuracy given up by not always escalating. Given a tolerance $\varepsilon$ set by the user, we
seek the cheapest policy whose risk stays within it:

$$\min_{\pi} \ \mathbb{E}\big[C_{\tau_\pi}\big] \quad \text{s.t.} \quad R(\pi) \le \varepsilon . \qquad (1)$$

The always-escalate policy has zero risk, so (1) is always feasible, and it is strictly feasible
for any $\varepsilon > 0$.

**Lagrangian relaxation.** For a price $\lambda \ge 0$ on tokens, (1) relaxes to the unconstrained
stopping problem

$$\max_{\pi} \ \mathbb{E}\big[\,Y_{\tau_\pi} - \lambda\, C_{\tau_\pi}\,\big], \qquad (2)$$

in which every token must buy at least $\lambda$ in expected accuracy. Problem (2) is solved by
backward induction: at rung $k$ the system stops if the expected reward of stopping is at least
the expected reward of continuing optimally. Writing $V_{k+1}^{\lambda}$ for the optimal
continuation reward, the net gain of escalating at rung $k$ is

$$G_k^{\lambda}(s) \;=\; \mathbb{E}\big[\,V_{k+1}^{\lambda} - (Y_k - \lambda C_k) \;\big|\; S_k = s\,\big],$$

and the optimal decision is to stop when $G_k^{\lambda}(S_k) \le 0$.

---

## 4. Method

CLIMB has two parts. Section 4.1 shows that, under a condition we can test, the solution of (1)
lies in a one-parameter family of confidence-threshold policies, which is what makes the problem
tractable with weak signals and small calibration sets. Section 4.2 selects a member of that
family by multiple hypothesis testing, which bounds the risk with no assumption on the model or
the signal. Optimality rests on the condition; validity does not.

### 4.1 Structure of the optimal policy

We make two assumptions. (A1) *Sufficiency*: given the score $S_k$, the outcomes of later rungs
are independent of earlier scores, so $S_k$ summarises everything the system knows at rung $k$
that bears on escalating. (A2) *Monotone gain*: for every $\lambda \ge 0$, the net gain
$G_k^{\lambda}(s)$ is non-increasing in $s$; the more confident the current answer, the less
escalating is expected to improve it per token spent.

**Proposition 1 (threshold optimality).** Under (A1) and (A2), for every $\lambda \ge 0$ there are
thresholds $t_0(\lambda), \dots, t_{K-1}(\lambda)$ such that the policy stopping at the first rung
$k$ with $S_k \ge t_k(\lambda)$ solves (2). Moreover, for every $\varepsilon > 0$, problem (1) is
solved by a mixture of two threshold policies of this form.

The first statement follows from backward induction: by (A1) the stopping decision at rung $k$
depends on the history only through $S_k$, and by (A2) the stopping region $\{s : G_k^\lambda(s)
\le 0\}$ is an upper interval. The second follows from strong duality: (1) is a linear program over
randomised policies with a single constraint that is strictly feasible, so there is no duality
gap, and its optimum is attained by a Lagrangian-optimal policy at the optimal price, mixed at the
boundary. The full proof is in Appendix D.

Condition (A2) is the substantive one. It says that confidence orders tasks by how much they stand
to gain from the multi-agent pipeline, and it is testable on data that records every rung's
outcome; Section 6.4 tests it. Where it fails, the threshold family need not contain the optimum,
but the guarantee of Section 4.2 still holds, and in that case calibration falls back to the full
pipeline rather than accept an unsafe threshold.

**Instantiation.** The score $S_k$ is the mean token log-probability of rung $k$'s answer, with an
entropy veto: rung $k$ may stop only if its mean top-$k$ entropy is at most 0.6. Because the
expected incremental cost varies little within a benchmark, we tie the thresholds of the cheap
rungs, $t_0 = t_1 = t$, reducing the policy family to a single parameter; Section 6.3 shows that
untying them, or replacing the threshold rule by richer estimates of $G_k^\lambda$, does not lower
cost under the same guarantee.

### 4.2 Risk-controlled calibration

The price $\lambda$ that solves (1) depends on the unknown distribution of tasks. Rather than
estimate it, we select the threshold directly with a distribution-free procedure: we cast the
choice as a family of hypothesis tests, one per candidate threshold, and keep only thresholds
whose risk the data can certify to be within the tolerance. This is the Learn-then-Test framework
for risk control, which we instantiate as follows.

**Candidates.** Let $\Lambda = (\varnothing, t_1, t_2, \dots, t_M)$ with
$t_1 > t_2 > \dots > t_M$, ordered from conservative to aggressive. The setting $\varnothing$ skips
the cheap rungs and runs the full pipeline, so its risk is zero.

**A loss that can be tested.** The per-task loss $Y_K - Y_\tau$ takes values in $\{-1, 0, 1\}$ and is
zero for most tasks. We bound it by the harm indicator $H = \mathbb{1}[Y_K = 1,\ Y_\tau = 0]$, the event
that stopping early turns a correct pipeline answer into a wrong one. Since $Y_K - Y_\tau \le H$,
controlling $\mathbb{E}[H]$ controls the risk, and $H$ is Bernoulli, which permits an exact test.

**Fixed-sequence testing.** On a calibration set of $n$ tasks with every rung's outcome recorded,
for $m = 1, 2, \dots$ we test the null hypothesis $\mathcal{H}_m : \mathbb{E}[H_{t_m}] > \varepsilon$
with the exact binomial p-value $p_m = \Pr[\mathrm{Bin}(n, \varepsilon) \le k_m]$, where $k_m$ is the
number of harms of threshold $t_m$ on the calibration set. We reject while $p_m \le \delta$ and stop
at the first non-rejection. The certified set $\mathcal{C}$ is $\varnothing$ together with all rejected
thresholds, and CLIMB deploys the member of $\mathcal{C}$ with the lowest mean calibration cost.

**Proposition 2 (risk control).** If calibration and test tasks are drawn i.i.d. from the same
distribution, then with probability at least $1 - \delta$ over the calibration set, every policy in
$\mathcal{C}$, and in particular the deployed one, satisfies $R \le \varepsilon$. The deployed policy's
mean cost on the calibration set is at most that of the full pipeline.

Testing in a fixed order controls the family-wise error rate at $\delta$ without a multiplicity
correction, because a true null can be rejected only if the first true null in the sequence is,
and each $p_m$ is valid by the exactness of the binomial test. On the event that no true null is
rejected, every member of $\mathcal{C}$ has $\mathbb{E}[H] \le \varepsilon$ and hence risk at most
$\varepsilon$, whichever member is then selected; choosing the cheapest therefore costs nothing in
validity. The cost statement holds because $\varnothing \in \mathcal{C}$. Proofs are in Appendix D.

Two properties of this construction matter in practice. First, the guarantee requires neither
(A1), (A2) nor any calibration of the confidence score: a poorly informative score simply leads
to fewer rejections and a more conservative policy. Second, the cost statement concerns the rung
costs observed at calibration. If a rung costs a different amount at deployment, as when the
pipeline receives the cheap rungs' draft (Section 6.5), the cost bound need not carry over, while
the accuracy guarantee is unaffected as long as the deployed rungs behave as calibrated.

The exactness of the test is not a technicality. A Hoeffding bound on the loss itself rejects no
threshold at our calibration sizes, because it ignores that almost all losses are zero.

### 4.3 Designing the rungs

CLIMB is agnostic to what the rungs are, but its savings depend on them. Two measurements guide
their design (Section 7).

**What to compress.** The plan is the largest part of what a pipeline re-sends, 41 to 49% of all
prompt characters. Reducing it to the current step costs no significant accuracy on either model.
The reviewer's critique is under 2% of prompt characters but determines whether revisions help.
Structured critiques discriminate better than free-form ones, but on reasoning models their
generation must not be capped tightly: at a 150-token cap, 81 to 98% of reviews are truncated to
nothing, an empty review is read as a request to revise, and correct answers are rewritten into
wrong ones.

**How to size calls.** No single per-call generation cap is right across budgets: a large cap
avoids truncation under a loose budget but makes each round too expensive under a tight one.
Allocating caps in proportion to the remaining budget, statically or at run time, matches the best
fixed cap at every budget we test without tuning it per budget.

---

## 5. Experimental Setup

**Models.** GLM-5.3-Flash (FP8, SGLang) and DeepSeek-V4.1-Flash (vLLM), both reasoning models.
Temperature 0. Every call in every system uses a per-call generation cap of 4,096 tokens.

**Benchmarks.** MATH-500 (400 items), GAIA validation (122), FRAMES (660) and MuSiQue (1,934), at
a per-task budget of 24,000 tokens.

**Rungs.** $L_0$: one executor call with the cheap reasoning setting (thinking disabled on
DeepSeek, `reasoning_effort: low` on GLM), 2,048-token cap. $L_1$: the same with a Python tool,
seeded with $L_0$'s answer. $L_2$: the plan-execute-review pipeline with the default reasoning
setting, up to five rounds, stopping when the reviewer accepts.

**Evaluation protocol.** Sections 6.1 to 6.4 use replay; Section 6.5 runs CLIMB live. For every item we record the outcome and cost of every rung: $L_0$ and
$L_1$ from a run in which every item passes through all cheap rungs, and $L_2$ from a run of the
full pipeline on the same item. Policies are evaluated by replaying these records, as is standard
for cascade evaluation. One difference from a live deployment is that the live $L_2$ would receive
$L_0$'s draft; here it runs without it. Each configuration is evaluated over 200 random 50/50
splits into calibration and test sets, stratified by benchmark, with $\delta = 0.1$.

**Baselines.** The full pipeline; a fixed threshold ($t = -0.10$, the value our earlier system was
tuned to); and four richer policies under the same certified selection (Section 6.3).

---

## 6. Results

### 6.1 Main result

**Table 1.** CLIMB against the full pipeline, 24k budget, cap 4,096. Accuracy and tokens as a
share of the pipeline's, averaged over 200 calibration/test splits. "Violations" is the share of
splits in which test accuracy loss exceeded $\varepsilon$ (the guarantee bounds it by $\delta = 10\%$).
(`analysis/risk_gate.py`, `results/risk_gate.json`)

| Model | Benchmark | Pipeline | ε = 2%: acc / tokens | ε = 5%: acc / tokens | violations (ε = 5%) |
|---|---|---|---|---|---|
| GLM | MATH | 69.2% / 5.1k | 68.7% / 97% | 67.7% / **75%** | 1% |
| | GAIA | 27.3% / 15.5k | 27.3% / 100% | 26.9% / **76%** | 2% |
| | FRAMES | 28.8% / 6.8k | 28.7% / 99% | **29.4%** / **58%** | 0% |
| | MuSiQue | 18.7% / 7.0k | 19.1% / 79% | **19.2%** / **25%** | 0% |
| DeepSeek | MATH | 62.0% / 3.5k | 62.0% / 100% | **63.4%** / **74%** | 0% |
| | GAIA | 32.8% / 13.6k | 32.8% / 100% | 32.6% / 100% | 2% |
| | FRAMES | 28.9% / 6.4k | 28.9% / 99% | **29.3%** / **61%** | 0% |
| | MuSiQue | 17.3% / 6.9k | 17.4% / 100% | **18.3%** / **47%** | 0% |

At $\varepsilon = 5\%$, CLIMB uses a quarter to three quarters of the pipeline's tokens in seven
of eight settings. Its accuracy is higher than the pipeline's in five of them, because the cheap
rungs are sometimes right where the pipeline's revisions are not, and at most 1.5 points lower in
the others. In the eighth, DeepSeek on GAIA, calibration finds no threshold that saves tokens
within the tolerance and CLIMB runs the full pipeline. The guarantee holds throughout: the
largest violation rate across all settings and tolerances is 4%, against a bound of 10%.

At $\varepsilon = 2\%$ the savings are small. With confidence this weak, certifying a two-point
bound on a few hundred calibration tasks leaves little room; CLIMB then behaves as the pipeline,
which is the intended failure mode.

### 6.2 The cost of an uncertified threshold

A fixed threshold, tuned once, is cheaper but carries no guarantee:

| | GLM MATH | GLM GAIA | DeepSeek GAIA | DeepSeek FRAMES |
|---|---|---|---|---|
| fixed $t = -0.10$: accuracy loss / tokens | 2.8 / 31% | 1.7 / 66% | **4.1** / 42% | 1.1 / 17% |
| CLIMB, ε = 5%: accuracy loss / tokens | 1.5 / 75% | 0.4 / 76% | 0.2 / 100% | −0.4 / 61% |

The fixed threshold loses 4.1 points on DeepSeek GAIA with no indication that it has done so.
CLIMB recognises that its signal is unreliable there and declines to save.

### 6.3 Richer policies do not save more

We compare CLIMB's shared threshold with four policies that use more information or fewer
assumptions, all under the same certified selection: (B) separate thresholds for $L_0$ and $L_1$,
with their offset chosen on a training split; (C) stopping at $L_1$ when its answer agrees with
$L_0$'s; (D) a non-parametric Bayes rule, estimating each rung's accuracy by confidence decile
and applying the Lagrangian stopping rule of Section 3; and (E) logistic-regression estimates of each rung's accuracy from confidence,
entropy, length, answer agreement and benchmark, applied through the same rule. Policies B to E need a
training split, so all five use 40% training, 20% calibration and 40% test.

**Table 2.** Tokens as a share of the pipeline's at ε = 5%, same splits for all policies; lower is
better. All satisfy the guarantee. (`analysis/escalation_variants.py`, `analysis/opt_escalation.py`)

| | A. shared threshold | B. two thresholds | C. agreement | D. Bayes deciles | E. learned values |
|---|---|---|---|---|---|
| GLM MATH | **83%** | 92% | 97% | 94% | 81% |
| GLM FRAMES | **71%** | 83% | 95% | 88% | 86% |
| GLM MuSiQue | 33% | **31%** | 59% | 47% | 42% |
| DeepSeek MATH | 87% | 93% | **74%** | 93% | 85% |
| DeepSeek FRAMES | **75%** | 96% | 88% | 95% | 87% |
| DeepSeek MuSiQue | **60%** | 91% | 83% | 79% | 87% |
| GAIA (both models) | 100% | 100% | 100% | 100% | 100% |

(E's numbers come from a separate run with the same split proportions.)

The shared threshold is the cheapest or within two points of it in five of six settings. Richer
policies spend part of the data on estimation, leaving less for certification, and the signals
they add carry little information beyond confidence. Answer agreement is the one exception, on
DeepSeek MATH, but it is the most expensive policy elsewhere. Proposition 1 explains why the
single-parameter family is enough when its condition holds.

### 6.4 When the threshold is optimal

Proposition 1 requires the gain from escalating to decrease with confidence (A2). We test a
directly measurable proxy: the gain in accuracy from escalating to the full pipeline,
$\Pr[Y_K = 1 \mid S_k] - \Pr[Y_k = 1 \mid S_k]$. When the incremental cost of escalating does not
depend on the score, as is approximately the case within a benchmark, the net gain $G_k^\lambda$ of
escalating straight to the pipeline differs from this proxy by a constant, so the two share their
monotonicity. Table 3 estimates the proxy by confidence quintile.

**Table 3.** Gain from escalating to the pipeline, $\Pr[Y_K] - \Pr[Y_k]$ in points, by quintile of
the rung's confidence (Q1 least confident), and Spearman's $\rho$ between quintile and gain.
(`analysis/monotone_gain.py`)

| Model | Benchmark | rung | Q1 | Q2 | Q3 | Q4 | Q5 | ρ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| GLM | MATH | $L_0$ | +10.0 | +12.5 | +3.8 | +2.5 | +2.5 | −0.87 |
| GLM | FRAMES | $L_1$ | +0.8 | +3.8 | 0.0 | −0.8 | −1.5 | −0.90 |
| DeepSeek | FRAMES | $L_0$ | +3.0 | +10.6 | +0.8 | −1.5 | −4.5 | −0.90 |
| DeepSeek | MuSiQue | $L_0$ | +5.7 | −0.5 | +1.8 | −0.3 | −4.1 | −0.70 |
| DeepSeek | GAIA | $L_0$ | 0.0 | +4.2 | +16.0 | 0.0 | +8.0 | **+0.36** |
| DeepSeek | GAIA | $L_1$ | +8.3 | 0.0 | +8.0 | 0.0 | +16.0 | **+0.21** |

The condition holds ($\rho \le 0$) in 11 of 16 (benchmark, rung) pairs; the full table is in
Appendix A. It fails most clearly on DeepSeek GAIA, where the most confident cheap answers are
the ones the pipeline improves most. That is the setting in which CLIMB saves nothing: where the
condition fails, no threshold is both cheap and safe, calibration detects this, and the method
falls back to the pipeline.

### 6.5 Live deployment

Replay assumes that each rung behaves at deployment as it did when its outcome was recorded. To
test this, we fix one random split per setting, certify the threshold on the calibration half from
recorded outcomes only ($\varepsilon = 5\%$, $\delta = 0.1$), and then run CLIMB live on the test
half: $L_0$ and $L_1$ are generated afresh, and items that escalate run the full pipeline seeded with
the cheap rungs' draft. We compare with the plain pipeline's recorded outcomes on the same items.

**Table 7.** Live CLIMB on the test halves. Tokens as a share of the pipeline's on the same items.
Paired exact McNemar. DeepSeek GAIA certified "run the pipeline" and is omitted.
(`analysis/climb_live.py`, `results/climb_live.json`)

| Model | Benchmark | n | Pipeline | Replay CLIMB | Live CLIMB | Δ acc (live) | p | stop at $L_0$/$L_1$/$L_2$ |
|---|---|---:|---|---|---|---:|---:|---|
| GLM | MATH | 200 | 69.5% / 5.1k | 68.5% / 87% | 66.0% / **80%** | −3.5 | 0.17 | 85 / 18 / 97 |
| | GAIA | 61 | 26.2% / 15.6k | 26.2% / 85% | 27.9% / **63%** | +1.6 | 1.00 | 14 / 14 / 33 |
| | FRAMES | 330 | 30.9% / 7.4k | 32.4% / 60% | 29.7% / **42%** | −1.2 | 0.62 | 166 / 78 / 86 |
| | MuSiQue | 967 | 18.5% / 7.0k | 19.1% / 25% | 19.0% / **20%** | +0.5 | 0.69 | 685 / 209 / 73 |
| DeepSeek | MATH | 200 | 68.5% / 3.2k | 69.5% / 70% | **71.5%** / 101% | +3.0 | 0.26 | 86 / 29 / 85 |
| | FRAMES | 330 | 27.3% / 6.5k | 28.8% / 69% | **32.4%** / 94% | **+5.2** | 0.01 | 57 / 66 / 207 |
| | MuSiQue | 967 | 17.0% / 6.9k | 17.1% / 38% | **18.0%** / **40%** | +1.0 | 0.39 | 516 / 244 / 207 |

Live accuracy stays within the tolerance in every setting: the largest loss is 3.5 points (GLM
MATH, not significant), and on DeepSeek FRAMES CLIMB is significantly more accurate than the
pipeline. Live cost differs from replay in two ways that act in opposite directions on the two
models. First, regenerated cheap answers have somewhat different confidence, so the escalation
rate drifts: GLM escalates fewer items than in replay (97 instead of 135 on MATH), DeepSeek more
(207 instead of 170 on FRAMES). Second, the seeded pipeline costs a different amount: on escalated
items GLM's seeded pipeline is cheaper than the replayed one (12.6k against 14.4k on MuSiQue),
DeepSeek's more expensive (5.9k against 4.9k on MATH). On GLM, live CLIMB uses 20 to 80% of the
pipeline's tokens, less than replay predicts. On DeepSeek its savings shrink, to 40% on MuSiQue,
94% on FRAMES and 101% on MATH, where the cost bound of Proposition 2, which holds for recorded
costs, is slightly exceeded. Replay is thus a reliable guide to accuracy and an approximate one to
cost, and calibrating on live rather than recorded runs of the cheap rungs would remove both
sources of drift.

---

## 7. Designing the Rungs: Where the Pipeline's Budget Goes

CLIMB decides whether to run the pipeline; how much it saves depends on what the pipeline costs
and what its cheap rungs can do. This section reports the measurements behind the rung design.

### 7.1 Coordination is most of the cost

**Table 4.** Share of total tokens by role, plain pipeline, cap 4,096, 24k, pooled over four
benchmarks. (`analysis/role_decomp.py`)

| Model | planner | executor | reviewer | planner + reviewer |
|---|---:|---:|---:|---:|
| GLM | 21.1% | 47.6% | 31.3% | **52.4%** |
| DeepSeek | 30.3% | 42.2% | 27.5% | **57.8%** |

On reasoning models more than half of a pipeline's budget is its coordinators' own reasoning.
This is what the cheap rungs avoid when CLIMB stops early.

### 7.2 What can be compressed

Within the pipeline, the plan is the largest slot re-sent each round: 41.2% (GLM) and 48.7%
(DeepSeek) of all prompt characters, larger than the question. We compare, at the same stop rule
and cap, carrying the full plan with carrying only the current step, and the free-form critique
with a structured one.

**Table 5.** Accuracy difference from the plain pipeline, 24k, paired exact McNemar.
(`analysis/state_policy.py` and inline scripts)

| Change to the carried state | GLM MATH | GLM GAIA | DeepSeek MATH | DeepSeek GAIA |
|---|---:|---:|---:|---:|
| plan reduced to the current step | **+3.0** (p=0.04) | 0.0 | −1.2 | −4.1 |
| tool output compressed to a 120-token note | +0.8 | 0.0 | −1.2 | 0.0 |
| full history carried (accumulation) | +2.5 | +2.5 | 0.0 | −1.6 |
| structured critique, generation capped at 150 | **−6.2** (p<0.01) | **−13.9** (p<0.01) | −2.0 | **−9.8** (p=0.01) |
| structured critique, generation capped at 1,024 | −0.8 | −4.1 | **−3.5** (p=0.03) | **−7.4** (p=0.04) |

Compressing the plan, the tool output, or the history costs no significant accuracy except one
GLM MATH cell where reducing the plan helps. Compressing the critique's generation is what hurts.
At a 150-token cap, 81 to 98% of reviews on both models are empty, because the reviewer's
reasoning consumes the allowance, and an empty review is parsed as "revise": correct answers are
accepted only 3 to 21% of the time and are then rewritten. Raising the cap to 1,024 recovers most
of the loss on GLM. On DeepSeek, whose reasoning is longer, 17 to 63% of reviews are still empty
at 1,024, and a smaller residual effect of the structured format remains. When it is not
truncated, the structured reviewer discriminates better than the free-form one: on GLM GAIA it
accepts 69% of correct answers and rejects 88% of wrong ones, where the free-form reviewer
accepts most wrong answers.

The design rule is therefore to compress the plan and the content of the critique, and not to
compress the reviewer's generation.

### 7.3 How to size each call

**Table 6.** Plain pipeline, per-call cap 4,096 against 1,024, by budget. Paired exact McNemar;
* p < 0.05.

| | 24k | 16k | 8k |
|---|---:|---:|---:|
| GLM MATH | +5.0* | +5.2* | **−10.2*** |
| GLM GAIA | +9.0* | +1.6 | **−9.8*** |
| DeepSeek MATH | +4.2* | +5.2* | **−4.8*** |
| DeepSeek GAIA | +13.9* | +6.6* | −4.1 |

A larger cap helps under a loose budget and hurts under a tight one, where each round becomes so
expensive that the task is stopped before it answers. Allocating each call's cap in proportion to
the budget, either fixed per role before execution or from the remaining budget at run time,
matches the better fixed cap at both 24k and 8k on both models and both benchmarks we test (no
setting significantly worse), without choosing the cap per budget. The evaluation in Section 6 runs at a 24k budget, where the
better fixed cap is 4,096, and uses it; at tighter budgets proportional caps should be used inside
$L_2$, and we have not evaluated CLIMB in that regime.

---

## 8. Limitations

**Replay calibration.** Thresholds are calibrated on recorded outcomes. A live run (Section 6.5)
confirms the accuracy guarantee but shows that cost can drift from replay, by up to the full
saving on DeepSeek MATH, because cheap answers are regenerated and the seeded pipeline costs a
different amount. The cost bound holds for recorded costs only.

**Calibration data.** CLIMB needs a labelled calibration set from the task distribution. Its
guarantee assumes calibration and deployment tasks are drawn i.i.d. from one distribution and does
not transfer to a shifted one.

**Weak signals limit savings.** At tight tolerances CLIMB saves little, because the confidence
signal is weak. Its guarantee is what prevents this from becoming silent accuracy loss, but a
stronger signal would be needed to save more.

**Two reasoning models, one pipeline.** We study two reasoning models and one plan-execute-review
topology. Other topologies or non-reasoning models may have different cost structures.

**Single seed.** Each item is run once at temperature 0. The guarantee accounts for sampling of
calibration tasks, not for run-to-run variation of the model.

**Small GAIA.** With 122 items, GAIA calibration sets are small, which limits certification.

---

## 9. Conclusion

Running a multi-agent pipeline on every task pays for coordination that many tasks do not need,
but deciding which tasks need it with weak confidence signals risks silent accuracy loss. CLIMB
casts the decision as cost minimisation under an accuracy-loss bound, uses a threshold policy that
is optimal when escalation's gain decreases with confidence, and certifies the threshold with a
finite-sample guarantee. It saves a quarter to three quarters of the pipeline's tokens where its
condition holds, and recognises and declines to save where it does not. Run live, it keeps
accuracy within the tolerance on every setting, while its savings track replay on one model and
shrink on the other.

---

## Appendix A. Monotone-gain test, all cells

See `results/monotone_gain.json` (16 rows). $\rho \le 0$ in 11 of 16; positive on GLM GAIA $L_1$
(+0.05), DeepSeek MATH $L_1$ (+0.21), DeepSeek GAIA $L_0$ (+0.36) and $L_1$ (+0.21), and DeepSeek
FRAMES $L_1$ (+0.31).

## Appendix B. Single-call reference

For reference, a single executor call at the same cap (4,096) and default reasoning, 24k:

| | single call | CLIMB, ε = 5% | pipeline |
|---|---|---|---|
| GLM MATH | 61.3% / 0.7k | 67.7% | 69.2% / 5.1k |
| GLM GAIA | 20.5% / 2.8k | 26.9% | 27.3% / 15.5k |
| GLM FRAMES | 30.2% / 1.4k | 29.4% | 28.8% / 6.8k |
| GLM MuSiQue | 20.6% / 1.6k | 19.2% | 18.7% / 7.0k |
| DeepSeek MATH | 58.0% / 0.7k | 63.4% | 62.0% / 3.5k |
| DeepSeek GAIA | 27.9% / 3.1k | 32.6% | 32.8% / 13.6k |
| DeepSeek FRAMES | 32.6% / 1.5k | 29.3% | 28.9% / 6.4k |
| DeepSeek MuSiQue | 20.9% / 1.9k | 18.3% | 17.3% / 6.9k |

On multi-hop retrieval a single call with default reasoning is as accurate as the pipeline or
more, which is consistent with CLIMB stopping early on most of those items. CLIMB's guarantee is
relative to the pipeline, the system it is designed to make cheaper.

## Appendix C. Scripts and results

| Result | Script | Output |
|---|---|---|
| Table 1, 6.2 | `analysis/risk_gate.py` | `results/risk_gate.json` |
| Table 2 | `analysis/escalation_variants.py`, `analysis/opt_escalation.py` | `results/escalation_variants.json`, `results/opt_escalation.log` |
| Table 3, App. A | `analysis/monotone_gain.py` | `results/monotone_gain.json` |
| Table 4 | `analysis/role_decomp.py` | |
| Table 5 | `analysis/state_policy.py`, inline | `results/state_policy.json` |
| Table 6, 7.3 | `analysis/final_rc.py` | `results/final_rc.json` |
| Table 7 (live) | `analysis/climb_live.py`, `chains/_chain_climb_live.sh` | `results/climb_live.json`, `results/climb_live_plan.json` |

## Appendix D. Proofs

**Setting.** Write $h_k = (S_0, \dots, S_k)$ for the history at rung $k$, and $c_j = C_j - C_{j-1}$ for
the incremental cost of rung $j$. Assumption (A1) states that, given $S_k$, the variables
$(Y_k, \dots, Y_K, c_{k+1}, \dots, c_K, S_{k+1}, \dots, S_{K-1})$ are independent of $h_{k-1}$. We assume
$Y$ and $C$ are integrable, which holds since answers are binary and costs bounded by the budget.

**Proof of Proposition 1, first part.** Fix $\lambda \ge 0$ and let $W_k = Y_k - \lambda C_k$ be the
reward of stopping at rung $k$. Define the optimal continuation values by backward induction:
$V_K = W_K$ and, for $k < K$, $V_k = \max\{W_k^{*}, \mathbb{E}[V_{k+1} \mid h_k]\}$, where
$W_k^* = \mathbb{E}[W_k \mid h_k]$. By standard optimal stopping theory for a finite horizon, the
policy that stops at the first $k$ with $W_k^* \ge \mathbb{E}[V_{k+1} \mid h_k]$ solves (2).

We show by backward induction that $\mathbb{E}[V_{k+1} \mid h_k] - W_k^*$ depends on $h_k$ only through
$S_k$. The sunk cost $C_{k-1}$ enters $W_k$ and every later reward identically, so it cancels from
the difference, leaving a function of $Y_k, \dots, Y_K$, of $c_k, \dots, c_K$ and of the later stopping
decisions. By the induction hypothesis the later decisions depend only on later scores, so by (A1)
the conditional expectation given $h_k$ equals that given $S_k$. The difference is therefore
$G_k^\lambda(S_k)$. The optimal decision at rung $k$ is to stop iff $G_k^\lambda(S_k) \le 0$; by (A2)
the set $\{s : G_k^\lambda(s) \le 0\}$ is an upper interval with left end $t_k(\lambda) = \inf\{s :
G_k^\lambda(s) \le 0\}$. On the set where $G_k^\lambda = 0$ the system is indifferent and either
decision is optimal. $\square$

**Proof of Proposition 1, second part.** Let $\Pi$ be the set of randomised stopping policies,
that is, probability mixtures of deterministic ones. Cost $J(\pi) = \mathbb{E}[C_{\tau_\pi}]$ and risk
$R(\pi)$ are linear in the mixture weights, so (1) is a linear program over the convex set $\Pi$
with one inequality constraint. The always-escalate policy has $R = 0 < \varepsilon$, so Slater's
condition holds and there is no duality gap: there is $\mu^* \ge 0$ such that every solution of (1)
minimises $J(\pi) + \mu^*(R(\pi) - \varepsilon)$ over $\Pi$ and satisfies complementary slackness.
Up to constants, this Lagrangian equals $-\mu^*\,\mathbb{E}[Y_{\tau_\pi} - \lambda^* C_{\tau_\pi}]$ with
$\lambda^* = 1/\mu^*$, so its minimisers are the solutions of (2) at $\lambda^*$, and their
mixtures. (If $\mu^* = 0$ the constraint is slack and the cheapest policy, stopping at rung 0, is
optimal; it is a threshold policy with $t_0 = -\infty$.) By the first part, the deterministic
solutions of (2) at $\lambda^*$ are threshold policies that differ only on the indifference set.
Complementary slackness is met by mixing the least and the most aggressive of them, so a mixture of
two threshold policies solves (1). $\square$

The proposition concerns the population problem. CLIMB searches a finite grid of thresholds, ties
the cheap rungs' thresholds, and uses deterministic policies, so it approximates the optimum to the
grid's resolution; Section 6.3 measures the effect of tying.

**Proof of Proposition 2.** *Validity of each p-value.* Under $\mathcal{H}_m$, $q = \mathbb{E}[H_{t_m}] >
\varepsilon$ and the number of harms is $k_m \sim \mathrm{Bin}(n, q)$. Let $F$ be the CDF of
$\mathrm{Bin}(n, \varepsilon)$. Since $F$ is non-decreasing and $\mathrm{Bin}(n, q)$ stochastically
dominates $\mathrm{Bin}(n, \varepsilon)$,
$\Pr[F(k_m) \le \delta] \le \Pr[F(K_\varepsilon) \le \delta] \le \delta$ for $K_\varepsilon \sim
\mathrm{Bin}(n, \varepsilon)$, the last step because a random variable's CDF evaluated at itself is
stochastically no smaller than a uniform.

*Family-wise error.* Let $m^*$ be the first index, in the fixed order, at which $\mathcal{H}_m$ is
true; it depends on the distribution only. The procedure rejects $\mathcal{H}_m$ only if it rejects
every earlier hypothesis, so rejecting any true null requires rejecting $\mathcal{H}_{m^*}$, which
has probability at most $\delta$.

*Risk.* On the complementary event, which has probability at least $1 - \delta$, every threshold in
$\mathcal{C}$ satisfies $\mathbb{E}[H] \le \varepsilon$, and $\varnothing$ has $H \equiv 0$. Since
$Y_K - Y_\tau \le H$ pointwise, $R \le \mathbb{E}[H] \le \varepsilon$ for every member of $\mathcal{C}$,
and hence for the deployed one, however it is selected from $\mathcal{C}$.

*Cost.* The deployed policy minimises mean calibration cost over $\mathcal{C}$, which contains
$\varnothing$, whose mean calibration cost is that of the full pipeline. $\square$
