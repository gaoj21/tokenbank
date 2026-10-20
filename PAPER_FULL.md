# When to Call the Team: Risk-Controlled Escalation for Multi-Agent LLM Pipelines

*Full draft, 2026-10-04. Method paper. Two reasoning models (GLM-5.3-Flash, DeepSeek-V4.1-Flash),
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
condition behind our analysis holds in 11 of 16 (benchmark, rung) pairs, failing most clearly in the
one setting where CLIMB declines to save.

<!-- Contributions -->
Our contributions are as follows.
1. We formulate the decision of when to invoke a multi-agent pipeline as constrained sequential
   stopping, and prove that under a testable monotone-gain condition its optimal solution is a
   confidence-threshold policy (Sections 3 and 4.3).
2. We propose CLIMB, a training-free escalation method that calibrates the threshold by
   multiple hypothesis testing and guarantees, with high probability, that accuracy loss relative
   to the full pipeline stays within a user-specified tolerance (Section 4.4).
3. We evaluate CLIMB in replay and live deployment on two reasoning models and four benchmarks,
   showing when it saves, when and why it declines to, and that richer policies do not save more
   under the same guarantee (Section 6).
4. We measure where a multi-agent pipeline's budget goes and which parts of its state can be
   compressed without loss, which informs the design of the escalation ladder (Section 7).

---

## 2. Related Work

**Multi-agent LLM systems.** A large body of work organises several LLM instances into roles that
plan, solve, critique and revise: debate among peers (Du et al., 2023; Liang et al., 2024),
conversational frameworks (AutoGen; Wu et al., 2023), role-playing societies (CAMEL; Li et al.,
2023), and software-engineering teams with standardised operating procedures (MetaGPT, Hong et
al., 2024; ChatDev, Qian et al., 2024). Self-refinement (Madaan et al., 2023) and reflection
(Shinn et al., 2023) are the single-agent analogues of the execute-review loop we study. These
systems are typically evaluated at a fixed configuration on every task. Our work takes such a
pipeline as given and asks on which tasks it should run at all.

**Efficiency of multi-agent systems.** Several methods reduce the cost of a pipeline that runs on
every task. AgentPrune (Zhang et al., 2025) and AgentDropout (Wang et al., 2025) prune redundant
communication edges or agents; Optima (Chen et al., 2025) trains agents to communicate more
concisely; recent analyses decompose where agent tokens go and find that context re-sent across
turns dominates (arXiv:2604.22750; arXiv:2609.23790) and that efficiency comparisons are sensitive
to generation limits (arXiv:2609.05933). These approaches shrink the per-task cost of the team.
CLIMB is orthogonal: it decides whether to invoke the team, and can be placed in front of any of
them.

**Budgets for agents.** Another line allocates a fixed token budget within an agentic pipeline.
ZEBRA (arXiv:2605.20485) estimates per-phase utility curves with an LLM and solves a continuous
knapsack by water-filling on the Lagrange multiplier. BATS makes a tool-using agent aware of its
remaining budget, and CoRL (arXiv:2511.02755) trains a controller with reinforcement learning to
respect a budget. These methods decide how to spend inside a pipeline that always runs, and none
offers a guarantee on the accuracy cost of saving. Our formulation shares ZEBRA's use of a
Lagrangian price on tokens, but the decision variable is when to stop escalating rather than how
to split a budget, and the price is not estimated but replaced by a calibrated threshold with a
finite-sample guarantee.

**Cascades and routing.** Model cascades answer with a cheap model and defer to an expensive one
on uncertain inputs. FrugalGPT (Chen et al., 2023) learns a scoring function and thresholds for a
cascade of API models; AutoMix (Madaan et al., 2024) uses few-shot self-verification to decide
whether to route; RouteLLM (Ong et al., 2025) and Hybrid LLM (Ding et al., 2024) train routers
from preference or quality data; Yue et al. (2024) use answer consistency across sampled
reasoning to decide when to call a stronger model. Signed Rescue Routing (arXiv:2609.07786)
argues that uncertainty is the wrong routing target, since escalation pays off only when it
repairs an error, and trains predictors of rescue and harm. Three features distinguish our
setting. The rungs are configurations of one model (direct, verified, multi-agent) rather than
different models; outputs are free-form answers without a fixed label space; and we seek a
guarantee relative to the most expensive rung rather than a good average trade-off. We also find,
in contrast to the trend toward learned routers, that with a weak confidence signal and small
calibration sets a single calibrated threshold saves more than learned estimates of the value of
escalating, and we give a condition under which thresholds are optimal.

**Confidence of LLM outputs.** Token log-probabilities, entropy, verbalised confidence and
sample consistency have all been studied as signals of correctness (Kadavath et al., 2022;
Kuhn et al., 2023; Xiong et al., 2024; Wang et al., 2023). They are informative but weak,
particularly for reasoning models whose final answers follow long hidden reasoning. CLIMB takes
the signal as given: its guarantee does not require the signal to be calibrated, and a weak signal
only makes the selected policy more conservative.

**Distribution-free risk control.** Conformal prediction (Vovk et al., 2005; Angelopoulos and
Bates, 2023) provides finite-sample coverage guarantees without distributional assumptions.
Conformal risk control (Angelopoulos et al., 2024) extends them to monotone losses, and
Learn-then-Test (Angelopoulos et al., 2021) to arbitrary losses by casting parameter selection as
multiple hypothesis testing. These tools have been applied to language generation, for example to
sampling until an admissible answer appears (Quach et al., 2024) and to filtering unsupported
claims (Mohri and Hashimoto, 2024), and Conformal Cascade (arXiv:2607.25018) applies them to
multi-tier model cascades. We use Learn-then-Test with fixed-sequence testing and an exact
binomial test on a one-sided harm bound, which we find is what makes certification possible at
realistic calibration sizes, and we pair it with an optimality analysis that justifies the
one-parameter family being tested.

*(Citation note for the authors: arXiv identifiers are given where we did not record author
names; all references should be verified against the published versions before submission.)*

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

CLIMB has four components: a ladder of increasingly expensive configurations of one model
(Section 4.1), a confidence score observed after each cheap rung (Section 4.2), a structural
result that restricts the stopping policy to a one-parameter family (Section 4.3), and a
risk-controlled procedure that selects a member of that family from calibration data
(Section 4.4). Section 4.5 states what the resulting guarantee does and does not cover, and
Section 4.6 describes how the measurements of Section 7 shaped the rungs. Figure 1 gives an
overview, and Algorithm 1 summarises calibration and deployment. A theme runs through the
design: the optimality of the threshold family rests on a condition that can fail, but the
validity of the guarantee does not, so a poor signal or a violated condition costs savings, never
accuracy beyond the tolerance.

> *[Figure 1, to be drawn (see `METHOD_FIGURE.md`): left, the ladder $L_0 \to L_1 \to L_2$ with a
> stop/continue gate after each cheap rung; right, calibration: recorded outcomes on the
> calibration set, the fixed sequence of tests from conservative to aggressive thresholds, the
> certified set, and the cheapest member deployed.]*

### 4.1 The escalation ladder

All rungs use the same model; they differ in how many roles are involved, how much the model is
allowed to reason, and what state is carried between calls. Every rung asks for the final answer
on a last line of the form `ANSWER: <answer>`, from which the answer is extracted.

**Rung 0: direct answer ($L_0$).** A single executor call with the question. The model may
emit a Python block for computation; if it does, the block is run in a separate Python process (10-second limit,
imports of `os`, `sys` and `subprocess`, file access and `eval`/`exec` rejected, output truncated
to 1,000 characters) and the output
is returned for one follow-up call that gives the final answer. $L_0$ uses the model's cheap
reasoning setting and a 2,048-token generation cap per call.

**Rung 1: verified answer ($L_1$).** A second executor pass that receives $L_0$'s answer as a
draft with the instruction to verify it, correct it if wrong, and answer. It has the same tool,
reasoning setting and cap as $L_0$. $L_1$ is a cheap form of self-verification; it does not
involve other roles.

**Rung 2: the multi-agent pipeline ($L_2$).** A planner decomposes the question into a
step-by-step plan. The executor receives the question, the plan and, when $L_2$ is reached by
escalation, the draft from the cheap rungs, and answers with the same Python tool. A reviewer
receives the question and the proposed answer and judges whether it is correct. If it accepts, the
pipeline stops; otherwise its feedback is returned to the executor, which revises. The loop runs
for at most five rounds. Between rounds the executor's context is rebuilt rather than accumulated:
it contains the question, the plan, the draft, and only the most recent answer and review. All
three roles use the model's default reasoning setting and a 4,096-token cap per call.

**Budget enforcement.** Every task runs under a hard budget of $B = 24{,}000$ tokens, counting
prompt and completion tokens, including hidden reasoning tokens, of every call at every rung. A
call is admitted only if its estimated cost fits in the remaining budget minus a closing reserve
of 256 tokens. When a call is refused, the reserve pays for one short closing call that returns
the best available answer, so every task ends with an answer.

**Reasoning settings.** The two models expose different controls. For GLM-5.3-Flash the cheap
setting is `reasoning_effort: low`; for DeepSeek-V4.1-Flash it is to disable thinking through the
chat template (`thinking: false`). The two controls must not be combined: on DeepSeek, sending a
reasoning-effort field re-enables thinking. Rung 2 uses each model's default.

**Table 1.** The escalation ladder.

| Rung | Roles | Calls per task | Reasoning | Cap per call | Input state |
|---|---|---|---|---|---|
| $L_0$ | executor | 1 to 2 | cheap | 2,048 | question |
| $L_1$ | executor | 1 to 2 | cheap | 2,048 | question, $L_0$ draft |
| $L_2$ | planner, executor, reviewer | 3 to 16 | default | 4,096 | question, plan, draft, latest answer and review |

### 4.2 Confidence score

After $L_0$ and after $L_1$, the system computes a confidence score from the rung's own final call,
using log-probabilities returned by the inference server at no extra cost. The score $S_k$ is the
mean log-probability of the generated tokens. As a veto, the mean entropy over generated tokens,
computed from the top-5 alternatives returned at each position, must not exceed $\tau = 0.6$ nats: a rung whose
output is uncertain at many positions never stops, whatever its mean log-probability. If the
server returns no log-probabilities, the rung escalates. We use this score because it is free,
available for every rung and model, and the kind of signal a practitioner would reach for; the
method does not depend on its particular form, and Section 6.3 compares it with richer signals.

### 4.3 Structure of the optimal policy

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
boundary. The full proof is in Appendix A.

Condition (A2) is the substantive one. It says that confidence orders tasks by how much they stand
to gain from the multi-agent pipeline, and it is testable on data that records every rung's
outcome; Section 6.4 tests it. Where it fails, the threshold family need not contain the optimum,
but the guarantee of Section 4.4 still holds, and in that case calibration falls back to the full
pipeline rather than accept an unsafe threshold.

**Tying the cheap rungs.** Proposition 1 allows a separate threshold per rung. Because the
expected incremental cost varies little within a benchmark, we tie the thresholds of the two cheap
rungs, $t_0 = t_1 = t$, reducing the policy family to a single parameter that is ordered from
conservative to aggressive. This ordering is what the calibration procedure of Section 4.4
requires. Section 6.3 shows that untying the thresholds, or replacing the threshold rule by
richer estimates of $G_k^\lambda$, does not lower cost under the same guarantee.

### 4.4 Risk-controlled calibration

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

**Algorithm 1.** CLIMB calibration and deployment.

```
Input:  calibration tasks x_1..x_n with recorded outcomes Y_0, Y_1, Y_2 and costs of every rung;
        tolerance eps; error level delta; threshold grid t_1 > t_2 > ... > t_M
Calibration:
  C <- { ∅ }                                   # ∅ = always run the full pipeline (risk 0)
  for m = 1..M:
      k_m <- #{ i : Y_2(x_i) = 1 and Y_{tau(t_m)}(x_i) = 0 }    # harms of threshold t_m
      p_m <- BinomCDF(k_m ; n, eps)
      if p_m <= delta:  C <- C ∪ { t_m }
      else:             break                  # fixed-sequence: stop at first non-rejection
  t* <- argmin_{t in C} mean_i cost_t(x_i)     # cheapest certified setting
Deployment on a new task x:
  if t* = ∅: run L_2 and return its answer
  run L_0; if S_0 > t* and H_0 <= 0.6: return L_0 answer
  run L_1 (seeded with L_0); if S_1 > t* and H_1 <= 0.6: return L_1 answer
  run L_2 (seeded with the draft) and return its answer
```

Here $\tau(t)$ is the stopping rung of the threshold policy $t$ on a calibration task, which is
computed from the recorded scores without running anything, and $\mathrm{cost}_t$ is the recorded
cumulative cost of the rungs that policy would run. Calibration requires one pass of every rung
over the calibration set and is otherwise free. The grid has 120 thresholds from $-0.005$ to
$-0.600$ in steps of $0.005$, and we use $\delta = 0.1$ throughout.

**Proposition 2 (risk control).** If calibration and test tasks are drawn i.i.d. from the same
distribution, then with probability at least $1 - \delta$ over the calibration set, every policy in
$\mathcal{C}$, and in particular the deployed one, satisfies $R \le \varepsilon$. The deployed policy's
mean cost on the calibration set is at most that of the full pipeline.

Testing in a fixed order controls the family-wise error rate at $\delta$ without a multiplicity
correction, because a true null can be rejected only if the first true null in the sequence is,
and each $p_m$ is valid by the exactness of the binomial test. On the event that no true null is
rejected, every member of $\mathcal{C}$ has $\mathbb{E}[H] \le \varepsilon$ and hence risk at most
$\varepsilon$, whichever member is then selected; choosing the cheapest therefore costs nothing in
validity. The cost statement holds because $\varnothing \in \mathcal{C}$. Proofs are in Appendix A.

Two properties of this construction matter in practice. First, the guarantee requires neither
(A1), (A2) nor any calibration of the confidence score: a poorly informative score simply leads
to fewer rejections and a more conservative policy. Second, the cost statement concerns the rung
costs observed at calibration. If a rung costs a different amount at deployment, as when the
pipeline receives the cheap rungs' draft (Section 6.5), the cost bound need not carry over, while
the accuracy guarantee is unaffected as long as the deployed rungs behave as calibrated.

The exactness of the test is not a technicality. A Hoeffding bound on the loss itself rejects no
threshold at our calibration sizes, because it ignores that almost all losses are zero.

### 4.5 What the guarantee covers

The guarantee is a statement about the population of tasks from which the calibration set was
drawn: with probability at least $1 - \delta$ over the draw of the calibration set, the deployed
policy's expected accuracy loss relative to the full pipeline is at most $\varepsilon$. Four
qualifications follow from its form.

*It is relative to the full pipeline.* CLIMB guarantees that it is not much worse than always
calling the team. It says nothing about whether the team is better than a single call, which on
some tasks it is not (Appendix C).

*It is an expectation over tasks.* Individual tasks may be answered wrongly where the pipeline was
right; the guarantee bounds how often this happens, in expectation, net of the opposite case.

*It assumes the deployed rungs behave as calibrated.* The decision at each rung is computed from
the deployed rung's own score. If the rungs' behaviour shifts between calibration and deployment,
for example because the model or prompts change, calibration must be repeated.

*It does not bound cost.* The cost statement of Proposition 2 is in-sample. Section 6.5 shows that
deployed cost can differ from calibrated cost when the seeded pipeline costs a different amount.

**Cost of calibration.** Calibrating requires labelled outcomes of every rung on the calibration
set, which costs one run of the full ladder per calibration task. With 60 to 967 calibration tasks
per setting in our experiments, this is a one-time cost of the same order as evaluating the
pipeline on the calibration set.

### 4.6 Designing the rungs

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

### 5.1 Models and serving

We use two open-weight reasoning models from different families. **GLM-5.3-Flash** is served in
FP8 with SGLang; **DeepSeek-V4.1-Flash** is served with vLLM. Both are queried through an
OpenAI-compatible API with temperature 0 and with token log-probabilities and the top-5
alternatives returned for every generated token. Reasoning settings are as in Section 4.1. Every
call in every system, including the baselines, has a per-call generation cap: 2,048 tokens on the
cheap rungs and 4,096 elsewhere. We fix the cap because it matters: with reasoning models a small
cap truncates answers, and comparisons across systems are only meaningful at a common cap
(Section 7.3).

### 5.2 Benchmarks

We use four benchmarks spanning three task types. Before any experiment, each benchmark was split
once at random (seed 42) into a development split (20%), used while building the pipeline and
choosing its prompts, and a test split (80%), used for every number reported in this paper.

| Benchmark | Task type | Source split | Test items | Scoring |
|---|---|---|---:|---|
| MATH-500 | competition mathematics | test | 400 | symbolic equivalence (math-verify) |
| GAIA | tool-assisted QA | validation, levels 1 to 3 | 121 / 122 | normalised exact match, order-free for lists |
| FRAMES | multi-hop factual QA | test | 660 | normalised exact match |
| MuSiQue | multi-hop QA, answerable | validation | 1,934 | normalised exact match |

GAIA tasks with image or audio attachments are excluded, since both models are text-only; text
attachments are appended to the question. One GLM GAIA task lacks a complete record of both cheap
rungs and is excluded from all GLM GAIA comparisons, leaving 121 tasks. FRAMES and MuSiQue are evaluated
closed-book: the models receive the question only, without Wikipedia pages or supporting
paragraphs, so they measure multi-hop recall and reasoning rather than reading. Accuracy is the
share of tasks answered correctly; for FRAMES and MuSiQue we report exact match, and token-level
F1 is recorded but not used.

### 5.3 Cost

Cost is the number of tokens charged to a task: prompt plus completion tokens, including hidden
reasoning tokens, summed over every call at every rung the task reaches. We report cost in
tokens rather than money because both models are served locally; for API-priced models, the same
analysis applies with prices as weights.

### 5.4 Evaluation protocols

**Replay (Sections 6.1 to 6.4).** For every test item we record the outcome and cost of every
rung. $L_0$ and $L_1$ come from a run in which every item passes through both cheap rungs
regardless of confidence (all thresholds disabled); $L_2$ comes from a separate run of the full
pipeline on the same item. Any threshold policy can then be evaluated on any subset of items by
replaying these records, which is the standard protocol for cascade evaluation and lets us
compare many policies on identical items. One difference from deployment is that the recorded
$L_2$ runs without a draft. For each setting we draw 200 random splits of the test items into a
calibration half and an evaluation half, stratified by benchmark; CLIMB is calibrated on the
first half and evaluated on the second, and we report means over splits and the share of splits
in which the evaluation-half accuracy loss exceeded $\varepsilon$ (the violation rate, which the
guarantee bounds by $\delta$). Policies that need training data (Section 6.3) use a 40/20/40 split
into training, calibration and evaluation.

**Live deployment (Section 6.5).** For each setting we fix one random split. The threshold is
calibrated on the calibration half from recorded outcomes only. CLIMB is then run live on the
evaluation half: $L_0$ and $L_1$ are generated afresh, and items that escalate run the full
pipeline seeded with the cheap rungs' draft. Live results are compared with the recorded
pipeline on the same items.

### 5.5 Baselines

* **Full pipeline.** $L_2$ on every task; the reference for the guarantee.
* **Fixed threshold.** The same ladder with a hand-set threshold $t = -0.10$ on both cheap rungs,
  the value our system used for $L_0$ before this work, chosen on the development split. It
  represents an uncertified cascade.
* **Richer escalation policies under the same guarantee.** Separate thresholds per rung,
  answer-agreement stopping, a non-parametric Bayes rule and learned value estimates (Section
  6.3). Each is calibrated with the same procedure, so differences are in cost only.
* **Single call** (Appendix C). One executor call with default reasoning, for reference.

### 5.6 Statistics

Paired accuracy differences on the same items are tested with an exact McNemar test (two-sided
binomial test on discordant pairs). We report $p$-values without multiple-comparison correction
and mark $p < 0.05$. All runs use one seed at temperature 0; Section 8 discusses the
consequences.

---

## 6. Results

### 6.1 Main result

**Table 2.** CLIMB against the full pipeline, 24k budget, cap 4,096. Accuracy and tokens as a
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

**Table 3.** Tokens as a share of the pipeline's at ε = 5%, same splits for all policies; lower is
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
monotonicity. Table 4 estimates the proxy by confidence quintile.

**Table 4.** Gain from escalating to the pipeline, $\Pr[Y_K] - \Pr[Y_k]$ in points, by quintile of
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
Appendix B. It fails most clearly on DeepSeek GAIA, where the most confident cheap answers are
the ones the pipeline improves most. That is the setting in which CLIMB saves nothing: where the
condition fails, no threshold is both cheap and safe, calibration detects this, and the method
falls back to the pipeline.

### 6.5 Live deployment

Replay assumes that each rung behaves at deployment as it did when its outcome was recorded. To
test this, we fix one random split per setting, certify the threshold on the calibration half from
recorded outcomes only ($\varepsilon = 5\%$, $\delta = 0.1$), and then run CLIMB live on the test
half: $L_0$ and $L_1$ are generated afresh, and items that escalate run the full pipeline seeded with
the cheap rungs' draft. We compare with the plain pipeline's recorded outcomes on the same items.

**Table 5.** Live CLIMB on the test halves. Tokens as a share of the pipeline's on the same items.
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

**Table 6.** Share of total tokens by role, plain pipeline, cap 4,096, 24k, pooled over four
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

**Table 7.** Accuracy difference from the plain pipeline, 24k, paired exact McNemar.
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

**Table 8.** Plain pipeline, per-call cap 4,096 against 1,024, by budget. Paired exact McNemar;
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

## Appendix A. Proofs

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

## Appendix B. Monotone-gain test, all settings

Gain from escalating to the full pipeline, $\Pr[Y_2] - \Pr[Y_k]$ in points, by quintile of rung
$k$'s confidence (Q1 least confident), and Spearman's $\rho$ between quintile index and gain.
(`analysis/monotone_gain.py`, `results/monotone_gain.json`)

| Model | Benchmark | Rung | Q1 | Q2 | Q3 | Q4 | Q5 | ρ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| GLM | MATH | $L_0$ | +10.0 | +12.5 | +3.8 | +2.5 | +2.5 | -0.87 |
| GLM | MATH | $L_1$ | +12.5 | +8.8 | -2.5 | +5.0 | +1.2 | -0.70 |
| GLM | GAIA | $L_0$ | +12.5 | -8.3 | +0.0 | +4.2 | -4.0 | -0.30 |
| GLM | GAIA | $L_1$ | +4.2 | -8.3 | +0.0 | +4.2 | +4.0 | +0.05 |
| GLM | FRAMES | $L_0$ | +6.8 | +3.8 | -1.5 | -6.8 | +0.8 | -0.70 |
| GLM | FRAMES | $L_1$ | +0.8 | +3.8 | +0.0 | -0.8 | -1.5 | -0.90 |
| GLM | MuSiQue | $L_0$ | +2.1 | +0.0 | +3.4 | -2.1 | -1.8 | -0.60 |
| GLM | MuSiQue | $L_1$ | +3.4 | +1.6 | -5.2 | -2.6 | +0.0 | -0.60 |
| DeepSeek | MATH | $L_0$ | +2.5 | -2.5 | -6.2 | -1.2 | +1.2 | -0.10 |
| DeepSeek | MATH | $L_1$ | -5.0 | +1.2 | -5.0 | -1.2 | -3.8 | +0.21 |
| DeepSeek | GAIA | $L_0$ | +0.0 | +4.2 | +16.0 | +0.0 | +8.0 | +0.36 |
| DeepSeek | GAIA | $L_1$ | +8.3 | +0.0 | +8.0 | +0.0 | +16.0 | +0.21 |
| DeepSeek | FRAMES | $L_0$ | +3.0 | +10.6 | +0.8 | -1.5 | -4.5 | -0.90 |
| DeepSeek | FRAMES | $L_1$ | +0.0 | -0.8 | -3.0 | +0.8 | +0.0 | +0.31 |
| DeepSeek | MuSiQue | $L_0$ | +5.7 | -0.5 | +1.8 | -0.3 | -4.1 | -0.70 |
| DeepSeek | MuSiQue | $L_1$ | +0.0 | +0.5 | -3.9 | +1.3 | -2.8 | -0.10 |

$\rho \le 0$ in 11 of 16 rows. Quintiles contain 24 to 387 items, so individual cells are noisy,
particularly on GAIA.

## Appendix C. Single-call reference

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

## Appendix D. Prompts

All prompts are reproduced verbatim. `{q}` is the question, `{d}` a draft answer, `{a}` a
proposed answer, `{r}` a review, `{out}` tool output. The answer-format suffix
`End your reply with a single line: "ANSWER: <final answer>".` is appended to every executor
system prompt.

**Executor, $L_0$ and $L_1$ (system).**
`You are an Executor. Answer concisely. End your reply with a single line: "ANSWER: <final answer>". For ANY computation, put Python code that prints the result in a ```python block; you will receive the output, then give the final answer.`

**Executor, $L_0$ (user).** `Question: {q}` / `Final answer.`

**Executor, $L_1$ (user).** `Question: {q}` / `Draft answer: {d}` / `Verify it, fix it if wrong, then answer.` / `Final answer.`

**Tool follow-up (user, any rung).** `[Computation result]` / `{out}` / `Now give the final answer.`

**Planner, $L_2$.** System: `You are a Planner. Break down the question into sub-tasks.`
User: `Question: {q}` / `Provide a step-by-step plan.`

**Executor, $L_2$ (user).** `Question: {q}` / `Plan: {plan}` / `Draft answer: {d}` (only when
reached by escalation) / `Final answer.`

**Reviewer, $L_2$.** System: `You are a Reviewer. Validate the answer.` User: `Question: {q}` /
`Proposed: {a}` / `Correct? YES/NO`. The answer is accepted if the review contains "yes"
(case-insensitive). This parser is deliberately simple; Section 7.2 discusses a structured
alternative and why it fails under tight generation caps.

**Revision turn (appended to the rebuilt executor context).** Assistant: `{a}`. User:
`Reviewer feedback: {r}` / `Revise and give the final answer.`

**Closing call (budget exhausted, 256-token cap).** `Budget exhausted. Give your best final answer now, concisely.`

## Appendix E. Scripts and results

| Result | Script | Output |
|---|---|---|
| Table 2, 6.2 | `analysis/risk_gate.py` | `results/risk_gate.json` |
| Table 3 | `analysis/escalation_variants.py`, `analysis/opt_escalation.py` | `results/escalation_variants.json`, `results/opt_escalation.log` |
| Table 4, App. A | `analysis/monotone_gain.py` | `results/monotone_gain.json` |
| Table 6 | `analysis/role_decomp.py` | |
| Table 7 | `analysis/state_policy.py`, inline | `results/state_policy.json` |
| Table 8, 7.3 | `analysis/final_rc.py` | `results/final_rc.json` |
| Table 5 (live) | `analysis/climb_live.py`, `chains/_chain_climb_live.sh` | `results/climb_live.json`, `results/climb_live_plan.json` |
| Table 3 (live) prompts, ladder | `_run_v2_experiment.py` (`run_gaia_task_v5`), `_budget_sweep_v2.py` (`CLIMB-Live`) | |
| Richer policies | `analysis/escalation_variants.py`, `analysis/opt_escalation.py` | `results/escalation_variants.json` |

