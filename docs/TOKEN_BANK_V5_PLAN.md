# Token Bank v5 改进计划

> 版本：v5 draft
> 日期：2026-08-18
> 基于：`experiment_findings_v3.md`（Qwen3.6-27B-FP8 × Gemma-4-31B 双模型验证）
> 目标：**在给定 token 预算下最大化正确数**

---

## 目录

- [0. 为什么要改](#0-为什么要改)
- [1. 设计骨架](#1-设计骨架)
- [Phase 0：零成本诊断](#phase-0零成本诊断)
- [Phase 1：评测地基](#phase-1评测地基)
- [Phase 2：机制重构](#phase-2机制重构)
- [Phase 3：实验矩阵](#phase-3实验矩阵)
- [Phase 4：时间线与门控](#phase-4时间线与门控)
- [Phase 5：风险与回退](#phase-5风险与回退)
- [附录 A：v3 结果的主要疑点](#附录-av3-结果的主要疑点)
- [附录 B：新增 instrumentation 字段清单](#附录-b新增-instrumentation-字段清单)

---

## 0. 为什么要改

### 0.1 诊断

现有七个机制中，**五个**（① final_reserve、② role_credit_mult、③ predictive_est、⑤ predictive_lending，加上跨 agent 借贷本身）都在优化同一个维度——**单任务内、agent 之间的配额搬运**。

这一维恰恰是可压缩空间最小的，v3 数据已经自证：

| 证据 | 数值 |
|---|---|
| Improved（①~⑤）在 GAIA 16K 的增益 | +0.6pp（噪声内） |
| 把 ①~⑤ 叠进 Compact 之后 | 24.3% vs 纯 Compact 26.3% = **−2.0pp** |
| 真正有效的两个机制（⑥⑦） | 都不在这一维上 |

> **结论**：改进方向不是把调度器做得更精细，而是**换维度**。

### 0.2 目标函数的重述

现在的隐含目标是"在每题 24K 内跑完管线"。真正的目标应该是：

> 给定总预算 $T$，为每个任务 $i$ 选择执行策略 $\pi_i$（组件组合 × 轮数 × 采样数），
> 最大化 $\sum_i P(\text{correct} \mid \pi_i)$，s.t. $\sum_i c(\pi_i) \le T$。

这是一个**带随机成本的背包问题**。写出来之后，一堆启发式立刻有了骨架。

---

## 1. 设计骨架

从目标函数推出四条原则，以及各自的机制归属：

| # | 原则 | 对应机制 | v3 现状 |
|---|---|---|---|
| P1 | 边际收益归零就停 | **M1 收敛早停** | ④ 的语法代理版（Jaccard），很弱 |
| P2 | 边际收益为负的组件跳过 | **M2 级联路由** | **完全没有** |
| P3 | 拉平各任务的边际收益 | **M4 跨任务池化** | **完全没有**（等分配额） |
| P4 | 同样信息用更少 token 携带 | **M3 上下文经济学** | ⑦ 做了一半 |
| P5 | 深度与广度按任务选型 | **M5 深度换广度** | **完全没有**（固定 5 轮 refine） |

**七个机制砍到三到四个、每个都有清晰归因，比七个机制更强。**

一句话叙事升级：从"预算治理"改为
> **决定要不要花，比决定花给谁更重要。**

---

## Phase 0：零成本诊断

**不跑任何新推理**，全部从已有的 `traces/tasks_sweep_*.jsonl` 计算。

> ⚠️ **前置条件**：若现有 trace 缺少逐轮 token 记账和逐轮答案快照，Phase 0 无法进行。
> 那么第一件事是给 `_run_v2_experiment.py` 加 instrumentation（见附录 B），
> 然后**只重跑 Unconstrained 一遍 GAIA + MuSiQue**（约 3h）。这笔钱必须花。

### D1 — 收敛轮次分布（最高优先级）

对 Unconstrained 的每个任务，逐轮判定答案是否正确，输出：

| 字段 | 含义 |
|---|---|
| `first_correct_round` | 首次答对的轮次（未答对记 ∞） |
| `final_round` | 实际结束轮次 |
| `flip_rate` | 第 k 轮对、最终错的任务比例 |
| `wasted_tokens` | `tokens(first_correct_round .. final_round)` |

**判据**：`median(first_correct_round) ≤ 2` 且 `wasted_tokens / total ≥ 25%`
→ **M1 早停是最大单点收益，最高优先级**。

> `flip_rate` 同时是 R2 机制假说（"过度修订把正确答案改错"）的**直接证据**，
> 比现在靠 3 道题的边际 SR 差值强得多。

### D2 — 各 flag 触发率

对 IC 的每个任务统计实际触发次数：

- borrowing 发生次数
- `final_reserve` 被动用次数
- stall 触发次数
- `kv_trigger` 触发次数
- `role_credit_mult` 实际改变行为的次数

**判据**：触发率 < 10% 的机制 → **Phase 2 直接删除**，不再做消融。

> 预期借贷类会落在这一档：24K 预算 vs 13.4K 平均消耗，配额对多数任务根本不绑定。
> 这也解释了 ①~⑤ 为什么是噪声——物理上不可能移动 SR 超过一两个点。

### D3 — stall 假阳率

取 IC 中被 stall 掐掉的任务集合 $S$，计算这些任务在 Unconstrained 下的正确率。

**判据**：$P(\text{correct} \mid \text{stalled, unconstrained}) > 15\%$
→ stall 阈值过激进，M1 必须带回滚机制。

### D4 — token 去向拆解

把每个任务的 token 按 **role × 类型（prompt / completion / tool output）** 拆开。回答：

- reviewer 吃掉多少？
- 工具原文占 prompt 的多少？
- plan 重复回传占多少？
- **`kv_trigger` 减少的到底是计费 token 还是显存？**（若只减显存，它不该和 ⑦ 归为一类"压缩机制"，R4 的归因会混）

这张表直接给 M3 的四个子项排序。

### D5 — 难度方差

统计每题 `first_correct_round` 与 token 消耗的分布，计算变异系数 $CV = \sigma/\mu$。

| CV | 结论 |
|---|---|
| > 0.8 | **M4 池化收益大**，优先做 |
| 0.4 ~ 0.8 | 池化值得做，但排在 M1/M2 之后 |
| < 0.4 | 池化不值得，**Phase 3 砍掉 M4** |

---

## Phase 1：评测地基

与 Phase 0 并行。**不修这些，后面所有数字都不可信。**

### P1.1 判题器

| 数据集 | 现状 | 改为 |
|---|---|---|
| MATH-500 | 包含匹配 | `math-verify` / sympy 等价判定 |
| GAIA | 包含匹配 | 官方 normalizer（数字/单位/列表规范化后 EM） |
| FRAMES | `gt in ans` 或前两词命中 | EM + F1 双报 |
| MuSiQue | 同上 | EM + F1 双报 |

**废弃"gt 前两词命中"规则**——它系统性偏袒长答案。

只需重跑 judge，不用重跑推理（几分钟）。
**预期所有 SR 会下降，尤其 MATH。掉了是好事——说明之前在注水，且所有方法同等下降，相对结论不变。**

### P1.2 数据划分

| 集合 | 比例 | 用途 |
|---|---|---|
| dev | 20%（seed 固定） | **所有**阈值/超参调优、探索性实验 |
| test | 80% | 只在最终表跑一次 |

需要在 dev 上重新确定并在文档中注明来源的常数：
`kv_trigger=0.65`、`stall_sim` 阈值、`runaway_frac` 分档（0.15/0.20/0.25）、`role_credit_mult`（0.5/1.5/1.0/0.75）。

> Combo16k 那种"试了发现负收益"的探索，今后只在 dev 上做。

### P1.3 修好上界

新增两个参照配置：

| 配置 | 定义 | 用途 |
|---|---|---|
| `Unc+Compact` | 无治理 + ⑥⑦ | **真上界**，替换现有 Unconstrained 作为主参照 |
| `Unc-legacy` | 现在的 Unconstrained | 保留，用于量化上下文膨胀本身的伤害 |

> ⑦ `exec_compact` 改的是 **prompt 构造方式**而非预算，属于上下文工程修复。
> 现有 Unconstrained 用的是旧构造（无限追加，涨到 4~6.5K），
> 因此"IC 反超上界"混淆了两个变量：治理机制 + 上下文卫生。
> **R2 的所有措辞必须按 `Unc+Compact` 重写。**

### P1.4 Oracle 上界

从 Unconstrained trace 算每题"首次答对所需最少 token" $c_i^*$（答不对记 ∞），
在总预算 $T$ 下按 $c_i^*$ 升序贪心（收益均为 1，贪心即 0-1 背包最优）→ `oracle_SR(T)`。

所有方法额外报 **`SR / oracle_SR`**，回答"还剩多少空间"。

### P1.5 报告口径

- **主图**：从"三个预算点的表"改为 **SR–token Pareto 曲线**
  （横轴平均消耗 token，纵轴 SR，每方法一条线，扫 6~8 个预算点）
  → R5 的"甜蜜点/饱和浪费"不再需要靠 2~3 题的非单调差异论证，曲线拐点直接可见
- 每个数据点带 **bootstrap 95% CI**
- 方法间比较一律用 **McNemar 配对检验**，报不一致对 $(b, c)$ 与 p 值
- 每个配置跑 **3 次**（同 seed 重跑即可，测 vLLM 连续批处理 + TP=2 + FP8 的非确定性），报 mean ± range

---

## Phase 2：机制重构

### M0 — 删除

依据 D2 触发率，预期删除：

- ② `role_credit_mult`
- ③ `predictive_est`
- ⑤ `predictive_lending`
- 跨 agent 借贷本身

**保留** ① `final_reserve`——其语义是"保证有答案交卷"，属于正确性保障而非优化。

### M1 — 收敛早停（替换 ④）

#### 信号（三路融合，全部免费）

| 信号 | 来源 | 成本 |
|---|---|---|
| `logprob` | vLLM `logprobs=1`，取答案 span 的平均 logprob 与熵；llama.cpp 同样支持 | **0 token** |
| `stability` | 本轮答案与上一轮经 P1.1 normalizer 后是否等价（比 Jaccard 语义得多） | 0 |
| `verdict` | reviewer 的结构化 verdict 字段（见 M3.1） | 0（复用） |

#### 停止规则

```python
stop     if stability and mean_logprob > tau_conf
stop     if verdict == "accept" and mean_logprob > tau_conf
continue if entropy > tau_high          # 模型自己不确定，值得投钱
force_stop at round 5                    # 保留硬上限
```

- $\tau_{conf}$、$\tau_{high}$ 在 **dev** 上扫
- 调参目标：**在 SR 掉幅 ≤ 0.5pp 的约束下最小化 token**

#### 回滚保护（应对 D3）

若被早停的任务是"低置信 + 首轮"，保留一次 `final_reserve` 复核机会。

> 熵信号**同时复用**给 M4 做边际收益排序——一个信号服务两个机制，机制数反而变少。

### M2 — 级联路由（新增，替代固定管线）

把 Planner→Executor→Reviewer 从固定管线改为**按需升级的三级级联**：

| 级别 | 组成 | 触发条件 | 相对成本 |
|---|---|---|---|
| **L0** | 单次直答，无 plan、无工具、无 review | 默认入口 | ~1/10 |
| **L1** | Executor + 工具，单轮 | L0 置信度 < $\theta_0$ | ~1/3 |
| **L2** | 完整 Plan → Executor → Reviewer 修订循环 | L1 置信度 < $\theta_1$ 或工具失败 | 1 |

- 升级判据复用 M1 的置信信号；$\theta_0, \theta_1$ 在 dev 上扫
- 升级时**复用**低级别产出（L0 答案作为 L1 的 draft，不从零开始）
- 预计 MATH-500 上 60%+ 的题在 L0 结束

> 这是整个计划中**学术价值最高**的一块。

### M3 — 上下文经济学（扩展 ⑦）

按 D4 的排序执行，每条独立 flag：

#### M3.1 Reviewer 结构化输出

```json
{
  "verdict": "accept | revise",
  "issues": ["≤3 条，每条 ≤20 词"],
  "hint": "≤30 词"
}
```

硬 `max_tokens=150`，禁止散文。
> reviewer 输出每轮都进 executor 上下文，是**复利级**浪费。

#### M3.2 工具输出笔记化（预计收益最大）

- **废弃** 800 字符截断——截断是无差别信息损失
- 改为：executor 读完原文后显式写 ≤120 token 的 `note`，
  **原文立即丢弃，只有 note 进入后续上下文**
- 对 GAIA / FRAMES / MuSiQue 这类 retrieval-heavy 任务，这条可能比 `exec_compact` 还大

#### M3.3 Plan 冻结

plan 生成一次后不再全文回传，每轮只带 `current_step`。

#### M3.4 输出侧收紧

强制最终答案单独一行 `ANSWER: xxx`，收紧 `max_tokens`。

> ⚠️ **必须等 P1.1 判题器修好再做**——包含匹配奖励啰嗦，现在做会拿到错误信号。

### M4 — 跨任务预算池化

> **前提**：budget 是**数据集级总预算**。若是每题硬 SLA，本机制不适用。
> 论文中**两种设定都应报告**——池化那组数会好看很多。
> **依赖 D5 判据**：CV < 0.4 则砍掉本机制。

#### Two-pass 实现

**Pass 1（探针）**
每题分配 $b_0 = 0.3 \times T/N$，跑 M2 级联的 L0/L1。
记录：`answered`、`confidence`、`tokens_used`、`escalation_level`。

**排序**
对未收敛任务估计边际价值：

$$v_i = \frac{\hat{P}(\text{correct} \mid \text{+追加预算}) - \hat{P}(\text{correct} \mid \text{现在})}{\hat{c}_i}$$

- $\hat{P}$：dev 上拟合的小 logistic 回归
  （特征：置信度、熵、已用 token、当前级别、是否 stall）
- $\hat{c}_i$：该级别的历史 P50

**Pass 2**
剩余池 $T - \sum b_0$ 按 $v_i$ 降序发放，直到池空。
$v_i$ 低于阈值的任务**直接放弃**（用 `final_reserve` 交当前最佳答案）。

#### 必需对照

**均分池化**（不排序，剩余预算平均追加）——用于隔离"排序"本身的价值。

### M5 — 深度换广度

在 M1/M2 之上增加一个策略维度：同样 token，串行修订 5 轮 vs 并行 3 个短采样 + 自洽投票？

- 实现：`strategy ∈ {refine, best_of_n, hybrid}`；`best_of_n` 用 temperature=0.7、n=3、多数投票
- R2 假说（过度修订把对的改错）本身就在预言 `refine` 边际收益为负
- MATH 类任务上 best-of-n 通常明显优于 self-refine
- **路由**：熵高（模型不确定）→ 广度；熵低但 reviewer 反对（模型确定但错）→ 深度

> **至少要做成基线。** 若同等 token 下 best-of-n 赢过 refine，
> 那"少 token 好性能"的最大杠杆其实在这里，前面的调度都是次要的。

---

## Phase 3：实验矩阵

### 3.1 消融（Qwen，dev split，24K 等效预算）

leave-one-out，每次关一个：

| Run | 配置 | 检验什么 |
|---|---|---|
| A0 | 全开（M1+M2+M3+M4+M5） | 上界 |
| A1 | −M1（无早停） | 早停贡献 |
| A2 | −M2（固定管线） | 级联贡献 |
| A3 | −M3.2（退回工具截断） | 笔记化贡献 |
| A4 | −M4（等分预算） | 池化贡献 |
| A5 | −M5（只 refine） | 广度贡献 |
| **A6** | **只 M3（纯上下文优化，无任何调度）** | **关键对照** |

> **A6 的意义**：若 A0 ≈ A6，则整套调度机制不值得存在，论文应改成上下文工程论文。
> **这个结果本身也是有价值的发现，不要害怕它。**

### 3.2 便宜基线（必做——审稿人一定会问）

| 基线 | 说明 |
|---|---|
| B1 | 固定 2 轮修订，无任何治理 |
| B2 | 只截断历史（保留最近 k 轮），无配额/借贷/stall |
| B3 | 单 agent CoT，同预算 |
| B4 | best-of-3 投票，同预算 |
| B5 | Static Quota（保留，朴素参照） |

**判据**：若 B1/B2 达到 A0 的 90%，方法的卖点必须重新定位。

### 3.3 主结果（test split，Pareto 曲线）

- **预算点**：8K / 12K / 16K / 20K / 24K / 32K / 48K（7 点）
- **方法**：`Static` / `TB-v2` / `IC(旧)` / `TBv5(新)` / `Unc+Compact` / `oracle`

#### 吞吐估算（Qwen vLLM TP=2，~92 调用/分钟）

| 数据集 | 题数 | 单 run 耗时 |
|---|---|---|
| GAIA | 152 | ~17 min |
| MATH-500 | 500 | ~27 min |
| FRAMES | 824 | ~72 min |
| MuSiQue | 2417 | ~2.6 h |
| **合计（一个完整 sweep）** | | **~4.5 h** |

全矩阵 7×6×3×4.5h ≈ **570 GPU·h，过多**。

#### 裁剪方案

| 数据集 | 预算点 | seed |
|---|---|---|
| GAIA、MATH | 全 7 点 | 3 |
| FRAMES、MuSiQue | 关键 3 点（16/24/32K） | 3 |
| FRAMES、MuSiQue | 其余 4 点 | 1 |

→ 总量降至 **~180 h，约一周**。

### 3.4 Gemma 确认

~8 调用/分钟，**只做最终配置的确认，不参与任何调参**：

- GAIA 全量 + 其余三集 300 题子样本，24K 单点，单 seed
- **必须查清**：v3 中 Gemma GAIA 32K 的 Static = IC = 上界 = 20.4%（均为 31/152）
  → 对比三份 trace 的逐题命中集合，若完全一致则为 bug

---

## Phase 4：时间线与门控

| 周 | 内容 | 门控条件 |
|---|---|---|
| **W1** | Phase 0 全部诊断 + Phase 1 判题器/划分 | D1/D5 判据决定 M1/M4 优先级；D2 决定删哪些机制 |
| **W2** | M1 + M3 实现，dev 上调阈值 | M1 若在 SR −0.5pp 内省不到 20% token → 重新设计信号 |
| **W3** | M2 级联实现 + dev 调 $\theta$ | L0 命中率 < 20% → 级联收益有限，降级为"仅 reviewer 门控" |
| **W4** | M4 + M5，跑 3.1 消融 + 3.2 基线 | A0 ≈ A6 → 转向上下文工程叙事 |
| **W5** | Phase 3.3 主结果（test） | — |
| **W6** | Gemma 确认 + 写作 | — |

---

## Phase 5：风险与回退

| 风险 | 征兆 | 回退方案 |
|---|---|---|
| logprob 信号在 FP8 / Q4 量化下不可靠 | dev 上置信度 vs 正确性 AUC < 0.65 | 退回 `stability` + `verdict` 两路信号；或加一次 20-token "确定吗"自评调用（有成本但可控） |
| 级联把简单题做错（L0 过度自信） | L0 早退任务 SR 显著低于全管线 | 提高 $\theta_0$；或对 L0 结果做一次廉价 verify |
| 池化在流式/在线场景不适用 | — | 论文同时报 offline pooling 与 per-task SLA 两种设定 |
| 判题器修正后 SR 大幅下降 | MATH 从 79.6% 掉到 ~60% | 这是诚实的数；所有方法同等下降，**相对结论不变** |
| A6 说明调度无用 | A0 − A6 < 1pp | 改标题为"上下文经济学"，把调度作为负结果报告——**依然是好论文** |

---

## 附录 A：v3 结果的主要疑点

供写作时逐条消解。

### A.1 R2 的量级在噪声里

换算回题数：

| 比较 | 百分比 | 题数差 |
|---|---|---|
| GAIA 24K：IC vs 上界 | 38.2% vs 36.2% | **3 题**（58 vs 55） |
| GAIA 32K | 40.1% vs 36.2% | 6 题 |
| FRAMES 24K | 53.4% vs 53.5% | **1 题**（n=824） |
| MuSiQue 32K | +0.9pp | ~22 题（n=2417，相对可信） |
| Gemma FRAMES | +0.3pp | ~1 题（n=300） |
| Gemma MuSiQue | +1.0pp | ~3 题（n=300） |

GAIA n=152、p≈0.4 时单点 SE ≈ 4pp。
"4/4 数据集贴平上界"中有三个点的差异小于一题的十分之一个标准误。

此外 **temperature=0 ≠ 可复现**：vLLM 连续批处理 + TP=2 + FP8，batch 组成变化会改变数值，同配置重跑 SR 差 1~2pp 属常见。当前为单次采样。

### A.2 Unconstrained 不是干净的上界

见 P1.3。GAIA 上 Unc 平均只烧 20.2K，离 200K 预算差一个数量级——预算根本没绑定，
所谓 "Unconstrained" 实为"无治理 + 旧上下文"。
IC 反超它的最简约解释是：**长而脏的 prompt 让模型变笨，而你把它修干净了**。

### A.3 超参在测试集上调

`role_credit_mult`(0.5/1.5/1.0/0.75)、`kv_trigger`(0.65)、`stall_sim` 阈值、
`runaway_frac` 分档——来源未注明。
Combo16k"验证为负收益、弃用"本身即测试集选择的证据。→ P1.2 修复。

### A.4 机制负交互未被承认

16K 上 IC(24.3%) < 纯 Compact(26.3%)，即加入 ①~⑤ 是 **−2.0pp**，
而 R4 写成"Improved 仅 +0.6pp，噪声内"。
可能原因：`final_reserve` 在紧预算下白扣 tokens；`predictive_est` 用 P80 低估导致中途被拦。
→ 3.1 leave-one-out 消融。

### A.5 tok/成功的定义偏差

若统计包含 prompt token，则 ⑦ 直接压 prompt，R3 的成本优势有一部分是**定义性的**。
→ 同时报 completion-only 成本与 wall-clock。

### A.6 stall 假阳率完全不可见

治理机制唯一的真实伤害面。→ D3。

### A.7 非单调点被过度解读

FRAMES 24K(53.4%) > 32K(53.0%)、MATH Static 24K(71.2%) > 32K(70.8%)，差异均 ≤ 2 题。
R5 解释为"饱和浪费"，更省事的解释是噪声。→ P1.5 Pareto 曲线 + CI。

### A.8 内部算术自洽（正面）

已逐行核对全部 `平均 tokens / SR = tok/成功`，12 行全部吻合（误差 < 0.2K）。记账层面无误。

---

## 附录 B：新增 instrumentation 字段清单

`_run_v2_experiment.py` 每任务 trace 需新增：

```jsonc
{
  "task_id": "...",
  "rounds": [
    {
      "round": 1,
      "role": "executor",
      "prompt_tokens": 0,          // 拆分计费
      "completion_tokens": 0,
      "tool_output_tokens": 0,     // 单列，供 D4
      "answer_snapshot": "...",    // 供 D1 逐轮判定
      "is_correct": false,         // judge 后回填
      "mean_logprob": 0.0,         // 供 M1
      "entropy": 0.0,
      "escalation_level": "L2"     // 供 M2
    }
  ],
  "flags_fired": {                 // 供 D2
    "borrowing": 0,
    "final_reserve": 0,
    "stall": 0,
    "kv_trigger": 0,
    "role_credit_mult_effective": 0
  },
  "stopped_by": "converged | stall | budget | max_rounds",
  "wall_clock_ms": 0
}
```

---

## 一句话总结

> **先花一周把 Phase 0 的五个诊断做完再动手。**
>
> 现在最大的问题不是机制不够多，而是**你不知道 token 具体浪费在哪里、哪些机制根本没触发**。
> D1 和 D4 两张表出来之后，这份计划里有一半的东西可能会被证明不必做，
> 剩下一半的优先级会完全清楚。
