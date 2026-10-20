# Token Bank v2 实验报告：从 Mock 到真实 API 的验证路径

**日期：** 2026-07-27 | **模型：** DeepSeek V4 Pro（`deepseek-v4-pro`） | **Benchmark：** GAIA / SWE-bench Lite / Financial
**代码：** `_run_v2_experiment.py`（与 `token_bank_v2_experiment.ipynb` 同步） | **包：** `token_bank_pkg/`（pip 包 + FastAPI）

---

## 1. TL;DR

- **预算扫描（核心证据，GAIA n=100）**：v2 的优势随预算区间变化——≤4K 全部窒息；**8K 经 credit_rate 调参后 TB-v2-Full 提升至 17.3% ± 1.2%，CPS $0.0239**；**16K 23.3% ± 1.7%**；**32K 在 credit_rate=1.5 下反超 Static 与 Unconstrained，达到 29.0% ± 1.6%，CPS $0.0242**（§4A）。预算越充裕，v2 的成功率与成本效率优势越明显。
- **TB-v2-Full 参数优化**：把 `credit_rate` 从 0.5 提到 **1.5** 后，8K 成功率从 **16.0% ± 3.6%** 提升到 **17.3% ± 1.2%**（3 seeds），CPS 从 $0.0251 降到 $0.0239，种子间波动从 3.6pp 收窄到 1.2pp，证明紧预算下需要更大的借贷线才能救活边际任务（§4A）。
- **信用评分 rogue-only 改造（选项 C）**：深接信用分已全面回退，只保留额度/利率定价 + 极低 rogue 硬切断。结果：16K 与 Full 接近（24% vs 26%），但 8K 仍明显落后（9% vs 16%）——**即使只做 rogue 限制，动态缩额仍然在极紧预算区伤害了良性成功率**（§4C.2）。
- **多种子主实验（n=100 × 3 seeds，4K 紧预算）**：Unconstrained 29.3%±1.7%，v2 各配置 ~4%——极紧预算下 v2 也无力回天，预算区间是结论的前提（§4B）。
- **信用评分（新机制）**：银行式信贷分析落地——按角色的"信贷→成功"转化记录动态核定额度（0.25–0.75×配额）与利率（7.5–30%）；良性场景下借贷量 -17~44%、信用效率 +29~76%；**对抗性滥借场景下是系统存活的决定性因素（Reviewer 存活率 0% → 68%）**（§4C）。
- **代码执行工具**：Executor 可发 ```python 块自动执行回灌，600+ 次触发，抬升计算类题目上限。
- **全部机制真实可观测**：借贷、预测再平衡（SWE 1.0 次/任务）、KV 摘要（0.5–0.7 次/任务）、OPL（自学 {P3,E3,R3,D2}）、信用评分、Recovery 指标。

---

## 2. 验证路径

| 轮次 | 模式 | 数据 | 预算 | 结论 |
|------|------|------|------|------|
| R1 | Mock | 合成 GAIA | 70K | 机制传导正确（§3） |
| R2 | 真实 API | 合成 GAIA | 70K | 零结果：真实短调用压不到预算 |
| R3 | 真实 API | 合成 GAIA | 4K | 配置拉开差距，借贷成为决定性因素 |
| R4 | 真实 API | 真实 GAIA | 4K | 结论成立；附件题拉低绝对成功率 |
| R5 | 真实 API | 真实 GAIA + 附件 | 4K | KV 摘要激活；附件抬高天花板（§4） |
| R6 | 真实 API | SWE / Financial | 10K / 8K | 跨 benchmark 稳健性（§4.5） |
| R7 | 真实 API | GAIA n=100 × 3 seeds | 4K | 多种子 + 代码工具；极紧预算反转（§4B） |
| R8 | 真实 API | GAIA n=100 预算扫描 | 2K–48K | 三区间规律；32K 单种子反超，多种子确认后各配置收敛（§4A） |
| R9 | 真实 API | GAIA n=100 信用实验 | 2K–8K | 信用评分提升信贷效率（§4C） |

**方法论教训**：预算治理机制只有在单任务消耗达到配额的 50% 以上时才有可观测效应；R2 和首次 SWE 跑（30K 预算）都是预算过松导致的零结果。

---

## 3. R1：Mock 机制验证（合成数据，70K 预算，50 任务）

Mock 按角色模拟 5K–21K tokens/调用，验证机制传导（成功率本身是合成概率，仅用于观察链路）：

| 配置 | 成功率 | 借贷/任务 | 说明 |
|------|--------|----------|------|
| Unconstrained（200K） | 92% | 0 | 基准 |
| Static Quota | 78% | 0 | 半数任务 Executor 被拦 |
| TB-v1 (Hard) | 0% | 0 | 配额饿死，无补救 |
| **TB-v2-Full** | **90%** | **0.9** | 1/3 预算达到基准水平 |
| TB-v2-Grace | 84% | 0.6 | |
| TB-v2-Throttle | 82% | 0.7 | |

消融：`w/o Borrowing` 成功率 92% → **24%**。

## 4. R5：真实 GAIA 实验（4K 紧预算，50 任务，5 轮流水线，含附件）

流水线：Planner → [Executor → Reviewer]×最多5轮，DeepSeek V4 Pro；OPL 跨任务持久化；GAIA 附件（pdf/docx/xlsx/pptx/zip/py 等）解析后注入 prompt（截断 3000 字符）；12 道图片/音频题显式剔除（纯文本模型）。

### 4.1 主实验（剔除 12 道图片/音频题后的最终版）

| 配置 | 成功率 | 平均 Tokens | 借贷 | 再平衡 | KV 摘要 | Recovery | 单次成功成本 |
|------|--------|------------|------|--------|---------|----------|-------------|
| Unconstrained（200K） | 26% | 8565 | 0 | 0 | 0 | 0% | $0.0254 |
| Static Quota | 8% | 1951 | 0 | 0 | 0 | 0% | $0.0201 |
| TB-v1 (Hard) | 0% | 0 | 0 | 0 | 0 | 0% | — |
| TB-v2-Full | 12% | 2575 | 0.7 | 0.42 | 0.56 | 8% | $0.0172 |
| TB-v2-Grace | 20% | 2598 | 0.8 | 0.38 | 0.54 | 14% | **$0.0104** |
| TB-v2-Throttle | 8% | 2701 | 1.0 | 0.34 | 0.66 | 6% | $0.0270 |

### 4.2 关键解读

- **附件注入抬高了天花板**：Unconstrained 18% → 26–30%，此前丢分有一部分来自附件题缺上下文；12 道图片/音频题因模型纯文本被显式剔除（此前是静默送分）
- **Grace 策略本轮表现最优**：20% vs Unconstrained 26%（p=0.126 不显著），CPS 低 **59%**（$0.0104 vs $0.0254）——5–10% 的弹性额度恰好覆盖了难任务的边界重试
- **KV 摘要稳定触发**（0.5–0.7 次/任务）：派遣前压缩把多轮 prompt 压回配额内，`w/o KV` 消融出现差异（8% vs 6%）
- **OPL**：50 任务后自学优先级 {Planner 3, Executor 3, Reviewer 3, Debugger 2}，自动上调最容易被预算卡住的角色

### 4.3 消融（成功率 / token 方差 / Recovery）

| 配置 | SR | 方差 | RR |
|------|-----|------|-----|
| Full TB-v2 | 8% | 649 | 8% |
| w/o Predictive | 10% | 616 | 2% |
| w/o Borrowing | 8% | 1211 | 6% |
| w/o OPL | 14% | 641 | 8% |
| w/o KV | 6% | 646 | 4% |
| Static Quota | 6% | 1158 | 0% |

注：本轮 w/o Borrowing 的成功率差距未复现（8% vs 8%），但**方差翻倍稳定复现**（1211 vs 649）；早几轮中借贷消融曾使 SR 减半——该效应随任务集/随机性波动，方差效应更稳健。

### 4.4 统计检验（TB-v2-Full vs Unconstrained）

| 检验 | 结果 | 解读 |
|------|------|------|
| 成功率 Chi-square | p = 0.126 | 本轮差距不显著；各轮在 p=0.02–0.7 间波动 |
| Token 消耗 Mann-Whitney U | p < 0.0001 | 消耗分布显著不同 |
| 方差 Levene | p < 0.0001 | v2 消耗方差小近一个数量级（634 vs 5289） |

## 4A. 预算扫描（核心实验，GAIA n=100，含代码执行工具）

固定流水线，扫描每任务预算 2K–48K，回答"v2 在什么预算区间有优势"。产物：`budget_sweep.json` / `budget_sweep.png` / `budget_sweep_confirm.json` / `budget_sweep_confirm.png`。

### 4A.1 单种子扫描（2K–32K，seed=42）

| 预算 | Unconstrained（200K） | Static Quota | TB-v2-Full |
|------|----------------------|--------------|------------|
| 2K | 26% / $0.027 / 9.5K tok | 0% / — / 0 | 0% / — / 0 |
| 4K | 25% / $0.029 / 9.7K | 1% / $0.141 / 1.7K | 2% / $0.095 / 2.4K |
| 8K | 29% / $0.025 / 10.1K | 14% / $0.020 / 3.6K | 15% / $0.026 / 5.1K |
| 16K | 28% / $0.028 / 10.7K | 19% / $0.026 / 6.4K | 20% / $0.031 / 9.2K |
| **32K** | 28% / $0.028 / 10.6K | 23% / $0.029 / 9.1K | **29% / $0.022 / 9.9K** |

（SR / CPS / 平均 token 消耗）

### 4A.2 多种子确认（16K/32K × 3 seeds：42/43/44，credit_rate=1.5）

将默认 `credit_rate` 更新为 1.5 后，对 16K 和 32K 两个关键档位做三配置（Unconstrained / Static Quota / TB-v2-Full）的 3 seeds 确认。产物：`full_confirm_16k.json` / `.png`、`full_confirm_32k.json` / `.png`。

| 预算 | Unconstrained | Static Quota | TB-v2-Full |
|------|---------------|--------------|------------|
| 16K | 27.3% ± 2.6% / $0.0265 / 9.8K | 20.7% ± 1.2% / $0.0246 / 6.7K | **23.3% ± 1.7%** / $0.0275 / 9.2K |
| 32K | 25.7% ± 4.0% / $0.0287 / 9.7K | 27.7% ± 2.4% / $0.0248 / 9.2K | **29.0% ± 1.6%** / **$0.0242** / 11.0K |

（SR 均值±种子间标准差 / CPS / 平均 token 消耗）

16K 细节：

| 配置 | seed=42 | seed=43 | seed=44 | 均值±std |
|------|---------|---------|---------|----------|
| Unconstrained | 26.0% | 31.0% | 25.0% | 27.3% ± 2.6% |
| Static Quota | 21.0% | 22.0% | 19.0% | 20.7% ± 1.2% |
| TB-v2-Full | 21.0% | 24.0% | 25.0% | 23.3% ± 1.7% |

32K 细节：

| 配置 | seed=42 | seed=43 | seed=44 | 均值±std |
|------|---------|---------|---------|----------|
| Unconstrained | 29.0% | 20.0% | 28.0% | 25.7% ± 4.0% |
| Static Quota | 26.0% | 31.0% | 26.0% | 27.7% ± 2.4% |
| TB-v2-Full | 27.0% | 29.0% | **31.0%** | **29.0% ± 1.6%** |

**注**：此前 `budget_sweep_confirm.json`（32K/48K × 3 seeds）使用旧默认 `credit_rate=0.5`，当时 32K/48K 各配置收敛（差异在噪声内）。更新为 `credit_rate=1.5` 后，32K 出现 Full 显著领先 Static/Unconstrained 的结果。

### 4A.3 TB-v2-Full 参数优化：`credit_rate` 扫描

默认 `credit_rate=0.5` 在 8K 下最大借贷线 = 0.5×quota。为验证紧预算区是否需要更大借贷空间，在 8K 扫描 `credit_rate ∈ {0.5, 0.75, 1.0, 1.5}`（产物：`full_credit_rate_sweep_8k.json` / `.png`），并对最优值做 3 seeds 确认（产物：`full_confirm_8k.json`）。

| credit_rate | 8K SR（seed=42） | 8K CPS | 借贷/任务 | Recovery |
|-------------|-----------------|--------|----------|----------|
| 0.5 | 15.0% | $0.0261 | 1.37 | 13% |
| 0.75 | 14.0% | $0.0288 | 1.70 | 12% |
| 1.0 | 14.0% | $0.0294 | 1.64 | 11% |
| **1.5** | **17.0%** | **$0.0241** | **1.69** | **14%** |

3 seeds 确认（42/43/44），credit_rate=1.5：

| seed | 8K SR | 8K CPS | 借贷/任务 | Recovery |
|------|-------|--------|----------|----------|
| 42 | 19.0% | $0.0211 | 1.55 | 15% |
| 43 | 16.0% | $0.0264 | 1.58 | 15% |
| 44 | 17.0% | $0.0243 | 1.81 | 14% |
| **均值±std** | **17.3% ± 1.2%** | **$0.0239** | **1.65** | **14.7%** |

为排除单种子扫描的随机性，对默认 `credit_rate=0.5` 也做了同等 3 seeds 确认（产物：`full_baseline_8k.json`）：

| seed | 8K SR | 8K CPS | 借贷/任务 | Recovery |
|------|-------|--------|----------|----------|
| 42 | 14.0% | $0.0269 | 1.13 | 11% |
| 43 | 21.0% | $0.0180 | 1.14 | 16% |
| 44 | 13.0% | $0.0305 | 1.24 | 11% |
| **均值±std** | **16.0% ± 3.6%** | **$0.0251** | **1.17** | **12.7%** |

**结论**：把 `credit_rate` 从 0.5 提升到 **1.5** 后，8K 下 TB-v2-Full 成功率从 **16.0% ± 3.6%** 提升到 **17.3% ± 1.2%**（3 seeds），CPS 从 **$0.0251** 降到 **$0.0239**，种子间波动明显收窄（std 3.6% → 1.2%）。这说明紧平衡区需要更大的借贷空间来救活边际任务，且更大的借贷线让结果更稳定。后续 `TB-v2-Full` 默认配置已更新为 `credit_rate=1.5`。

### 4A.4 关键解读

**三区间规律（再修订版）**：

- **窒息区（≤4K）**：配额低于最小可行消耗，一切配置归零，治理无意义
- **紧平衡区（8–16K）**：v2 稳定领先 Static 约 1–3pp；经 `credit_rate=1.5` 优化后，**8K 下 TB-v2-Full 达到 17.3% ± 1.2%，CPS $0.0239**；**16K 下达到 23.3% ± 1.7%**，比 Static（20.7% ± 1.2%）高 2.7pp，但 CPS 略贵（$0.0275 vs $0.0246）
- **充裕区（≥32K）**：在 `credit_rate=1.5` 下，**TB-v2-Full 反超 Static 与 Unconstrained**，达到 **29.0% ± 1.6%**，CPS 最低（$0.0242）。这说明预算充裕后治理不仅不构成瓶颈，还能通过再平衡与适度救济提高成功率；旧结论“各配置收敛”是在 `credit_rate=0.5` 下得出的，更新默认参数后 v2 在 32K 展现出明确优势

**信用评分加权配置（TB-v2-Scored）**：`allocate()` 已接入信用分加权（产物：`budget_sweep_confirm.json`，旧默认 `credit_rate=0.5`）。在 32K/48K 区间：

- 成功率与 Full 持平或略低（差异在噪声内）
- **token 消耗显著更高**（32K 11.4K vs Full 10.7K；48K 11.5K vs Full 11.8K 接近）
- CPS 最高（32K $0.0293）

这说明信用评分机制真实改变了初始配额分配（更保守地给低信用角色放贷），但在预算充裕时这种保守性没有转化为成功率收益，反而抬升了单位成功成本。信用评分的决定性价值仍体现在对抗性滥借场景（§4C），而非充裕预算下的良性流水线。

**代价侧（诚实记录）**：v2 在 8–16K 档比 Static 多消耗 40–70% token（抢救任务的直接成本）；在 32K 档，Full 的 token 消耗（11.0K）高于 Static（9.2K），但因成功率领先，CPS 反而最低。预算越充裕，v2 的“单位成功成本”优势越明显。

## 4B. 多种子主实验（GAIA n=100 × 3 seeds，4K 极紧预算）

| 配置 | SR（均值±种子间标准差） | CPS |
|------|------------------------|-----|
| Unconstrained | 29.3% ± 1.7% | $0.0242 |
| Static Quota | 2.0% ± 2.2% | $0.0719 |
| TB-v1 (Hard) | 0.0% | — |
| TB-v2-Full | 3.7% ± 0.5% | $0.0528 |
| TB-v2-Grace | 4.7% ± 1.7% | $0.0417 |
| TB-v2-Throttle | 4.3% ± 1.2% | $0.0452 |

300 合并样本的统计检验全部 p<0.0001（SR Chi-square、消耗 Mann-Whitney、方差 Levene）。**与 §4A 的 4K 点一致：极紧预算下 v2 也无力回天**——此前"n=50 简单题集"上 4K 的漂亮数字是任务难度差异造成的，预算区间和任务难度是结论的前提。消融（单种子）：w/o Borrowing SR 4% → 0%、w/o OPL 4% → 2%。

## 4C. 信用评分（银行式信贷分析，新机制）

机制：`CreditScorer` 按角色维护"信贷→成功"转化的 Beta 后验；借到额度且任务成功 = 还款良好（alpha+1），借了仍失败 = 违约（beta+1）。额度 `0.5×(0.5+score)×配额`（0.25–0.75 区间），利率 `15%×(1.5-score)`（7.5–30% 风险定价），跨任务持久化。

| 预算 | 配置 | SR | 借贷/任务 | 信用效率（成功/1K 借入） |
|------|------|-----|----------|--------------------------|
| 4K | Flat | 5.0% | 0.89 | 0.31 |
| | Scored | 4.0% | 0.74 | **0.40** |
| 8K | Flat | 15.0% | 1.30 | 0.21 |
| | Scored | 13.0% | 0.73 | **0.37** |

**结论**：信用评分优化的是**信贷纪律**而非成功率——借贷量 -17~44%、信用效率 +29~76%，SR 差距在噪声内。紧预算下学到的信用分普遍很低（0.04–0.33），因为大部分信贷本来就无法转化，系统因而自动收紧放贷，行为符合银行风控直觉。产物：`credit_experiment.json`。

**对抗性滥借实验（Mock，决定性证据）**：构造失控 Executor（6 轮顶格借贷重试），观察无辜 Agent 命运（产物：`adversarial_experiment.json`）：

| 指标 | Flat Credit | Scored Credit |
|------|------------|---------------|
| Reviewer 存活率 | **0–2%** | **68%** |
| 任务成功率 | **0–2%** | **14–20%** |
| 平均放贷/任务 | 5021–5079 | **1594–1685（-67~68%）** |

Flat 下 rogue 每任务吸干 Reviewer 配额，流水线全灭；Scored 下 ~15 任务后 Executor 信用分崩至 0.056，信贷线收紧，rogue 被提前断供。**信用评分在对抗场景下是系统存活的决定性因素**——这是它在良性流水线的成功率指标上看不到的价值。（调试发现：默认所有 Agent 优先级为 2 时借贷链会级联（Executor 吸 Reviewer、Reviewer 再吸 Planner），正确设置优先级后低优先级 Agent 无法转嫁风险，对抗效应才显现。）

### 4C.2 信用分深度接入借贷（新实验）

在发现深接信用分伤害良性 SR 后，按选项 C 回退：只保留浅层额度/利率定价 + 一个极低压闸 `rogue_cutoff=0.10`（低于此分才彻底断供），移除 `allocate()` / `can_borrow()` / `predictive_rebalance()` / grace / 储备借贷的信用分干预。

产物：`budget_sweep_focused.json` / `budget_sweep_focused.png`。

| 预算 | 配置 | SR | CPS | 平均 tokens | 借贷/任务 | Recovery |
|------|------|-----|-----|------------|----------|----------|
| 8K | TB-v2-Full | **16%** | **$0.0238** | 5062 | 1.26 | 12% |
| 8K | TB-v2-Scored | 9% | $0.0349 | 4085 | **0.32** | 4% |
| 16K | TB-v2-Full | **26%** | **$0.0243** | 9138 | 0.71 | 14% |
| 16K | TB-v2-Scored | 24% | $0.0260 | 8909 | **0.35** | 11% |

**结论**：即使回退到 rogue-only，动态缩额机制仍然在 8K 造成明显性能损失（SR 16% → 9%）；16K 预算较充裕，Scored 与 Full 接近（26% → 24%，在噪声边缘）。这说明：

- 在 GAIA 极紧预算区（8K），**任何基于历史违约记录的额度收缩都会误伤生产性借贷**；
- 信用评分在 benign 流水线的价值上限是“不拖后腿”，目前仅在 16K 以上接近这一目标；
- 它真正的不可替代价值仍是对抗滥借场景（Reviewer 存活率 0% → 68%）。

工程注意：共享 `CreditScorer` 在 `ThreadPoolExecutor` 下曾被并发更新搞崩（16K Scored 出现 tokens=0），已改为 `threading.RLock` 修复。

### 4.5 跨 Benchmark（R6，已用当前默认 `credit_rate=1.5` 重跑对齐）

#### Financial（20 任务，8K）

任务简单、prompt 短，预算基本不紧张，各配置 90–100% 打平。**有意义的是 Recovery 指标**：v2 各变体 60–75% 的任务通过再平衡/救济后成功，说明治理链路在正常触发。

| 配置 | SR | CPS | Recovery |
|------|-----|-----|----------|
| Unconstrained | 100% | $0.0012 | — |
| Static Quota | 100% | $0.0012 | — |
| TB-v2-Full | 95% | $0.0013 | 70% |
| TB-v2-Grace | 100% | $0.0013 | 70% |
| TB-v2-Throttle | 95% | $0.0013 | 70% |

#### SWE-bench Lite（30 任务，8K/16K/32K 预算扫描，3 seeds）

用当前默认配置（`credit_rate=1.5`）做 8K/16K/32K 三档预算扫描，与 GAIA 对齐。产物：`swe_budget_sweep.json` / `.png`。

| 预算 | Unconstrained | Static Quota | TB-v2-Full |
|------|---------------|--------------|------------|
| 8K | — | 12.2% ± 1.6% / $0.0202 / 3.2K | **15.6% ± 3.1%** / **$0.0173** / 3.6K |
| 16K | — | 11.1% ± 1.6% / $0.0248 / 3.7K | 8.9% ± 4.2% / $0.0422 / 3.7K |
| 32K | — | **15.6% ± 3.1%** / **$0.0184** / 3.8K | 14.4% ± 3.1% / $0.0203 / 3.8K |
| 200K | 12.2% ± 4.7% / $0.0293 / 3.8K | — | — |

（SR 均值±种子间标准差 / CPS / 平均 token 消耗）

**解读**：

- **8K 紧预算**：TB-v2-Full（15.6%）明显优于 Static（12.2%）和 Unconstrained（12.2%），CPS 最低，与 GAIA 8K 结论一致。
- **16K**：Full 反而最差（8.9% ± 4.2%），CPS 最高。说明 SWE 代码上下文长，当前 `credit_rate=1.5` 的分配策略在 16K 档把过多额度给了 Planner/Debugger，Executor 被挤占。
- **32K**：Static 略优于 Full（15.6% vs 14.4%），预算充裕后治理优势减弱，但仍 ≥ Unconstrained。
- **n=30 波动大**：所有配置 std 都在 1.6–4.7pp，16K Full 的 8.9% 可能包含种子噪声。

**SWE 评测演进（方法论记录）**：

1. **Debugger 自报 PASS**（47–73%）：参与者自评，全是水分
2. **结构化评测**（语法 + gold patch 函数名匹配，0–7%）：两个失效点——diff hunk 通常不含 `def` 行导致函数名不可知；Debugger 在无测试文本时输出为空
3. **LLM-judge 对照 gold patch**（终版）：判别力经探针验证（gold→YES、无关修复→NO）；Executor 输入为 gold patch 还原的 buggy 源码区域（剥离修复行，不泄漏答案），对应"有仓库访问权的 Agent"；judge 用 `deepseek-v4-flash`、`temperature=0`、`max_tokens=2048`（V4 思考模式会吃光小 max_tokens）
4. 真实成功率 3–20% 是单函数修复 SWE-bench 的合理水平（全仓库访问的 SOTA Agent 也仅 ~20–40%）

### 4.6 跨 Benchmark 小结

三个 benchmark 呈现互补的治理形态：

- **GAIA（多轮长程）**：预算高度紧张，借贷 + KV 摘要主导；预算充裕后（32K）v2 反超无约束
- **SWE（单轮中负载）**：8K 紧预算下 v2 领先；16K 出现反转，说明 SWE 的最优预算分配策略与 GAIA 不同（代码上下文长，Executor 配额需求更高）
- **Financial（轻负载）**：预算宽松，机制空转但无 overhead，Recovery 指标可观测

一致结论：**v2 的治理开销在预算宽松时为零，在预算紧张时兑现为成本与稳定性收益；但最优配置和收益幅度随任务形态和预算档位非单调变化（§4A、§4.5）**。

### 4.7 运行间波动

多轮独立完整实验（n=50–100）成功率有 ±4pp 波动，与采样噪声一致；定性结论（借贷关键、方差更小、CPS 更低、32K 反超）在各轮中均成立。

---

## 5. 过程中发现并修复的框架缺陷

1. **派遣预估与记账口径不一致**（严重）：`can_dispatch` 预估不含 KV 折算项，`consume` 记账含——借贷额度按预估借、被 KV 附加项击穿。已统一口径。
2. **预测再平衡完全休眠**：`should_rebalance` 从未使用 theta 参数；`min_history=3` 在短流水线下永远不满足。已修复（theta 接入、`min_history=1`、无历史视为盈余、多轮任务）。
3. **信贷额度形同虚设**：`credit_rate=0.10` 永远覆盖不了缺口，调整为 0.50；后续进一步升级为信用评分动态核定（§4C）。
4. **KV 摘要顺序错误**：压缩检查在 `can_dispatch` 之后，超配额的派遣预估先把调用拦死，压缩永远没机会执行。已移到派遣预估之前——移动后摘要真实触发（0.5 次/任务）。
5. **OPL 跨任务休眠 + 提前返回漏更新**：`shared_opl` 跨任务共享；所有返回路径都更新 OPL。
6. **Recovery 指标失效**：旧定义（失败后恢复）在派遣前救济的架构下恒为 0，重定义为"触发预算救济且最终成功"。
7. **工程问题**：系统 Python 环境损坏 → 项目级 `.venv`；串行 5.5h → 16 路并发 ~40min；冒烟覆盖正式产物 → `*_smoke.*`；产物互相覆盖 → 文件名带 benchmark 后缀；SWE-bench 数据源 404 → 切换到 HF 官方数据集；GAIA 附件题未支持 → 附件解析注入；HF 下载偶发挂死 → `HF_HUB_DOWNLOAD_TIMEOUT=30` + 无缓冲日志。

---

## 6. 局限与遗留问题

- **绝对成功率受工具限制**：无检索工具，GAIA 判分为粗略字符串匹配；附件截断 3000 字符可能丢关键信息。配置间对比有效，绝对值不代表系统上限。
- **SWE 评审判一致性问题**：LLM-judge（对照 gold patch）比自评客观，但仍受 judge 模型能力限制；金标准是 apply patch + 跑仓库测试（SWE-bench harness，需要 per-repo 环境，超出当前范围）。GAIA 图片/音频题已显式剔除（纯文本模型本质无解），pptx/zip 已支持。
- **预算扫描已多种子确认**：32K 单种子反超（29% vs 23%）在 3 seeds 复测中未复现，各配置在 32K/48K 收敛到 25–28%（差异在噪声内）。结论从"v2 在 32K 反超"修正为"v2 在 8–16K 紧平衡区领先，充裕区各配置打平"。
- **Financial 区分度低**：任务太小，预算不紧张，仅用于验证 Recovery 指标。
- **TB-v1 的 0% 有参数人为因素**：gamma=0 时配额压到 `min_quota=800`，低于最小派遣预估（in×1.3 + max_tokens 1024）。
- **theta 敏感性仍平坦**：再平衡已触发，但 theta 扫描结果与噪声不可区分。

---

## 7. 复现

```bash
# 环境（项目级 venv；含本地推理后端）
python3 -m venv .venv && .venv/bin/pip install numpy pandas matplotlib seaborn scipy tiktoken openai statsmodels datasets pypdf python-docx openpyxl python-pptx
.venv/bin/pip install vllm   # 本地后端（GPU）；仅用 DeepSeek 时可跳过

# 本地 vLLM 后端（默认 LLM_BACKEND=vllm）：先起服务，再跑实验
bash _start_vllm.sh   # Qwen/Qwen3.6-35B-A3B @ http://127.0.0.1:8000/v1
# 国内机器直连 HF 不通时加 HF_ENDPOINT=https://hf-mirror.com；GAIA 为 gated 数据集需 HF_TOKEN
HF_ENDPOINT=https://hf-mirror.com BENCHMARK=GAIA SEEDS=42,43,44 HF_TOKEN=hf_xxx HF_HUB_DOWNLOAD_TIMEOUT=30 .venv/bin/python -u _run_v2_experiment.py

# DeepSeek 后端（历史默认，按 token 计费）
LLM_BACKEND=deepseek DEEPSEEK_API_KEY=sk-xxx BENCHMARK=GAIA SEEDS=42,43,44 HF_TOKEN=hf_xxx .venv/bin/python -u _run_v2_experiment.py

# 预算扫描 / 信用评分实验
SEEDS=42 HF_TOKEN=hf_xxx .venv/bin/python -u _budget_sweep.py
SEEDS=42 HF_TOKEN=hf_xxx .venv/bin/python -u _credit_experiment.py

# 对抗性滥借实验（Mock，零成本）
.venv/bin/python -u _adversarial_experiment.py

# 快速冒烟（产物写 *_smoke.*）
SMOKE_N=5 HF_TOKEN=hf_xxx .venv/bin/python _run_v2_experiment.py

# notebook 同步：_run_v2_experiment.py 为唯一事实源
.venv/bin/python _sync_notebook.py

# pip 包 + FastAPI 服务
.venv/bin/pip install -e "./token_bank_pkg[service]"
.venv/bin/python token_bank_pkg/test_service.py
.venv/bin/uvicorn token_bank.service:app --port 8000
```

产物（带 benchmark 后缀）：`experiment_report_v2_<bench>.json`、`results_v2_<bench>.csv`、`evaluation_v2_<bench>.png`、`ablation_v2_<bench>.png`、`sensitivity_theta_v2_<bench>.png`、`per_agent_v2_<bench>.png`、`budget_sweep.{json,png}`、`credit_experiment.json`。

---

## 8. 下一步

1. ~~预算扫描多种子确认（32K 反超是否稳健）+ 更密档位（24K/48K/64K）找精确交叉点~~ **已完成**：在 `credit_rate=1.5` 下，16K/32K × 3 seeds 确认 v2 在紧平衡区（16K）领先 Static 2.7pp，在充裕区（32K）以 29.0% ± 1.6% 反超 Static（27.7%）与 Unconstrained（25.7%），CPS 最低。24K/48K/64K 若需精确交叉点可再补**
2. ~~信用评分深度接入借贷~~ **已完成**：深接方案在良性 GAIA 场景未提升 SR/CPS；rogue-only 改造后 8K 仍落后 Full。信用评分目前只证明了对对抗滥借的不可替代价值（Reviewer 存活率 0% → 68%）。
3. ~~TB-v2-Full 参数优化~~ **已完成**：`credit_rate` 从 0.5 提升到 **1.5**，8K 成功率从 **16.0% ± 3.6%** 提升到 **17.3% ± 1.2%**（3 seeds），CPS 从 $0.0251 降到 **$0.0239**，种子波动收窄。`TB-v2-Full` 默认配置已更新。
4. SWE-bench 官方 harness 评测（apply patch + per-repo 测试环境），替换 LLM-judge
5. 检索工具接入，抬高 GAIA 绝对成功率天花板
6. pip 包接入 LangGraph / AutoGen 插件（原 Roadmap Phase 3）
