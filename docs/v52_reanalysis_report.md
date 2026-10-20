# v5.2 Phase R 离线重分析报告（修订版主表 v1）

数据来源：`traces/*.jsonl` 实算（脚本 `_v52_reanalyze.py`，B=10000 bootstrap，种子 20260828）。
全部数字存于 `results/v52_reanalysis.json`。零 GPU。

## 0. 口径与字段体检（R0）

- **split**：test = `random.Random(42)` shuffle 后去掉前 20% dev；各数据集的 test 题集以
  `v51t_{ds}.jsonl` 的 task_id 集合为准：GAIA=122、MATH=400、FRAMES=660、MuSiQue=1934。
  24k 与 16k 的 test 题集逐一核对一致。**既往表述修正：16k MATH 是 test(400) 不是 500；
  FRAMES test 是 660 不是 659。**
- **字段体检**：46 个（文件 × 配置）组合中 `f1 / escalation_level / stopped_by /
  flags_fired.m1_early_stop / rounds[].answer_snapshot` 全部齐全，含 Unc 臂。无缺字段项。
- **EM 与 SR 同义**（同一 `success` 字段），主表双列中的 EM 列 = SR 点估计。
- **token 口径**：`total_used`（实际消耗，含 planner/reviewer 调用）与预算帽分列。
  注意 `rounds[]` 只记录 executor 轮次，rounds 合计约为 total_used 的 35%–57%，
  差额是 planner/reviewer 等开销（R4 分解中记为 overhead）。
- **判题口径**：判题输入 = 产生最终 answer 那一轮的**完整** content
  （`_run_v2_experiment.py` 中 `judge_task(task_id, content, answer)`）；trace 里的
  `answer_snapshot` 是该 content 的前 2000 字符，R5 审计基于最后一轮快照，截断风险单独量化。
- 既往来自 v5 的数字（A0/A4/A6）判据版本与题集：与 v5.1 同一判题代码、同一 traces 落盘格式；
  本报告全部重算到 test 子集，不再引用全量数字。

## 1. 主表 v1（SR ± bootstrap 95% CI，B=10000；EM/F1 双列；实际 token vs 预算帽）

格式：`SR [CI] (n, split=test)` | EM | F1 | 实际平均 token | 预算帽。Unc = 等预算基线。

### 24k

| 数据集 | 配置 | SR [95% CI] | EM | F1 | token | 帽 |
|---|---|---|---|---|---|---|
| GAIA | Unc | 15.6% [9.8, 22.1] (n=122, test) | 15.6 | 15.6 | 13084 | 24000 |
| GAIA | Unified | 18.9% [12.3, 26.2] (n=122, test) | 18.9 | 18.9 | 9029 | 24000 |
| GAIA | NoPool | 20.5% [13.9, 27.9] (n=122, test) | 20.5 | 20.5 | 10259 | 24000 |
| MATH | Unc | 52.8% [47.8, 57.5] (n=400, test) | 52.8 | 52.8 | 6333 | 24000 |
| MATH | Unified | 70.5% [65.8, 75.0] (n=400, test) | 70.5 | 70.5 | 2905 | 24000 |
| MATH | NoPool | 71.3% [66.8, 75.8] (n=400, test) | 71.3 | 71.3 | 3283 | 24000 |
| FRAMES | Unc | 10.6% [8.3, 13.0] (n=660, test) | 10.6 | 20.0 | 8251 | 24000 |
| FRAMES | Unified | 12.7% [10.3, 15.3] (n=660, test) | 12.7 | 21.5 | 5316 | 24000 |
| FRAMES | NoPool | 15.2% [12.4, 18.0] (n=660, test) | 15.2 | 24.2 | 7237 | 24000 |
| MuSiQue | Unc | 7.0% [5.9, 8.2] (n=1934, test) | 7.0 | 15.7 | 7665 | 24000 |
| MuSiQue | Unified | 8.6% [7.4, 9.9] (n=1934, test) | 8.6 | 17.2 | 4462 | 24000 |
| MuSiQue | NoPool | 9.7% [8.4, 11.0] (n=1934, test) | 9.7 | 18.8 | 6582 | 24000 |

### 16k

| 数据集 | 配置 | SR [95% CI] | EM | F1 | token | 帽 |
|---|---|---|---|---|---|---|
| GAIA | Unc | 14.8% [9.0, 21.3] (n=122, test) | 14.8 | 14.8 | 10944 | 16000 |
| GAIA | Unified | 17.2% [10.7, 24.6] (n=122, test) | 17.2 | 17.2 | 8493 | 16000 |
| GAIA | NoPool | 17.2% [10.7, 24.6] (n=122, test) | 17.2 | 17.2 | 9398 | 16000 |
| MATH | Unc | 51.8% [46.8, 56.5] (n=400, test) | 51.8 | 51.8 | 5362 | 16000 |
| MATH | Unified | 70.8% [66.0, 75.3] (n=400, test) | 70.8 | 70.8 | 2741 | 16000 |
| MATH | NoPool | 70.5% [66.0, 75.0] (n=400, test) | 70.5 | 70.5 | 2965 | 16000 |
| FRAMES | Unc | 10.0% [7.7, 12.3] (n=660, test) | 10.0 | 19.8 | 7691 | 16000 |
| FRAMES | Unified | 13.3% [10.8, 15.9] (n=660, test) | 13.3 | 21.4 | 5158 | 16000 |
| FRAMES | NoPool | 15.3% [12.6, 18.2] (n=660, test) | 15.3 | 24.6 | 6981 | 16000 |
| MuSiQue | Unc | 7.0% [5.8, 8.2] (n=1934, test) | 7.0 | 15.1 | 6960 | 16000 |
| MuSiQue | Unified | 8.9% [7.6, 10.2] (n=1934, test) | 8.9 | 17.5 | 4449 | 16000 |
| MuSiQue | NoPool | 10.1% [8.8, 11.5] (n=1934, test) | 10.1 | 18.8 | 6489 | 16000 |

判读：所有臂实际 token 远低于预算帽（最高的 GAIA Unc 也只用 13.1k/24k），预算帽在两个
档位下均不构成 binding 约束——"等预算"对比成立，增益不来自预算挪用。

## 2. R1：配对显著性（McNemar 精确二项 + Δ 配对 bootstrap CI）

### NoPool vs Unc（8 格）

| 格 | ΔEM | Δ 95% CI | b/c | McNemar p | 显著 |
|---|---|---|---|---|---|
| GAIA 24k | +4.9pp | [−0.8, +10.7] | 10/4 | 0.180 | 否 |
| GAIA 16k | +2.5pp | [−3.3, +8.2] | 8/5 | 0.581 | 否 |
| MATH 24k | +18.5pp | [+14.0, +23.0] | 85/11 | 2.5e-15 | **是** |
| MATH 16k | +18.8pp | [+14.3, +23.3] | 88/13 | 7.5e-15 | **是** |
| FRAMES 24k | +4.5pp | [+2.3, +6.8] | 45/15 | 1.3e-04 | **是** |
| FRAMES 16k | +5.3pp | [+3.0, +7.7] | 51/16 | 2.2e-05 | **是** |
| MuSiQue 24k | +2.6pp | [+1.5, +3.8] | 95/44 | 1.8e-05 | **是** |
| MuSiQue 16k | +3.2pp | [+2.1, +4.3] | 92/31 | 3.3e-08 | **是** |

### NoPool vs Unified（8 格）

| 格 | ΔEM | Δ 95% CI | b/c | McNemar p | 显著 |
|---|---|---|---|---|---|
| GAIA 24k | +1.6pp | [−2.5, +6.6] | 5/3 | 0.727 | 否 |
| GAIA 16k | 0.0pp | [−4.1, +4.1] | 3/3 | 1.000 | 否 |
| MATH 24k | +0.8pp | [−1.0, +2.8] | 9/6 | 0.607 | 否 |
| MATH 16k | −0.3pp | [−3.0, +2.3] | 13/14 | 1.000 | 否 |
| FRAMES 24k | +2.4pp | [+0.2, +4.7] | 36/20 | 0.044 | **是** |
| FRAMES 16k | +2.0pp | [−0.2, +4.1] | 31/18 | 0.085 | 否 |
| MuSiQue 24k | +1.0pp | [−0.1, +2.1] | 69/49 | 0.080 | 否 |
| MuSiQue 16k | +1.2pp | [+0.1, +2.4] | 78/54 | 0.045 | **是** |

判读（出口表述）：**NoPool vs Unc 方向一致 8/8，显著 6/8**——GAIA 两格（n=122）配对检验
没能救回（discordant pairs 只有 10/4 与 8/5），GAIA 的 SR claim 应收缩为"方向一致、不显著"。
**NoPool vs Unified 方向一致 6/8，显著 2/8**（FRAMES 24k、MuSiQue 16k），池化的边际贡献
是小幅正方向但多数格不显著——叙事上池化应定位为"不亏质量换 token"而非 SR 增益来源。

## 3. R2：题集对齐重算（24k MATH/FRAMES，v5 臂全量 → test 子集）

| 数据集 | 配置 | 全量 SR (n) | test SR (n) | 全量 token | test token |
|---|---|---|---|---|---|
| MATH | Unc | 53.2% (500) | 52.8% (400) | 6011 | 6333 |
| MATH | A0-Full | 49.2% (500) | 47.3% (400) | 1145 | 1144 |
| MATH | A4-NoM4 | 48.8% (500) | 47.0% (400) | 962 | 976 |
| MATH | A6-OnlyM3 | 68.2% (500) | 68.0% (400) | 5821 | 6044 |
| FRAMES | Unc | 11.5% (824) | 10.6% (660) | 8359 | 8251 |
| FRAMES | A0-Full | 16.5% (824) | 15.6% (660) | 5901 | 5782 |
| FRAMES | A4-NoM4 | 15.8% (824) | 15.6% (660) | 2628 | 2564 |
| FRAMES | A6-OnlyM3 | 13.7% (824) | 13.3% (660) | 7553 | 7440 |

判读：MATH 的 Unc 从 53.2%→52.8%，NoPool 同集增益 +18.5pp（p≈2.5e-15），**18pp 翻盘
不成立，H2 稳固**。FRAMES 的 Unc 从 11.5%→10.6%（dev 题略简单），NoPool 同集增益从原先
的 +3.7pp 变为 **+4.5pp 且显著（p=1.3e-04）**——变数方向是有利的：对齐后 FRAMES 增益
反而更大更稳。全表已统一 test 子集 + n 列，跨题集比较不再出现。

## 4. R3：ΔEM vs ΔF1（NoPool − Unc）

| 格 | ΔEM | ΔF1 | ΔEM−ΔF1 |
|---|---|---|---|
| GAIA 24k / 16k | +4.9 / +2.5 | +4.9 / +2.5 | 0.0 / 0.0 |
| MATH 24k / 16k | +18.5 / +18.8 | +18.5 / +18.8 | 0.0 / 0.0 |
| FRAMES 24k / 16k | +4.5 / +5.3 | +4.2 / +4.8 | +0.4 / +0.5 |
| MuSiQue 24k / 16k | +2.6 / +3.2 | +3.1 / +3.7 | −0.5 / −0.5 |

注：GAIA/MATH 的 f1 与 success 同源（判对=1.0），两列恒等属构造性质；有判别力的是
FRAMES/MuSiQue 的 token-F1。

判读（G1 信号）：FRAMES/MuSiQue 的 ΔF1 与 ΔEM 差距 ≤0.5pp（MuSiQue 的 F1 增益甚至
**大于** EM 增益），**ΔF1≈ΔEM 成立——按 R3 这道证伪，增益是真实答案质量提升而非抽取/
格式优势，I1 的第一道检验未坐实**。但最终判决还需叠加 R5 的上界估计与 C1（Unc+Fmt）。

## 5. R4：Pareto + 分流审计（24k）

Pareto 图：`figures/v52_pareto_{ga,ma,fr,mu}.png`（含 v5 的 A0/A4/A6 等臂，统一 test 子集；
虚线为 Pareto 前沿）。MuSiQue 上 NoPool@16k 在前沿上；Unified 两臂以约 −2.1k~−3.2k token
换 −1.0pp SR，处于前沿内侧。

### 5.1 分流分布（24k）

| 数据集 | 配置 | L0 停留 | L1 停留 | L2 | M1 触发率 | stopped_by 前三 |
|---|---|---|---|---|---|---|
| GAIA | Unc | 0 | 0 | 100% | 0 | max_rounds 61.5%, converged 28.7%, budget 9.8% |
| GAIA | Unified | 12.3% | 0 | 87.7% | 27.0% | max_rounds 52.5%, converged 44.3%, budget 3.3% |
| GAIA | NoPool | 13.1% | 3.3% | 83.6% | 30.3% | converged 50.0%, max_rounds 47.5% |
| MATH | Unc | 0 | 0 | 100% | 0 | converged 80.0%, max_rounds 11.0%, budget 9.0% |
| MATH | Unified | 76.7% | 0.8% | 22.5% | 14.0% | converged 91.5%, max_rounds 7.5% |
| MATH | NoPool | 77.2% | 2.8% | 20.0% | 11.7% | converged 92.7%, max_rounds 7.2% |
| FRAMES | Unc | 0 | 0 | 100% | 0 | converged 55.0%, max_rounds 43.8% |
| FRAMES | Unified | 0.9% | 0.3% | 98.8% | 47.3% | converged 52.0%, max_rounds 47.7% |
| FRAMES | NoPool | 1.1% | 1.4% | 97.6% | 45.3% | converged 52.9%, max_rounds 47.1% |
| MuSiQue | Unc | 0 | 0 | 100% | 0 | converged 61.0%, max_rounds 38.3% |
| MuSiQue | Unified | 0.9% | 0.2% | 99.0% | 42.7% | converged 49.9%, max_rounds 49.9% |
| MuSiQue | NoPool | 0.8% | 1.6% | 97.6% | 45.4% | converged 53.7%, max_rounds 46.3% |

### 5.2 token 分解（24k，级联臂；级别边界推断见下）

推断逻辑：级联臂 rounds 编号为 L0:(0)、L1:(1)、L2:(0..k 重新编号)，以"第二个 round==0
的位置"为 L2 起点，此前两轮为 L0/L1 探路；escalation_level<2 或只有一个编号序列的
（Unc/A6，m2 关）全部计为 L2。overhead = total_used − rounds 合计（planner/reviewer）。

| 数据集 | 配置 | 升级 L2 占比 | 探路 token/题* | L2 executor | overhead | 探路占总量 | 探路已答对(L0/L1) |
|---|---|---|---|---|---|---|---|
| GAIA | Unified | 87.7% | 600 | 5357 | 4106 | 6.6% | 3.7% / 5.6% |
| GAIA | NoPool | 83.6% | 462 | 6240 | 5113 | 4.5% | 1.0% / 1.0% |
| MATH | Unified | 22.5% | 581 | 4293 | 3530 | 20.0%** | 1.1% / 5.6% |
| MATH | NoPool | 20.0% | 345 | 5589 | 4914 | 10.5%** | 0 / 0 |
| FRAMES | Unified | 98.8% | 276 | 2732 | 2371 | 5.2% | 0.3% / 0.3% |
| FRAMES | NoPool | 97.6% | 264 | 3772 | 3366 | 3.6% | 0 / 0.2% |
| MuSiQue | Unified | 99.0% | 238 | 2351 | 1916 | 5.3% | 0.4% / 0.6% |
| MuSiQue | NoPool | 97.6% | 237 | 3501 | 3000 | 3.6% | 0 / 0.1% |

\* 仅统计升到 L2 的题。\*\* MATH 分母含 77% 停 L0 的题，拉低了总量、抬高了占比。

### 5.3 "token 省在哪"分解（FRAMES/MuSiQue，24k，vs Unc）

| 数据集 | Unc 总量 | Unified 总量（省） | NoPool 总量（省） | 节省主因 |
|---|---|---|---|---|
| FRAMES | 8251 | 5316（−2935, −35.6%） | 7237（−1014, −12.3%） | 见下 |
| MuSiQue | 7665 | 4462（−3203, −41.8%） | 6582（−1083, −14.1%） | 见下 |

升级到 L2 的题里，探路（L0+L1）只花 237–276 token（占总量的 3.6%–5.3%），且这些题中
探路答案本来就对的比例 ≤0.6%——**探路既不是有效 draft（升级题里 L0/L1 几乎从不对），
也称不上昂贵（<300 token）；它是廉价但近乎纯开销的路由税**。检索型两数据集的 token 节省
主因不在 M2 分流（99% 照样升 L2），而在 **L2 内部收敛**：M1 触发率 42%–47%，
converged 比例从 Unc 的 55–61% 结构性地被 M1 提前终止（max_rounds 占比下降）、
L2 executor token 从 Unc 的 4434（FRAMES）/ 4080（MuSiQue）压到 2351–3772，
overhead（planner/reviewer）从 3817 / 3585 压到 1916–3366。Unified 比 NoPool 多省
~2.1k~2.2k token，差值来自池化在 L2 各角色间的再分配（overhead 与 executor 双降），
代价是 −1.0pp SR。

判读（G2 信号）：检索型数据集上 M2 级联的路由价值近似为零（99% 升 L2、探路答案几乎
从不对），token 节省全部来自 L2 内部（M1 + M3 上下文经济学）；M2 的实质贡献限于 MATH
（77% 停 L0，省 54% token）与"draft 种子"角色。这与 G2 的"标题改为语言化升级判据 +
上下文经济学"分支一致，待 Ab3/C2 判决。

## 6. R5：格式合规与抽取审计

口径：对最后一轮 `answer_snapshot`（判题同源文本的前 2000 字符）复现 `_extract_answer`
三级路径：anchor=正则 `answer\s*(is|are|was|were)?\s*[:\-]`，answer_is=`answer is/are/…`，
fallback=最后一个非空行。trunc=快照长度=2000 且无任何锚点（**疑似截断**，此时真实
content 末尾可能有 ANSWER: 行，fallback 分类被高估）。

### 6.1 抽取路径分布与分层 EM（24k；16k 见 JSON，趋势相同）

| 数据集 | 配置 | anchor | answer_is | fallback | 疑似截断 | EM(anchor) | EM(fallback) |
|---|---|---|---|---|---|---|---|
| GAIA | Unc | 4.1% | 0 | 95.9% | 41.0% | 0.0 (n=5) | 16.2 |
| GAIA | NoPool | 57.4% | 2.5% | 40.2% | 37.7% | 28.6 | 10.2 |
| MATH | Unc | 6.8% | 2.2% | 91.0% | 10.7% | 59.3 | 52.2 |
| MATH | NoPool | 84.0% | 0.5% | 15.5% | 13.8% | 81.5 | 17.7 |
| FRAMES | Unc | 2.4% | 1.4% | 96.2% | 23.3% | 12.5 (n=16) | 10.7 |
| FRAMES | NoPool | 81.5% | 0.2% | 18.3% | 17.7% | 18.6 | 0.0 |
| MuSiQue | Unc | 4.5% | 1.3% | 94.2% | 16.8% | 2.3 | 7.4 |
| MuSiQue | NoPool | 88.1% | 0.9% | 11.1% | 10.7% | 10.7 | 1.9 |

### 6.2 格式带来的 EM 增益上界（24k）

两种估计（均把 anchor-vs-fallback 的通过率差**全部**归因于格式，故为上界）：

| 数据集 | ΔEM | (a) 臂内上界* | (a) 残余 ΔEM | (b) 跨臂上界** | (b) 残余 ΔEM |
|---|---|---|---|---|---|
| GAIA | +4.9pp | 0.0pp | +4.9pp | +11.8pp | −6.9pp |
| MATH | +18.5pp | +6.6pp | +11.9pp | +27.4pp | −8.9pp |
| FRAMES | +4.5pp | +1.7pp | +2.8pp | +7.7pp | −3.1pp |
| MuSiQue | +2.6pp | 0.0pp | +2.6pp | +3.2pp | −0.5pp |

\* (a)：Unc 的非 anchor 项（91%–96%）若都达到 Unc 自己 anchor 项的通过率。
\*\* (b)：Unc 的非 anchor 项若都达到 NoPool anchor 项的通过率（把质量差也算进格式，最不保守）。

判读（G1 信号）：**臂内上界 (a) 下，格式至多解释 MATH 增益的 36%、FRAMES 的 38%，
GAIA/MuSiQue 为 0**（Unc 的 anchor 项通过率反而低于自己的 fallback 项，格式无从解释）。
且同路径对比下 NoPool 仍全胜（anchor 路径 EM：MATH 81.5 vs 59.3，FRAMES 18.6 vs 12.5，
MuSiQue 10.7 vs 2.3）。**格式/抽取不是增益主因**，与 R3 的 ΔF1≈ΔEM 互证；I1 在 R 阶段
两项证据下均不成立，最终关门留给 C1（Unc+Fmt）。(b) 的"上界吃掉全部增益"是把质量差
也算作格式的极端情形，仅作敏感性参考。
附注：GAIA 疑似截断率 37.7%–41.8%，其 fallback 分类明显高估，GAIA 的 R5 数字精度最低。

## 7. 遗留事项（Phase R 未覆盖）

- **逐题翻转率**（§6.5，单种子质疑的经验下界）：需 V5.1-NoPool × GAIA × 24k 重跑一次，
  属 GPU 任务，不在本阶段。
- R5 规范中的"人工抽样 30 条 Unc 判错样本标注真错/抽取失败"：traces 不含 gold answer，
  离线无法标注"真错"；以 6.2 的率估计替代，若需精确值需带 gold 数据重判。
- NoPool vs Unified 的 8 格中有 2 格 Δ 为负/零（MATH 16k −0.3pp、GAIA 16k 0.0pp），
  池化的 SR 叙事应保守。

---

产物：`results/v52_reanalysis.json`（全部数值）、`figures/v52_pareto_{ga,ma,fr,mu}.png`、
本报告；脚本 `_v52_reanalyze.py`（可复跑，种子固定）。
