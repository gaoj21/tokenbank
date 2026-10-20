# Token Bank v5 实验全记录：从"预算治理"到"决定要不要花"

> 日期：2026-08-18 ~ 2026-08-21
> 平台：3× NVIDIA RTX A5000 (24GB)
> 服务栈：vLLM（Qwen/Qwen3.6-27B-FP8，TP=2，max-model-len 8192，max-num-seqs 128，端口 8001）
> 实验框架：`_run_v2_experiment.py`（新增 `run_gaia_task_v5`）+ `_budget_sweep_v2.py`（新增 `CONFIGS_V5` + `run_m4_pool`）
> 计划文档：`TOKEN_BANK_V5_PLAN.md`；本文是其执行结果。用户指令：按 plan 执行，**不做多种子**。

---

## 0. v5 的出发点：v3 结论里的四个疑点

v3（`experiment_findings_v3.md`）的核心结论是"Improved+Compact 全矩阵反超 Unconstrained 上界"。v5 计划（附录 A）逐条指出其中的水分：

1. **判题器过松**：旧规则是"gold in answer 或前两词匹配"，啰嗦答案也能蒙对，SR 普遍虚高。
2. **n=152（GAIA）噪声大**，且超参是在测试集上调的（无 dev/test 分离）。
3. **Unconstrained 不是干净上界**：它用的是无限追加的 legacy prompt，"IC beats Unc"混杂了上下文卫生因素。
4. **触发率未审计**：治理机制可能根本没被触发，"全胜"故事可能讲错了因果。

v5 因此分三步：Phase 0 诊断（先测量，再决定做什么）→ Phase 2 机制重构（M0~M5）→ Phase 3 dev 上调参、test 上出表。

---

## 1. Phase 1：基础设施修复（判题器 / 划分 / instrumentation）

### 1.1 判题器（P1.1，`judge_task`，按 task_id 前缀分发）

| 数据集 | 判据 | 实现 |
|---|---|---|
| GAIA | 数字/列表规范化后 EM | `judge_gaia`：`_norm_number`（去逗号/$/%、整数化）+ `_norm_basic`（小写、去标点、去冠词），列表按 `,;` 切分排序比较 |
| MATH | 符号等价 | `judge_math`：math_verify（`parse`+`verify`），先 `\boxed{}` 后 `_extract_answer`；math_verify 不可用时退回规范化字符串比较 |
| MuSiQue/FRAMES/Financial | EM + token 级 F1 | `em_f1`：先 `_extract_answer` 再规范化 |

`_extract_answer` 按优先级提取：`ANSWER: x` / `the answer is x`（取最后一次出现）→ 最后一个非空行。**旧的 substring/前两词规则全部删除**（3 处判题点 + runner 末尾）。

**脱水效应实测**：GAIA 16k Unconstrained 从 v3 的 ~36% 降到 19.1%；MuSiQue 2417 题从 v3 的 ~44%（32k）降到 6.0%（16k）。另有 8% 的 MuSiQue 答案 f1≥0.5 但不过严格 EM（"差边"案例）。v3 的高分里大量是判题水分。

### 1.2 dev/test 划分（P1.2）

`SPLIT=dev/test/full` 环境变量：独立 `Random(42)` 洗牌后 dev=前 20%（调参专用）、test=后 80%（只出最终表）。GAIA 152 → dev 30 / test 122；MuSiQue 2417 → dev 483 / test 1934。

### 1.3 Instrumentation（附录 B）

每任务 trace 新增：`rounds`（逐轮 role/prompt_tokens/completion_tokens/tool_output_tokens/answer_snapshot(≤2000字符)/is_correct/mean_logprob/entropy）、`stopped_by`、`flags_fired`（计数 dict）、`wall_clock_ms`、`f1`、`escalation_level`（v5 新增）。`capture_logprobs=True` 时 vLLM 请求 logprobs+top_logprobs=5，算 mean_logprob 与近似熵（top-5）。

### 1.4 真上界配置（P1.3）

`Unc+Compact`：无治理 + kv_trigger=0.65 + exec_compact，把"上下文卫生"从上界参照里剥离。

---

## 2. Phase 0：五项诊断（16k，Qwen，Unconstrained + Unc+Compact）

| 诊断 | 结果 | 判据 | 结论 |
|---|---|---|---|
| D1 收敛轮次 | median first_correct_round=**1**（两数据集）；首次答对后浪费 GAIA **31~38%**、MuSiQue 8.5~18.5%；correct→wrong 翻转个位数 | median≤2 且 wasted≥25% → M1 最高优先 | **M1 成立（GAIA 型任务）**，MuSiQue 大部分题从未答对、无可省 |
| D2 flag 触发率 | borrowing/final_reserve/stall/role_credit **全 0%**；kv_trigger 仅 5%（GAIA compact） | <10% → Phase 2 直接删除 | **治理机制基本是死代码**，v5 转向被强力验证 |
| D3 stall 假阳率 | stall **0 次触发** | — | 无假阳问题，但 stall 本身是摆设，删 |
| D4 token 去向 | **executor 只占 50~56%**，planner/reviewer 脚手架吃掉另一半 | 排序 M3 子项 | **M2 级联（L0 跳过管线）是最大杠杆** |
| D5 难度方差 | MuSiQue token CV≈**0.96~0.99**；GAIA 0.59~0.65 | >0.8 → M4 优先 | M4 在 MuSiQue 优先，GAIA 中等 |

额外发现：logprob 单信号区分度弱（答对 −0.070 vs 答错 −0.077），M1 必须三路信号融合；Unc+Compact 在 GAIA 反而更差（16.4% vs 19.1%）且 kv 只触发 5% → compact 思路让位给 M3 重设计。

---

## 3. Phase 2：机制重构（M0~M5）

### M0 — 删除

依据 D2：跨 agent 借贷、role_credit_mult、predictive_est、predictive_lending、stall 全部不进 v5 配置。**保留 final_reserve**（语义是"保证有答案交卷"，属正确性保障）。同时**取消 per-role 配额天花板**（`bank.q_min = budget`）——冒烟中发现 min_quota 地板（budget/5）会把 executor 配额压死（960/4800），探针全被 quota 挡死；v5 只留全局 reserve 约束。

### M1 — 收敛早停（替换 stall）

三路免费信号融合：`stability`（本轮答案与上轮经 P1.1 normalizer 后等价）+ `verdict`（reviewer 结构化 verdict）+ `mean_logprob`。停止规则：

```
stop     if (stability or verdict==accept) and mean_logprob > tau_conf
continue if entropy > tau_high          # 熵 override：模型自己不确定，继续投钱
force_stop at round 5                    # 硬上限
```

回滚保护（应对 D3）：低置信首轮永远不会被早停（规则本身要求 logprob > tau_conf）。

### M2 — 级联路由（替代固定管线）

| 级别 | 组成 | 升级条件 | 成本 |
|---|---|---|---|
| L0 | 单次直答（无 plan/工具/review，max_tokens=768） | 默认入口 | ~1/10 |
| L1 | Executor+代码工具单轮，L0 答案作 draft | lp < θ0 或 entropy > τ_high | ~1/3 |
| L2 | 完整 Plan→Execute→Review 循环（≤5 轮），L1 答案作 draft | lp < θ1 | 1 |

熵信号同时服务 M1/M2/M4/M5 四个机制。

### M3 — 上下文经济学

- **M3.1**：reviewer 强制 JSON `{"verdict","issues"≤3,"hint"≤30词}`，max_tokens=150，禁散文
- **M3.2**：工具输出读完原文后写 ≤120 token 笔记（独立调用，max_tokens=150），**原文立即丢弃**，后续只带笔记
- **M3.3**：plan 只在第一轮全文出现，之后每轮只带 current_step
- **M3.4**：最终答案强制单独一行 `ANSWER: xxx`，L0/L1 max_tokens=768

### M4 — 跨任务预算池化（two-pass）

Pass 1：每题 b0=0.3×B 探针（只跑 L0/L1，`v5_max_level=1`）；已收敛的直接交卷。排序：未收敛任务按边际价值 v_i=(1−conf_i)/c_i 降序（conf 来自探针 logprob 映射，c_i=探针成本 P50；plan 里的 logistic 回归以该启发式近似）。Pass 2：剩余池按 v_i 降序以 B 为单位发放，任务带 draft 从 L2 续跑（`_v5_start_level=2, _v5_draft`），探针花费并入总账。**均分池化对照**（`m4_pool_equal`）隔离"排序"本身的价值。

冒烟中修复的关键 bug：触达 max_level 上限的任务被误标 `converged`（pass 2 永远不触发）→ 拆出独立停止原因 `max_level`。

### M5 — 深度换广度

`strategy ∈ {refine, best_of_n, hybrid}`：best_of_n 用 temp=0.7、n=3、规范化后多数投票；hybrid 路由：L1 处 entropy > τ_high（模型不确定）→ 广度采样，否则 refine。

---

## 4. dev 阈值调优（离线重放，不烧真机）

校准跑法：`V5_CALIB=1` 关掉所有早停（θ=+∞、M1 off、refine），temp=0 下观测每题完整 L0/L1/L2 轨迹；然后**离线重放**扫描阈值组合。

L0 logprob 校准表（模型"知道自己知道"）：

| 数据集 | θ=-0.02 | θ=-0.05 | θ=-0.10 | θ=-0.15 | θ=-0.25 |
|---|---|---|---|---|---|
| GAIA-dev：停/acc | 7%/1.00 | 20%/0.33 | 27%/0.50 | 57%/0.24 | 80%/0.25 |
| MuSiQue-dev：停/acc | 5%/0.12 | 7%/0.14 | 13%/0.17 | 28%/0.13 | 84%/0.08 |

调参目标（plan）：SR 掉幅 ≤0.5pp 约束下最小化 token。**选定 θ0=-0.10, θ1=-0.15, τ_conf=-0.05, τ_high=0.6**：

| | 全升级基线 | 选定组合 | 节省 |
|---|---|---|---|
| GAIA-dev（n=30） | SR 23.3% / 10,696 tok | SR 23.3%（**零损失**）/ 2,632 tok | **−75%** |
| MuSiQue-dev（n=483） | SR 7.9% / 6,485 tok | SR 8.7%（**+0.8pp**）/ 2,523 tok | **−61%** |

---

## 5. Phase 3.1：dev 消融（24k 等效预算）

MuSiQue-dev（n=483）：A0 全开 SR 8.5% 最优；A2−M2 掉 1.7pp 且 token +48%；A6 只M3 7.2% ≠ A0 → 调度有真实贡献；M4-Equal 7.9% < A0 → 排序有价值。GAIA-dev（n=30，噪声大）：A0 20.0% vs Unc 10.0%，方向一致。

## 6. Phase 3：test 主结果（24k，最终表）

### MuSiQue test（n=1934，决定性数据集）

| 配置 | SR | tokens | CPS | vs A0 |
|---|---|---|---|---|
| **A5 −M5** | **10.1%** | 5,343 | $0.0013 | +0.8pp |
| A1 −M1 | 10.0% | 5,327 | $0.0012 | +0.7pp |
| A4 −M4 | 9.8% | **3,618** | $0.0011 | +0.5pp, −33% tok |
| M4-Equal | 9.8% | 5,325 | $0.0013 | +0.5pp |
| A3 −M3.2 | 9.5% | 5,137 | $0.0013 | +0.2pp |
| A0 全开 | 9.3% | 5,365 | $0.0014 | — |
| A6 只M3 | 9.3% | 6,330 | $0.0020 | ±0 |
| A2 −M2 | 8.7% | 7,945 | $0.0017 | **−0.6pp, +48% tok** |
| Unc 上界 | 7.0% | 7,665 | $0.0032 | — |

级联分流实测：13% 停 L0、4% 停 L1、**84% 升 L2**（比 dev 模拟保守——MuSiQue 多跳题模型确实不自信，置信信号诚实）；M1 在 45% 任务上触发。

### GAIA test（n=122）

| 配置 | SR | tokens |
|---|---|---|
| **A6 只M3** | **18.9%** | 11,231 |
| A2 −M2 | 18.0% | 14,302 |
| A0 全开 / A1 −M1 | 17.2% | 10,486 / 10,643 |
| Unc 上界 | 15.6% | 13,084 |
| A3/A4/A5/M4-Eq | 13.1~14.8% | 4,219~10,567 |

---

## 7. 结论

1. **v5 全线反超上界**：MuSiQue 最佳 10.1% vs Unc 7.0%（+3.1pp、省 30% token、CPS −59%）；GAIA 最佳 18.9% vs 15.6%。判题器脱水后的反超是真反超。
2. **真正有效的机制是 M2 级联 + M3 上下文经济学**：A2（去级联）是两个 test 集上唯一稳定负贡献项；A6（只 M3）在两 test 集都 ≈/＞ A0。
3. **A6≈A0 判据触发**（两 test 集）：调度三件套 M1/M4/M5 在 24k 宽松预算下零/负贡献——M1 在 test 上误停本可答对的题（A1 反超 A0 +0.7pp，dev→test 阈值泛化不佳）；M5 的 best-of-n 是净浪费（A5 +0.8pp）；M4 在池子不紧时是纯探针开销（见第 8 点）。
4. **M4 的价值依赖预算松紧（机制级发现）**：24k 下池子 46.4M、探针只花 2.2M、85% 的题照样拿全额 → 池化约束不绑定，探针（1.1k/题）+ L2 带 draft 重跑是纯开销；A4（不池化）反而 +0.5pp 且省 33% token。plan 的前提"预算紧才适用"被证实，紧预算（8k/16k）验证见第 8 节。
5. **dev→test 的教训**：M1 的 τ_conf 在 GAIA dev（n=30）上主导选择，泛化到 MuSiQue test 后偏激进。小 dev 集上调早停阈值要保守。

## 8. 紧预算验证（8k/16k，test split）

配置：A0 / A4-NoM4 / Unc × 8k,16k。验证假说：池子真正绑定时 M4 转为正贡献。**结果：假说在 MuSiQue 上成立（弱正向），在 GAIA 上不成立——M4 的收益是数据集条件性的，正好对应 D5 的难度方差判据。**

### MuSiQue test（n=1934）

| 配置 | 8k | 16k | 24k（第 6 节） |
|---|---|---|---|
| A0（含 M4 池化） | 9.2% / 3,384 tok | 9.4% / 5,191 tok | 9.3% / 5,365 tok |
| A4（不池化） | 9.0% / 2,446 tok | 9.7% / 3,424 tok | 9.8% / 3,618 tok |
| Unc | 6.3% / 4,309 tok | 6.8% / 6,944 tok | 7.0% / 7,665 tok |

- A0 从 24k→8k SR 几乎不掉（9.3→9.2）：池化+级联把"预算砍 2/3、性能不变"做出来了
- 池化在紧预算转为**正贡献**（8k：+0.2pp over A4），但幅度小；A4 始终是 token 最省的
- 两配置全程碾压 Unc（+2.4~2.9pp）

### GAIA test（n=122）

| 配置 | 8k | 16k |
|---|---|---|
| A0（含 M4 池化） | 10.7% / 4,918 tok | 13.1% / 8,901 tok |
| A4（不池化） | 13.1% / 2,956 tok | **17.2% / 4,629 tok** |
| Unc | 13.9% / 6,033 tok | 13.1% / 10,925 tok |

- GAIA 上池化全程负贡献（探针在紧预算下吃掉 30% 预算的固定开销）
- **A4（M1+M2+M3，不池化）是 GAIA 全场最优**：16k 下 17.2% 且只花 Unc 42% 的 token

### 结论修正

**v5 最终推荐形态 = M2 级联 + M3 上下文经济学（+ M1 保守早停），M4 按 D5 难度方差条件性启用**（CV>0.8 的检索型数据集如 MuSiQue 开启，CV≈0.6 的 GAIA 关闭），M5 删除。这比 plan 原设想的全开系统更简单也更强。

## 8.1 MATH/FRAMES 对齐实验（24k，全量，Qwen）

配置：A0 / A4 / A6 / Unc。这两个数据集从未参与调参，直接全量跑。

### MATH-500

| 配置 | SR | tokens |
|---|---|---|
| **A6 只M3（固定管线）** | **68.2%** | 5,821 |
| Unc | 53.2% | 6,011 |
| A0 全开 | 49.2% | 1,145 |
| A4 不池化 | 48.8% | 962 |

**级联在 MATH 上严重反噬**（A0 比 Unc 低 4pp、比 A6 低 19pp）：数学题的 logprob 置信信号不可靠——模型不做计算时"自信地错"；且 L0/L1 的 max_tokens=768 会截断 CoT。D1 早已预示：MATH 的浪费结构是"需要更深"，不是"答对后空转"。**级联适合检索/知识型任务，不适合计算密集型任务。**

### FRAMES（n=824）

| 配置 | SR | tokens |
|---|---|---|
| **A0 全开** | **16.5%** | 5,901 |
| A4 不池化 | 15.8% | 2,628 |
| A6 只M3 | 13.7% | 7,553 |
| Unc | 11.5% | 8,359 |

检索型任务上级联恢复正贡献：A0 +5.0pp over Unc 且省 29% token。

（注：MATH/FRAMES 的 SR 与 v3 数字不可比——math_verify 符号判题和严格 EM 比 v3 的 substring 匹配严得多。）

### 数据集条件性总结（v5 最终结论）

| 数据集类型 | 代表 | 级联 M2 | 池化 M4 | 最优配置 |
|---|---|---|---|---|
| 检索/多跳，难度方差大 | MuSiQue (CV≈1.0) | + | 紧预算 + | A0/A4 |
| 知识问答，方差中等 | GAIA (CV≈0.6) | + | − | A4 |
| 检索型 | FRAMES | + | 微负 | A0 |
| 计算密集 | MATH | **−−** | − | A6（固定管线+M3） |

---

## 8.2 v5.1：双配置方案（NoPool 默认 + 计算型可选池化）

### 动机

第 8.1 节的"按数据集类型选配置"不是合格的系统结论。级联在 MATH 失败的根因不是级联本身，而是 **L0 升级判据失效**：logprob 区分不了"知道答案"和"需要计算/检索"（模型自信地错），且 L0/L1 max_tokens=768 截断 CoT。

### 改动（相对 v5）

1. **语言化升级通道（`m2_verbal`）**：L0/L1 prompt 显式允许模型拒绝直答——"需要计算、多步推理或不确定的信息时，回复 ESCALATE"。路由从"猜 logprob"变为"模型自声明"，ESCALATE 优先于 logprob 门控（嘴硬但低置信也升级）
2. **L0/L1 max_tokens 768→2048**：不截断 CoT
3. **M4 池化降级为可选**：紧预算验证 + 本节的 dev/test 显示 two-pass 探针开销对检索型任务是净负担（且发现 ESCALATE×池化交互 bug：ESCALATE 只有 ~4 个高置信 token，被池化排序器误读为"自信已答"而丢弃，已修 conf=0 强制挂起）；但池化对计算型任务同分省 token，故不作为默认、保留为计算型负载的可选开关
4. M5 已于 v5 删除；M1/M3 保留

### dev 验证（24k，阈值沿用 v5 dev 调优，未重调）

| 数据集 | V5.1-NoPool | v5 最优 | Unc |
|---|---|---|---|
| MATH (100) | **74.0%** / 2,139 tok | 70.0% (A6) | 56.0% |
| MuSiQue (483) | **7.2%** | 8.5% (A0) | 5.2% |
| GAIA (30) | **23.3%** | 20.0% (A0) | 10.0% |

### test 主结果（24k，单 seed，temp=0）

| 数据集 | **V5.1-NoPool（统一配置）** | V5.1-Unified(含池化) | v5 最优 | Unc |
|---|---|---|---|---|
| GAIA (122) | **20.5%** / 10,259 tok | 18.9% / 9,029 tok | 18.9% (A6) / 11,231 tok | 15.6% / 13,084 tok |
| MATH (400) | **71.2%** / 3,283 tok | 70.5% / 2,905 tok | 68.2% (A6, 全量) / 5,821 tok | 53.2% (全量) / 6,011 tok |
| FRAMES (659) | **15.2%** / 7,237 tok | 12.7% / 5,316 tok | 16.5% (A0, 全量) / 5,901 tok | 11.5% (全量) / 8,359 tok |
| MuSiQue (1934) | **9.7%** / 6,582 tok | 8.6% / 4,462 tok | 10.1% (A5) / 5,343 tok | 7.0% / 7,665 tok |

token 效率（V5.1-NoPool vs Unc，每任务平均 token）：GAIA −21.6%、MATH −45.4%、FRAMES −13.4%、MuSiQue −14.1%——**四个数据集全部同时更准且更省**。token 数均为每任务均值（prompt+completion）。

### 16k 紧预算验证（test split，单 seed，temp=0）

| 数据集 | **V5.1-NoPool** | V5.1-Unified | Unc（等预算） |
|---|---|---|---|
| GAIA (122) | **17.2%** / 9,398 tok | 17.2% / 8,493 tok | 14.8% / 10,944 tok |
| MATH (500) | **70.5%** / 2,965 tok | **70.8%** / 2,741 tok | 51.8% / 5,362 tok |
| FRAMES (659) | **15.3%** / 6,981 tok | 13.3% / 5,158 tok | 10.0% / 7,691 tok |
| MuSiQue (1934) | **10.1%** / 6,489 tok | 8.9% / 4,449 tok | 7.0% / 6,960 tok |

NoPool vs Unc（16k）：SR +2.4 / +18.7 / +5.3 / **+3.1**pp；token −14.1% / −44.7% / −9.2% / −6.8%。结论与 24k 完全一致——预算收紧后各配置 SR 普降 1–3pp，但相对排序不变，统一方案在紧预算下同样"更准且更省"。注意 v5/v5.1 的 Unc 是**等预算对照组**（全局 budget 帽、无治理机制），不是 v3 的 200k 真·无约束。

### 结论

- **推荐方案：双配置**——**V5.1-NoPool（M1+M2 语言化级联+M3）为默认统一配置**，在全部 4 个数据集 × 2 个预算上反超 Unc 且更省 token；**V5.1-Unified（+M4 池化）为计算型负载的可选配置**
- **池化的作用是有符号的**：MATH/GAIA 上 Unified 与 NoPool 同分（差 ≤0.7pp，噪声内）且稳定省 8–12% token（16k MATH 甚至 +0.3pp）；但 FRAMES/MuSiQue × 16k/24k **4/4 全是 NoPool 赢 1.1–2.5pp**（相对损失 10–25%）。机制原因：池化按置信度排名截流，检索题大量 ESCALATE（conf=0）排在池尾被牺牲。故"利于计算型、损于检索型"，不作默认
- **语言化升级是关键创新**：MATH 上 v5 级联 49.2%（反噬）→ v5.1 71.2%（+22pp 逆转）；检索任务上模型诚实声明"要查资料"（MuSiQue 99% 升级 L2），放弃了部分 v5 靠 logprob 便宜答对的 L0 题，换来跨任务类型的稳健性
- **Gemma-4-31B 跨模型确认**（GAIA+MATH test）：级联反噬 MATH 的模式跨模型成立（Gemma：A0 36.0%/A4 34.0% < Unc 39.0%）——正是 v5.1 修复的失效模式；GAIA 上 A0 13.1% > Unc 12.3% 方向一致
- **机制净贡献终版**：M2（语言化级联）+ M3（上下文经济学）扛大梁，M1 保守辅助，M5 删除，M4 降级为计算型可选

## 8.3 v5.2：结论加固（G1–G4 全部判决）

> 计划文档：`TOKEN_BANK_V52_PLAN.md`。离线重分析脚本 `_v52_reanalyze.py`，全部数值在 `results/v52_reanalysis.json`，完整报告 `v52_reanalysis_report.md`（含 CI/McNemar/EM-F1/Pareto/审计表）。本节只记判决。

### R（离线重算，0 卡）

- **R1 显著性**：NoPool vs Unc 方向一致 8/8，**显著 6/8**——MATH（+18.5/+18.8pp，p≈1e-15）、FRAMES（+4.5/+5.3pp，p≤1.3e-4）、MuSiQue（+2.6/+3.2pp，p≤1.8e-5）显著；**GAIA 两格不显著**（p=0.180/0.581），GAIA 的 SR claim 收缩为"方向一致"
- **R2 题集对齐**：24k MATH Unc 同集 52.8%（原用全量 53.2%）；FRAMES Unc 同集 10.6%（原 11.5%），增益反而升至 +4.5pp 且显著。更正既往口径：16k MATH=test(400)、FRAMES test=660
- **R3/R5（I1 离线证据）**：ΔF1≈ΔEM（MuSiQue 的 ΔF1 甚至更大）；格式增益上界估计 MATH ≤36%、FRAMES ≤38%、GAIA/MuSiQue =0——离线证据不支持格式假说，但 C1 给出了相反结论（见下），**以 C1 为准**
- **R4 分流审计**：检索型任务 99% 升 L2 时，L0/L1 探路仅 237–276 tok（3.6–5.3%）且探路答案本就正确的 ≤0.6%——**探路是廉价但近乎纯开销的路由税**；token 节省主因是 M1 在 L2 内部收敛（触发率 42–47%）

### C1 — Unc+Fmt 对照【G1 判决：I1 坐实】

Unc 仅加 `ANSWER:` 指令（新配置 `V5-Unc-Fmt`），GAIA/FRAMES × 24k/16k：

| 格 | NoPool | Unc+Fmt | Unc | NoPool vs Unc+Fmt（McNemar） |
|---|---|---|---|---|
| GAIA 24k | 20.5% | 19.7% | 15.6% | p=1.000（7/6） |
| GAIA 16k | 17.2% | **19.7%** | 14.8% | p=0.508（3/6） |
| FRAMES 24k | 15.2% | 14.8% | 10.6% | p=0.894（29/27） |
| FRAMES 16k | 15.3% | 14.2% | 10.0% | p=0.371（26/19） |

**GAIA/FRAMES 的 SR 增益几乎全部是 `ANSWER:` 抽取格式的贡献**——M3.4 重新归类为"评测协议对齐"而非治理机制，Unc+Fmt 成为这两集的正式基线。对该基线，v5.1 仍保：token −12%~−23%（如 GAIA 24k 10,259 vs 13,264）。MATH（R5 上界 ≤36%，+18.5pp 中 ≥64% 是真实质量）与 MuSiQue（上界 0）的 SR 增益不受影响。

### Phase Ab — v5.1 自身消融【G2 判决：级联只省 token，不贡献 SR】

GAIA+MATH @24k test，McNemar 全部不显著：

| 配置 | MATH SR / tok | GAIA SR / tok |
|---|---|---|
| NoPool（完整） | 71.2% / 3,283 | 20.5% / 10,259 |
| Ab3 **−M2**（固定管线+M1+M3） | 70.8% / **5,648** | 19.7% / 10,343 |
| Ab1 −M1 | 70.2% / 3,281 | 18.9% / 10,438 |
| Ab2 −M3.2 | 70.8% / 3,139 | 19.7% / 10,379 |

- **M2 级联的 SR 贡献在所有已测数据集上 ≈0**（MATH p=0.82）：MATH 的 +18.5pp 逆转不是级联的功劳（无级联的 Ab3 同样 70.8%），扛 SR 的是 M3 上下文经济学（+格式对齐）——Ab1/Ab2 排除了 M1、M3.2 单独的必要性
- M2 的真实价值 = **计算型任务的 token 效率开关**：MATH 上同分省 42%（77% 的题停在 L0）
- v5 级联在 MATH 的反噬（49.2% < Unc 52.8%）本质是 logprob 路由把该升级的题按在 L0；v5.1 语言化升级判据修好了"不错误降档"，但"完全不路由"（Ab3）在 SR 上等价——只是贵 72%

### X1 — Gemma × V5.1-NoPool【G3 判决：修复跨模型成立】

Gemma-4-31B-it（llama.cpp QAT q4_0），24k test：

| 数据集 | V5.1-NoPool | v5 A0（纯 logprob 级联） | Unc |
|---|---|---|---|
| MATH (400) | **64.8%** / 2,035 tok | 36.0% / 2,223 tok（反噬） | 39.0% / 9,112 tok |
| GAIA (122) | **15.6%** / 9,569 tok | 13.1% / 11,488 tok | 12.3% / 12,337 tok |

MATH **+25.8pp 且 token −78%**（比 Qwen 的 +18.5pp 还强），同一模型上 v5 级联是反噬的——**语言化升级判据的修复效果跨模型成立**。GAIA +3.3pp 同向（n=122，弱证据）。

> **引用限制（v5.3 Z1b 追加）**：这里的 +25.8pp 基准是 **Unc（无格式）**，与 Qwen 的 +18.5pp 同一口径——而 E1 已证明该口径在 Qwen 上有 57% 是格式贡献。Gemma 没有 Unc+Fmt 对照臂，所以 **+25.8pp 不能作为方法 SR 增益的跨模型证据**。本节真正成立的是**反噬-修复** claim：A0（纯 logprob 级联）36.0% < Unc 39.0% 是 v5 臂之间的内部比较，两臂都带格式，与格式无关，跨模型结论不受影响。

### V1 — 池化失效的真实机制【G4 判决：分支 (a)，文档原解释错误纠正】

代码 + trace 双重核实：

1. conf=0 修复**生效**：ESCALATE 任务按 v_i=(1−conf)/c_i 确实排**池首**；原"排池尾被牺牲"的解释错误
2. 池容量充足：没有任务被困在 ESCALATE 状态（6 个 trace 文件 0 例），排除"池耗尽截流"
3. 真实机制 = **draft 锚定的提前收敛**：Unified 丢分的题 100% 是 L2 题（MuSiQue 69/69、FRAMES 36/36），这些题上 Unified 平均少花 ~25% token；其中 stopped_by=converged 的 31 题里 **21 题最终答案与探针轮次 snapshot 完全相同**——pass-2 带 `_v5_draft` 从 L2 起步，M1 稳定性信号看到"答案与 draft 一致"提前收火
4. 这解释了池化的符号性：MATH 探针收敛是真会（池化白赚 8–12% token），检索题探针 draft 是假稳定（池化亏 1.1–2.5pp）
5. 探针花费确实并入总账（`run_m4_pool` 合并逻辑核实），Unified 的 token 数字口径无误

### V2 — Unc 基线确切定义（复现清单）

与 V5.1-NoPool **相同**：全局 hard budget 帽（=当次 sweep 预算）、final_reserve=256 兜底、exec_compact、max_rounds=5、同模型同工具（代码执行）、temp=0、logprob 采集。
**Unc 缺的**：M2 级联（所有题直接 L2 完整管线）与语言化升级通道；M1 早停（跑满 5 轮或预算耗尽）；M3.1 结构化 reviewer（自由文本）；M3.2 工具笔记（原文留存）；M3.3 plan 冻结（每轮全文）；**M3.4 `ANSWER:` 指令**（I1 的根源，已由 C1 定量）。

### §6.5 run-to-run 翻转率（替代多种子）

NoPool × GAIA × 24k 同配置重跑：**逐题翻转 8/122 = 6.6%**，SR 20.5%→17.2%（±3.3pp），平均 token 相对差 4.5%。**经验下界：GAIA（n=122）上 <3pp 的差值不值得解释**；vLLM 连续批处理在 temp=0 下也非严格确定。

### v5.2 总判决

- **保住的**：H2（语言化升级修复级联反噬，跨模型 ×2）；MATH/MuSiQue 的 SR 增益（显著、抗格式检验）；全线 token 节省（对任何基线都成立）
- **收缩的**：GAIA/FRAMES 的 SR 增益归入格式对齐（基线换 Unc+Fmt 后剩 token 效率 claim）；GAIA 整体证据弱（不显著 + 翻转率 6.6%）
- **改写的**：M2 定位 = 计算型 token 效率开关（非"核心创新"）；扛 SR 的是 M3+M1；M4 失效机制 = draft 锚定提前收敛（非池尾截流）

## 8.4 v5.3：收口（归因定案 + 边界定稿）

> 计划：`TOKEN_BANK_V53_PLAN.md`。Phase Z 报告 `v53_phasez_report.md` + `results/v53_phasez.json` + `figures/v53_pareto_*.png`。本节记最终判决。

### Z（离线）

- **Z1 过度修订审计**：Unc 臂"曾答对→最终答错"翻转——MATH 33 题（含 17 个 budget-finalize 伪影：预算耗尽后收尾调用答错）→ 净翻转 +24 题（vs NoPool 的 9 题），占 MATH 无归属增益的 50%；MuSiQue 净翻转 **−31（反向）**，过度修订解释不了 MuSiQue。条件翻转率：MATH 13.5%→3.1%、GAIA 29.6%→13.8%（M1 压制）；FRAMES/MuSiQue 反向（NoPool 在 max_rounds 跑满后改错更多）
  > **MATH 一格已被 Z1b 取代**：Z1 跑在 E1 之前，`_v53_phasez.py` 的 `FC1` 只映射了 GAIA/FRAMES 的 Unc+Fmt 轨迹，MATH 的 `v53e1_ma{24,16}` 从未进过审计——于是"增益对 Unc+Fmt 测、归因对 Unc 审"，口径不一致。MuSiQue/GAIA/FRAMES 的结论不受影响（这三集的 gate 判决只用净翻转的符号）。
- **Z1b MATH 归因复核（对正式基线 Unc+Fmt，补跑）**：见下方专节。要点：净翻转份额从 75% 降到 **53%**（24k），逐题因果口径只有 **47%**；`−M1` 消融在 MATH 上是构造性 no-op；**+8.0pp 中 +4.5pp 长在 L0（单轮直答）子集上**
- **Z2**：Unc+Fmt 4 格补齐 CI/McNemar，NoPool vs Unc+Fmt 全部不显著（p=0.37~1.0）
- **Z3 + E3（G-C 判决）**：8k 第三预算点补上后，GAIA/FRAMES 上 NoPool 与 Unc+Fmt **三点前沿重合**（8k 差值 +1.6/+0.9pp，均在噪声下界内；FRAMES 8k 处 Unc+Fmt 甚至更省 token）。两集最终表述：**同一条 Pareto 前沿，无 SR 增益、无稳定 token 优势**
- **Z4（重要发现）**：GAIA 两次 Unc+Fmt（24k/16k）SR 同为 19.7%，逐题正确集对称差 = 6.6%，**恰好等于同配置重跑翻转率**——正确集差异就是运行噪声本身：**GAIA 上预算帽对朴素管线不绑定**
- **Z5 + E4**：MATH 翻转率 22/400=5.5% ≈ GAIA 6.6%，运行噪声 ~6% 是 vLLM 连续批处理的系统特性。双噪声合并后：MATH 7.1×SE、MuSiQue 3.1×SE（当时口径），GAIA/FRAMES 为零

### E1 — MATH × Unc+Fmt（G-A 判决：通过，分解 57/43）

| MATH | NoPool | Unc+Fmt | Unc |
|---|---|---|---|
| 24k | 71.3% / 3,283 | 63.2% / 6,489 | 52.8% / 6,333 |
| 16k | 70.5% / 2,965 | 62.5% / 5,158 | 51.8% / 5,362 |

格式占 18.5pp 中的 10.5pp（**57%**，R5 离线估计 ≤36% 失真）；**真实质量 +8.0pp，McNemar p=1.9e-8（34/2）**。这一格是全文最硬的数字：四个消融臂（−M1/−M2/−M3.2）全部保住 +7.0~+8.0pp，p 均 ≤6e-7。

**真实增益的机制归属见 §8.4b——不是原先写的"M1 主体（75%）"。**

### E2 + E2b — MuSiQue 归因（G-B 判决：全部 null，MuSiQue 的 SR claim 撤销）

| MuSiQue 24k | SR | tokens | vs NoPool |
|---|---|---|---|
| NoPool | 9.7% | 6,582 | — |
| −M1 | 9.8% | 6,564 | null |
| −M3ctx | 10.0% | 8,451 | p=0.61，null |
| −M2 | 9.2% | 6,350 | p=0.362，null |
| **Unc+Fmt** | **9.8%** | 8,220 | **p=0.842，null** |
| Unc | 7.0% | 7,665 | — |

**没有任何单一机制对 MuSiQue 的增益是必要的；Unc+Fmt 单独即达到 9.8%——格式解释了 +2.8pp ≥ 全部增益 +2.6pp。** R5 的"MuSiQue 格式上界 0"判断错误（其离线界在 MATH 上也失真：估 36% 实测 57%）——教训：**格式/抽取效应只能由对照臂实测，离线界不可信**。MuSiQue 残余 claim = 同分省 token 20%（6,582 vs 8,220）。

## 8.4b Z1b：MATH 机制归因复核（推翻 §8.4 的"M1 主体"）

> 全部数值 `results/v53_phasez.json` → `z1b_math_attribution_vs_unc_fmt`，报告 `v53_phasez_report.md` §Z1b。零 GPU，纯 trace 重算。

**发现的口径错误**：Z1 的翻转审计做在 **Unc** 臂上，但 E1 之后 MATH 的正式基线是 **Unc+Fmt**。增益对 Unc+Fmt 测（+8.0pp），归因对 Unc 审（gap 18.5pp）——两个不同的 gap 混用，系统性高估过度修订的份额。原因很直接：`ANSWER:` 格式本身就消掉了大部分 budget-finalize 伪影（24k：Unc 17 → Unc+Fmt **9**），这部分不该记在 M1 头上。

| MATH 24k | SR | token | 翻转(strict+budget) | 条件翻转率 |
|---|---|---|---|---|
| Unc | 52.8% | 6,333 | 33 (16+17) | 13.5% |
| **Unc+Fmt**（正式基线） | 63.2% | 6,489 | **26 (17+9)** | 9.3% |
| NoPool | 71.2% | 3,283 | 9 (9+0) | 3.1% |

**改正后的归因**（24k / 16k）：

| 口径 | 24k | 16k |
|---|---|---|
| 净翻转占真实增益 | **53%**（+17/32 题） | 72%（+23/32 题） |
| 逐题因果口径（NoPool 赢的题中基线"曾答对又改错"的比例） | **47%**（16/34） | 52%（21/40） |
| ~~旧口径（对 Unc）~~ | ~~75%~~ | ~~—~~ |

逐题口径比净翻转更严：净翻转是聚合差，会把两边各自的翻转相消。NoPool 赢的 34 题里，**18 题基线从头到尾没答对过**——那不是过度修订，是朴素管线本身没解出来，M1 无从施力。

### 消融臂复核：`−M1` 在 MATH 上测不到 M1

| 臂 | SR | token | 翻转 | 路由 | Δ vs Unc+Fmt | McNemar |
|---|---|---|---|---|---|---|
| NoPool（完整） | 71.2% | 3,283 | 9 (9+0) | L0 309 / L1 11 / L2 80 | +8.0pp | 34/2, p=1.9e-8 |
| Ab1 −M1 | 70.2% | 3,281 | 7 (7+0) | L0 **305** / L1 18 / L2 77 | +7.0pp | 32/4, p=1.9e-6 |
| Ab3 −M2（强制全 L2） | 70.8% | 5,648 | **17 (17+0)** | L2 400 | +7.5pp | 34/4, p=6.0e-7 |
| Ab2 −M3.2 | 70.8% | 3,139 | 7 (7+0) | L0 310 / L1 12 / L2 78 | +7.5pp | 34/4, p=6.0e-7 |

两个结构性读数：

1. **Ab1（−M1）是构造性 no-op，不能当证据用**。级联开着时 305/400 题走 L0 单轮直答——只有一轮，就没有"改错"的机会，M1 天然无事可做。Ab1 只掉 1.0pp 不说明"M1 不重要"，只说明这个消融设计在 MATH 上测不到它。§8.3 Phase Ab 用 Ab1 排除"M1 的单独必要性"，该推断对 MATH **不成立**。
2. **Ab3（强制全 L2）才是 M1 能施力的环境**：翻转 26 → 17，其中 budget-finalize 9 → 0、strict 17 → 17。
   > ⚠️ **该读数已被 Z1c 修正**：Ab3 相对 Unc+Fmt 同时改了 M1 **和** M3，把两者的效果混在一起了。补上中间臂 Ab6（+M3、−M1）后可见，budget-finalize 的消失是 **M3** 的功劳（省 token → 不再耗尽预算），而 M1 消掉的恰恰是**真·中途改错**（strict 25 → 17）。详见 §8.4c。

### 增益长在哪一层：+8.0pp 中 +4.5pp 在 L0

| NoPool 路由层 | n | NoPool SR | Unc+Fmt SR | 不一致(赢/输) | 占总增益 |
|---|---|---|---|---|---|
| **L0（单轮直答）** | 309 | 79.0% | 73.1% | 20/2 | **+4.5pp** |
| L1 | 11 | 63.6% | 36.4% | 3/0 | +0.8pp |
| L2（完整管线） | 80 | 42.5% | 28.7% | 11/0 | +2.8pp |

**头条战果的主体形态是"数学题不要跑多 agent 管线"**，不是预算治理。L0 那一层的"方法"字面上就是单轮直答；朴素管线在同一批题上反而低 5.9pp。这个发现本身站得住（20/2 的不一致分布），但它是关于**多轮管线在计算型任务上有害**的证据，不是关于预算分配的证据。

> **§8.4c 的补充**：这里的"+4.5pp 在 L0"不等于"L0 这个动作创造了增益"——它度量的是**朴素管线在这批题上造成的损害**。Ab6/Ab3 证明同一份损害也可以靠把管线修好（M3+M1）来消除。两条路径互替，这才是各消融看似 null 的根因。

### 遗留缺口（v5.3 未闭合，写作时必须声明）

| 缺口 | 状态 | 补法（成本） |
|---|---|---|
| ~~无任何单机制消融能复现 +8.0pp~~ | ✅ **已闭合**（Z1c/Ab6）：所有消融都在级联阴影下做，机制无题可管；关掉级联后可加法分解 5.2+2.2+0.4≈8.0pp。见 §8.4c | 已跑，400 题·次 |
| M3ctx 与 M5 未拆分 | **新缺口**：+5.2pp 是 M3.1/3.2/3.3 与 `m5_strategy=hybrid` 的合计（`V5-Unc-Fmt` 不带 m5 键） | Ab6 再去掉 M5，**400 题·次** |
| 跨模型 Gemma 无 Unc+Fmt 对照 | +25.8pp 是对 **Unc**（无格式）测的——正是 C1/E1 在 Qwen 上证明会虚高 57% 的那种比较 | Gemma × Unc+Fmt × MATH，**400 题·次**（已排队，`_chain_v54_p2.sh`） |
| 无 `V5.1 − 格式` 臂 | 格式/方法的分解只有单边（有"基线+格式"，无"方法−格式"） | 可选，400 题·次 |

跨模型的**反噬-修复** claim 不受影响：Gemma 上 A0（纯 logprob 级联）36.0% < Unc 39.0% 是 v5 臂之间的内部比较，与格式无关。受影响的只有把 +25.8pp 当作方法 SR 增益的跨模型证据这一句。

## 8.4c Z1c：Ab6 隔离臂——MATH 的机制归因终于打通

> 配置 `V5.1-Ab6-NoM2NoM1`（`_budget_sweep_v2.py`），链脚本 `_chain_v54_z1b.sh`，产物 `results/budget_sweep_v2_Qwen3.6-27B-FP8_v54a_ma_ab6.json` + `traces/..._v54a_ma_ab6.jsonl`，统计见 `results/v53_phasez.json` → `z1c_gap_closing`。MATH test(400) × 24k × seed42 × temp0。

**为什么要这个臂**：此前所有单机制消融都是在**级联开着**的条件下做的，而级联把 305~310/400 道 MATH 题停在 L0 单轮直答——任何"改善管线"的机制都只剩 ~23% 的题可作用，拿掉它自然只掉 ~1pp。把级联关掉（强制全 L2），机制才进入可观测区。

### 固定管线下的加法分解

| 臂（均为固定 L2 管线） | SR [95% CI] | token | 翻转(strict+budget) | 增量 | McNemar |
|---|---|---|---|---|---|
| Unc+Fmt（仅格式） | 63.2% | 6,489 | 26 (17+**9**) | — | — |
| **Ab6** = +M3ctx+M5 | **68.5%** [64.0, 73.0] | 5,750 | 25 (**25**+0) | **+5.2pp** | 36/15, **p=0.0046** |
| **Ab3** = +M1 | **70.8%** [66.2, 75.2] | 5,648 | 17 (**17**+0) | **+2.2pp** | 22/13, p=0.175 |
| NoPool = +M2 级联 | 71.2% | **3,283** | 9 (9+0) | +0.4pp | — |

**5.2 + 2.2 + 0.4 = 7.8 ≈ 8.0pp**——总增益回来了。此前"无单一机制归属"的困惑，根源就是所有消融都在级联的阴影里做。

### 两条被修正的结论

1. **M1 确实在防真·过度修订，不是只防截断伪影**。§8.4b 从 Ab3 vs Unc+Fmt 读出的"strict 17→17、budget-finalize 9→0"是**两个机制混在一起**的假象。插入 Ab6 后各归各位：
   - **M3（省 token）消掉 budget-finalize**：9 → 0。管线每轮花得少（6,489 → 5,750），任务不再耗尽预算，也就没有劣质收尾
   - **M1 消掉真·中途改错**：strict 25 → 17，条件翻转率 8.4% → 5.7%

   （Ab6 的 strict 比 Unc+Fmt 的 17 还高，是因为 M3 把原本"死于预算耗尽"的 9 题救活进了循环，它们转而暴露在多轮改错风险下。）

2. **M1 的 SR 效应是方向性的，不显著**：+2.2pp，配对 CI [−0.5, +5.2]，**p=0.175**。n=400 上分辨不出 2.2pp。可以说"M1 减少过度修订（翻转 25→17）且 SR 方向为正"，**不能说"M1 带来 +2.2pp"**。

### 归因中真正扛 SR 的是 M3ctx（+M5）

**+5.2pp，p=0.0046**——这是 MATH 上唯一显著的单机制增量。但两点限定：

- **含 M5 混淆**：`V5_FULL` 同时带 `m5_strategy='hybrid'`，而 `V5-Unc-Fmt` 没有该键。所以 +5.2pp 是 **M3.1+M3.2+M3.3 与 M5 的合计**，不是纯 M3ctx。要拆开还需一臂（400 题·次），本轮未做
- **与 MuSiQue 的 −M3ctx null 不矛盾**：那次是在级联下测的（大部分题不走管线），且数据集不同。M3ctx 的价值条件化于"管线真的在跑"

### M2 级联的最终定位

级联在 SR 上只加 +0.4pp（Ab3 → NoPool），却把 token 从 5,648 砍到 3,283（**−42%**）。所以它**不是增益来源，是同增益下的省钱开关**——这与 §8.3 的结论一致，且现在有了正确的对照臂（Ab3 而非 Unc+Fmt）。

同时这解释了 §8.4b 的路由拆分：+4.5pp 长在 L0 子集上，并不是"L0 这个动作创造了增益"，而是 **L0 题在朴素管线下被多轮改坏了**（Unc+Fmt 在该子集 73.1%，NoPool 79.0%）。级联通过"根本不跑管线"规避了这个损失，M1+M3 通过"把管线修好"规避了同一个损失——**两条路径治的是同一个病**，这才是它们互为冗余的真正原因。

### 最终归因表（v5.3 定案）

| 数据集 | SR 增益归属 | 残余 token claim（vs Unc+Fmt，24k） |
|---|---|---|
| MATH | 格式 **57%** + 真实质量 **43%**（+8.0pp，p=1.9e-8）。真实质量在固定管线下可加法分解（§8.4c）：**M3ctx+M5 +5.2pp（p=0.005，唯一显著项）**、M1 +2.2pp（p=0.175，方向性）、M2 级联 +0.4pp（但省 42% token） | −49%（3,283 vs 6,489） |
| MuSiQue | **全部是格式**（Unc+Fmt 追平） | −20%（6,582 vs 8,220） |
| GAIA | **全部是格式**（C1）；且预算帽对朴素管线不绑定（Z4） | 同前沿（−23% 但 8k 处不稳） |
| FRAMES | **全部是格式**（C1） | 同前沿（−12% 但 8k 处被反超） |

机制定位终版（Z1b + Z1c 修订）：

- **M3.4 格式 = 评测协议对齐**——最大单一块（MATH 57%，其余三集 100%），但不算治理功劳
- **M3.1/3.2/3.3 上下文经济学（+M5）= MATH 上唯一显著的 SR 贡献项**：固定管线下 +5.2pp，p=0.0046。机制是省 token → 任务不再耗尽预算 → 消灭 budget-finalize 劣质收尾（9→0）。**注意 M5 混淆未拆**（见 §8.4c）。此前"M3ctx 对 SR 无必要贡献"的结论只在 MuSiQue、且只在级联下成立
- **M1 收敛早停 = 防真·过度修订**：固定管线下把 strict 中途改错 25→17、条件翻转率 8.4%→5.7%；SR +2.2pp 但 **p=0.175 不显著**，只能作方向性陈述
- **M2 语言化级联 = 防反噬（跨模型实证，不受格式混淆）+ 省钱开关**：SR 上仅 +0.4pp（vs Ab3），token −42%。§8.3"M2 对 SR 贡献 ≈0"的结论在正确对照臂下依然成立
- **M4** 对 SR 无独立必要贡献

**两条已作废的说法**：①"M1 是 MATH 真实增益主体（75%）"——错基线审计 + 测不到 M1 的 Ab1；②"M1 只防预算截断、不防真改错"——Ab3 相对 Unc+Fmt 同时动了 M1 和 M3，是混淆读数，Ab6 已纠正。

### 最终叙事（plan 形状 1 与 3 的混合）

*"固定预算下：格式对齐是免费但必要的基线修正，它解释了四个数据集上全部或大部分的表观增益；在此之上只有计算型任务还剩真实增益——MATH +8.0pp（p≈2e-8）。该增益在固定管线下可加法分解：**上下文经济学 +5.2pp（p=0.005）**、防过度修订 +2.2pp（方向性）、级联 +0.4pp 但省 42% token。朴素多轮管线在计算题上会把已答对的答案改坏、并在预算耗尽时给出劣质收尾；**修好管线（M3+M1）与不跑管线（M2 级联降档）治的是同一个病**，因而互为冗余——这也是此前每个单机制消融都看似 null 的原因。语言化升级判据让级联在计算型任务上不再反噬（双模型实证）；全线对格式对齐基线同分省 token 12~49%。知识问答型任务上与格式对齐基线同前沿——方法的边界在此。"*

> **叙事强度评估（Z1c 后上修）**：这落回 V5.3 计划 §3 的"形状 1"——机制是清楚的，只是主角从计划预期的 M1 换成了 M3ctx。可写的核心命题有两条，都有对照臂支撑：①**多轮 agent 管线在计算型任务上有净损害**（Unc+Fmt 63.2% vs 单轮直答路由 71.2%）；②**这个损害有两条互替的修法**——把管线修好（+7.5pp，但贵）或干脆降档（同等 SR，省 42% token）。仍需诚实声明的是：M3ctx 与 M5 未拆分；M1 的 SR 效应不显著；GAIA/FRAMES/MuSiQue 上方法与格式对齐基线同前沿。

## 9. 产物索引






| 内容 | 路径 |
|---|---|
| Phase0 诊断 | `results/budget_sweep_v2_Qwen3.6-27B-FP8_v5diag_{ga,mu}.json` + 同名 traces |
| dev 校准（全轨迹） | `results/..._v5cal_{ga,mu}.json` + traces |
| dev 消融 | `results/..._v5abl_{ga,mu}.json` + traces |
| test 主结果 | `results/..._v5test_{ga,mu}.json` + traces |
| 紧预算（8k/16k） | `results/..._v5tight_{ga,mu}.json` + traces |
| Gemma 确认 | `results/budget_sweep_v2_gemma-4-31B-it_v5gem_{ga,ma}.json` + traces |
| v5.1 dev 验证 | `results/..._v51_{ma,mu,ga}.json`、`..._v51mufix.json`、`..._v51gafix.json` + traces |
| v5.1 test 主结果 | `results/..._v51t_{ga,ma,fr,mu}.json` + traces |
| v5.1 16k 紧预算 | `results/..._v51t16_{ga,ma,fr,mu}.json` + traces（日志 `logs/_v51test16_*.log`，链脚本 `_chain_v51_16k.sh`） |
| v5.2 离线重分析 | `_v52_reanalyze.py`、`results/v52_reanalysis.json`、`v52_reanalysis_report.md`、`figures/v52_pareto_*.png` |
| v5.2 C1（Unc+Fmt） | `results/..._v52c1_{ga24,ga16,fr24,fr16}.json` + traces（`_chain_v52_c1.sh`） |
| v5.2 X1（Gemma 修复确认） | `results/budget_sweep_v2_gemma-4-31B-it_v52x1_{ma,ga}.json` + traces（`_chain_v52_x1.sh`） |
| v5.2 消融（v5.1 自身） | `results/..._v52ab_{ga,ma}.json` + traces（`_chain_v52_ab.sh`） |
| v5.2 run-to-run 重跑 | `results/..._v52rr_ga.json` + traces（`_chain_v52_rr.sh`） |
| v5.3 离线分析 | `_v53_phasez.py`、`results/v53_phasez.json`、`v53_phasez_report.md`、`figures/v53_pareto_*.png` |
| v5.3 Z1b 归因复核（§8.4b） | 同上脚本的 `z1b_*` 段 + 报告 §Z1b；输入是 E1 的 `..._v53e1_ma{24,16}` 与消融的 `..._v52ab_ma` |
| v5.3 E1（MATH×Unc+Fmt） | `results/..._v53e1_ma{24,16}.json` + traces |
| v5.3 E3（8k 第三预算点） | `results/..._v53e3_{ga,fr}8.json` + traces |
| v5.3 E4（MATH 翻转率） | `results/..._v53e4_ma.json` + traces |
| v5.3 E2/E2b（MuSiQue 归因） | `results/..._v53e2_mu_nom{1,3}.json`、`..._v53e2b_mu_{nom2,fmt}.json` + traces（`_chain_v53_e2.sh`、`_chain_v53_e2b.sh`） |
| 日志 | `logs/_v5diag_*.log`、`logs/_v5cal_*.log`、`logs/_v5abl_*.log`、`logs/_v5test_*.log`、`logs/_v5tight_*.log` |
| 链脚本 | `_chain_v5_phase0.sh`、`_chain_v5_calib.sh`、`_chain_v5_abl.sh`、`_chain_v5_test.sh`、`_chain_v5_tight.sh` |

## 10. 复现要点

- v5 实验统一入口：`V5=1` 启用 `CONFIGS_V5`；`V5_CALIB=1` 为校准模式（关所有早停）；阈值可用 `THETA0/THETA1/TAU_CONF/TAU_HIGH/M5_STRATEGY` 环境变量覆盖
- 判题器脱水后 SR 与 v3 数字**不可比**，引用时必须注明判据版本
- v5 配置取消 per-role 配额（`bank.q_min=budget`），只留全局 reserve——这是 M0 删除借贷后的必要配套
- `pgrep -f` 会匹配自身 bash wrapper 导致假阳性，核实进程用 `pgrep -af ... | grep -v grep`
- **离线分析脚本的 CI 复现性**：`_v53_phasez.py` 的 `get()/get_uf()` 必须按 `sorted(TEST_IDS)` 取记录。bootstrap 重采样对记录顺序敏感，而 `TEST_IDS` 是 set、迭代顺序随 `PYTHONHASHSEED` 变化——不排序则同一份数据逐次运行 CI 会漂 0.2~0.8pp（SR/token 是顺序无关的，不受影响）。该问题已修，修复前后所有结论不变
- **新增对照臂后必须回头复查既有的离线归因**：Z1/Z1b 的教训是"增益对新基线测、归因还对旧基线审"——`FC1` 之类的轨迹映射表加了数据集就要同步，否则整段因果叙事会挂在错误的对照上
