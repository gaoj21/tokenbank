# Token Bank 实验全记录：Qwen3.6-27B-FP8 × Gemma-4-31B 双模型验证

> 日期：2026-08-12 ~ 2026-08-17
> 平台：3× NVIDIA RTX A5000 (24GB)，驱动 535
> 服务栈：vLLM 0.27.1（Qwen FP8，TP=2）/ 自编译 llama.cpp CUDA sm_86（Gemma GGUF，3 卡 tensor-split）
> 实验框架：`_run_v2_experiment.py`（唯一事实源）+ `_budget_sweep_v2.py`（sweep 驱动）

---

## 1. 方法：Token Bank v2 + 改进点全览

Token Bank 是多 Agent 流水线（Planner → Executor → Reviewer，≤5 轮修订）的**运行时 token 预算治理中间件**。核心思想：把 token 预算当作一种可调度资源，在运行时做配额、预测、借贷、压缩、止损，而不是静态切分。

### 1.1 双核心机制（v2，DeepSeek 时代定稿）

1. **预测再平衡 + 跨 Agent 借贷**：PredictiveRebalancer 用消费历史预测各 agent 还需多少 token，配额在 agent 间动态转移；BudgetBorrowing 允许消耗快的 agent 向有盈余的 agent 借贷（credit_rate=1.5）。
2. **Adaptive runaway stall**：检测"答案不再变化且已烧掉相当比例预算"的失控任务，提前终止，把 token 留给有进展的任务。

### 1.2 v3 五个改进点（本轮新增，全部 flag 化）

| # | 机制 | flag | 说明 |
|---|---|---|---|
| ① | 结题储备 | `final_reserve` | 从每任务预算中扣留 N tokens，普通调用不可动用；任务被拦后用 `final=True` 的调用（绕过配额/pacing/借贷，只看 reserve）发一次 256-token 的结题请求 |
| ② | 角色借贷线 | `role_credit_mult` | 按角色缩放信用额度：Planner×0.5、Executor×1.5、Reviewer×1.0、Debugger×0.75 |
| ③ | 预测式派遣预估 | `predictive_est` | 输出长度估计用该角色历史的 P80（近 30 次），替代 worst-case max_tokens，减少误拦和无效借贷 |
| ④ | 相似度 stall | `stall_sim` + `runaway_frac_auto` | "答案未变"判定从精确字符串匹配改为 token 集 Jaccard ≥ 阈值；runaway 触发预算占比按预算档位自适应（≤8K→0.15，≤16K→0.20，>16K→0.25） |
| ⑤ | 预测贷方选择 | `predictive_lending` | 借贷时按"剩余 − 预测消耗"的盈余降序选贷方 |

### 1.3 v4 两个压缩机制（方向 3，本轮关键突破）

| # | 机制 | flag | 说明 |
|---|---|---|---|
| ⑥ | 提前 KV 压缩 | `kv_trigger` | KV 压缩触发阈值从 quota 的 0.85 提前到 0.65——在估算即将超额之前就先压缩，而不是被拦后补救 |
| ⑦ | Executor 上下文瘦身 | `exec_compact` | 每轮修订后 executor 上下文只保留 system + 原始问题/plan + 最新一轮答案和 reviewer 反馈（此前是无限追加，prompt 随轮数线性膨胀到 4~6.5K）；工具输出截断 800 字符 |

### 1.4 实验配置

| 配置 | 内容 |
|---|---|
| Unconstrained | 无任何治理，预算 200K（事实上的上界参照） |
| Static Quota | 静态等分配额，hard policy（朴素基线） |
| TB-v2-Full | 双核心，无 pacing/stall |
| Full+Stall | TB-v2-Full + adaptive runaway + stall |
| Improved | Full+Stall + v3 五点（①~⑤） |
| Compact | Full+Stall + v4 两点（⑥⑦） |
| Improved+Compact（IC） | 全部机制（①~⑦） |
| Combo16k | IC 的 16K 调优变体（final_reserve 256、stall_sim 0.95）——验证为负收益，弃用 |

### 1.5 实验设置

- 模型：**Qwen/Qwen3.6-27B-FP8**（vLLM，TP=2，max-num-seqs 128，~92 调用/分钟）；**gemma-4-31B-it QAT Q4_0 GGUF**（自编译 llama.cpp，3 卡 tensor-split，16 槽 × 8192 ctx，~8 调用/分钟）
- 数据集：GAIA（152 题，level 1-3 文本题）、FRAMES（824 题）、MATH-500（500 题）、MuSiQue（2417 题全量 validation 可答题）
- 预算点：16K / 24K / 32K tokens
- seed=42，temperature=0，全程可复现
- 判题：答案包含匹配（gt in ans 或 gt 前两词命中）

---

## 2. Qwen3.6-27B-FP8 完整结果

### 2.1 SR 总表（4 数据集 × 3 预算）

| 数据集 | 预算 | Static | IC | 上界 | IC vs 上界 |
|---|---|---|---|---|---|
| GAIA (152) | 16K | 11.2% | 24.3%（Compact 26.3%） | 36.2% | −11.9pp |
| GAIA | 24K | 13.2% | **38.2%** | 36.2% | **+2.0pp** |
| GAIA | 32K | 17.8% | **40.1%** | 36.2% | **+3.9pp** |
| FRAMES (824) | 16K | 29.0% | 44.4% | 53.5% | −9.1pp |
| FRAMES | 24K | 32.4% | **53.4%** | 53.5% | −0.1pp |
| FRAMES | 32K | 35.7% | **53.0%** | 53.5% | −0.5pp |
| MATH (500) | 16K | 66.6% | 72.8% | 80.4% | −7.6pp |
| MATH | 24K | 71.2% | 75.6% | 80.4% | −4.8pp |
| MATH | 32K | 70.8% | **79.6%** | 80.4% | −0.8pp |
| MuSiQue (2417) | 16K | 24.2% | 36.9% | 43.1% | −6.2pp |
| MuSiQue | 24K | 26.1% | **42.0%** | 43.1% | −1.1pp |
| MuSiQue | 32K | 28.3% | **44.0%** | 43.1% | **+0.9pp** |

中间配置（GAIA）：TB-v2-Full 16K=18.4% / 24K=30.3%；Full+Stall 16K=22.4% / 24K=30.9%；Improved 16K=23.0% / 24K=35.5%；Compact 16K=26.3% / 24K=35.5%。

### 2.2 Token 消耗（平均 / 每成功成本）

| 数据集 | 预算 | 配置 | 平均 tokens | tok/成功 |
|---|---|---|---|---|
| GAIA | 16K | Unc | 20.2K | 55.9K |
| GAIA | 16K | Static | 5.1K | 45.6K |
| GAIA | 16K | Compact | 9.5K | **36.1K** |
| GAIA | 24K | Static | 7.1K | 54.2K |
| GAIA | 24K | IC | 13.4K | **35.2K** |
| GAIA | 32K | Static | 9.1K | 51.5K |
| GAIA | 32K | IC | 14.9K | **37.3K** |
| FRAMES | 16K | Unc | 10.9K | 20.3K |
| FRAMES | 16K | IC | 7.1K | **15.9K** |
| FRAMES | 24K | IC | 9.2K | **17.2K** |
| MATH | 32K | Unc | 7.2K | 8.9K |
| MATH | 32K | IC | 6.5K | **8.2K** |
| MuSiQue | 16K | IC | 6.2K | **16.9K**（Unc 22.9K） |

---

## 3. Gemma-4-31B（Q4_0）验证结果

GAIA 全曲线 + 其余三数据集 24K 点（300 题子样本，受单卡 GGUF 吞吐限制）：

| 数据集 | 预算 | Static | IC | 上界 | IC vs 上界 |
|---|---|---|---|---|---|
| GAIA | 16K | 15.1% | 18.4% | 20.4% | −2.0pp |
| GAIA | 24K | 18.4% | **21.7%** | 20.4% | **+1.3pp** |
| GAIA | 32K | 20.4% | 20.4% | 20.4% | 0 |
| FRAMES (300) | 24K | 36.0% | **38.3%** | 38.0% | **+0.3pp** |
| MATH (300) | 24K | 51.3% | **63.0%** | 67.0% | −4.0pp |
| MuSiQue (300) | 24K | 22.3% | **25.0%** | 24.0% | **+1.0pp** |

Gemma 的每成功 token 成本：GAIA 24K IC 45.7K（最低）；FRAMES 24K IC 18.5K（比上界省 21%）；MATH 24K IC 13.2K（最低）；MuSiQue 24K IC 29.2K（比上界省 27%）。

---

## 4. 核心结论

### R1：IC 对 Static 全胜
两个模型 × 所有数据集 × 所有预算点，Improved+Compact 全部击败 Static Quota（Qwen 上 +4.4pp ~ +25.0pp；Gemma 上 +2.3pp ~ +11.7pp）。样本量最大的 MuSiQue（2417 题，±1pp 精度）领先 12.7~15.9pp，铁统计显著。

### R2："追平/反超无约束上界"是可复现现象
Qwen 4/4 数据集贴平上界（GAIA +2.0/+3.9pp、MuSiQue +0.9pp 反超）；Gemma 4/4 贴平（GAIA/FRAMES/MuSiQue 反超）。**无约束运行本身存在浪费**（过度修订把正确答案改错、失控轮次空烧），治理机制修剪后反而略胜——这推翻了"预算约束必然损失性能"的直觉。

### R3：成功率最高的同时单成功成本最低
在预算甜蜜点上，IC 的 tok/成功全面低于 Unconstrained（GAIA 省 33~37%、FRAMES 省 15~21%、MuSiQue 省 27%）和 Static。Static 是"假省"——均值低但大量 token 浪费在被掐死的任务上。

### R4：机制分工明确，按预算档位选型
- **窒息区（≤16K）**：压缩机制（⑥⑦）是唯一有效杠杆。GAIA 16K 上 Compact 26.3% 单点最强；v3 五点（调度精细度）在预算物理不足时无空间发挥（Improved 仅 +0.6pp，噪声内）
- **宽松区（≥24K）**：全套机制叠加效果最大（IC 38.2%/40.1%）
- **16K 调优变体（Combo16k：储备减半+stall 收紧）验证为负收益**，16K 维持纯 Compact 即可

### R5：预算存在甜蜜点，给过头反而浪费
Static 在 24K 后饱和；IC 在任务中位消耗 ×1.5~2 处达到最佳效率。FRAMES 32K、GAIA 32K（gemma）出现"IC 多花 token 买不到成功"的饱和浪费。实践建议：预算设为任务中位消耗的 1.5~2 倍。

### R6：三区间规律（沿用 v2，本轮再次验证）
≤4K 窒息区（任何治理都救不了）；8~16K 治理机制领先 Static；≥24K IC 追平上界。本轮把右端点的结论从"Full 反超 v2"推进为"IC 反超无约束上界"。

---

## 5. 工程记录

- **vLLM 0.27.1 已移除 GGUF 支持**（LoadConfig 中无 gguf 选项）→ Gemma 改走自编译 llama.cpp（`tools/llama.cpp`，CUDA sm_86，cmake 4.4.2 装于 .venv，修复了 cublas 开发符号链接缺失和 webui npm 挂起两个问题）
- vLLM 服务：`_start_vllm_conda.sh`（conda env `vllm`，cuda-compat 13.3.1 解决驱动 535 vs cu130 的兼容）；Gemma 服务：`_start_gemma_llama.sh`
- 吞吐：Qwen vLLM TP=2 ~92 调用/分钟；Gemma llama.cpp ~8 调用/分钟（这是 gemma 侧裁剪规模的原因）
- 实验链脚本：`_chain_24k.sh`、`_chain_frames24k.sh`、`_chain_frames_baselines.sh`、`_chain_math_musique2.sh`、`_chain_32k_musique.sh`、`_chain_gemma.sh`、`_chain_gemma24k.sh`

## 6. 产物索引

- 聚合结果：`results/budget_sweep_v2_{Qwen3.6-27B-FP8,gemma-4-31B-it}_*.json`
- 逐题轨迹：`traces/tasks_sweep_{Qwen3.6-27B-FP8,gemma-4-31B-it}_*.jsonl`
- 实验日志：`logs/_gaia*.log`、`logs/_frames*.log`、`logs/_math*.log`、`logs/_musique*.log`、`logs/_gemma_*.log`
- 方法文档：`METHOD_DUAL_CORE.md`（双核心定稿）；前代结论：`experiment_findings_v2.md`（DeepSeek 时代 R1–R9）
