# Token Bank v2 项目说明文档
## 面向工程团队的快速上手指南

**版本：** v2.0 | **日期：** 2026-07-26 | **状态：** 实验验证阶段

---

## 1. 项目概述

### 1.1 这是什么？

Token Bank v2 是一个**多 Agent LLM 系统的运行时预算治理中间件**。简单来说：

> 当你有多个 AI Agent（如 Planner、Executor、Reviewer）协作完成任务时，Token Bank 负责给每个 Agent 分配 Token 预算、监控实时消耗、在预算不足时自动调配资源，确保总成本不超支。

### 1.2 为什么需要它？

当前多 Agent 系统（如 AutoGen、CrewAI）存在三个痛点：

| 痛点 | 后果 |
|------|------|
| **无预算上限** | 一个 Agent 可能无限调用工具，单次任务消耗 124K+ tokens |
| **静态分配** | 预分配的配额无法适应任务实际复杂度，导致部分 Agent 饿死、部分浪费 |
| **无恢复机制** | 预算耗尽后任务直接失败，没有补救手段 |

Token Bank v2 解决上述问题，并支持与现有预执行优化器（如 ZEBRA、BAMAS）互补协作。

---

## 2. v2 核心改进（vs v1）

v1 是一个基础版的"Token 银行"，v2 针对实际落地做了 5 项关键增强：

### 2.1 BudgetAdapter — 接入外部优化器
- **作用：** 把 ZEBRA/BAMAS 等预执行优化器算出的初始预算，导入为 Token Bank 的初始配额
- **价值：** 不推翻现有优化流程，只在运行时做修正和兜底
- **配置参数：** `gamma=0.8`（80% 信任外部计划，20% 信任运行时自适应）

### 2.2 PredictiveRebalancing — 预测驱动再平衡
- **作用：** 用 ARIMA/EWMA 预测每个 Agent 的未来消耗，提前调配闲置配额
- **vs v1：** v1 是"利用率超过 80% 才触发"的被动策略；v2 是"预测到即将超支就提前行动"的主动策略
- **代码：** `PredictiveRebalancer.should_rebalance()` 计算"不行动的机会成本" vs "再平衡开销"

### 2.3 BudgetBorrowing — 跨 Agent 借贷
- **作用：** Agent A 预算耗尽时，可向 Agent B 短期借调配额（利率 15%）
- **安全保证：** 总信贷额度 ≤ 全局储备，确保不会引发全局溢出（Theorem 2）
- **优先级规则：** 高优先级 Agent 可向低优先级 Agent 借贷，反之不行

### 2.4 OnlinePriorityLearning (OPL) — 在线优先级学习
- **作用：** 用 Thompson Sampling 多臂老虎机，根据历史任务成败自动调整各 Agent 角色的优先级
- **vs 手动配置：** 无需人工调参，运行 10 个任务后自动进入自适应模式
- **遗憾界：** $O(\\sqrt{KT \\log T})$，零训练数据

### 2.5 KVCacheAwareQuota — KV-Cache 内存感知
- **作用：** 不仅统计 token 数，还追踪多轮对话中 KV-Cache 的内存膨胀
- **触发机制：** 当 `effective_tokens = prompt + 0.3 * kv_cache` 超过配额 85% 时，自动触发上下文摘要压缩
- **价值：** 防止长对话任务因内存膨胀导致隐性成本爆炸

---

## 3. 代码结构

```
token_bank_v2/
├── 核心引擎
│   ├── TokenBankV2              # 主类：预算分配、消耗记账、策略执行
│   ├── BudgetAdapter            # 外部计划导入器
│   ├── PredictiveRebalancer     # ARIMA/EWMA 预测 + 机会成本再平衡
│   ├── BudgetBorrowing          # 跨 Agent 信贷市场
│   ├── OnlinePriorityLearner    # Thompson Sampling 优先级自适应
│   └── KVCacheTracker           # KV-Cache 追踪与摘要触发
│
├── 执行层
│   ├── LLMClientV2              # OpenAI API 封装（支持 Mock 模式）
│   └── Task Runners             # GAIA / SWE-bench / Financial 任务编排
│
└── 实验层
    ├── run_experiment_v2        # 实验编排器
    ├── 可视化                    # Matplotlib 6 图输出
    └── 统计检验                  # Mann-Whitney U / Chi-square / Levene
```

---

## 4. 数据集说明

实验使用三套数据集，均提供**自动 fallback**，无需担心网络或权限问题：

### 4.1 GAIA Benchmark
- **来源：** huggingface `gaia-benchmark/GAIA`
- **内容：** 多步推理问答（如"2018 年世界杯冠军国家的首都是哪？人口多少？"）
- **使用：** 加载 validation split 的 Level 1–2 任务，随机采样 50 题
- **Fallback：** 若 huggingface 不可用，自动生成 5 类模板任务（人口查询、电影配乐、复利计算等）

### 4.2 SWE-bench Lite
- **来源：** Princeton-NLP GitHub 官方 JSON
- **内容：** Python 代码修复任务（如 off-by-one 错误、KeyError 处理、递归优化）
- **使用：** 下载官方 `swe-bench-lite.json`，取前 30 题
- **Fallback：** 3 类经典代码修复模板循环生成

### 4.3 Financial Document Extraction
- **来源：** 公开 SEC 10-K 财报摘录（Apple、Tesla、Microsoft）
- **内容：** 从财报文本中提取结构化字段（CEO、营收、净利润、审计机构等）
- **使用：** 3 份真实财报文档循环构造 20 道提取题
- **标注：** 每题含 8–10 个 ground-truth 字段，支持 F1 评估

### 4.4 数据集加载代码

```python
# 自动尝试在线加载，失败则切换本地 fallback
GAIA_TASKS = load_gaia_dataset(n_tasks=50)        # ~50 题
SWE_TASKS = load_swebench_lite(n_tasks=30)        # ~30 题
FINANCIAL_TASKS = load_financial_dataset(n_tasks=20)  # ~20 题
```

---

## 5. 快速上手

### 5.1 环境准备

```bash
pip install openai tiktoken numpy pandas matplotlib seaborn scipy pyyaml datasets statsmodels
```

本地推理后端（GPU 机器，默认）：`pip install vllm`，并用 `_start_vllm.sh` 启动
Qwen/Qwen3.6-35B-A3B 的 OpenAI 兼容服务（`http://127.0.0.1:8000/v1`）。

### 5.2 零成本调试（Mock 模式）

无需 LLM 服务，秒级跑完全部实验：

```python
# Cell 2 中设置
MOCK_MODE = True
```

### 5.3 真实模型实验

通过 `LLM_BACKEND` 环境变量选择后端（默认 `vllm`）：

```bash
# 本地 vLLM（默认，边际成本≈电费/GPU 租金）
LLM_BACKEND=vllm .venv/bin/python -u _run_v2_experiment.py

# DeepSeek API（历史默认，按 token 计费）
LLM_BACKEND=deepseek DEEPSEEK_API_KEY=sk-... .venv/bin/python -u _run_v2_experiment.py
```

注意：本地后端下 PRIMARY/THROTTLE/SUMMARIZE 均为同一模型，PRICING 为名义价格
（GPU 租金摊销），仅用于保持 CPS 指标的配置间可比性；Qwen3.6 的思考模式已通过
`chat_template_kwargs.enable_thinking=False` 关闭，避免小 max_tokens 被思考链耗尽。

### 5.3.1 接入任意 OpenAI 兼容 API（OpenAI / OpenRouter / 硅基流动等）

`LLM_BACKEND=vllm` 分支本质是一个通用 OpenAI 兼容客户端，三个环境变量即可接任何服务：

```bash
LLM_BACKEND=vllm \
OPENAI_BASE_URL=https://api.openai.com/v1 \
VLLM_API_KEY=sk-... \
VLLM_MODEL=gpt-4o \
.venv/bin/python -u _run_v2_experiment.py
```

- `VLLM_MODEL` 同时决定 PRICING 的键名（默认为名义价 $0.03/1M tok，可在 Cell 2 按供应商实际价格改）
- token 记账优先用本地 tokenizer（`TOKENIZER_PATH` 指向任意 HF 模型目录），
  没有时自动回退 tiktoken gpt-4o 近似——远程 API 场景建议设一个词表相近的本地 tokenizer
- 预算扫描：`BUDGETS=8000,16000 .venv/bin/python -u _budget_sweep_vllm.py`
  （`BENCHMARK=GAIA|FRAMES|MUSIQUE|MATH|SWE|FINANCIAL`，GAIA 需 HF_TOKEN）

### 5.4 切换 Benchmark

```python
# Cell 9 中修改
BENCHMARK = 'GAIA'   # 可选: 'GAIA' | 'SWE' | 'FINANCIAL'
```

### 5.5 运行全流程

1. 打开 `token_bank_v2_experiment.ipynb`
2. 依次执行 Cell 1–15
3. 查看输出图表和 CSV 结果

---

## 6. 关键配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BUDGET` | 100000 | 每任务总 Token 预算 |
| `policy` | `'hard'` | 执行策略：`hard` / `grace` / `throttle` |
| `gamma` | 0.8 | 外部计划信任度（0=全自适应，1=全外部） |
| `theta` | 0.8 | 再平衡触发阈值 |
| `credit_rate` | 0.10 | 单 Agent 最大借贷额度（配额的 10%） |
| `interest_rate` | 0.15 | 借贷利率 |
| `cold_start` | 10 | OPL 手动优先级冷启动任务数 |
| `compression_factor` | 0.3 | KV-Cache 折算系数 |

---

## 7. 实验输出

运行完成后自动生成以下文件：

| 文件 | 内容 |
|------|------|
| `results_v2_*.csv` | 各配置的成功率、Token 消耗、成本汇总表 |
| `evaluation_v2_*.png` | 6 张评估图（成功率、消耗、方差、溢出率、恢复率、Pareto 前沿） |
| `ablation_v2.png` | 消融实验柱状图 |
| `sensitivity_theta_v2.png` | 阈值敏感性分析图 |
| `per_agent_v2.png` | 各 Agent 配额/消耗/借贷细粒度分析 |
| `experiment_report_v2.json` | 完整实验 JSON 报告 |

---

## 8. 与现有系统的集成建议

### 8.1 如果你在用 ZEBRA
```python
# ZEBRA 输出预分配计划
zebra_plan = {'planner': 30000, 'executor': 50000, 'reviewer': 20000}

# Token Bank 包裹 ZEBRA，运行时修正偏差
bank = TokenBankV2(total_budget=100000, external_plan=zebra_plan, gamma=0.8)
```

### 8.2 如果你在用 BAMAS
```python
# BAMAS 输出拓扑 + 模型选择
# Token Bank 负责运行时执行和预算兜底
```

### 8.3 如果你从零开始
```python
# 直接使用 Token Bank 内置的自适应分配
bank = TokenBankV2(total_budget=100000, gamma=0.0)  # 全自适应
```

---

## 9. 避坑指南

| 问题 | 解决方案 |
|------|---------|
| huggingface 下载失败 | 自动 fallback 到本地合成数据，无需处理 |
| statsmodels 安装失败 | 自动降级为 EWMA 预测，功能不受影响 |
| API 额度不足 | 设 `MOCK_MODE=True`，逻辑验证零成本 |
| 某 Agent 频繁耗尽配额 | 检查 OPL 是否已冷启动完成（10 任务后自适应） |
| 长对话任务内存爆炸 | 确认 `enable_kv=True`，KV-Cache 压缩会自动触发 |

---

## 10. 后续 Roadmap

| 阶段 | 目标 |
|------|------|
| **Phase 1**（当前） | Notebook 实验验证，Mock + 真实 API 双模式 |
| **Phase 2** | 封装为 pip 包 `token-bank`，提供 Flask/FastAPI 服务接口 |
| **Phase 3** | 对接 LangGraph / AutoGen / CrewAI 插件 |
| **Phase 4** | 支持多模型路由（GPT-4o ↔ Claude ↔ Llama）的跨模型预算共享 |

---

## 11. 联系人 & 资源

- **Notebook 文件：** `token_bank_v2_experiment.ipynb`
- **策略文档：** `token_bank_v2_improved_method.md`
- **碰撞检测报告：** `token_bank_collision_survey.md`
- **实验设计文档：** `token_bank_experiment_design.md`

---

*本文档供内部工程团队使用，如有问题请联系项目负责人。*
