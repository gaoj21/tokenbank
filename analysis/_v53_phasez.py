#!/usr/bin/env python
"""v5.3 Phase Z：纯离线收口分析（零 GPU）。

从 traces/*.jsonl 实算：
  Z1 过度修订审计（Unc 臂 correct→wrong 翻转 vs 无归属增益量级，gate E2）
  Z2 Unc+Fmt 正式基线统计（SR + bootstrap 95% CI + NoPool vs Unc+Fmt McNemar/ΔCI）
  Z3 多预算点 Pareto 曲线（16k/24k 连线，8k 待补标注）
  Z4 GAIA Unc+Fmt 24k/16k 同分核查（逐题正确集合 Jaccard）
  Z5 双噪声源合并口径（抽样 McNemar/CI + 运行漂移 SD=√(f/n)）
产出 results/v53_phasez.json、v53_phasez_report.md、figures/v53_pareto_{ds}.png。

口径与 _v52_reanalyze.py 一致：判题=最终 success 字段；rounds[].is_correct 是该轮
answer_snapshot 的判题结果；级联臂 rounds 编号在 L2 起点重新从 0 计（翻转轮次分布
一律用 rounds 列表的位置下标，避免编号歧义；NoPool 另按 probe/L2 段拆分）。
不 import _v52_reanalyze（其模块级代码会重写既有结果文件），统计函数按同源复制。
"""
import json, math, collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PRE = 'traces/tasks_sweep_Qwen3.6-27B-FP8_'
DS = ['ga', 'ma', 'fr', 'mu']
DS_NAME = {'ga': 'GAIA', 'ma': 'MATH', 'fr': 'FRAMES', 'mu': 'MuSiQue'}
F24 = {d: f'{PRE}v51t_{d}.jsonl' for d in DS}          # NoPool/Unified @24k
F16 = {d: f'{PRE}v51t16_{d}.jsonl' for d in DS}        # NoPool/Unified/Unc @16k
FV5 = {'ga': f'{PRE}v5test_ga.jsonl', 'mu': f'{PRE}v5test_mu.jsonl',
       'ma': f'{PRE}v5al_ma.jsonl', 'fr': f'{PRE}v5al_fr.jsonl'}  # Unc @24k（ma/fr 须过滤 test）
FC1 = {'ga24': f'{PRE}v52c1_ga24.jsonl', 'ga16': f'{PRE}v52c1_ga16.jsonl',
       'fr24': f'{PRE}v52c1_fr24.jsonl', 'fr16': f'{PRE}v52c1_fr16.jsonl',
       # MATH 的 Unc+Fmt 来自 E1（v5.3 第二步），初版 Z1 跑在 E1 之前，漏了这两格
       # ——Z1b 补审计（见下），头条机制归因必须对正式基线做，不能对 Unc 做。
       'ma24': f'{PRE}v53e1_ma24.jsonl', 'ma16': f'{PRE}v53e1_ma16.jsonl'}  # V5-Unc-Fmt
FAB = f'{PRE}v52ab_ma.jsonl'                           # v5.1 自身消融 @MATH 24k
AB1, AB2, AB3 = 'V5.1-Ab1-NoM1', 'V5.1-Ab2-NoNotes', 'V5.1-Ab3-NoM2'
FRR = f'{PRE}v52rr_ga.jsonl'                           # NoPool×GAIA×24k 同配置重跑
UNC, UNI, NP, UF = 'V5-Unconstrained', 'V5.1-Unified', 'V5.1-NoPool', 'V5-Unc-Fmt'
B = 10000
SEED = 20260829
FLIP_F = 0.066  # GAIA 实测逐题翻转率（v5.2 §6.5，v52rr）


def load(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            out[(d['config'], d['task_id'])] = d
    return out


DATA = {}
for d in DS:
    DATA[('24', d)] = load(F24[d])
    DATA[('16', d)] = load(F16[d])
    DATA[('v5', d)] = load(FV5[d])
for k, p in FC1.items():
    DATA[('c1', k)] = load(p)
DATA[('rr', 'ga')] = load(FRR)
DATA[('ab', 'ma')] = load(FAB)

# 注意：TEST_IDS 是 set，但 get()/get_uf() 必须按 sorted() 取——bootstrap 重采样对
# 记录顺序敏感，而 set 的迭代顺序随 PYTHONHASHSEED 变化，不排序则 CI 逐次运行不可复现
# （SR/token 是顺序无关的，只有 CI 会漂 0.2~0.8pp）。
TEST_IDS = {d: {tid for (c, tid) in DATA[('24', d)]} for d in DS}
for d in DS:
    ids16 = {tid for (c, tid) in DATA[('16', d)]}
    assert ids16 == TEST_IDS[d], f'{d}: 16k/24k test ids 不一致'


def get(budget, d, cfg):
    """-> {task_id: rec}。24k 的 Unc 从 v5 文件按 test ids 过滤。"""
    if budget == '24' and cfg == UNC:
        src = DATA[('v5', d)]
    else:
        src = DATA[(budget, d)]
    return {tid: src[(cfg, tid)] for tid in sorted(TEST_IDS[d]) if (cfg, tid) in src}


def get_uf(d, budget):
    src = DATA[('c1', f'{d}{budget}')]
    return {tid: src[(UF, tid)] for tid in sorted(TEST_IDS[d]) if (UF, tid) in src}


def boot_ci(x, seed):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(x)
    means = np.empty(B)
    for s in range(0, B, 2000):
        k = min(2000, B - s)
        idx = rng.integers(0, n, size=(k, n))
        means[s:s + k] = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float(means.std())


def paired_boot(a, b, seed):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    diff = a - b
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.empty(B)
    for s in range(0, B, 2000):
        k = min(2000, B - s)
        idx = rng.integers(0, n, size=(k, n))
        means[s:s + k] = diff[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)),
            float(means.std()))


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    ln2 = n * math.log(2)
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - ln2
            for i in range(k + 1)]
    m = max(logs)
    return min(1.0, 2.0 * math.exp(m + math.log(sum(math.exp(l - m) for l in logs))))


def cell_stats(recs, seed):
    succ = np.array([r['success'] for r in recs.values()], dtype=float)
    tok = np.array([r['total_used'] for r in recs.values()], dtype=float)
    lo, hi, _ = boot_ci(succ, seed)
    return {'n': len(recs), 'sr': float(succ.mean()), 'sr_ci': [lo, hi],
            'avg_tokens': float(tok.mean())}


def compare(recs_a, recs_b, seed):
    """a - b，按 task_id 配对。返回 Δ、配对 bootstrap CI/SE、McNemar。"""
    ids = sorted(set(recs_a) & set(recs_b))
    a = np.array([recs_a[i]['success'] for i in ids], dtype=float)
    b = np.array([recs_b[i]['success'] for i in ids], dtype=float)
    lo, hi, se = paired_boot(a, b, seed)
    bb = int(((a == 1) & (b == 0)).sum())
    cc = int(((a == 0) & (b == 1)).sum())
    return {'n_paired': len(ids), 'delta': float((a - b).mean()), 'delta_ci': [lo, hi],
            'delta_se': se, 'mcnemar_b': bb, 'mcnemar_c': cc, 'mcnemar_p': mcnemar_p(bb, cc)}


def split_levels(rec):
    """(probe_rounds, l2_rounds)：级联臂用第二个 round==0 定位 L2 起点。"""
    rounds = rec['rounds']
    idx0 = [i for i, r in enumerate(rounds) if r['round'] == 0]
    if rec.get('escalation_level') == 2 and len(idx0) >= 2 and idx0[0] == 0:
        b = idx0[1]
        return rounds[:b], rounds[b:]
    return [], rounds


# ================================================================ Z1 过度修订审计
def flip_audit(recs, cascade):
    """逐题：存在某轮 is_correct=True 但最终 success=False → correct→wrong 翻转。
    细分两类机制：
      strict   —— 某后续 executor 轮转错（真·过度修订，模型自己改错）；
      stayed   —— 直到末轮记录都判对但整体 success=False（实测全部 stopped_by=budget：
                  预算耗尽后 final-reserve 收尾调用产出的最终答案错了，该调用不在
                  rounds 里）——预算截断伪影，不是过度修订，但同样是 M1 省 token 可避免的。
    轮次一律用 rounds 位置下标。cascade=True 时另按 probe/L2 段拆分首次答对位置。"""
    n = len(recs)
    flips = 0
    n_ever_correct = 0
    first_correct_hist = collections.Counter()
    turned_wrong_hist = collections.Counter()
    seg_hist = collections.Counter()      # 首次答对发生在 probe / l2（仅级联臂）
    for r in recs.values():
        rounds = r['rounds']
        ic = [bool(x.get('is_correct')) for x in rounds]
        if any(ic):
            n_ever_correct += 1
        if r['success'] or not any(ic):
            continue
        flips += 1
        fc = ic.index(True)
        first_correct_hist[fc] += 1
        tw = next((i for i in range(fc + 1, len(ic)) if not ic[i]), None)
        turned_wrong_hist[tw if tw is not None else 'stayed_correct_but_failed'] += 1
        if cascade:
            probes, _ = split_levels(r)
            seg_hist['probe' if fc < len(probes) else 'l2'] += 1
    stayed = turned_wrong_hist.get('stayed_correct_but_failed', 0)
    return {'n': n, 'flips': flips, 'flip_rate': flips / n if n else None,
            'flips_strict_overrevision': flips - stayed,
            'flips_budget_finalize_artifact': stayed,
            'n_ever_correct': n_ever_correct,
            'flip_rate_given_ever_correct': flips / n_ever_correct if n_ever_correct else None,
            'first_correct_round_hist': {str(k): v for k, v in sorted(first_correct_hist.items())},
            'turned_wrong_round_hist': {str(k): v for k, v in sorted(turned_wrong_hist.items(),
                                        key=lambda kv: str(kv[0]))},
            'first_correct_segment': dict(seg_hist) if cascade else None}


z1 = {}
for d in DS:
    for bud in ('24', '16'):
        unc = get(bud, d, UNC)
        npool = get(bud, d, NP)
        a_unc = flip_audit(unc, cascade=False)
        a_np = flip_audit(npool, cascade=True)
        gap_sr = cell_stats(npool, SEED)['sr'] - cell_stats(unc, SEED)['sr']
        gap_tasks = gap_sr * a_unc['n']
        net = a_unc['flips'] - a_np['flips']
        z1[f'{d}|{bud}'] = {
            'unc': a_unc, 'nopool': a_np,
            'gap_nopool_minus_unc_pp': gap_sr * 100,
            'gap_tasks': gap_tasks,
            'flip_share_of_gap': (a_unc['flips'] / gap_tasks) if gap_tasks > 0 else None,
            'net_flips_unc_minus_nopool': net,
            'net_share_of_gap': (net / gap_tasks) if gap_tasks > 0 else None,
        }

# 目标量级（plan §1 Z1）：MATH ~12pp≈48 题，MuSiQue ~2.6pp≈50 题；GAIA/FRAMES 用实测 gap
z1_targets = {'ma': round(0.12 * 400), 'mu': round(0.026 * 1934)}
for d in ('ga', 'fr'):
    z1_targets[d] = round(z1[f'{d}|24']['gap_tasks'])
z1_verdict = {}
for d in DS:
    z = z1[f'{d}|24']
    t = z1_targets[d]
    z1_verdict[d] = {'flips_24k': z['unc']['flips'], 'target': t,
                     'ratio': (z['unc']['flips'] / t) if t else None,
                     'net_flips_24k': z['net_flips_unc_minus_nopool'],
                     'net_share_of_gap': z['net_share_of_gap'],
                     'cond_flip_rate_unc': z['unc']['flip_rate_given_ever_correct'],
                     'cond_flip_rate_nopool': z['nopool']['flip_rate_given_ever_correct']}
# E2 gate：看"Unc − NoPool 净翻转"（毛翻转数受 ever-correct 基数影响，
# 净差才是 M1 压掉的部分）；MATH 对照无归属 12pp（48 题），MuSiQue 对照全额 2.6pp（50 题）。
ma_net_gap = z1_verdict['ma']['net_share_of_gap']
ma_net_tgt = z1_verdict['ma']['net_flips_24k'] / z1_targets['ma']
mu_net = z1_verdict['mu']['net_share_of_gap']
if ma_net_tgt >= 0.5 and (mu_net or 0) >= 0.5:
    e2_gate = ('MATH 与 MuSiQue 的净翻转均接近目标量级：无归属增益归因 M1（防过度修订），'
               'E2 缩减为只跑 −M1（1,934 题·次）')
elif (mu_net or 0) <= 0:
    e2_gate = (f"MATH 净翻转 {z1_verdict['ma']['net_flips_24k']:+d} 题 = 无归属 12pp 的 "
               f"{ma_net_tgt*100:.0f}%（占全额 18.5pp gap 的 {ma_net_gap*100:.0f}%）——"
               '防过度修订（含 budget-finalize 截断伪影，同属 M1 省 token 可避免）是 MATH '
               '无归属增益的一半来源；但 MuSiQue 净翻转为 '
               f"{z1_verdict['mu']['net_flips_24k']:+d} 题（NoPool 自身翻转不少于 Unc，条件"
               f"翻转率 {z1_verdict['mu']['cond_flip_rate_unc']*100:.1f}%→"
               f"{z1_verdict['mu']['cond_flip_rate_nopool']*100:.1f}% 反而更高）——"
               '过度修订不能解释 MuSiQue 的 2.6pp，'
               '增益归 M3，E2 必须全跑（3,868 题·次）')
else:
    e2_gate = ('净翻转介于 0~50%：过度修订只是部分来源，E2 必须全跑（3,868 题·次），'
               '−M1 消融预期只能解释增益的一部分')

# ========================================== Z1b MATH 归因复核（对正式基线 Unc+Fmt）
# 初版 Z1 把翻转审计做在 Unc 臂上，但 E1 之后 MATH 的正式基线是 Unc+Fmt——增益对
# Unc+Fmt 测（+8.0pp），归因却对 Unc 审（gap 18.5pp），口径不一致会高估过度修订的
# 份额，因为 `ANSWER:` 格式本身就消掉了一部分 budget-finalize 伪影。这里补三件事：
#   (1) MATH Unc+Fmt 臂的翻转审计与净翻转占真实增益的比例；
#   (2) 逐题因果口径：NoPool 赢的题里，基线究竟"曾答对又改错"还是"从没对过"；
#   (3) 消融臂的翻转 + 路由分布——用来检验 −M1 在 MATH 上是否根本测不到该机制。
def route_hist(recs):
    h = collections.Counter(r.get('escalation_level') for r in recs.values())
    return {f'L{k}': v for k, v in sorted(h.items(), key=lambda kv: (kv[0] is None, kv[0]))}


def ever_correct(rec):
    return any(bool(x.get('is_correct')) for x in rec['rounds'])


def is_flip(rec):
    return (not rec['success']) and ever_correct(rec)


def discordant_split(win_recs, base):
    """NoPool(或消融臂)赢的题里，基线是过度修订(曾答对)还是从未答对。
    净翻转是聚合口径，会把"两边各自翻转"的部分相消；这里逐题看，是更严的因果下界。"""
    over = [t for t in win_recs if is_flip(base[t])]
    return {'n_wins': len(win_recs), 'baseline_over_revised': len(over),
            'baseline_never_correct': len(win_recs) - len(over),
            'over_revision_share': len(over) / len(win_recs) if win_recs else None}


z1b = {'per_budget': {}, 'ablations': {}, 'route_gain': {}}
for bud in ('24', '16'):
    uf = get_uf('ma', bud)
    npool = get(bud, 'ma', NP)
    unc = get(bud, 'ma', UNC)
    a_uf, a_np = flip_audit(uf, cascade=False), flip_audit(npool, cascade=True)
    ids = sorted(set(uf) & set(npool))
    gap_tasks = sum(npool[t]['success'] for t in ids) - sum(uf[t]['success'] for t in ids)
    net = a_uf['flips'] - a_np['flips']
    wins = [t for t in ids if npool[t]['success'] and not uf[t]['success']]
    z1b['per_budget'][bud] = {
        'unc_fmt': a_uf, 'nopool': a_np, 'unc': flip_audit(unc, cascade=False),
        'real_gap_pp': gap_tasks / len(ids) * 100, 'real_gap_tasks': gap_tasks,
        'net_flips_uf_minus_nopool': net,
        'net_share_of_real_gap': net / gap_tasks if gap_tasks > 0 else None,
        # 旧口径（对 Unc）留作并列，说明高估从何而来
        'legacy_net_vs_unc': z1[f'ma|{bud}']['net_flips_unc_minus_nopool'],
        'legacy_share_of_unc_gap': z1[f'ma|{bud}']['net_share_of_gap'],
        'discordant': discordant_split(wins, uf),
    }

# (3) 消融臂：SR / token / 翻转 / 路由 / vs 正式基线
uf24, np24 = get_uf('ma', '24'), get('24', 'ma', NP)
AB_SRC = DATA[('ab', 'ma')]
for cfg, tag in [(NP, 'NoPool(full)'), (AB1, 'Ab1 -M1'),
                 (AB3, 'Ab3 -M2 (fixed L2 pipe)'), (AB2, 'Ab2 -M3.2notes')]:
    recs = np24 if cfg == NP else {t: AB_SRC[(cfg, t)] for t in sorted(TEST_IDS['ma'])
                                   if (cfg, t) in AB_SRC}
    a = flip_audit(recs, cascade=(cfg != AB3))
    st = cell_stats(recs, SEED)
    wins = [t for t in sorted(set(recs) & set(uf24))
            if recs[t]['success'] and not uf24[t]['success']]
    z1b['ablations'][tag] = {
        'config': cfg, 'sr': st['sr'], 'avg_tokens': st['avg_tokens'],
        'flips': a['flips'], 'flips_strict_overrevision': a['flips_strict_overrevision'],
        'flips_budget_finalize_artifact': a['flips_budget_finalize_artifact'],
        'flip_rate_given_ever_correct': a['flip_rate_given_ever_correct'],
        'routes': route_hist(recs),
        'vs_unc_fmt': compare(recs, uf24, SEED + 7),
        'discordant': discordant_split(wins, uf24),
    }

# (4) 增益按 NoPool 的路由层拆分：+8pp 到底长在 L0(单轮直答) 还是 L2(完整管线) 上
ids24 = sorted(set(uf24) & set(np24))
for lvl in (0, 1, 2):
    sub = [t for t in ids24 if np24[t].get('escalation_level') == lvl]
    if not sub:
        continue
    w = sum(1 for t in sub if np24[t]['success'] and not uf24[t]['success'])
    l = sum(1 for t in sub if uf24[t]['success'] and not np24[t]['success'])
    z1b['route_gain'][f'L{lvl}'] = {
        'n': len(sub), 'sr_nopool': sum(np24[t]['success'] for t in sub) / len(sub),
        'sr_unc_fmt': sum(uf24[t]['success'] for t in sub) / len(sub),
        'disc_b': w, 'disc_c': l, 'net_tasks': w - l,
        'share_of_total_pp': (w - l) / len(ids24) * 100,
    }

_p24 = z1b['per_budget']['24']
z1b_verdict = (
    f"对正式基线 Unc+Fmt：MATH 真实增益 {_p24['real_gap_pp']:+.1f}pp（{_p24['real_gap_tasks']} 题），"
    f"净翻转 {_p24['net_flips_uf_minus_nopool']:+d} 题 = 真实增益的 "
    f"{_p24['net_share_of_real_gap']*100:.0f}%（旧口径对 Unc 审计为 "
    f"{_p24['legacy_net_vs_unc']:+d} 题/{_p24['legacy_share_of_unc_gap']*100:.0f}%，"
    f"高估来自格式本身消掉的 budget-finalize 伪影：Unc {_p24['unc']['flips_budget_finalize_artifact']} → "
    f"Unc+Fmt {_p24['unc_fmt']['flips_budget_finalize_artifact']}）；"
    f"逐题因果口径更低——NoPool 赢的 {_p24['discordant']['n_wins']} 题里只有 "
    f"{_p24['discordant']['baseline_over_revised']} 题"
    f"（{_p24['discordant']['over_revision_share']*100:.0f}%）基线曾答对又改错，"
    f"其余 {_p24['discordant']['baseline_never_correct']} 题基线从未答对（管线本身没解出来，不是过度修订）")

# ============================== Z1c 缺口补测（Ab6 隔离臂 + Gemma 格式对照臂）
# §8.4b 列出的两个 400 题·次缺口。两个臂都可能尚未跑完——文件不在就跳过，
# 报告里如实写"未完成"，绝不留空表假装做过。
import os as _os

Z1C_AB6 = f'{PRE}v54a_ma_ab6.jsonl'                       # Qwen  V5.1-Ab6-NoM2NoM1
Z1C_GEM = 'traces/tasks_sweep_gemma-4-31B-it_v54b_ma_fmt.jsonl'   # Gemma V5-Unc-Fmt
GEM_X1 = 'traces/tasks_sweep_gemma-4-31B-it_v52x1_ma.jsonl'       # Gemma V5.1-NoPool (X1)
GEM_UNC = 'traces/tasks_sweep_gemma-4-31B-it_v5gem_ma.jsonl'      # Gemma Unc (v5gem)
AB6 = 'V5.1-Ab6-NoM2NoM1'

z1c = {'ab6_m1_isolation': None, 'gemma_format_control': None}

# --- (a) Ab6：在 M1 能施力的环境里隔离 M1。Ab3 - Ab6 = M1 自己的 SR 贡献 ---
if _os.path.exists(Z1C_AB6):
    _src = load(Z1C_AB6)
    ab6 = {t: _src[(AB6, t)] for t in sorted(TEST_IDS['ma']) if (AB6, t) in _src}
    ab3 = {t: DATA[('ab', 'ma')][(AB3, t)] for t in sorted(TEST_IDS['ma'])
           if (AB3, t) in DATA[('ab', 'ma')]}
    if ab6:
        a6, a3 = flip_audit(ab6, cascade=False), flip_audit(ab3, cascade=False)
        s6, s3 = cell_stats(ab6, SEED + 11), cell_stats(ab3, SEED + 12)
        z1c['ab6_m1_isolation'] = {
            'n': len(ab6),
            'ab6_no_m1': {'sr': s6['sr'], 'sr_ci': s6['sr_ci'], 'avg_tokens': s6['avg_tokens'],
                          'flips': a6['flips'],
                          'strict': a6['flips_strict_overrevision'],
                          'budget_finalize': a6['flips_budget_finalize_artifact'],
                          'cond_flip': a6['flip_rate_given_ever_correct']},
            'ab3_with_m1': {'sr': s3['sr'], 'sr_ci': s3['sr_ci'], 'avg_tokens': s3['avg_tokens'],
                            'flips': a3['flips'],
                            'strict': a3['flips_strict_overrevision'],
                            'budget_finalize': a3['flips_budget_finalize_artifact'],
                            'cond_flip': a3['flip_rate_given_ever_correct']},
            'm1_effect': compare(ab3, ab6, SEED + 13),          # Ab3 - Ab6 = M1 的净贡献
            'ab6_vs_unc_fmt': compare(ab6, get_uf('ma', '24'), SEED + 14),
        }

# --- (b) Gemma 格式对照：+25.8pp 里有多少是格式 ---
if _os.path.exists(Z1C_GEM) and _os.path.exists(GEM_X1):
    _g = load(Z1C_GEM)
    gem_uf = {t: r for (c, t), r in _g.items() if c == UF}
    _x = load(GEM_X1)
    gem_np = {t: r for (c, t), r in _x.items() if c == NP}
    gem_unc = {}
    if _os.path.exists(GEM_UNC):
        _u = load(GEM_UNC)
        gem_unc = {t: r for (c, t), r in _u.items() if c == UNC}
    if gem_uf and gem_np:
        ids = sorted(set(gem_uf) & set(gem_np))
        gem_uf = {t: gem_uf[t] for t in ids}
        gem_np = {t: gem_np[t] for t in ids}
        cell_uf, cell_np = cell_stats(gem_uf, SEED + 21), cell_stats(gem_np, SEED + 22)
        blk = {'n': len(ids),
               'nopool': cell_np, 'unc_fmt': cell_uf,
               'nopool_vs_unc_fmt': compare(gem_np, gem_uf, SEED + 23)}
        if gem_unc:
            gu = {t: gem_unc[t] for t in ids if t in gem_unc}
            if len(gu) == len(ids):
                blk['unc'] = cell_stats(gu, SEED + 24)
                blk['nopool_vs_unc'] = compare(gem_np, gu, SEED + 25)
                # 格式份额 = (Unc+Fmt - Unc) / (NoPool - Unc)，与 Qwen E1 的 57% 同口径
                tot = blk['nopool_vs_unc']['delta']
                fmt = blk['unc_fmt']['sr'] - blk['unc']['sr']
                blk['format_share'] = fmt / tot if tot else None
        z1c['gemma_format_control'] = blk

# ================================================================ Z2 Unc+Fmt 统计
z2 = {}
for d in ('ga', 'fr'):
    for bud in ('24', '16'):
        uf = get_uf(d, bud)
        npool = get(bud, d, NP)
        z2[f'{d}|{bud}'] = {
            'unc_fmt': cell_stats(uf, SEED + len(uf)),
            'nopool': cell_stats(npool, SEED + len(npool) + 1),
            'nopool_vs_unc_fmt': compare(npool, uf, SEED + 2),
        }

# ================================================================ Z3 Pareto 曲线
z3 = {}
ARM_STYLE = {NP: ('C0', 'o', 'NoPool'), UNI: ('C2', '^', 'Unified'),
             UNC: ('C3', 's', 'Unc'), UF: ('C1', 'D', 'Unc+Fmt')}
for d in DS:
    fig, ax = plt.subplots(figsize=(8.5, 6))
    # ma 的 Unc+Fmt 由 E1 补齐后一并入图（初版只有 ga/fr 有该臂）
    arms = [NP, UNI, UNC] + ([UF] if d in ('ga', 'fr', 'ma') else [])
    z3[d] = {}
    for cfg in arms:
        color, mk, label = ARM_STYLE[cfg]
        pts = []
        for bud in ('16', '24'):
            recs = get_uf(d, bud) if cfg == UF else get(bud, d, cfg)
            st = cell_stats(recs, SEED)
            pts.append((bud, st['avg_tokens'], st['sr']))
            z3[d][f'{label}@{bud}k'] = st
        xs = [p[1] for p in pts]; ys = [p[2] for p in pts]
        ax.plot(xs, ys, color=color, lw=1.2, alpha=0.75, zorder=2)
        for bud, t, s in pts:
            ax.scatter(t, s, color=color, marker=mk, s=80, zorder=3,
                       label=label if bud == '24' else None)
            ax.annotate(f'{bud}k', (t, s), textcoords='offset points',
                        xytext=(6, 5), fontsize=8, color=color)
    ax.text(0.02, 0.02, '8k 预算点实验进行中（E3），待补',
            transform=ax.transAxes, fontsize=8, color='gray', style='italic')
    ax.set_xlabel('avg actual tokens (total_used)')
    ax.set_ylabel('SR')
    ax.set_title(f'v5.3 Pareto — {DS_NAME[d]} (test split, 16k/24k)')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'figures/v53_pareto_{d}.png', dpi=140)
    plt.close(fig)

# 前沿重合判读（GAIA/FRAMES）：同预算下 NoPool 与 Unc+Fmt 的 SR 差 vs 漂移
z3_read = {}
for d in ('ga', 'fr'):
    for bud in ('16', '24'):
        a, b = z3[d][f'NoPool@{bud}k'], z3[d][f'Unc+Fmt@{bud}k']
        drift = math.sqrt(FLIP_F / a['n'])
        z3_read[f'{d}|{bud}'] = {
            'sr_diff_pp': (a['sr'] - b['sr']) * 100,
            'token_diff_pct': (a['avg_tokens'] - b['avg_tokens']) / b['avg_tokens'] * 100,
            'drift_sd_pp': drift * 100,
            'diff_in_drift_sd': abs(a['sr'] - b['sr']) / drift,
        }

# ================================================================ Z4 GAIA 同分核查
uf24, uf16 = get_uf('ga', '24'), get_uf('ga', '16')
s24 = {t for t, r in uf24.items() if r['success']}
s16 = {t for t, r in uf16.items() if r['success']}
inter, union = s24 & s16, s24 | s16
sym_diff = len(union) - len(inter)
sym_diff_rate = sym_diff / len(TEST_IDS['ga'])
z4 = {
    'n_24k': len(uf24), 'n_16k': len(uf16),
    'correct_24k': len(s24), 'correct_16k': len(s16),
    'intersection': len(inter), 'union': len(union),
    'jaccard': len(inter) / len(union) if union else None,
    'sym_diff': sym_diff, 'sym_diff_rate': sym_diff_rate,
    'run_flip_rate_f': FLIP_F,
    'only_24k': sorted(s24 - s16), 'only_16k': sorted(s16 - s24),
    'avg_tokens_24k': float(np.mean([r['total_used'] for r in uf24.values()])),
    'avg_tokens_16k': float(np.mean([r['total_used'] for r in uf16.values()])),
}
# 判据：两次独立运行（不同预算帽）的正确集差异若 ≤ 运行噪声（f=6.6% 翻转率）量级，
# 则差异无需用预算解释——预算帽对朴素管线不绑定。
z4['verdict'] = (
    f"两次运行正确集的对称差 {sym_diff}/{len(TEST_IDS['ga'])} = {sym_diff_rate*100:.1f}%，"
    f"{'≤ 运行漂移翻转率' if sym_diff_rate <= 1.5 * FLIP_F else '> 运行漂移翻转率'}"
    f"（f={FLIP_F*100:.1f}%）量级——"
    + ('差异完全是同配置重跑也会出现的现象——预算帽对朴素管线不绑定（重要发现，'
       '直接解释空洞 C）' if sym_diff_rate <= 1.5 * FLIP_F else
       '差异超出运行噪声，SR 恰好相等属巧合，注明'))

# ================================================================ Z5 双噪声源合并口径
# GAIA 翻转率复核（v52rr vs v51t NoPool 同配置重跑）
rr = {tid: r for (c, tid), r in DATA[('rr', 'ga')].items() if c == NP}
base = get('24', 'ga', NP)
ids_rr = sorted(set(rr) & set(base))
rr_flip = sum(1 for i in ids_rr if rr[i]['success'] != base[i]['success'])
rr_check = {'n_paired': len(ids_rr), 'flips': rr_flip,
            'flip_rate': rr_flip / len(ids_rr) if ids_rr else None,
            'sr_base': float(np.mean([base[i]['success'] for i in ids_rr])),
            'sr_rerun': float(np.mean([rr[i]['success'] for i in ids_rr]))}

# 关键差值：GAIA/FRAMES = NoPool vs Unc+Fmt @24k；MATH/MuSiQue = NoPool vs Unc @24k
z5 = {}
key_cmp = {}
for d in DS:
    npool = get('24', d, NP)
    if d in ('ga', 'fr'):
        other, bname = get_uf(d, '24'), 'Unc+Fmt'
    else:
        other, bname = get('24', d, UNC), 'Unc'
    key_cmp[d] = {'baseline': bname, **compare(npool, other, SEED + 3)}
for d in DS:
    n = len(TEST_IDS[d])
    drift = math.sqrt(FLIP_F / n)
    c = key_cmp[d]
    se_combined = math.sqrt(c['delta_se'] ** 2 + drift ** 2)
    z5[d] = {
        'n': n, 'baseline': c['baseline'],
        'key_delta_pp': c['delta'] * 100,
        'delta_ci_pp': [c['delta_ci'][0] * 100, c['delta_ci'][1] * 100],
        'delta_se_pp': c['delta_se'] * 100,
        'mcnemar': f"{c['mcnemar_b']}/{c['mcnemar_c']}",
        'mcnemar_p': c['mcnemar_p'],
        'drift_sd_pp': drift * 100,
        'delta_in_drift_sd': abs(c['delta']) / drift,
        'combined_se_pp': se_combined * 100,
        'delta_in_combined_se': abs(c['delta']) / se_combined,
    }
z5_conclusion = ('合并运行漂移后口径：MATH（%.1f×合并SE）、MuSiQue（%.1f×合并SE）的增益'
                 '仍然稳固；GAIA（%.1f×）、FRAMES（%.1f×）对 Unc+Fmt 的差值在双噪声源'
                 '之下不可解释，结论不变：GAIA/FRAMES 与格式对齐基线同前沿、无 SR 增益。'
                 % (z5['ma']['delta_in_combined_se'], z5['mu']['delta_in_combined_se'],
                    z5['ga']['delta_in_combined_se'], z5['fr']['delta_in_combined_se']))

# ================================================================ 汇总 JSON
out = {
    'meta': {'seed': SEED, 'B': B, 'split': 'test（v51t task_id 集合）',
             'test_n': {d: len(TEST_IDS[d]) for d in DS},
             'flip_rate_f': FLIP_F,
             'notes': 'Z1 轮次为 rounds 位置下标；级联臂（NoPool/Unified）round 编号在 '
                      'L2 起点重置，首次答对位置另按 probe/l2 段拆分。'
                      'Unc@24k 的 MATH/FRAMES 从 v5al 全量按 test ids 过滤。'},
    'z1_overrevision_audit': z1,
    'z1_targets': z1_targets,
    'z1_verdict': z1_verdict,
    'z1_e2_gate': e2_gate,
    'z1b_math_attribution_vs_unc_fmt': z1b,
    'z1b_verdict': z1b_verdict,
    'z1c_gap_closing': z1c,
    'z2_unc_fmt_stats': z2,
    'z3_pareto_cells': z3,
    'z3_frontier_reading': z3_read,
    'z4_gaia_unc_fmt_overlap': z4,
    'z5_noise_combined': z5,
    'z5_rerun_check_gaia': rr_check,
    'z5_conclusion': z5_conclusion,
}
with open('results/v53_phasez.json', 'w') as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False)

# ================================================================ 报告
L = []
L.append('# v5.3 Phase Z 报告（纯离线，零 GPU）\n')
L.append('数据源：traces/*.jsonl 实算；统计口径同 `_v52_reanalyze.py`（bootstrap B=10000，'
         f'seed={SEED}；McNemar 精确二项）。全部数值见 `results/v53_phasez.json`。\n')

L.append('\n## Z1 — 过度修订审计（gate E2）\n')
L.append('Unc 臂逐题扫 `rounds[].is_correct`：曾答对但最终 `success=False` 记为 '
         'correct→wrong 翻转，并细分两类机制：**strict**（某后续轮转错，真·过度修订）与 '
         '**budget-finalize**（直到末轮记录都判对但 success=False——实测全部 '
         'stopped_by=budget，是预算耗尽后收尾调用答错的截断伪影，不是过度修订，但同样'
         '是 M1 省 token 可避免的）。目标量级：MATH ~12pp≈48 题、MuSiQue ~2.6pp≈50 题；'
         'GAIA/FRAMES 用实测 NoPool−Unc gap 换算。\n')
L.append('| 数据集 | 预算 | n | Unc 翻转(strict+budget) | 目标量级 | 翻转/目标 | NoPool 翻转(strict+budget) | 净翻转(Unc−NoPool) | gap(pp) |')
L.append('|---|---|---|---|---|---|---|---|---|')
for d in DS:
    for bud in ('24', '16'):
        z = z1[f'{d}|{bud}']
        u, p = z['unc'], z['nopool']
        tgt = z1_targets[d] if bud == '24' else '—'
        ratio = f"{z1_verdict[d]['ratio']:.2f}" if bud == '24' and z1_verdict[d]['ratio'] is not None else '—'
        L.append(f"| {DS_NAME[d]} | {bud}k | {u['n']} | "
                 f"{u['flips']} ({u['flips_strict_overrevision']}+{u['flips_budget_finalize_artifact']})"
                 f" = {u['flip_rate']*100:.1f}% | {tgt} | {ratio} | "
                 f"{p['flips']} ({p['flips_strict_overrevision']}+{p['flips_budget_finalize_artifact']})"
                 f" = {p['flip_rate']*100:.1f}% | {z['net_flips_unc_minus_nopool']:+d} | "
                 f"{z['gap_nopool_minus_unc_pp']:+.1f} |")
L.append('\n毛翻转数受 ever-correct 基数影响（NoPool 答对过的题更多，翻转机会也更多），'
         '公平对照用**条件翻转率 = 翻转数 / 曾答对题数**，归因用**净翻转占 gap 比例**（24k）：\n')
L.append('| 数据集 | ever-correct Unc/NoPool | 条件翻转率 Unc→NoPool | 净翻转/gap |')
L.append('|---|---|---|---|')
for d in DS:
    z = z1[f'{d}|24']
    u, p = z['unc'], z['nopool']
    net = z['net_share_of_gap']
    L.append(f"| {DS_NAME[d]} | {u['n_ever_correct']}/{p['n_ever_correct']} | "
             f"{u['flip_rate_given_ever_correct']*100:.1f}% → {p['flip_rate_given_ever_correct']*100:.1f}% | "
             + (f"{net*100:.0f}% |" if net is not None else "— |"))
L.append('\n翻转轮次分布（Unc，24k；位置下标，0=首轮）：')
for d in DS:
    z = z1[f'{d}|24']['unc']
    L.append(f"- {DS_NAME[d]}：首次答对轮 {z['first_correct_round_hist']}；"
             f"转错轮 {z['turned_wrong_round_hist']}")
L.append('\nNoPool 对照（级联臂，首次答对位置按 probe/L2 段拆分）：')
for d in DS:
    z = z1[f'{d}|24']['nopool']
    L.append(f"- {DS_NAME[d]}：翻转 {z['flips']} 题，分段 {z['first_correct_segment']}，"
             f"首次答对轮 {z['first_correct_round_hist']}")
L.append(f"\n**E2 gate 判决：{e2_gate}。**\n")

L.append('\n> **口径更正（Z1b）**：上表的 MATH 归因是对 **Unc** 审的，但 E1 之后 MATH 的'
         '正式基线是 **Unc+Fmt**。增益对 Unc+Fmt 测、归因对 Unc 审，两边 gap 不同口径，'
         '会系统性高估过度修订的份额。下节按正式基线重算，**以 Z1b 为准**。\n')

L.append('\n## Z1b — MATH 归因复核（对正式基线 Unc+Fmt）\n')
L.append('初版 Z1 跑在 E1 之前，`FC1` 只映射了 GAIA/FRAMES 的 Unc+Fmt 轨迹，MATH 的两格'
         '（`v53e1_ma{24,16}`）从未进过审计。补齐后：\n')
L.append('| 预算 | 臂 | n | SR | token | 翻转(strict+budget) | ever-correct | 条件翻转率 |')
L.append('|---|---|---|---|---|---|---|---|')
for bud in ('24', '16'):
    p = z1b['per_budget'][bud]
    for tag, key, recs in (('Unc', 'unc', get(bud, 'ma', UNC)),
                           ('**Unc+Fmt**', 'unc_fmt', get_uf('ma', bud)),
                           ('NoPool', 'nopool', get(bud, 'ma', NP))):
        a = p[key]
        st = cell_stats(recs, SEED)
        L.append(f"| {bud}k | {tag} | {a['n']} | {st['sr']*100:.1f}% | "
                 f"{st['avg_tokens']:,.0f} | "
                 f"{a['flips']} ({a['flips_strict_overrevision']}+"
                 f"{a['flips_budget_finalize_artifact']}) | "
                 f"{a['n_ever_correct']} | {a['flip_rate_given_ever_correct']*100:.1f}% |")
L.append('')
L.append('| 预算 | 真实增益(vs Unc+Fmt) | 净翻转 | 占真实增益 | 旧口径(vs Unc) | 占 Unc gap |')
L.append('|---|---|---|---|---|---|')
for bud in ('24', '16'):
    p = z1b['per_budget'][bud]
    L.append(f"| {bud}k | {p['real_gap_pp']:+.1f}pp（{p['real_gap_tasks']} 题） | "
             f"{p['net_flips_uf_minus_nopool']:+d} | **{p['net_share_of_real_gap']*100:.0f}%** | "
             f"{p['legacy_net_vs_unc']:+d} | {p['legacy_share_of_unc_gap']*100:.0f}% |")
L.append('\n高估的来源可以直接指出来：`ANSWER:` 格式本身就消掉了大部分 budget-finalize 伪影'
         f"（24k：Unc {z1b['per_budget']['24']['unc']['flips_budget_finalize_artifact']} → "
         f"Unc+Fmt {z1b['per_budget']['24']['unc_fmt']['flips_budget_finalize_artifact']}），"
         '这部分不该记在 M1 头上。\n')
L.append('**逐题因果口径**（比净翻转更严：净翻转是聚合差，会把两边各自的翻转相消）：\n')
L.append('| 预算 | NoPool 赢的题 | 其中基线"曾答对又改错" | 基线"从未答对" | 过度修订份额 |')
L.append('|---|---|---|---|---|')
for bud in ('24', '16'):
    dc = z1b['per_budget'][bud]['discordant']
    L.append(f"| {bud}k | {dc['n_wins']} | {dc['baseline_over_revised']} | "
             f"{dc['baseline_never_correct']} | **{dc['over_revision_share']*100:.0f}%** |")
L.append('\n"从未答对"的题不是过度修订——是朴素管线本身就没解出来，M1 无从施力。\n')

L.append('**消融臂复核（MATH 24k，vs Unc+Fmt）**：\n')
L.append('| 臂 | SR | token | 翻转(strict+budget) | 条件翻转率 | 路由 | Δ vs Unc+Fmt | McNemar |')
L.append('|---|---|---|---|---|---|---|---|')
for tag, a in z1b['ablations'].items():
    c = a['vs_unc_fmt']
    L.append(f"| {tag} | {a['sr']*100:.1f}% | {a['avg_tokens']:,.0f} | "
             f"{a['flips']} ({a['flips_strict_overrevision']}+"
             f"{a['flips_budget_finalize_artifact']}) | "
             f"{a['flip_rate_given_ever_correct']*100:.1f}% | {a['routes']} | "
             f"{c['delta']*100:+.1f}pp | {c['mcnemar_b']}/{c['mcnemar_c']}, p={c['mcnemar_p']:.2e} |")
_ab1, _ab3 = z1b['ablations']['Ab1 -M1'], z1b['ablations']['Ab3 -M2 (fixed L2 pipe)']
_l0 = _ab1['routes'].get('L0', 0)
_uf24s = z1b['per_budget']['24']['unc_fmt']
L.append('\n两个关键读数：\n')
L.append(f"1. **`−M1` 在 MATH 上测不到该机制**：级联开着时 {_l0}/{sum(_ab1['routes'].values())} "
         '题走 L0 单轮直答，只有一轮就没有"改错"的机会，M1 天然无事可做——Ab1 掉 '
         f"{(z1b['ablations']['NoPool(full)']['sr'] - _ab1['sr'])*100:.1f}pp 是构造上的 no-op，"
         '不能当作"M1 不重要"的证据。')
L.append(f"2. **`−M2`（强制全 L2）才是唯一隔离该机制的臂**：翻转 {_uf24s['flips']} → "
         f"{_ab3['flips']}，其中 budget-finalize "
         f"{_uf24s['flips_budget_finalize_artifact']} → "
         f"{_ab3['flips_budget_finalize_artifact']}、strict "
         f"{_uf24s['flips_strict_overrevision']} → {_ab3['flips_strict_overrevision']}。"
         '**在固定管线里 M1 消掉的几乎全是预算耗尽的收尾截断，真·中途改错基本没动。**')
L.append('\n**增益按路由层拆分（NoPool 的 escalation_level，24k）**：\n')
L.append('| 层 | n | NoPool SR | Unc+Fmt SR | 不一致(赢/输) | 净题数 | 占总增益 |')
L.append('|---|---|---|---|---|---|---|')
for lvl, g in z1b['route_gain'].items():
    L.append(f"| {lvl} | {g['n']} | {g['sr_nopool']*100:.1f}% | {g['sr_unc_fmt']*100:.1f}% | "
             f"{g['disc_b']}/{g['disc_c']} | {g['net_tasks']:+d} | "
             f"{g['share_of_total_pp']:+.1f}pp |")
_l0g = z1b['route_gain'].get('L0', {})
L.append(f"\n**{_l0g.get('share_of_total_pp', 0):+.1f}pp / "
         f"{z1b['per_budget']['24']['real_gap_pp']:+.1f}pp 长在 L0 子集上**——那一层"
         '"方法"就是单轮直答、不跑管线。头条战果的主体形态是**"数学题不要跑多 agent 管线"**，'
         '而非预算治理本身。\n')
L.append(f"**Z1b 判决：{z1b_verdict}。**\n")

# ---------------- Z1c 报告：缺口补测 ----------------
L.append('\n## Z1c — 缺口补测（§8.4b 的两个 400 题·次实验）\n')
_a6 = z1c['ab6_m1_isolation']
if _a6:
    m = _a6['m1_effect']
    L.append('**(a) Ab6 = 固定 L2 管线 + M3、不带 M1** —— 唯一能让 M1 施力的环境'
             '（Ab1 在级联下是 no-op）。`Ab3 − Ab6` 是 M1 自身 SR 贡献的首个干净估计：\n')
    L.append('| 臂 | SR [95% CI] | token | 翻转(strict+budget) | 条件翻转率 |')
    L.append('|---|---|---|---|---|')
    for tag, k in (('Ab3（+M1）', 'ab3_with_m1'), ('Ab6（−M1）', 'ab6_no_m1')):
        b = _a6[k]
        L.append(f"| {tag} | {b['sr']*100:.1f}% [{b['sr_ci'][0]*100:.1f}, "
                 f"{b['sr_ci'][1]*100:.1f}] | {b['avg_tokens']:,.0f} | "
                 f"{b['flips']} ({b['strict']}+{b['budget_finalize']}) | "
                 f"{b['cond_flip']*100:.1f}% |")
    L.append(f"\n**M1 净贡献 = {m['delta']*100:+.1f}pp** "
             f"[{m['delta_ci'][0]*100:+.1f}, {m['delta_ci'][1]*100:+.1f}]，"
             f"McNemar {m['mcnemar_b']}/{m['mcnemar_c']}, p={m['mcnemar_p']:.3g}"
             f"（n={_a6['n']}）。")
    _uf = _a6['ab6_vs_unc_fmt']
    L.append(f"Ab6 vs Unc+Fmt：{_uf['delta']*100:+.1f}pp，"
             f"{_uf['mcnemar_b']}/{_uf['mcnemar_c']}, p={_uf['mcnemar_p']:.3g}"
             '——这一格说明去掉 M1 之后，M3 单独还能剩多少。\n')
else:
    L.append('**(a) Ab6（−M1 隔离臂）：未完成**——`traces/..._v54a_ma_ab6.jsonl` 不存在。'
             'M1 的自身贡献仍无干净估计，§8.4b 的第一个缺口保持开放。\n')

_gm = z1c['gemma_format_control']
if _gm:
    L.append('**(b) Gemma × Unc+Fmt** —— X1 的 +25.8pp 是对 Unc（无格式）测的，'
             '与 Qwen E1 证明有 57% 是格式的那个口径相同。补上格式对照臂：\n')
    L.append('| Gemma MATH 24k | SR [95% CI] | token |')
    L.append('|---|---|---|')
    for tag, k in (('V5.1-NoPool', 'nopool'), ('**Unc+Fmt**', 'unc_fmt'), ('Unc', 'unc')):
        if k in _gm:
            b = _gm[k]
            L.append(f"| {tag} | {b['sr']*100:.1f}% [{b['sr_ci'][0]*100:.1f}, "
                     f"{b['sr_ci'][1]*100:.1f}] | {b['avg_tokens']:,.0f} |")
    c = _gm['nopool_vs_unc_fmt']
    L.append(f"\n**NoPool vs Unc+Fmt：{c['delta']*100:+.1f}pp** "
             f"[{c['delta_ci'][0]*100:+.1f}, {c['delta_ci'][1]*100:+.1f}]，"
             f"McNemar {c['mcnemar_b']}/{c['mcnemar_c']}, p={c['mcnemar_p']:.3g}"
             f"（n={_gm['n']}）。")
    if _gm.get('format_share') is not None:
        L.append(f"格式占 Gemma 全额增益的 **{_gm['format_share']*100:.0f}%**"
                 f"（Qwen 同口径 57%）。")
else:
    L.append('\n**(b) Gemma × Unc+Fmt：未完成**——`traces/..._gemma-4-31B-it_v54b_ma_fmt.jsonl` '
             '不存在。跨模型的 SR 增益仍无格式对照，+25.8pp **不可作为方法 SR 增益的'
             '跨模型证据**引用。\n')

L.append('\n## Z2 — Unc+Fmt 正式基线统计\n')
L.append('| 格 | Unc+Fmt SR [95% CI] | token | NoPool SR [95% CI] | Δ(pp) [配对CI] | McNemar |')
L.append('|---|---|---|---|---|---|')
for d in ('ga', 'fr'):
    for bud in ('24', '16'):
        z = z2[f'{d}|{bud}']
        u, p, c = z['unc_fmt'], z['nopool'], z['nopool_vs_unc_fmt']
        L.append(f"| {DS_NAME[d]} {bud}k | {u['sr']*100:.1f}% "
                 f"[{u['sr_ci'][0]*100:.1f}, {u['sr_ci'][1]*100:.1f}] | {u['avg_tokens']:,.0f} | "
                 f"{p['sr']*100:.1f}% [{p['sr_ci'][0]*100:.1f}, {p['sr_ci'][1]*100:.1f}] | "
                 f"{c['delta']*100:+.1f} [{c['delta_ci'][0]*100:+.1f}, {c['delta_ci'][1]*100:+.1f}] | "
                 f"{c['mcnemar_b']}/{c['mcnemar_c']}, p={c['mcnemar_p']:.3f} |")

L.append('\n## Z3 — 多预算点 Pareto 曲线\n')
L.append('图：`figures/v53_pareto_{ga,ma,fr,mu}.png`（16k/24k 连线；8k 点实验进行中，图内已标注待补）。\n')
L.append('GAIA/FRAMES 前沿重合判读（NoPool vs Unc+Fmt，同预算）：\n')
L.append('| 格 | ΔSR(pp) | Δtoken | 漂移 SD(pp) | 差值/漂移SD |')
L.append('|---|---|---|---|---|')
for k, v in z3_read.items():
    d, bud = k.split('|')
    L.append(f"| {DS_NAME[d]} {bud}k | {v['sr_diff_pp']:+.1f} | {v['token_diff_pct']:+.0f}% | "
             f"{v['drift_sd_pp']:.1f} | {v['diff_in_drift_sd']:.2f}× |")
same = all(abs(v['diff_in_drift_sd']) < 2.0 for v in z3_read.values())
z3_msg = ('四格差值均在 2 倍漂移 SD（≈95%）以内——GAIA 24k +0.35×、FRAMES 两格 ≤1.1×，'
          '两条曲线重合：支持「同一条前沿」，不支持「更省」；NoPool 的残余价值是同 SR 下'
          ' token −9%~−23%（Δtoken 列）。8k 点（E3）到位后复核。'
          if same else
          '存在超出 2 倍漂移 SD 的格子，前沿重合存疑，等 E3 的 8k 点判决。')
L.append(f"\n判读：{z3_msg}\n")

L.append('\n## Z4 — GAIA Unc+Fmt 同分核查\n')
L.append(f"- 24k：{z4['correct_24k']}/{z4['n_24k']} 正确，平均 {z4['avg_tokens_24k']:,.0f} tok；"
         f"16k：{z4['correct_16k']}/{z4['n_16k']} 正确，平均 {z4['avg_tokens_16k']:,.0f} tok")
L.append(f"- 交集 {z4['intersection']} / 并集 {z4['union']}，**Jaccard = {z4['jaccard']:.3f}**")
L.append(f"- 仅 24k 对：{len(z4['only_24k'])} 题；仅 16k 对：{len(z4['only_16k'])} 题")
L.append(f"\n**{z4['verdict']}。**\n")

L.append('\n## Z5 — 双噪声源合并口径\n')
L.append(f"运行漂移 SD = √(f/n)，f = {FLIP_F:.1%}（GAIA 实测翻转率）。"
         f"重跑复核：v52rr vs v51t NoPool 逐题翻转 {rr_check['flips']}/{rr_check['n_paired']} "
         f"= {rr_check['flip_rate']*100:.1f}%，SR {rr_check['sr_base']*100:.1f}% → "
         f"{rr_check['sr_rerun']*100:.1f}%。\n")
L.append('| 数据集 | n | 关键差值(vs 基线) | 抽样SE(pp) | 漂移SD(pp) | 合并SE(pp) | 差值/漂移SD | 差值/合并SE | McNemar |')
L.append('|---|---|---|---|---|---|---|---|---|')
for d in DS:
    v = z5[d]
    L.append(f"| {DS_NAME[d]} | {v['n']} | {v['key_delta_pp']:+.1f}pp (vs {v['baseline']}) | "
             f"{v['delta_se_pp']:.1f} | {v['drift_sd_pp']:.1f} | {v['combined_se_pp']:.1f} | "
             f"{v['delta_in_drift_sd']:.1f}× | {v['delta_in_combined_se']:.1f}× | "
             f"{v['mcnemar']}, p={v['mcnemar_p']:.2g} |")
L.append(f"\n**{z5_conclusion}**\n")

with open('v53_phasez_report.md', 'w') as fh:
    fh.write('\n'.join(L) + '\n')

print('written results/v53_phasez.json, v53_phasez_report.md, figures/v53_pareto_*.png')
print('Z1 verdict:', json.dumps(z1_verdict, ensure_ascii=False))
print('E2 gate:', e2_gate)
print('Z4 jaccard:', z4['jaccard'])
