#!/usr/bin/env python
"""v5.2 Phase R: 纯离线重分析（零 GPU）。

从 traces/*.jsonl 实算：
  R0 字段体检 / R1 bootstrap CI + McNemar / R2 题集对齐 / R3 EM-F1 双列 /
  R4 Pareto + 分流审计 + token 分解 / R5 格式抽取审计。
产出 results/v52_reanalysis.json、figures/v52_pareto_{ds}.png、v52_reanalysis_report.md。

口径备注（从 _run_v2_experiment.py 核实）：
- 判题文本 = 产生最终 answer 的那一轮的完整 content；trace 里的 answer_snapshot
  是 content 的前 2000 字符，R5 审计基于"最后一轮 answer_snapshot"，与判题同源但
  存在截断风险（单独量化）。
- rounds[] 只记录 executor 轮次；planner/reviewer 调用计入 total_used 但不在 rounds
  里。因此 token 分解中 L2 成本 = L2 executor 轮 + (total_used - rounds 合计) 的余量。
- 级别边界推断：级联臂（m2 开）rounds 编号为 L0:(0), L1:(1), L2:(0..k 重新编号)，
  用"第二个 round==0 的位置"作为 L2 起点；Unc/A6（m2 关）全程 L2，只有一个编号序列。
"""
import json, math, re, collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PRE = 'traces/tasks_sweep_Qwen3.6-27B-FP8_'
DS = ['ga', 'ma', 'fr', 'mu']
DS_NAME = {'ga': 'GAIA', 'ma': 'MATH', 'fr': 'FRAMES', 'mu': 'MuSiQue'}
BUDGET_CAP = {24: 24000, 16: 16000}
F24 = {d: f'{PRE}v51t_{d}.jsonl' for d in DS}
F16 = {d: f'{PRE}v51t16_{d}.jsonl' for d in DS}
FV5 = {'ga': f'{PRE}v5test_ga.jsonl', 'mu': f'{PRE}v5test_mu.jsonl',
       'ma': f'{PRE}v5al_ma.jsonl', 'fr': f'{PRE}v5al_fr.jsonl'}
V5_ARMS = ['V5-A0-Full', 'V5-A1-NoM1', 'V5-A2-NoM2', 'V5-A3-NoNotes', 'V5-A4-NoM4',
           'V5-A5-NoM5', 'V5-A6-OnlyM3', 'V5-M4-Equal', 'V5-Unconstrained']
UNC, UNI, NP = 'V5-Unconstrained', 'V5.1-Unified', 'V5.1-NoPool'
B = 10000
SEED = 20260828


def load(path):
    """-> {(config, task_id): record}"""
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

# test split 定义 = v51t_{ds} 的 task_id 集合
TEST_IDS = {d: {tid for (c, tid) in DATA[('24', d)]} for d in DS}
for d in DS:
    ids16 = {tid for (c, tid) in DATA[('16', d)]}
    assert ids16 == TEST_IDS[d], f'{d}: 16k/24k test ids 不一致'


def get(budget, d, cfg):
    """取某格记录 {task_id: rec}；24k 的 v5 臂（含 Unc）从 v5 文件按 test ids 过滤。"""
    if budget == '24' and cfg in V5_ARMS:
        src = DATA[('v5', d)]
        return {tid: src[(cfg, tid)] for tid in TEST_IDS[d] if (cfg, tid) in src}
    src = DATA[(budget, d)]
    return {tid: src[(cfg, tid)] for tid in TEST_IDS[d] if (cfg, tid) in src}


def boot_ci(x, seed):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(x)
    means = np.empty(B)
    for s in range(0, B, 2000):
        k = min(2000, B - s)
        idx = rng.integers(0, n, size=(k, n))
        means[s:s + k] = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_boot_ci(a, b, seed):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    diff = a - b
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.empty(B)
    for s in range(0, B, 2000):
        k = min(2000, B - s)
        idx = rng.integers(0, n, size=(k, n))
        means[s:s + k] = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mcnemar_p(b, c):
    """discordant pairs 精确二项检验（双侧）。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    ln2 = n * math.log(2)
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - ln2
            for i in range(k + 1)]
    m = max(logs)
    return min(1.0, 2.0 * math.exp(m + math.log(sum(math.exp(l - m) for l in logs))))


def cell_stats(recs):
    succ = np.array([r['success'] for r in recs.values()], dtype=float)
    f1 = np.array([r['f1'] for r in recs.values()], dtype=float)
    tok = np.array([r['total_used'] for r in recs.values()], dtype=float)
    lo, hi = boot_ci(succ, SEED + len(succ))
    return {'n': len(recs), 'sr': float(succ.mean()), 'sr_ci': [lo, hi],
            'em': float(succ.mean()), 'f1': float(f1.mean()), 'avg_tokens': float(tok.mean())}


def compare(recs_a, recs_b, seed):
    """a - b，按 task_id 配对。返回 Δ、配对 bootstrap CI、McNemar。"""
    ids = sorted(set(recs_a) & set(recs_b))
    a = np.array([recs_a[i]['success'] for i in ids], dtype=float)
    b = np.array([recs_b[i]['success'] for i in ids], dtype=float)
    delta = float((a - b).mean())
    lo, hi = paired_boot_ci(a, b, seed)
    bb = int(((a == 1) & (b == 0)).sum())
    cc = int(((a == 0) & (b == 1)).sum())
    return {'n_paired': len(ids), 'delta': delta, 'delta_ci': [lo, hi],
            'mcnemar_b': bb, 'mcnemar_c': cc, 'mcnemar_p': mcnemar_p(bb, cc)}


# ---------------------------------------------------------------- R0 字段体检
R0_FIELDS = ['f1', 'escalation_level', 'stopped_by', 'flags_fired', 'rounds']
r0 = {}
for key, tab in DATA.items():
    for (cfg, tid), rec in tab.items():
        row = r0.setdefault((key, cfg), {f: True for f in R0_FIELDS + ['answer_snapshot', 'm1_flag']})
        for f in R0_FIELDS:
            if f not in rec:
                row[f] = False
        if not rec.get('rounds') or 'answer_snapshot' not in rec['rounds'][-1]:
            row['answer_snapshot'] = False
        if 'm1_early_stop' not in rec.get('flags_fired', {}):
            row['m1_flag'] = False
r0_out = {'|'.join(str(x) for x in (k[0] if isinstance(k[0], tuple) else (k[0],)) + (k[1],)): v
          for k, v in r0.items()}

# ---------------------------------------------------------------- R1+R3 主表与比较
main, comps = {}, {}
for d in DS:
    for bud in ('24', '16'):
        for cfg in (UNC, UNI, NP):
            main[f'{d}|{bud}|{cfg}'] = cell_stats(get(bud, d, cfg))
        npool, unc, uni = get(bud, d, NP), get(bud, d, UNC), get(bud, d, UNI)
        comps[f'{d}|{bud}|NoPool_vs_Unc'] = compare(npool, unc, SEED + 1)
        comps[f'{d}|{bud}|NoPool_vs_Unified'] = compare(npool, uni, SEED + 2)

r1_summary = {}
for cmp_name in ('NoPool_vs_Unc', 'NoPool_vs_Unified'):
    cells = [comps[f'{d}|{bud}|{cmp_name}'] for d in DS for bud in ('24', '16')]
    r1_summary[cmp_name] = {
        'direction_positive': sum(1 for c in cells if c['delta'] > 0),
        'total': len(cells),
        'significant': sum(1 for c in cells if c['delta'] > 0 and c['mcnemar_p'] < 0.05),
    }

# ΔEM vs ΔF1（NoPool − Unc）
em_f1 = {}
for d in DS:
    for bud in ('24', '16'):
        npool, unc = get(bud, d, NP), get(bud, d, UNC)
        ids = sorted(set(npool) & set(unc))
        d_em = float(np.mean([npool[i]['success'] for i in ids]) - np.mean([unc[i]['success'] for i in ids]))
        d_f1 = float(np.mean([npool[i]['f1'] for i in ids]) - np.mean([unc[i]['f1'] for i in ids]))
        em_f1[f'{d}|{bud}'] = {'delta_em': d_em, 'delta_f1': d_f1, 'gap': d_em - d_f1}

# ---------------------------------------------------------------- R2 题集对齐
r2 = {}
for d in ('ma', 'fr'):
    src = DATA[('v5', d)]
    all_ids = {tid for (c, tid) in src}
    for cfg in (UNC, 'V5-A0-Full', 'V5-A4-NoM4', 'V5-A6-OnlyM3'):
        full = {tid: src[(cfg, tid)] for tid in all_ids if (cfg, tid) in src}
        test = {tid: src[(cfg, tid)] for tid in TEST_IDS[d] if (cfg, tid) in src}
        r2[f'{d}|{cfg}'] = {'full': cell_stats(full), 'test_aligned': cell_stats(test)}

# ---------------------------------------------------------------- R4 分流审计 + token 分解
def split_levels(rec):
    """(probe_rounds, l2_rounds)：级联臂用第二个 round==0 定位 L2 起点。"""
    rounds = rec['rounds']
    idx0 = [i for i, r in enumerate(rounds) if r['round'] == 0]
    if rec.get('escalation_level') == 2 and len(idx0) >= 2 and idx0[0] == 0:
        b = idx0[1]
        return rounds[:b], rounds[b:]
    return [], rounds


def rtok(rs):
    return sum(r['prompt_tokens'] + r['completion_tokens'] + r['tool_output_tokens'] for r in rs)


r4 = {}
for d in DS:
    for cfg in (UNC, UNI, NP):
        recs = get('24', d, cfg)
        n = len(recs)
        esc = collections.Counter(r['escalation_level'] for r in recs.values())
        stop = collections.Counter(r['stopped_by'] for r in recs.values())
        m1 = sum(1 for r in recs.values() if r['flags_fired'].get('m1_early_stop', 0) > 0)
        r4[f'{d}|{cfg}'] = {
            'n': n,
            'esc_pct': {f'L{k}': esc.get(k, 0) / n for k in (0, 1, 2)},
            'stopped_by': {k: v / n for k, v in stop.most_common()},
            'm1_rate': m1 / n,
        }
    # token 分解（级联臂）
    unc_recs = get('24', d, UNC)
    unc_tok = float(np.mean([r['total_used'] for r in unc_recs.values()]))
    unc_exec = float(np.mean([rtok(r['rounds']) for r in unc_recs.values()]))
    for cfg in (UNI, NP):
        recs = get('24', d, cfg)
        groups = {0: [], 1: [], 2: []}
        probe_tok, l2_exec_tok, overhead = [], [], []
        l0_correct_esc, l1_correct_esc = 0, 0
        for r in recs.values():
            lv = r['escalation_level']
            groups[lv].append(r['total_used'])
            if lv == 2:
                probes, l2 = split_levels(r)
                rs_all = rtok(r['rounds'])
                probe_tok.append(rtok(probes))
                l2_exec_tok.append(rtok(l2))
                overhead.append(r['total_used'] - rs_all)
                if len(probes) >= 1 and probes[0].get('is_correct'):
                    l0_correct_esc += 1
                if len(probes) >= 2 and probes[1].get('is_correct'):
                    l1_correct_esc += 1
        ne = len(probe_tok)
        np_tok = float(np.mean([r['total_used'] for r in recs.values()]))
        r4[f'{d}|{cfg}']['token_decomp'] = {
            'unc_avg_tokens': unc_tok, 'unc_executor_tokens': unc_exec,
            'unc_overhead_tokens': unc_tok - unc_exec, 'arm_avg_tokens': np_tok,
            'savings_vs_unc': unc_tok - np_tok,
            'n_stay_L0': len(groups[0]), 'n_stay_L1': len(groups[1]), 'n_esc_L2': ne,
            'avg_tokens_stay_L0': float(np.mean(groups[0])) if groups[0] else None,
            'avg_tokens_stay_L1': float(np.mean(groups[1])) if groups[1] else None,
            'avg_tokens_esc_L2': float(np.mean(groups[2])) if groups[2] else None,
            'esc_L2_probe_tokens': float(np.mean(probe_tok)) if ne else None,
            'esc_L2_executor_tokens': float(np.mean(l2_exec_tok)) if ne else None,
            'esc_L2_overhead_tokens': float(np.mean(overhead)) if ne else None,
            'probe_share_of_total': (float(np.mean(probe_tok)) / np_tok) if ne else None,
            'esc_L2_l0_already_correct': l0_correct_esc / ne if ne else None,
            'esc_L2_l1_already_correct': l1_correct_esc / ne if ne else None,
        }

# Pareto 图
for d in DS:
    pts = []  # (label, tokens, sr, marker)
    v5arms = V5_ARMS if d in ('ga', 'mu') else [UNC, 'V5-A0-Full', 'V5-A4-NoM4', 'V5-A6-OnlyM3']
    for cfg in v5arms:
        recs = get('24', d, cfg)
        pts.append((cfg.replace('V5-', ''), np.mean([r['total_used'] for r in recs.values()]),
                    np.mean([r['success'] for r in recs.values()]), 'o'))
    for bud in ('24', '16'):
        for cfg in ((UNI, NP) if bud == '24' else (UNC, UNI, NP)):
            recs = get(bud, d, cfg)
            mk = '*' if bud == '24' else 's'
            pts.append((f"{cfg.replace('V5', 'v5')[:12]}@{bud}k",
                        np.mean([r['total_used'] for r in recs.values()]),
                        np.mean([r['success'] for r in recs.values()]), mk))
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, t, s, mk in pts:
        ax.scatter(t, s, marker=mk, s=90 if mk != 'o' else 45, zorder=3)
        ax.annotate(label, (t, s), textcoords='offset points', xytext=(6, 4), fontsize=7.5)
    srt = sorted(pts, key=lambda p: p[1])
    front, best = [], -1.0
    for p in srt:
        if p[2] > best + 1e-9:
            front.append(p); best = p[2]
    ax.plot([p[1] for p in front], [p[2] for p in front], 'k--', lw=1, alpha=0.6, label='Pareto front')
    ax.set_xlabel('avg actual tokens (total_used)')
    ax.set_ylabel('SR')
    ax.set_title(f'v5.2 Pareto — {DS_NAME[d]} (test split)')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'figures/v52_pareto_{d}.png', dpi=140)
    plt.close(fig)

# ---------------------------------------------------------------- R5 格式/抽取审计
ANCH1 = re.compile(r'answer\s*(?:is|are|was|were)?\s*[:\-]\s*(.+)', re.I)
ANCH2 = re.compile(r'answer\s+(?:is|are|was|were)\s+(.+)', re.I)


def extract_path(text):
    if ANCH1.search(text):
        return 'anchor'
    if ANCH2.search(text):
        return 'answer_is'
    return 'fallback'


r5 = {}
for d in DS:
    for bud in ('24', '16'):
        for cfg in (UNC, UNI, NP):
            recs = get(bud, d, cfg)
            n = len(recs)
            paths, em_by_path = collections.Counter(), collections.defaultdict(list)
            trunc_suspect = 0
            for r in recs.values():
                snap = r['rounds'][-1].get('answer_snapshot') or ''
                p = extract_path(snap)
                paths[p] += 1
                em_by_path[p].append(r['success'])
                if len(snap) >= 2000 and p == 'fallback':
                    trunc_suspect += 1
            r5[f'{d}|{bud}|{cfg}'] = {
                'n': n,
                'anchor_rate': paths['anchor'] / n,
                'answer_is_rate': paths['answer_is'] / n,
                'fallback_rate': paths['fallback'] / n,
                'trunc_suspect_rate': trunc_suspect / n,
                'em_by_path': {p: float(np.mean(v)) for p, v in em_by_path.items()},
                'n_by_path': dict(paths),
            }

# 格式增益上界（24k）：ΔEM 中至多有多少可归因于抽取路径
fmt_bound = {}
for d in DS:
    unc = get('24', d, UNC)
    npool = get('24', d, NP)
    u, v = r5[f'{d}|24|{UNC}'], r5[f'{d}|24|{NP}']
    em_unc = main[f'{d}|24|{UNC}']['em']
    em_np = main[f'{d}|24|{NP}']['em']
    delta = em_np - em_unc
    p_a_u = u['em_by_path'].get('anchor')
    p_f_u = u['em_by_path'].get('fallback')
    share_f_u = u['fallback_rate'] + u['answer_is_rate']
    # (a) 臂内上界：Unc 的非 anchor 项若都达到 Unc 自己 anchor 项的通过率
    ub_a, resid_a = None, None
    if p_a_u is not None and p_f_u is not None:
        ub_a = share_f_u * max(0.0, p_a_u - p_f_u)
        resid_a = delta - ub_a
    # (b) 跨臂版：Unc 的非 anchor 项若都达到 NoPool anchor 项的通过率
    p_a_np = v['em_by_path'].get('anchor')
    ub_b, resid_b = None, None
    if p_a_np is not None:
        ub_b = share_f_u * max(0.0, p_a_np - (p_f_u or 0.0))
        resid_b = delta - ub_b
    fmt_bound[d] = {'delta_em': delta, 'unc_anchor_em': p_a_u, 'unc_fallback_em': p_f_u,
                    'unc_nonanchor_share': share_f_u, 'nopool_anchor_em': p_a_np,
                    'upper_bound_within_arm': ub_a, 'residual_delta_within': resid_a,
                    'upper_bound_cross_arm': ub_b, 'residual_delta_cross': resid_b}

# ---------------------------------------------------------------- 汇总 JSON
out = {
    'meta': {'seed': SEED, 'B': B, 'split': 'test (random.Random(42) shuffle, drop first 20% dev)',
             'test_n': {d: len(TEST_IDS[d]) for d in DS},
             'notes': 'EM==SR（同一 success 字段）；token=total_used 实际消耗（含 planner/reviewer），预算帽分列。'},
    'r0_field_matrix': r0_out,
    'main_table': main,
    'comparisons': comps,
    'r1_summary': r1_summary,
    'r2_alignment': r2,
    'r3_em_f1': em_f1,
    'r4_routing_audit': r4,
    'r5_extraction_audit': r5,
    'r5_format_gain_bound': fmt_bound,
}
with open('results/v52_reanalysis.json', 'w') as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False)
print('written results/v52_reanalysis.json')
print('r1_summary:', json.dumps(r1_summary))
for k, v in fmt_bound.items():
    print(k, json.dumps(v))
