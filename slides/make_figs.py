"""Charts for the talk, drawn from the paper's data (paper/data/*.json and the recorded runs).

Run with the project venv: .venv/bin/python slides/make_figs.py  ->  slides/img/c_*.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
PAPER = HERE.parent / "paper"
OUT = HERE / "img"
M = json.loads((PAPER / "data" / "main_results.json").read_text())
A = json.loads((PAPER / "data" / "final_analyses.json").read_text())
PLUG = json.loads((PAPER / "data" / "plugin_baseline.json").read_text())

NAVY, BODY, MUTED, GRID = "#14213D", "#3A4150", "#6B7280", "#E4E1D8"
ORANGE, BLUE, LBLUE, RED, GREEN, GREY = "#B4541A", "#2E5C99", "#9DB9DE", "#B03A2E", "#2F6B34", "#A7A9AE"
MODELS = [("gpt-oss-20b", "20b"), ("gpt-oss-120b", "120b"), ("Gemma-4-26B-A4B", "Gemma")]
BENCH = [("MATH", "MATH"), ("FRAMES", "FRAMES"), ("MuSiQue", "MuSiQue"), ("BBEH", "BBEH")]
KEYS = [f"{m}|{b}" for m, _ in MODELS for b, _ in BENCH]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 20, "axes.edgecolor": MUTED, "axes.labelcolor": BODY,
                     "xtick.color": BODY, "ytick.color": BODY, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.dpi": 160, "savefig.facecolor": "white"})


def setting_axis(ax, label_y=-0.22):
    """12 settings grouped by model: benchmark ticks, model names below."""
    xs = np.array([g * 5 + j for g in range(3) for j in range(4)], float)
    ax.set_xticks(xs, [b for _ in MODELS for _, b in BENCH], rotation=40, ha="right", rotation_mode="anchor", fontsize=14)
    for g, (_, short) in enumerate(MODELS):
        ax.text(g * 5 + 1.5, label_y, short, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=19, fontweight="bold", color=NAVY)
    ax.set_xlim(-0.8, 14.8)
    return xs


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(name)


def accuracy():
    fig, ax = plt.subplots(figsize=(13, 5.6))
    xs = setting_axis(ax)
    for off, key, col, lab in ((-0.26, "L0", GREY, "direct answer"), (0.0, "L1", LBLUE, "single agent"),
                               (0.26, "pipe_acc", ORANGE, "agent team")):
        ax.bar(xs + off, [M[k][key] for k in KEYS], 0.26, color=col, label=lab)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("accuracy (%)")
    ax.legend(loc="upper center", fontsize=17, ncol=3, bbox_to_anchor=(0.5, 1.08))
    save(fig, "c_accuracy.png")


def cost():
    fig, ax = plt.subplots(figsize=(13, 5.6))
    xs = setting_axis(ax)
    v = [A["team"][k]["cost_vs_L1"] for k in KEYS]
    ax.bar(xs, v, 0.72, color=ORANGE)
    for x, y in zip(xs, v):
        ax.text(x, y + 0.12, f"{y:.1f}×", ha="center", fontsize=15, color=BODY)
    ax.axhline(1, color=BLUE, lw=3, ls="--")
    ax.text(14.7, 6.9, "dashed line: single agent = 1×", ha="right", va="top", color=BLUE, fontsize=17, fontweight="bold")
    ax.set_ylim(0, 7.2)
    ax.set_ylabel("team tokens ÷ single agent")
    save(fig, "c_cost.png")


def flows():
    fig, ax = plt.subplots(figsize=(13, 5.8))
    xs = setting_axis(ax)
    r = [A["team"][k]["rescue_L1"] for k in KEYS]
    h = [-A["team"][k]["harm_L1"] for k in KEYS]
    ax.bar(xs, r, 0.72, color=GREEN, label="team fixes a wrong answer")
    ax.bar(xs, h, 0.72, color=RED, label="team breaks a right answer")
    ax.axhline(0, color=BODY, lw=1.2)
    ax.set_ylabel("% of tasks (vs. single agent)")
    ax.set_ylim(-11, 17)
    ax.legend(loc="upper left", fontsize=17, ncol=2)
    save(fig, "c_flows.png")


def agree():
    fig, ax = plt.subplots(figsize=(13, 5.6))
    xs = setting_axis(ax)
    a = [M[k]["L1_acc_agree"] for k in KEYS]
    d = [M[k]["L1_acc_disagree"] for k in KEYS]
    ax.bar(xs - 0.19, a, 0.38, color=GREEN, label="L₀ and L₁ agree")
    ax.bar(xs + 0.19, d, 0.38, color=GREY, label="they disagree")
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("single agent accuracy (%)")
    ax.legend(loc="upper center", fontsize=17, ncol=2, bbox_to_anchor=(0.5, 1.08))
    save(fig, "c_agree.png")


def auroc():
    fig, ax = plt.subplots(figsize=(13, 5.6))
    xs = setting_axis(ax)
    v = [M[k]["auroc_S0"] for k in KEYS]
    ax.bar(xs, [y - 0.5 for y in v], 0.72, bottom=0.5, color=[BLUE if y >= 0.5 else RED for y in v])
    ax.axhline(0.5, color=BODY, lw=1.5, ls="--")
    ax.text(14.7, 0.515, "coin flip", ha="right", fontsize=16, color=BODY)
    ax.set_ylim(0.15, 0.9)
    ax.set_ylabel("how well confidence\nseparates right from wrong")
    ax.text(12.5, 0.62, "backwards:\nconfident but wrong", ha="center", va="bottom", fontsize=16, color=RED, fontweight="bold")
    save(fig, "c_auroc.png")


def records():
    sys.path.insert(0, str(PAPER / "analysis"))
    from climb_core import p_betting
    from final_analyses import frame
    from main_results import families
    x = frame("gpt-oss-120b", "MuSiQue")
    F = families(x, x.ag.values)
    rng = np.random.default_rng(0)
    cal = rng.permutation(len(x))[: len(x) // 2]
    Y, C, L, YK, CK = F[1]                      # thresholds on L0's confidence
    return L, C, CK, cal, p_betting


def sweep_and_wealth():
    L, C, CK, cal, p_betting = records()
    e, lvl = 0.05, 0.05
    risk = L.mean(1) * 100
    tok = C.mean(1) / CK.mean() * 100
    p = np.r_[0.0, p_betting(L[1:, cal], e, lvl)]
    first_fail = next(k for k in range(1, len(p)) if p[k] > lvl)
    keep = [0] + [k for k in range(1, len(risk)) if (round(risk[k], 2), round(tok[k], 1)) != (round(risk[k - 1], 2), round(tok[k - 1], 1))]

    fig, ax = plt.subplots(figsize=(12, 6.2))
    ok = [k for k in keep if k < first_fail]
    bad = [k for k in keep if k >= first_fail]
    ax.plot(tok[keep], risk[keep], color=GREY, lw=1.5, zorder=1)
    ax.scatter(tok[ok], risk[ok], s=90, color=GREEN, zorder=3, label="passes the test")
    ax.scatter(tok[bad], risk[bad], s=90, color=GREY, zorder=3, label="not certified")
    best = min(ok, key=lambda k: tok[k])
    ax.scatter([tok[best]], [risk[best]], s=420, marker="*", color=ORANGE, zorder=4, label="deployed: cheapest that passes")
    ax.axhline(5, color=RED, lw=2.5, ls="--")
    ax.text(101, 5.25, "tolerance ε = 5 points", ha="left", color=RED, fontsize=17, fontweight="bold")
    ax.annotate("", xy=(30, 8.7), xytext=(96, 8.7), arrowprops=dict(arrowstyle="-|>", lw=2.5, color=NAVY))
    ax.text(63, 9.05, "test in this order: safe → risky", ha="center", fontsize=17, color=NAVY)
    ax.set_xlim(102, 15)
    ax.set_ylim(-0.4, 10)
    ax.set_xlabel("tokens (% of always calling the team)")
    ax.set_ylabel("accuracy lost (points)")
    ax.legend(loc="lower right", fontsize=16)
    save(fig, "c_sweep.png")

    Lc = L[:, cal]
    t = np.arange(1, Lc.shape[1] + 1)

    def wealth(l):
        s1, s2 = np.cumsum(l) - l, np.cumsum(l * l) - l * l
        var = np.maximum((s2 + 0.25) / t - (s1 / t) ** 2, 1e-3)
        lam = np.minimum(0.5 / (1 - e), np.sqrt(2 * np.log(1 / lvl) / (var * len(l))))
        return np.cumprod(1 + lam * (e - l))

    def near(target):
        return min(keep[1:], key=lambda k: abs(risk[k] - target))

    fig, ax = plt.subplots(figsize=(12, 6.2))
    for k, col in ((near(0.5), GREEN), (near(2.2), BLUE), (near(6.0), RED)):
        w = wealth(Lc[k])
        status = "passes" if w.max() >= 1 / lvl else "fails"
        ax.plot(t, w, color=col, lw=3, label=f"loses {risk[k]:.1f} points: {status}")
    ax.axhline(1 / lvl, color=NAVY, lw=2.5, ls="--")
    ax.text(t[-1], 1 / lvl * 1.6, "pass line: wealth × 20", ha="right", fontsize=17, color=NAVY, fontweight="bold")
    ax.axhline(1, color=GREY, lw=1.2)
    ax.set_yscale("log")
    ax.set_ylim(0.01, 1e6)
    ax.set_xlabel("calibration tasks seen")
    ax.set_ylabel("gambler’s wealth (start = 1)")
    ax.legend(loc="upper left", fontsize=16)
    ax.grid(True, axis="y", which="major")
    save(fig, "c_wealth.png")
    print("first fail", first_fail, "best", best, round(tok[best]), round(risk[best], 1))


def viol():
    fig, ax = plt.subplots(figsize=(13, 5.6))
    xs = setting_axis(ax)
    v = [max(M[k]["climb"][e]["viol"] for e in ("0.02", "0.05", "0.1")) for k in KEYS]
    ax.bar(xs, v, 0.72, color=GREEN)
    ax.axhline(10, color=RED, lw=3, ls="--")
    ax.text(14.7, 10.4, "allowed: 10%", ha="right", color=RED, fontsize=18, fontweight="bold")
    ax.set_ylim(0, 12)
    ax.set_ylabel("splits where CLIMB\nlost more than ε (%)")
    save(fig, "c_viol.png")


def savings():
    fig, ax = plt.subplots(figsize=(13, 5.6))
    xs = setting_axis(ax)
    for off, e, col, lab in ((-0.19, "0.05", LBLUE, "ε = 5 points"), (0.19, "0.1", BLUE, "ε = 10 points")):
        ax.bar(xs + off, [M[k]["climb"][e]["tok"] for k in KEYS], 0.38, color=col, label=lab)
    ax.axhline(100, color=ORANGE, lw=3)
    ax.text(14.7, 102, "always call the team = 100%", ha="right", color=ORANGE, fontsize=16, fontweight="bold")
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("tokens used (%)")
    ax.legend(loc="upper left", fontsize=16, ncol=2, bbox_to_anchor=(0.0, 0.98))
    save(fig, "c_savings.png")


def tuned():
    eps = ("0.02", "0.05", "0.1")
    t = [np.mean([PLUG[k][e]["plugin"]["viol"] for k in KEYS]) for e in eps]
    c = [np.mean([PLUG[k][e]["climb"]["viol"] for k in KEYS]) for e in eps]
    fig, ax = plt.subplots(figsize=(10, 6))
    xs = np.arange(3)
    ax.bar(xs - 0.2, t, 0.4, color=RED, label="threshold tuned on data, no test")
    ax.bar(xs + 0.2, c, 0.4, color=GREEN, label="CLIMB")
    for x, y in zip(xs, t):
        ax.text(x - 0.2, y + 1, f"{y:.0f}%", ha="center", fontsize=18, color=RED, fontweight="bold")
    for x, y in zip(xs, c):
        ax.text(x + 0.2, y + 1, f"{y:.1f}%", ha="center", fontsize=18, color=GREEN, fontweight="bold")
    ax.axhline(10, color=NAVY, lw=2.5, ls="--")
    ax.text(2.6, 11, "allowed 10%", ha="right", fontsize=16, color=NAVY)
    ax.set_xticks(xs, ["ε = 2 points", "ε = 5 points", "ε = 10 points"])
    ax.set_ylim(0, 48)
    ax.set_ylabel("splits that break the promise (%)")
    ax.legend(loc="upper right", fontsize=16)
    save(fig, "c_tuned.png")


def maxloss():
    rows = [("Always direct answer", 20.6, RED), ("Fixed threshold, same cost", 15.9, RED),
            ("Always single agent", 9.4, RED), ("CLIMB, ε = 5", 2.8, GREEN)]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ys = np.arange(len(rows))[::-1]
    ax.barh(ys, [r[1] for r in rows], 0.6, color=[r[2] for r in rows])
    for y, r in zip(ys, rows):
        ax.text(r[1] + 0.3, y, f"{r[1]:.1f}", va="center", fontsize=19, fontweight="bold", color=r[2])
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.axvline(5, color=NAVY, lw=2.5, ls="--")
    ax.text(5.3, 3.45, "ε = 5", fontsize=16, color=NAVY)
    ax.set_xlim(0, 23)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.set_xlabel("worst accuracy loss over 12 settings (points)")
    save(fig, "c_maxloss.png")


def tests():
    rows = [("Worst-case bound (HB)", 100), ("Plain test, no sequence", 93), ("Ignore saved answers", 79),
            ("CLIMB’s test", 62)]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ys = np.arange(len(rows))[::-1]
    cols = [GREY, GREY, GREY, GREEN]
    ax.barh(ys, [r[1] for r in rows], 0.6, color=cols)
    for y, r, c in zip(ys, rows, cols):
        ax.text(r[1] + 1, y, f"{r[1]}%", va="center", fontsize=19, fontweight="bold", color=c if c != GREY else BODY)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_xlim(0, 112)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.set_xlabel("tokens certified at ε = 5 (% of the team’s, lower is better)")
    save(fig, "c_tests.png")


if __name__ == "__main__":
    for f in (accuracy, cost, flows, agree, auroc, viol, savings, tuned, maxloss, tests, sweep_and_wealth):
        f()
