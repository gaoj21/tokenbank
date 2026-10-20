"""Build slides/CLIMB_talk.pptx from the deck in slides/ (speaker notes come from each slide's <aside>).

Run with the project venv: .venv/bin/python slides/make_pptx.py
"""
import html
import json
import re
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Pt

HERE = Path(__file__).parent
IMG = HERE / "img"

NAVY, NAVY2, LIGHT, PAPER, BAND = "14213D", "1F2F52", "F7F6F2", "FDFCF9", "ECEAE3"
BODY, MUTED, RULE, SOFT = "3A4150", "6B7280", "DDD8CC", "C9D1E0"
ORANGE, ORANGE_D, BLUE, RED, GREEN, GREENBG = "B4541A", "F0A66B", "2E5C99", "B03A2E", "2F6B34", "E3ECE2"
SERIF, SANS = "Georgia", "Arial"


def px(v):
    return Emu(int(v * 6350))  # 1920 px = 13.333 in


def rgb(h):
    return RGBColor.from_string(h)


def notes_of(sid):
    m = re.search(r"<aside>(.*?)</aside>", (HERE / "slides" / f"{sid}.html").read_text(), re.S)
    return html.unescape(m.group(1)).strip() if m else ""


def runs_of(text):
    """'plain **bold** [[c=HEX|colored]]' -> [(text, bold, color)]."""
    out = []
    for tok in re.split(r"(\*\*.*?\*\*|\[\[c=[0-9A-F]{6}\|.*?\]\])", text):
        if not tok:
            continue
        if tok.startswith("**"):
            out.append((tok[2:-2], True, None))
        elif tok.startswith("[[c="):
            out.append((tok[11:-2], True, tok[4:10]))
        else:
            out.append((tok, False, None))
    return out


def fill_tf(tf, paras, margin=0):
    tf.word_wrap = True
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, px(margin))
    first = True
    for p in paras:
        para = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        para.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[p.get("align", "l")]
        para.line_spacing = p.get("ls", 1.15)
        para.space_after = Pt(p.get("after", 0))
        for text, bold, color in runs_of(p["t"]):
            r = para.add_run()
            r.text = text
            f = r.font
            f.size = Pt(p.get("size", 15))
            f.name = p.get("font", SANS)
            f.bold = bold or p.get("bold", False)
            f.italic = p.get("italic", False)
            f.color.rgb = rgb(color or p.get("color", BODY))


def text(slide, x, y, w, h, paras, anchor="t"):
    tb = slide.shapes.add_textbox(px(x), px(y), px(w), px(h))
    fill_tf(tb.text_frame, paras)
    tb.text_frame.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]
    return tb


def box(slide, x, y, w, h, fill, line=None, paras=None, margin=40, radius=True, anchor="t"):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
                                 px(x), px(y), px(w), px(h))
    if radius:
        shp.adjustments[0] = min(0.08, 16 / min(w, h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = rgb(fill)
    if line:
        shp.line.color.rgb = rgb(line)
        shp.line.width = Pt(0.75)
    else:
        shp.line.fill.background()
    shp.shadow.inherit = False
    if paras:
        fill_tf(shp.text_frame, paras, margin)
        shp.text_frame.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE}[anchor]
    return shp


def image(slide, name, x, y, w, h):
    return slide.shapes.add_picture(str(IMG / name), px(x), px(y), px(w), px(h))


def new_slide(prs, sid, dark=False):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb(NAVY if dark else LIGHT)
    n = notes_of(sid)
    if n:
        s.notes_slide.notes_text_frame.text = n
    return s


def header(s, eyebrow, title, dark=False, size=28, h=80):
    text(s, 128, 118, 1664, 40, [dict(t=eyebrow.upper(), size=12, bold=True, color=ORANGE_D if dark else ORANGE)])
    text(s, 128, 160, 1664, h, [dict(t=title, size=size, bold=True, font=SERIF, color=LIGHT if dark else NAVY, ls=1.05)])


def footer(s, n, dark=False):
    text(s, 128, 1000, 1664, 34, [dict(t=str(n), size=12, align="r", color="8B97AE" if dark else MUTED)])


def big(t, color=ORANGE, size=40):
    return dict(t=t, size=size, bold=True, font=SERIF, color=color, after=8, ls=1.0)


def body(t, size=15, color=BODY, after=0, **kw):
    return dict(t=t, size=size, color=color, after=after, **kw)


def h3(t, color=NAVY, size=16):
    return dict(t=t, size=size, bold=True, color=color, after=6)


def table(s, x, y, w, rows, widths, size=14, row_h=56, right_from=1, highlight=()):
    shp = s.shapes.add_table(len(rows), len(rows[0]), px(x), px(y), px(w), px(row_h * len(rows)))
    tbl = shp.table
    tbl.first_row = True
    for j, frac in enumerate(widths):
        tbl.columns[j].width = px(w * frac)
    for i, row in enumerate(rows):
        tbl.rows[i].height = px(row_h)
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.fill.solid()
            if i == 0:
                cell.fill.fore_color.rgb = rgb(NAVY)
            elif i in highlight:
                cell.fill.fore_color.rgb = rgb(GREENBG)
            else:
                cell.fill.fore_color.rgb = rgb(PAPER if i % 2 else BAND)
            cell.margin_left = cell.margin_right = px(14)
            cell.margin_top = cell.margin_bottom = px(4)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            fill_tf(cell.text_frame, [dict(t=val, size=size, bold=(i == 0 or i in highlight),
                                           color=LIGHT if i == 0 else BODY,
                                           align="r" if j >= right_from else "l")])
            for side in ("margin_left", "margin_right"):
                setattr(cell.text_frame, side, px(14))
    return shp


def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = px(1920), px(1080)
    order = json.loads((HERE / "deck.json").read_text())["order"]
    assert len(order) == 19

    # 1 cover
    s = new_slide(prs, "cover", dark=True)
    text(s, 128, 300, 1664, 40, [dict(t="CLIMB", size=14, bold=True, color=ORANGE_D)])
    text(s, 128, 350, 1664, 130, [dict(t="When to Call the Team", size=52, bold=True, font=SERIF, color=LIGHT)])
    text(s, 128, 490, 1664, 60, [dict(t="Risk-controlled escalation for multi-agent LLM pipelines", size=22, color=SOFT)])
    box(s, 128, 590, 240, 4, ORANGE_D, radius=False)
    text(s, 128, 630, 1664, 50, [dict(t="Jianjun Gao, Jinxin Liu, Dongyun Lin, Hongyun Cai, Zheng Ma", size=15, color=SOFT)])

    # 2 cost
    s = new_slide(prs, "cost")
    header(s, "Motivation", "A plan–execute–review team is expensive")
    cards = [("1.3–6.3×", "the tokens of a single agent that reasons once"),
             ("5.4k–175k", "tokens per task, across three open models and four benchmarks"),
             ("28–57%", "of that cost goes to the planner’s and the reviewer’s own calls")]
    for i, (n, d) in enumerate(cards):
        box(s, 128 + i * 565, 330, 533, 300, PAPER, RULE, [big(n), body(d)])
    text(s, 128, 690, 1664, 120, [body("A planner writes a plan, an executor answers, a reviewer accepts or rejects, for up to five rounds. "
                                       "Each role adds calls, and on open-book tasks every call re-reads the documents.", size=16, color=NAVY)])
    footer(s, 2)

    # 3 flows
    s = new_slide(prs, "flows")
    header(s, "Motivation", "It rescues some answers and loses others")
    image(s, "fig_flows.png", 128, 360, 1000, 388)
    text(s, 128, 760, 1000, 40, [body("Team vs. single agent L₁ (% of tasks). M, F, Q, E: MATH, FRAMES, MuSiQue, BBEH.", size=12, color=MUTED)])
    text(s, 1184, 340, 608, 560, [
        body(f"[[c={GREEN}|Rescues]] 1.2–14.9% of tasks that L₁ gets wrong.", after=14),
        body(f"[[c={RED}|Loses]] up to 9.3% that L₁ gets right.", after=14),
        body("**84%** of those losses are wrong in the first draft, and the reviewer does not correct them.", after=14),
        body("Rerun, the team again loses **66 of 142** tasks it had lost against L₀.")])
    footer(s, 3)

    # 4 signals
    s = new_slide(prs, "signals")
    header(s, "Motivation", "Cheap rungs often suffice, but confidence is weak")
    image(s, "fig1abc.png", 285, 290, 1350, 549)
    text(s, 128, 860, 1664, 80, [body("(b) Many tasks the team solves are already solved by L₀ or L₁. "
                                      "(c) L₀’s log-probability AUROC ranges 0.22–0.81 and is inverted on three Gemma settings.", size=14)])
    footer(s, 4)

    # 5 uncert
    s = new_slide(prs, "uncert")
    header(s, "Motivation", "Uncertified thresholds can lose a lot")
    image(s, "fig1d.png", 128, 310, 520, 618)
    text(s, 728, 360, 1064, 560, [
        big("up to 16 points", RED, 48),
        body("lost by a fixed confidence threshold that spends as much as CLIMB on average. "
             "It loses over 2 points on 7 of 12 settings.", size=16, after=14),
        body("Always stopping at the single agent loses up to 9.4.", size=16, after=14),
        body("**Without a bound on the accuracy given up, the savings are hard to trust.**", size=16, color=NAVY)])
    footer(s, 5)

    # 6 goal
    s = new_slide(prs, "goal", dark=True)
    header(s, "The problem", "Minimise cost, subject to a certified bound on the accuracy lost", dark=True, size=36, h=160)
    box(s, 128, 400, 1664, 150, NAVY2, paras=[dict(t="min J(π)    s.t.    Pr( R(π̂) > ε ) ≤ δ", size=30, font=SERIF, color=LIGHT)],
        margin=56, anchor="m")
    defs = [(f"[[c={LIGHT}|J]]: expected tokens per task."),
            (f"[[c={LIGHT}|R]]: accuracy lost relative to always running the team."),
            (f"[[c={LIGHT}|ε]]: the user’s tolerance, e.g. 5 points. [[c={LIGHT}|δ]] = 10%.")]
    for i, d in enumerate(defs):
        text(s, 128 + i * 571, 610, 523, 120, [body(d, color=SOFT)])
    footer(s, 6, dark=True)

    # 7 ladder
    s = new_slide(prs, "ladder")
    header(s, "Method", "CLIMB: a ladder of cheaper configurations")
    image(s, "figure_ladder.png", 128, 300, 800, 620)
    text(s, 990, 340, 802, 580, [
        body(f"[[c={BLUE}|L₀, direct answer]] at the cheapest reasoning setting, greedy.", after=14),
        body(f"[[c={BLUE}|L₁, single agent]] at the team’s own setting, from scratch, without L₀’s answer.", after=14),
        body(f"[[c={ORANGE}|Team]]: planner, executor, reviewer, up to 5 rounds. The reference.", after=14),
        body("Scores: S₀ = L₀’s mean token log-probability. S₁ = 10·1[a₀ = a₁] + s̄₁, so agreeing tasks rank first.")])
    footer(s, 7)

    # 8 signed
    s = new_slide(prs, "signed")
    header(s, "Method", "A signed loss: skipping the team can keep answers")
    text(s, 128, 300, 1664, 70, [dict(t="ℓ  =  Y(team)  −  Y(answer kept)  ∈  {−1, 0, +1}", size=24, font=SERIF, color=NAVY)])
    cards = [("+1", RED, "**Stopping loses**: the cheap answer is wrong, the team’s is right."),
             ("0", "5B6170", "**No change**: both right or both wrong. Most stopped tasks."),
             ("−1", GREEN, "**Stopping saves**: the cheap answer is right, the team’s is wrong.")]
    for i, (n, c, d) in enumerate(cards):
        x = 128 + i * 565
        box(s, x, 410, 533, 290, PAPER, RULE, [big(n, c), body(d)])
        box(s, x, 410, 533, 10, c, radius=False)
    text(s, 128, 750, 1664, 120, [body("Clipping the loss at zero ignores the answers stopping saves. Since most losses are zero, "
                                       "a test that adapts to the variance can certify much more.", color=NAVY)])
    footer(s, 8)

    # 9 threshold
    s = new_slide(prs, "threshold")
    header(s, "Method · structure", "When is a confidence threshold optimal?")
    box(s, 128, 320, 812, 400, NAVY, paras=[
        h3("(M) Monotone gain", ORANGE_D, 18),
        dict(t="g(s) = E[ Y(team) − Y(cheap) | s ]", size=18, font=SERIF, color=LIGHT, after=10),
        body("g(s) and g(s)/c(s) are non-increasing in the score s. Escalation buys less, per token too, "
             "the more confident the cheap answer is.", color=SOFT)], margin=48)
    box(s, 980, 320, 812, 185, PAPER, RULE, [h3("Threshold optimality"),
        body("Under (M), the constrained problem is a fractional knapsack solved by a threshold on the score.", size=14)], margin=32)
    box(s, 980, 535, 812, 185, PAPER, RULE, [h3("Ordered testing is lossless"),
        body("The safe thresholds form a prefix of the conservative-to-aggressive sequence, and cost falls along it.", size=14)], margin=32)
    text(s, 128, 770, 1664, 80, [body("**(M) shapes the savings, not the guarantee.** The risk bound holds whether or not it does.", color=NAVY)])
    footer(s, 9)

    # 10 certify
    s = new_slide(prs, "certify")
    header(s, "Method · certification", "Test thresholds in order with a betting test")
    steps = [("1", "Candidates", "Thresholds t₁ > t₂ > …, fixed before any label, from conservative to aggressive. "
                                 "Each gives a null: its mean loss exceeds ε."),
             ("2", "Betting test", "Bet that the mean loss is below ε. Wealth is a supermartingale under the null, so p = 1 / max W is valid."),
             ("3", "Fixed sequence", "Stop at the first threshold not rejected. Deploy the cheapest rejected one. No multiplicity correction needed.")]
    for i, (n, t, d) in enumerate(steps):
        box(s, 128 + i * 565, 320, 533, 390, PAPER, RULE, [big(n, size=32), h3(t), body(d, size=14)])
    box(s, 128, 750, 1664, 110, BAND, paras=[dict(t="Wⱼ = ∏ ( 1 + λᵢ (ε − ℓᵢ) ) over i ≤ j,   λᵢ set from the running variance of earlier losses",
                                                   size=20, font=SERIF, color=NAVY)], margin=48, anchor="m")
    footer(s, 10)

    # 11 depth
    s = new_slide(prs, "depth")
    header(s, "Method · deployment", "Choose the depth, fall back to the team")
    left = [("Depth 0, tested at δ/2", "Threshold on S₀: stop at L₀ when it is confident."),
            ("Depth 1, tested at δ/2", "Threshold on S₁ after both rungs: agreement, then L₁’s confidence.")]
    right = [("Deploy the cheaper certified policy", "A union bound keeps both certificates valid at once with probability ≥ 1 − δ."),
             ("Nothing certified? Run the team", "Always running the team has zero risk, so it is always admissible.")]
    for i, (t, d) in enumerate(left):
        box(s, 128, 320 + i * 200, 812, 176, PAPER, RULE, [h3(t), body(d, size=14)], margin=32)
        box(s, 128, 320 + i * 200, 10, 176, BLUE, radius=False)
    for i, (t, d) in enumerate(right):
        box(s, 980, 320 + i * 200, 812, 176, NAVY, paras=[h3(t, ORANGE_D), body(d, size=14, color=SOFT)], margin=32)
    text(s, 128, 760, 1664, 120, [body("**If tasks and outputs are i.i.d., the deployed policy loses at most ε with probability ≥ 1 − δ.** "
                                       "A weak or misleading signal tends to cost savings rather than accuracy.", color=NAVY)])
    footer(s, 11)

    # 12 setup
    s = new_slide(prs, "setup")
    header(s, "Experiments", "Three open models, four benchmarks, twelve settings")
    cols = [("Models", ["gpt-oss-20b", "gpt-oss-120b", "Gemma-4-26B-A4B-it"],
             "L₀ at the cheapest reasoning setting. L₁ and the team at the default."),
            ("Benchmarks", ["MATH-500 (400)", "FRAMES (660, open-book)", "MuSiQue (1,934, open-book)", "BBEH (874)"],
             "Cost counts all prompt, completion and reasoning tokens."),
            ("Replay protocol", ["Record every rung and the team on every task. Draw 200 random 50/50 calibration and evaluation splits."],
             "Violation rate: share of splits where the deployed policy loses more than ε on the full pool.")]
    for i, (t, lines, foot) in enumerate(cols):
        paras = [h3(t)] + [body(l, size=14, after=2) for l in lines]
        paras[-1]["after"] = 14
        paras.append(body(foot, size=13, color="5B6170"))
        box(s, 128 + i * 565, 320, 533, 430, PAPER, RULE, paras)
    footer(s, 12)

    # 13 holds
    s = new_slide(prs, "holds")
    header(s, "Results", "The guarantee holds in practice")
    cards = [("≤ 3.4%", "failure rate over 5,000 i.i.d. calibration draws per setting and tolerance, against δ = 10%"),
             ("≤ 2%", "violation rate in the replay at ε = 2, 5 and 10%, and 3% anywhere in the tolerance sweep"),
             ("12 / 12", "settings where the mean loss stays within ε: at most 2.8 points at ε = 5%, 7.2 at 10%")]
    for i, (n, d) in enumerate(cards):
        box(s, 128 + i * 565, 340, 533, 380, PAPER, RULE, [big(n, GREEN, 48), body(d)])
    footer(s, 13)

    # 14 savings
    s = new_slide(prs, "savings")
    header(s, "Results", "The tolerance sets the price of the savings")
    image(s, "tradeoff_a.png", 128, 310, 990, 600)
    blocks = [("ε = 2%", "97% of tokens", NAVY), ("ε = 5%, range 31–96%", "62% · saves 38%", ORANGE),
              ("ε = 10%, range 18–62%", "35% · saves 65%", ORANGE)]
    for i, (a, b, c) in enumerate(blocks):
        text(s, 1180, 340 + i * 190, 612, 160, [body(a, size=14, color="5B6170", after=4), big(b, c, 28)])
        if i < 2:
            box(s, 1180, 500 + i * 190, 612, 2, RULE, radius=False)
    footer(s, 14)

    # 15 table
    s = new_slide(prs, "table")
    header(s, "Results", "Per-setting results")
    rows = [["Model", "Benchmark", "Team (acc / tokens)", "ε = 5%", "ε = 10%", "Viol."],
            ["gpt-oss-20b", "MATH", "94.8 / 6.1k", "92.9 / 62%", "89.6 / 33%", "0.0%"],
            ["", "FRAMES", "76.1 / 93.9k", "73.6 / 66%", "70.8 / 33%", "1.0%"],
            ["", "MuSiQue", "64.9 / 15.0k", "63.0 / 44%", "58.1 / 31%", "1.0%"],
            ["", "BBEH", "28.0 / 59.7k", "26.2 / 63%", "24.7 / 22%", "2.0%"],
            ["gpt-oss-120b", "MATH", "97.0 / 5.4k", "95.3 / 41%", "92.7 / 18%", "0.0%"],
            ["", "FRAMES", "79.8 / 84.8k", "78.4 / 63%", "75.1 / 32%", "1.0%"],
            ["", "MuSiQue", "71.1 / 12.4k", "70.0 / 46%", "64.0 / 29%", "2.0%"],
            ["", "BBEH", "45.1 / 53.3k", "42.8 / 75%", "39.5 / 40%", "0.5%"],
            ["Gemma-4-26B-A4B-it", "MATH", "99.8 / 10.4k", "98.1 / 31%", "98.0 / 20%", "0.0%"],
            ["", "FRAMES", "67.9 / 174.6k", "66.4 / 86%", "62.8 / 62%", "1.0%"],
            ["", "MuSiQue", "65.3 / 50.2k", "62.7 / 69%", "61.3 / 33%", "0.0%"],
            ["", "BBEH", "36.4 / 120.2k", "33.5 / 96%", "29.2 / 62%", "1.5%"]]
    table(s, 128, 270, 1664, rows, [0.22, 0.14, 0.19, 0.17, 0.17, 0.11], size=13, row_h=50, right_from=2)
    text(s, 128, 940, 1664, 40, [body("Accuracy / tokens as a share of the team’s, means over 200 splits. Viol.: largest violation rate over ε = 2, 5, 10%.",
                                      size=12, color=MUTED)])
    footer(s, 15)

    # 16 baselines
    s = new_slide(prs, "baselines")
    header(s, "Results", "Uncertified rules vs. CLIMB")
    rows = [["Policy", "Tokens", "Mean loss", "Max loss"],
            ["Always L₀", "15%", "8.7", "20.6"],
            ["Always L₁", "43%", "3.3", "9.4"],
            ["Threshold, same cost", "61%", "5.0", "15.9"],
            ["CLIMB, ε = 5% (certified)", "62%", "1.9", "2.8"],
            ["CLIMB, ε = 10% (certified)", "35%", "5.0", "7.2"],
            ["Oracle (not deployable)", "65%", "−7.2", "0.0"]]
    table(s, 128, 320, 1000, rows, [0.46, 0.18, 0.18, 0.18], size=15, row_h=66, highlight=(4, 5))
    text(s, 128, 800, 1000, 40, [body("Means over the twelve settings. Loss in accuracy points against the team.", size=12, color=MUTED)])
    box(s, 1176, 320, 616, 520, NAVY, paras=[
        h3("Tuned per setting, without certification", ORANGE_D),
        big("31%", LIGHT),
        body("of splits exceed ε = 5% on average, above δ on 9 of 12 settings. CLIMB: at most 2%.", size=14, color=SOFT, after=12),
        body("The extra tokens CLIMB spends are largely the price of the guarantee.", size=14, color=SOFT)], margin=44)
    footer(s, 16)

    # 17 tests
    s = new_slide(prs, "tests")
    header(s, "Results · ablation", "The choice of test matters as much as the signal")
    rows = [["Test (ε = 5%)", "20b", "120b", "Gemma", "Best"],
            ["Hoeffding–Bentkus", "100", "100", "100", "0/12"],
            ["Betting, clipped", "82", "75", "80", "0/12"],
            ["Binomial, clipped", "79", "73", "77", "3/12"],
            ["Betting, Bonferroni", "92", "90", "97", "0/12"],
            ["Betting, signed (ours)", "59", "56", "71", "9/12"]]
    table(s, 128, 320, 1000, rows, [0.40, 0.14, 0.14, 0.16, 0.16], size=15, row_h=70, highlight=(5,))
    text(s, 128, 760, 1000, 40, [body("Tokens certified, % of the team’s. Best: settings where the test is cheapest.", size=12, color=MUTED)])
    box(s, 1176, 320, 616, 210, PAPER, RULE, [h3("Adapt to the variance"),
        body("Most losses are zero. A worst-variance bound certifies nothing cheaper than the team.", size=14)], margin=32)
    box(s, 1176, 560, 616, 210, PAPER, RULE, [h3("Keep the sign"),
        body("Clipping certifies much less where the team often loses answers: 83 vs. 44% on 20b MuSiQue.", size=14)], margin=32)
    footer(s, 17)

    # 18 limits
    s = new_slide(prs, "limits")
    header(s, "Results · analysis", "What limits the savings, never the guarantee")
    cols = [("The signal", ["Where L₀’s confidence ranks well, the gain falls from 21.6 points in the least confident tenth to 4–7. "
                            "Where it is inverted, CLIMB relies on agreement or the team.", "Even a random score keeps the guarantee."]),
            ("Calibration data", ["50 tasks certify almost nothing. Savings at ε = 5% need a few hundred.",
                                  "Recording them is typically repaid within 340–2,000 queries at ε = 10%."]),
            ("Headroom", ["An oracle stopping at the cheapest correct rung beats the team by 7.2 points on average.",
                          "Much of the headroom lies in better signals, which plug in directly."])]
    for i, (t, ps) in enumerate(cols):
        x = 128 + i * 565
        box(s, x, 320, 533, 420, PAPER, RULE, [h3(t, size=17)] + [body(p, size=14, after=12) for p in ps])
        box(s, x, 320, 533, 10, ORANGE, radius=False)
    footer(s, 18)

    # 19 close
    s = new_slide(prs, "close", dark=True)
    header(s, "Takeaways", "Call the team only when a cheaper answer is not certified", dark=True, size=32, h=120)
    cols = [("Teams are not uniformly better", "They cost several times a single agent and sometimes lose its correct answers."),
            ("Signed loss, adaptive test", "A betting test in fixed sequence certifies the cheapest safe policy, with the team as fallback."),
            ("Savings within the bound", "38% of tokens saved at 5 points, 65% at 10, with failure rates at most 3.4%.")]
    for i, (t, d) in enumerate(cols):
        x = 128 + i * 565
        box(s, x, 380, 533, 4, ORANGE_D, radius=False)
        text(s, x, 400, 533, 200, [h3(t, LIGHT), body(d, size=14, color=SOFT)])
    text(s, 128, 640, 1664, 110, [body("Scope: the guarantee assumes calibration and deployment share one distribution, so thresholds are best "
                                       "calibrated on the target tasks. Recalibration takes under a second once records exist.", size=14, color=SOFT)])
    text(s, 128, 800, 1664, 80, [dict(t="Thank you. Questions?", size=28, bold=True, font=SERIF, color=ORANGE_D)])

    out = HERE / "CLIMB_talk.pptx"
    prs.save(out)
    print(out, len(prs.slides))


if __name__ == "__main__":
    build()
