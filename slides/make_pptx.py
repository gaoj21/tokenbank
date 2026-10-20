"""Build slides/CLIMB_talk.pptx: a 20-slide talk, one idea per slide, charts from make_figs.py.

Run with the project venv: .venv/bin/python slides/make_figs.py && .venv/bin/python slides/make_pptx.py
"""
import re
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Pt

HERE = Path(__file__).parent
IMG = HERE / "img"

NAVY, NAVY2, LIGHT, PAPER, BAND = "14213D", "1F2F52", "F7F6F2", "FDFCF9", "ECEAE3"
BODY, MUTED, RULE, SOFT = "3A4150", "6B7280", "DDD8CC", "C9D1E0"
ORANGE, ORANGE_D, BLUE, RED, GREEN = "B4541A", "F0A66B", "2E5C99", "B03A2E", "2F6B34"
REDBG, GREENBG, BLUEBG, ORANGEBG = "F6E3E0", "E3ECE2", "E2EAF5", "F8E8DC"
SERIF, SANS = "Georgia", "Arial"
TEAM_ACC = [[94.8, 76.1, 64.9, 28.0], [97.0, 79.8, 71.1, 45.1], [99.8, 67.9, 65.3, 36.4]]


def px(v):
    return Emu(int(v * 6350))  # 1920 px = 13.333 in


def rgb(h):
    return RGBColor.from_string(h)


def runs_of(text):
    """'plain **bold** [[c=HEX|colored bold]]' -> [(text, bold, color)]."""
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
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[p.get("align", "l")]
        para.line_spacing = p.get("ls", 1.15)
        para.space_after = Pt(p.get("after", 0))
        for t, bold, color in runs_of(p["t"]):
            r = para.add_run()
            r.text = t
            f = r.font
            f.size = Pt(p.get("size", 18))
            f.name = p.get("font", SANS)
            f.bold = bold or p.get("bold", False)
            f.color.rgb = rgb(color or p.get("color", BODY))


def text(s, x, y, w, h, paras, anchor="t"):
    tb = s.shapes.add_textbox(px(x), px(y), px(w), px(h))
    fill_tf(tb.text_frame, paras)
    tb.text_frame.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]
    return tb


def box(s, x, y, w, h, fill, line=None, paras=None, margin=32, shape=MSO_SHAPE.ROUNDED_RECTANGLE, anchor="m", lw=1.0):
    shp = s.shapes.add_shape(shape, px(x), px(y), px(w), px(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        shp.adjustments[0] = min(0.1, 18 / min(w, h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = rgb(fill)
    if line:
        shp.line.color.rgb = rgb(line)
        shp.line.width = Pt(lw)
    else:
        shp.line.fill.background()
    shp.shadow.inherit = False
    if paras:
        fill_tf(shp.text_frame, paras, margin)
        shp.text_frame.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE}[anchor]
    return shp


def arrow(s, x, y, w, h, color=MUTED, shape=MSO_SHAPE.RIGHT_ARROW):
    return box(s, x, y, w, h, color, shape=shape)


def picture(s, name, x, y, w, h):
    """Fit the image inside the box, centred, keeping its aspect ratio."""
    iw, ih = Image.open(IMG / name).size
    k = min(w / iw, h / ih)
    pw, ph = iw * k, ih * k
    return s.shapes.add_picture(str(IMG / name), px(x + (w - pw) / 2), px(y + (h - ph) / 2), px(pw), px(ph))


def new_slide(prs, notes, dark=False):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb(NAVY if dark else LIGHT)
    s.notes_slide.notes_text_frame.text = notes
    return s


def header(s, eyebrow, title, dark=False, size=30):
    text(s, 128, 100, 1664, 40, [dict(t=eyebrow.upper(), size=13, bold=True, color=ORANGE_D if dark else ORANGE)])
    text(s, 128, 140, 1664, 90, [dict(t=title, size=size, bold=True, font=SERIF, color=LIGHT if dark else NAVY, ls=1.05)])


def footer(s, n, dark=False):
    text(s, 128, 1010, 1664, 34, [dict(t=str(n), size=12, align="r", color="8B97AE" if dark else MUTED)])


def big(t, color=ORANGE, size=54):
    return dict(t=t, size=size, bold=True, font=SERIF, color=color, after=6, ls=1.0)


def say(t, size=20, color=BODY, after=0, **kw):
    return dict(t=t, size=size, color=color, after=after, **kw)


def callout(s, x, y, w, h, number, label, color=ORANGE, fill=PAPER):
    box(s, x, y, w, h, fill, RULE, [big(number, color, 36), say(label, 17)], margin=30)


def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = px(1920), px(1080)
    n = 0

    def slide(notes, dark=False):
        nonlocal n
        n += 1
        s = new_slide(prs, notes, dark)
        if n > 1:
            footer(s, n, dark)
        return s

    # 1 cover
    s = slide("Teams of LLM agents can be more accurate than one model call, but they cost several times as much. "
              "This talk is about when the team is worth calling, and how to skip it safely.", dark=True)
    text(s, 128, 300, 1664, 40, [dict(t="CLIMB", size=16, bold=True, color=ORANGE_D)])
    text(s, 128, 350, 1664, 130, [dict(t="When to Call the Team", size=56, bold=True, font=SERIF, color=LIGHT)])
    text(s, 128, 495, 1664, 60, [say("Save tokens on multi-agent LLM pipelines, with a guarantee on accuracy", 24, SOFT)])
    box(s, 128, 600, 240, 5, ORANGE_D, shape=MSO_SHAPE.RECTANGLE)
    text(s, 128, 640, 1664, 50, [say("Jianjun Gao, Jinxin Liu, Dongyun Lin, Hongyun Cai, Zheng Ma", 18, SOFT)])

    # team: what it is
    s = slide("First, what is an agent team? Instead of asking one model once, we split the work into roles, all played by "
              "the same model. A planner writes a step-by-step plan. An executor solves the task following the plan. "
              "A reviewer checks the answer. If the reviewer rejects it, the executor revises, for up to five rounds. "
              "This plan, execute, review pattern is common in multi-agent frameworks.")
    header(s, "Background", "What is an agent team?")
    text(s, 128, 270, 600, 50, [dict(t="Single agent", size=22, bold=True, color=BLUE)])
    box(s, 128, 330, 300, 130, PAPER, RULE, [dict(t="Question", size=22, bold=True, color=NAVY, align="c")])
    arrow(s, 445, 370, 90, 50)
    box(s, 550, 330, 300, 130, BLUEBG, BLUE, [dict(t="Model", size=22, bold=True, color=NAVY, align="c"),
                                              dict(t="thinks once", size=16, color=BLUE, align="c")], lw=2)
    arrow(s, 867, 370, 90, 50)
    box(s, 972, 330, 300, 130, PAPER, RULE, [dict(t="Answer", size=22, bold=True, color=NAVY, align="c")])
    text(s, 128, 575, 800, 50, [dict(t="Agent team: plan, execute, review", size=22, bold=True, color=ORANGE)])
    roles = [("Question", "", PAPER, RULE), ("Planner", "writes a plan", ORANGEBG, ORANGE),
             ("Executor", "follows the plan", ORANGEBG, ORANGE), ("Reviewer", "accept or reject", ORANGEBG, ORANGE),
             ("Answer", "", PAPER, RULE)]
    for i, (t, d, f, c) in enumerate(roles):
        x = 128 + i * 345
        paras = [dict(t=t, size=21, bold=True, color=NAVY, align="c", after=4)] + ([dict(t=d, size=15, color=c, align="c")] if d else [])
        box(s, x, 640, 280, 170, f, c, paras, margin=8, lw=2)
        if i < 4:
            arrow(s, x + 286, 700, 52, 50)
    arrow(s, 818, 840, 400, 50, ORANGE, shape=MSO_SHAPE.LEFT_ARROW)
    text(s, 700, 900, 640, 50, [dict(t="rejected? revise, up to 5 rounds", size=18, bold=True, color=ORANGE, align="c")])

    # team: why it helps
    s = slide("Why use a team? It is more accurate. Grey is a direct answer, blue a single agent that reasons, orange the team. "
              "On every one of our twelve settings the team is the most accurate: 2 to 21 points above the direct answer, "
              "and up to 9 points above the single agent. The gains are largest on hard reasoning, BBEH.")
    header(s, "Background", "Teams are more accurate")
    picture(s, "c_accuracy.png", 128, 260, 1150, 700)
    callout(s, 1330, 330, 462, 260, "+2 to +21", "points over a direct answer", ORANGE, ORANGEBG)
    callout(s, 1330, 630, 462, 260, "12 / 12", "settings where the team is best, up to 9 points over a single agent", ORANGE, ORANGEBG)

    # the catch: cost and harm
    s = slide("But the accuracy comes at a price, in two ways. Left: the team uses 1.3 to 6.3 times the tokens of a single agent, "
              "because planner, executor and reviewer each add calls, for up to five rounds. Right: against the single agent, "
              "green bars are wrong answers the team fixes, red bars are right answers it breaks, up to 9 percent of tasks. "
              "So the team is better on balance, but it is expensive and not uniformly better.")
    header(s, "But…", "But teams are costly, and sometimes harmful")
    for x, img, head, col in ((128, "c_cost_n.png", "Cost: up to [[c=B4541A|6.3×]] a single agent", ORANGE),
                              (992, "c_flows_n.png", "Harm: up to [[c=B03A2E|9%]] of tasks broken", RED)):
        text(s, x, 270, 800, 50, [dict(t=head, size=24, bold=True, color=NAVY)])
        picture(s, img, x, 340, 800, 620)

    # 4 why the team breaks answers
    s = slide("Where do the broken answers come from? In 84 percent of them the executor's first draft is already wrong, "
              "and the reviewer accepts it. Reviewers accept 23 to 58 percent of wrong answers. "
              "This is not just noise: rerunning the team reproduces about half of these losses.")
    header(s, "The problem", "Why does the team break right answers?")
    steps = [("Planner", "writes a plan", BLUEBG, BLUE), ("Executor", "first draft is WRONG", REDBG, RED),
             ("Reviewer", "“Looks good” ACCEPTED", BLUEBG, BLUE), ("Final answer", "WRONG", REDBG, RED)]
    for i, (t, d, f, c) in enumerate(steps):
        x = 128 + i * 430
        box(s, x, 330, 340, 220, f, c, [dict(t=t, size=22, bold=True, color=NAVY, align="c", after=8),
                                        dict(t=d, size=17, bold=True, color=c, align="c")], margin=16, lw=2.5)
        if i < 3:
            arrow(s, x + 352, 415, 66, 50)
    callout(s, 128, 640, 800, 250, "84%", "of broken answers start as a wrong first draft the reviewer did not catch", RED)
    callout(s, 992, 640, 800, 250, "23–58%", "of wrong answers are accepted by the reviewer", RED)

    # the idea: a ladder (lead-in)
    s = slide("Since the team is expensive and not always better, the idea is a ladder. Try the cheapest configuration first, "
              "a direct answer. If it looks right, stop. Otherwise try a single agent. Only if that also looks unreliable, "
              "call the full team. All three use the same model. The ladder itself is simple. The real question, "
              "and the rest of the method, is when a task should climb.")
    header(s, "The idea", "Climb a ladder: call the team only when needed")
    box(s, 128, 255, 1060, 80, NAVY, paras=[dict(t="The key question: when should a task climb?", size=22, bold=True, color=ORANGE_D)],
        margin=10, anchor="m")
    rungs = [("Direct answer", "cheapest", 250, BLUEBG, BLUE), ("Single agent", "medium", 390, BLUEBG, BLUE),
             ("Agent team", "most expensive", 530, ORANGEBG, ORANGE)]
    base = 900
    for i, (t, c, h, f, col) in enumerate(rungs):
        x = 160 + i * 560
        box(s, x, base - h, 400, h, f, col, [dict(t=t, size=24, bold=True, color=NAVY, align="c", after=6),
                                            dict(t=c, size=18, color=col, align="c", bold=True)], margin=12, lw=2.5,
            shape=MSO_SHAPE.RECTANGLE)
        if i < 2:
            text(s, x + 400, base - h - 10, 160, 60, [dict(t="not sure?\ngo up", size=15, bold=True, color=MUTED, align="c")])
            arrow(s, x + 415, base - h + 70, 130, 44, MUTED)
    for i in range(2):
        x = 160 + i * 560
        text(s, x, base + 20, 400, 50, [dict(t="sure? stop and answer", size=17, bold=True, color=GREEN, align="c")])
    text(s, 1280, base + 20, 400, 50, [dict(t="always answers", size=17, bold=True, color=ORANGE, align="c")])

    # roadmap: how to climb
    s = slide("Deciding when to climb breaks into four questions, and the rest of the method answers them in order. "
              "First, a signal: does the cheap answer look right? Second, a rule: where do we draw the line on that signal? "
              "Third, a check: how do we know the line is safe? Fourth, how far up the ladder to look before deciding. "
              "And if no rule passes the check, we simply always call the team.", dark=True)
    header(s, "How to climb", "Deciding when to climb: four questions", dark=True, size=34)
    qs = [("1", "Signal", "Does the cheap answer look right?", "confidence and agreement"),
          ("2", "Rule", "Where do we draw the line?", "a threshold on the score"),
          ("3", "Check", "Is that line safe?", "a test with a guarantee"),
          ("4", "Depth", "Look one rung up or two?", "let the data choose")]
    for i, (k, t, q, a) in enumerate(qs):
        x = 128 + i * 426
        box(s, x, 290, 386, 540, NAVY2, paras=[dict(t=k, size=36, bold=True, font=SERIF, color=ORANGE_D, after=4),
                                               dict(t=t, size=24, bold=True, color=LIGHT, after=12),
                                               dict(t=q, size=18, color=SOFT, after=16),
                                               dict(t="→ " + a, size=18, bold=True, color=ORANGE_D)], margin=28, anchor="t")
    text(s, 128, 870, 1664, 60, [say("No rule passes the check?  →  always call the team.  Worst case: no savings, never lost accuracy.", 20, SOFT)])

    # 1 signal: confidence
    s = slide("Question one: what tells us a cheap answer is right? The first signal is free: the model's own confidence, "
              "its average token probability. Above the dashed line, confident answers are more often right. "
              "On three Gemma settings it points the wrong way: the model is confidently wrong. "
              "So any rule built on this signal must be checked, not trusted.")
    header(s, "How to climb · 1 signal", "Is the direct answer confident?")
    picture(s, "c_auroc.png", 128, 260, 1150, 700)
    callout(s, 1330, 330, 462, 260, "usually helps", "confident answers tend to be right", BLUE, BLUEBG)
    callout(s, 1330, 630, 462, 260, "3 of 12", "settings where it points the wrong way", RED, REDBG)

    # 1 signal: agreement
    s = slide("The second signal is also nearly free. The direct answer and the single agent answer independently. "
              "When they agree, the single agent is much more often right, for example 99 against 77 percent "
              "on gpt-oss-20b MATH. CLIMB combines the two into one score: agreeing tasks rank first, "
              "then by confidence.")
    header(s, "How to climb · 1 signal", "Do two cheap answers agree?")
    picture(s, "c_agree.png", 128, 260, 1150, 700)
    callout(s, 1330, 330, 462, 260, "agree", "→ the single agent is usually right", GREEN, GREENBG)
    box(s, 1330, 630, 462, 260, NAVY, paras=[dict(t="One score", size=26, bold=True, color=ORANGE_D, after=8),
                                             say("agreeing tasks first, then by confidence", 17, LIGHT)], margin=30)

    # 2 rule: guessing fails
    s = slide("Question two: where do we draw the line? A rule stops the task when its score is above a threshold. "
              "The natural shortcut is to pick the threshold by hand. Such rules can look fine on average "
              "but lose a lot on some settings. A fixed confidence threshold with the same average cost as CLIMB "
              "loses up to 16 points. Always using the single agent loses up to 9. CLIMB at a 5 point tolerance loses at most 2.8. "
              "And nothing at deployment tells you which setting is the bad one.")
    header(s, "How to climb · 2 rule", "Where to draw the line? Guessing can go badly")
    picture(s, "c_maxloss.png", 128, 260, 1150, 700)
    callout(s, 1330, 360, 462, 300, "16 points", "lost by a hand-picked threshold on one setting", RED, REDBG)
    text(s, 1330, 700, 462, 200, [say("Nothing at deployment says which setting will go wrong.", 18, MUTED)])

    # 3 check: the promise
    s = slide("Question three: how do we know a line is safe? CLIMB turns this into a promise. You choose how much accuracy "
              "you can give up, say 5 points. CLIMB checks candidate rules on a few hundred labelled tasks and returns the "
              "cheapest rule that keeps the promise, with 90 percent probability. If no cheaper rule qualifies, it always "
              "calls the team, so the worst case is no savings, not lost accuracy.")
    header(s, "How to climb · 3 check", "Replace guessing with a promise")
    items = [("You choose", "“I can lose at most 5 points of accuracy”", ORANGEBG, ORANGE),
             ("CLIMB checks", "candidate rules on a few hundred labelled tasks", NAVY, None),
             ("You get", "the cheapest rule that keeps the promise, 9 times out of 10", GREENBG, GREEN)]
    for i, (t, d, f, c) in enumerate(items):
        x = 128 + i * 590
        dark = f == NAVY
        box(s, x, 330, 490, 320, f, c, [dict(t=t, size=24, bold=True, color=ORANGE_D if dark else c, align="c", after=12),
                                        dict(t=d, size=19, color=LIGHT if dark else NAVY, align="c")], margin=24, lw=2.5)
        if i < 2:
            arrow(s, x + 502, 450, 76, 60)
    box(s, 128, 720, 1664, 150, PAPER, RULE, [dict(t="No cheaper rule passes?  →  call the team every time.  Worst case: no savings, not lost accuracy.",
                                                    size=20, bold=True, color=NAVY, align="c")])

    # 3 check: sweep
    s = slide("Here is how the check runs, on real data: gpt-oss-120b on MuSiQue. Each dot is one candidate threshold. "
              "On the left, almost everything goes to the team: safe but expensive. Moving right, more tasks stop early: "
              "cheaper but riskier. CLIMB tests them in this order and stops at the first one that fails. "
              "It deploys the cheapest one that passed, the star, at 57 percent of the tokens. "
              "Some grey dots are still under the line: the test is cautious, it needs evidence.")
    header(s, "How to climb · 3 check", "Test from safe to risky, keep the cheapest that passes")
    picture(s, "c_sweep.png", 128, 250, 1664, 700)
    text(s, 128, 960, 1664, 40, [say("gpt-oss-120b on MuSiQue, thresholds on the direct answer’s confidence, one calibration half.", 15, MUTED)])

    # 3 check: betting
    s = slide("How does the test decide pass or fail for one threshold? Think of a gambler who bets that skipping the team "
              "is harmless. Start with one dollar. For each calibration task, if skipping did no harm, the gambler wins a little. "
              "If it broke an answer, the gambler loses. If the wealth ever reaches 20 times the start, the rule passes. "
              "A safe rule makes the gambler rich fast. A rule that loses 6 points makes the gambler broke. "
              "The bet size adapts to how often skipping changes anything, which makes the test efficient.")
    header(s, "How to climb · 3 check", "Each test is a betting game")
    picture(s, "c_wealth.png", 128, 250, 1100, 700)
    steps = [("1", "Start with $1."), ("2", "Each task: bet that skipping the team does no harm."),
             ("3", "No harm: win a little. Broken answer: lose."), ("4", "Reach $20: the rule passes.")]
    for i, (k, d) in enumerate(steps):
        y = 290 + i * 170
        box(s, 1290, y, 70, 70, ORANGE, paras=[dict(t=k, size=28, bold=True, color=LIGHT, align="c")], margin=0,
            shape=MSO_SHAPE.OVAL)
        text(s, 1385, y + 4, 407, 160, [say(d, 19)], anchor="t")

    # 3 check: signed loss
    s = slide("One detail makes the check much sharper. Skipping the team is not only a risk: sometimes the cheap answer "
              "is right and the team would have broken it. Usual tests count only the losses. CLIMB's bet counts both sides, "
              "so saved answers offset lost ones. On the right, every test is valid, but they certify very different savings: "
              "a worst-case bound certifies nothing, ignoring saved answers keeps 79 percent of tokens, CLIMB's test 62.")
    header(s, "How to climb · 3 check", "Count the answers skipping saves, too")
    cases = [("Cheap wrong, team right", "lose a point", REDBG, RED),
             ("Same result", "no change (most tasks)", BAND, MUTED),
             ("Cheap right, team wrong", "save a point", GREENBG, GREEN)]
    for i, (t, d, f, c) in enumerate(cases):
        box(s, 128, 290 + i * 190, 600, 160, f, c, [dict(t=t, size=22, bold=True, color=NAVY, align="c", after=6),
                                                     dict(t=d, size=20, bold=True, color=c, align="c")], margin=12, lw=2.5)
    picture(s, "c_tests.png", 790, 300, 1002, 520)
    text(s, 790, 840, 1002, 80, [say("Tokens each valid test can certify at ε = 5. Lower is better.", 17, MUTED)])

    # 4 depth
    s = slide("Question four: how far up the ladder should we look before deciding? Option A stops after the direct answer "
              "when it is confident. Option B also runs the single agent and stops when the two agree. Which is cheaper depends "
              "on the model and the task, so CLIMB checks both on the calibration data and deploys the cheaper one that passes, "
              "splitting the error budget between them. If neither passes, it always calls the team.")
    header(s, "How to climb · 4 depth", "One rung or two? The data decides")
    rows = [("Option A: one rung", "Direct answer", "confident? stop"),
            ("Option B: two rungs", "Direct answer + single agent", "agree and confident? stop")]
    for i, (lab, rung, cond) in enumerate(rows):
        y = 280 + i * 260
        text(s, 128, y, 900, 50, [dict(t=lab, size=22, bold=True, color=NAVY)])
        box(s, 128, y + 60, 420, 150, BLUEBG, BLUE, [dict(t=rung, size=21, bold=True, color=NAVY, align="c")], margin=14, lw=2)
        arrow(s, 560, y + 110, 70, 50)
        box(s, 645, y + 60, 420, 150, GREENBG, GREEN, [dict(t=cond, size=21, bold=True, color=GREEN, align="c")], margin=14, lw=2)
        text(s, 1085, y + 105, 250, 60, [dict(t="otherwise → team", size=19, bold=True, color=ORANGE)])
    box(s, 1380, 300, 412, 500, NAVY, paras=[dict(t="Check both", size=24, bold=True, color=ORANGE_D, after=12),
                                             say("on calibration data, then deploy the cheaper one that passes.", 18, LIGHT, after=12),
                                             say("Neither passes? Always call the team.", 18, SOFT)], margin=28)

    # 14 setup
    s = slide("We test on three open models and four benchmarks: math, two open-book question answering sets, "
              "and hard reasoning. We record every rung for every task once, then replay 200 random splits: "
              "half to calibrate CLIMB, half to evaluate it.")
    header(s, "Experiments", "3 models × 4 benchmarks = 12 settings")
    models = ["gpt-oss-20b", "gpt-oss-120b", "Gemma-4-26B"]
    benches = [("MATH", "math"), ("FRAMES", "open-book QA"), ("MuSiQue", "open-book QA"), ("BBEH", "hard reasoning")]
    for j, (b, d) in enumerate(benches):
        text(s, 480 + j * 320, 290, 300, 80, [dict(t=b, size=22, bold=True, color=NAVY, align="c"),
                                              dict(t=d, size=16, color=MUTED, align="c")])
    for i, m in enumerate(models):
        text(s, 128, 395 + i * 150, 320, 120, [dict(t=m, size=22, bold=True, color=NAVY)], anchor="m")
        for j, (b, _) in enumerate(benches):
            acc = TEAM_ACC[i][j]
            box(s, 480 + j * 320, 395 + i * 150, 300, 120, BLUEBG if i < 2 else ORANGEBG, RULE,
                [dict(t=f"{acc}%", size=24, bold=True, color=NAVY, align="c"), dict(t="team accuracy", size=13, color=MUTED, align="c")], margin=8)
    box(s, 128, 870, 1664, 100, NAVY, paras=[dict(t="Record every task once, then replay [[c=F0A66B|200 random splits]]: half to calibrate CLIMB, half to test it.",
                                             size=18, color=LIGHT, align="c")])

    # 15 promise holds
    s = slide("Does the promise hold? For each setting, the share of splits where CLIMB lost more than its tolerance. "
              "The guarantee allows up to 10 percent. We see at most 2 percent. A separate check with 5,000 random "
              "calibration draws per setting gives at most 3.4 percent.")
    header(s, "Results", "The promise holds on every setting")
    picture(s, "c_viol.png", 128, 260, 1150, 700)
    callout(s, 1330, 360, 462, 300, "≤ 2%", "of splits broke the promise. Allowed: 10%.", GREEN, GREENBG)

    # 16 savings
    s = slide("How much does it save? Each bar is the share of the team's tokens CLIMB uses. At a 5 point tolerance "
              "it saves 38 percent on average, at 10 points 65 percent. Savings are smallest where confidence points the "
              "wrong way, on Gemma, and the guarantee still holds there: CLIMB just calls the team more often.")
    header(s, "Results", "How much it saves")
    picture(s, "c_savings.png", 128, 260, 1150, 700)
    callout(s, 1330, 330, 462, 260, "38% saved", "allowing 5 points of loss", BLUE, BLUEBG)
    callout(s, 1330, 630, 462, 260, "65% saved", "allowing 10 points of loss", BLUE, BLUEBG)

    # 17 tuned without test
    s = slide("Why not just tune the threshold on the same data, without the test? It is cheaper on paper, "
              "but it breaks the promise: at 5 points it fails on 31 percent of splits on average, against at most "
              "2 percent for CLIMB. A rule tuned to the edge on one sample tends to cross it on the next. "
              "The extra tokens CLIMB spends are the price of the guarantee.")
    header(s, "Results", "Without the test, the promise breaks")
    picture(s, "c_tuned.png", 128, 260, 1150, 700)
    callout(s, 1330, 360, 462, 300, "31%", "of splits broken at ε = 5 by the tuned rule. CLIMB: 0.6%.", RED, REDBG)

    # 18 takeaways
    s = slide("To sum up. Agent teams are expensive and sometimes hurt. CLIMB calls them only when cheaper answers "
              "cannot be certified, keeps a user-chosen accuracy promise, and saves 38 to 65 percent of tokens. "
              "The guarantee assumes calibration tasks look like deployment tasks. Thank you.", dark=True)
    header(s, "Takeaways", "Call the team only when you must", dark=True, size=36)
    cols = [("Teams are costly", "and sometimes break right answers."),
            ("CLIMB keeps a promise", "you set the loss, it finds the cheapest safe rule."),
            ("38–65% saved", "at 5 and 10 points, with the promise kept.")]
    for i, (t, d) in enumerate(cols):
        x = 128 + i * 565
        box(s, x, 330, 533, 6, ORANGE_D, shape=MSO_SHAPE.RECTANGLE)
        text(s, x, 360, 533, 260, [dict(t=t, size=28, bold=True, font=SERIF, color=LIGHT, after=10),
                                   say(d, 20, SOFT)])
    text(s, 128, 740, 1664, 80, [say("Assumes calibration tasks resemble deployment tasks. Recalibrating takes under a second.", 18, SOFT)])
    text(s, 128, 850, 1664, 90, [dict(t="Thank you. Questions?", size=36, bold=True, font=SERIF, color=ORANGE_D)])

    out = HERE / "CLIMB_talk.pptx"
    prs.save(out)
    print(out, len(prs.slides))


if __name__ == "__main__":
    build()
