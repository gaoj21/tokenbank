"""Build slides/CLIMB_talk.pptx: an 18-slide talk, one idea per slide, charts from make_figs.py.

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

    # 2 cost
    s = slide("Our team is a standard plan, execute, review pipeline. Each bar is one model on one benchmark. "
              "The team uses 1.3 to 6.3 times the tokens of a single agent answering on its own.")
    header(s, "The problem", "Agent teams cost several times more")
    picture(s, "c_cost.png", 128, 260, 1150, 700)
    callout(s, 1330, 360, 462, 300, "up to 6.3×", "the tokens of one agent answering alone")
    text(s, 1330, 700, 462, 200, [say("Planner, executor and reviewer each add calls, for up to 5 rounds.", 18, MUTED)])

    # 3 flows
    s = slide("Green: tasks the single agent got wrong and the team fixed. Red: tasks the single agent got right "
              "and the team got wrong. The team wins on balance, but it also breaks answers, up to 9 percent of tasks.")
    header(s, "The problem", "…and they are not always better")
    picture(s, "c_flows.png", 128, 260, 1150, 700)
    callout(s, 1330, 330, 462, 260, "up to 15%", "of tasks: the team fixes a wrong answer", GREEN, GREENBG)
    callout(s, 1330, 630, 462, 260, "up to 9%", "of tasks: the team breaks a right answer", RED, REDBG)

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

    # 5 the idea: a ladder
    s = slide("The idea is a ladder. Try the cheapest configuration first: a direct answer. If it looks reliable, stop. "
              "Otherwise try a single agent. Only if that also looks unreliable, call the full team. "
              "All three use the same model. The question is how to decide 'looks reliable'.")
    header(s, "The idea", "Climb a ladder: call the team only when needed")
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

    # 6 signal 1: confidence
    s = slide("How do we know a cheap answer is reliable? The first signal is free: the model's own confidence, "
              "its average token probability. Above the dashed line, confident answers are more often right. "
              "On three Gemma settings it points the wrong way: the model is confidently wrong. "
              "So any rule built on this signal must be checked, not trusted.")
    header(s, "Signals", "Signal 1: how confident is the direct answer?")
    picture(s, "c_auroc.png", 128, 260, 1150, 700)
    callout(s, 1330, 330, 462, 260, "usually helps", "confident answers tend to be right", BLUE, BLUEBG)
    callout(s, 1330, 630, 462, 260, "3 of 12", "settings where it points the wrong way", RED, REDBG)

    # 7 signal 2: agreement
    s = slide("The second signal is also nearly free. The direct answer and the single agent answer independently. "
              "When they agree, the single agent is much more often right, for example 99 against 77 percent "
              "on gpt-oss-20b MATH. So CLIMB stops on agreeing tasks first.")
    header(s, "Signals", "Signal 2: do two cheap answers agree?")
    picture(s, "c_agree.png", 128, 260, 1150, 700)
    callout(s, 1330, 360, 462, 300, "agree", "→ the single agent is usually right", GREEN, GREENBG)
    text(s, 1330, 700, 462, 200, [say("Neither answer sees the other, so agreement costs nothing extra.", 18, MUTED)])

    # 8 guessing fails
    s = slide("A natural shortcut is to pick a threshold by hand or by intuition. These rules can look fine on average "
              "but lose a lot on some settings. A fixed confidence threshold with the same average cost as CLIMB "
              "loses up to 16 points. Always using the single agent loses up to 9. CLIMB at a 5 point tolerance loses at most 2.8. "
              "In deployment, nothing tells you which setting is the bad one.")
    header(s, "The problem", "Guessing a rule can go badly")
    picture(s, "c_maxloss.png", 128, 260, 1150, 700)
    callout(s, 1330, 360, 462, 300, "16 points", "lost by a hand-picked threshold on one setting", RED, REDBG)
    text(s, 1330, 700, 462, 200, [say("Nothing at deployment says which setting will go wrong.", 18, MUTED)])

    # 9 the promise
    s = slide("CLIMB turns this into a promise. You choose how much accuracy you can give up, say 5 points. "
              "CLIMB looks at a few hundred labelled tasks and returns the cheapest rule that keeps that promise, "
              "with 90 percent probability. If no cheaper rule qualifies, it simply calls the team every time, "
              "so the worst case is no savings, not lost accuracy.")
    header(s, "CLIMB", "What CLIMB promises")
    items = [("You choose", "“I can lose at most 5 points of accuracy”", ORANGEBG, ORANGE),
             ("CLIMB checks", "a few hundred labelled tasks", NAVY, None),
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

    # 10 sweep
    s = slide("Here is a real example: gpt-oss-120b on MuSiQue. Each dot is one possible threshold. On the left, "
              "almost everything goes to the team: safe but expensive. Moving right, more tasks stop early: cheaper "
              "but riskier. CLIMB tests them in this order and stops at the first one that fails. "
              "It deploys the cheapest one that passed, the star, at 57 percent of the tokens. "
              "Some grey dots are still under the line: the test is cautious, it needs evidence.")
    header(s, "How it works", "Keep the cheapest rule that passes")
    picture(s, "c_sweep.png", 128, 250, 1664, 700)
    text(s, 128, 960, 1664, 40, [say("gpt-oss-120b on MuSiQue, thresholds on the direct answer’s confidence, one calibration half.", 15, MUTED)])

    # 11 betting
    s = slide("How does the test decide pass or fail? Think of a gambler who bets that skipping the team is harmless. "
              "Start with one dollar. For each calibration task, if skipping did no harm, the gambler wins a little. "
              "If it broke an answer, the gambler loses. If the wealth ever reaches 20 times the start, the rule passes. "
              "A safe rule makes the gambler rich fast. A rule that loses 6 points makes the gambler broke. "
              "The bet size adapts to how often skipping changes anything, which makes the test efficient.")
    header(s, "How it works", "The test is a betting game")
    picture(s, "c_wealth.png", 128, 250, 1100, 700)
    steps = [("1", "Start with $1."), ("2", "Each task: bet that skipping the team does no harm."),
             ("3", "No harm: win a little. Broken answer: lose."), ("4", "Reach $20: the rule passes.")]
    for i, (k, d) in enumerate(steps):
        y = 290 + i * 170
        box(s, 1290, y, 70, 70, ORANGE, paras=[dict(t=k, size=28, bold=True, color=LIGHT, align="c")], margin=0,
            shape=MSO_SHAPE.OVAL)
        text(s, 1385, y + 4, 407, 160, [say(d, 19)], anchor="t")

    # 12 signed
    s = slide("One detail matters a lot. Skipping the team is not only a risk. Sometimes the cheap answer is right "
              "and the team would have broken it. Standard tests count only the losses. CLIMB counts both sides, "
              "so these saved answers offset the lost ones.")
    header(s, "How it works", "Skipping the team can also save answers")
    cases = [("Cheap answer wrong,\nteam right", "we lose a point", REDBG, RED),
             ("Both right,\nor both wrong", "no change (most tasks)", BAND, MUTED),
             ("Cheap answer right,\nteam wrong", "we save a point", GREENBG, GREEN)]
    for i, (t, d, f, c) in enumerate(cases):
        x = 128 + i * 565
        box(s, x, 330, 533, 360, f, c, [dict(t=t, size=26, bold=True, color=NAVY, align="c", after=20),
                                        dict(t=d, size=24, bold=True, color=c, align="c")], lw=2.5)
    box(s, 128, 760, 1664, 130, PAPER, RULE, [dict(t="Usual tests count only the left box. CLIMB counts both sides.",
                                                    size=24, bold=True, color=NAVY, align="c")])

    # 13 tests
    s = slide("Does the choice of test matter? Every test here is valid, they differ only in how much they can certify. "
              "A worst-case bound certifies nothing cheaper than the team. Ignoring the saved answers keeps 79 percent "
              "of tokens. CLIMB's test, which counts both sides and adapts to the variance, keeps 62 percent.")
    header(s, "How it works", "Counting both sides certifies much more")
    picture(s, "c_tests.png", 128, 280, 1150, 620)
    callout(s, 1330, 360, 462, 300, "62%", "of the team’s tokens with CLIMB’s test. A worst-case bound: 100%.", GREEN, GREENBG)

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
            ("CLIMB keeps a promise", "you choose the accuracy loss, it finds the cheapest safe rule."),
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
