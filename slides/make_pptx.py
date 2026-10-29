"""Build slides/CLIMB_talk.pptx: a 15-slide main talk with backup slides, one idea per slide, charts from make_figs.py.

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

NAVY, NAVY2, LIGHT, PAPER, BAND = "00175A", "0A2A78", "F6F8FB", "FFFFFF", "E9EEF5"
BODY, MUTED, RULE, SOFT = "333B47", "6B7480", "D3DCE8", "C3D3EC"
ORANGE, ORANGE_D, BLUE, RED, GREEN = "006FCF", "66B2F0", "00175A", "B03A2E", "2F6B34"  # Amex bright blue accent
REDBG, GREENBG, BLUEBG, ORANGEBG = "F6E3E0", "E3ECE2", "E6EBF4", "DDEDFB"
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
    text(s, 128, 1010, 1664, 34, [dict(t=str(n), size=12, align="r", color="8FA3C8" if dark else MUTED)])


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
        return new_slide(prs, notes, dark)

    # 1 cover
    s = slide("Teams of LLM agents can be more accurate than one model call, but they cost several times as much. "
              "This talk is about when the team is worth calling, and how to skip it safely.")
    text(s, 128, 300, 1664, 40, [dict(t="CLIMB", size=16, bold=True, color=ORANGE)])
    text(s, 128, 350, 1664, 130, [dict(t="When to Call the Team", size=56, bold=True, font=SERIF, color=NAVY)])
    text(s, 128, 495, 1664, 60, [say("Save tokens on multi-agent LLM pipelines, with a guarantee on accuracy", 24, BODY)])
    box(s, 128, 600, 240, 5, ORANGE, shape=MSO_SHAPE.RECTANGLE)
    text(s, 128, 640, 1664, 50, [say("Jianjun Gao, Jinxin Liu, Dongyun Lin, Hongyun Cai, Zheng Ma", 18, MUTED)])

    # team: what it is
    s = slide("First, what is an agent team? Instead of asking one model once, we split the work into roles, all played by "
              "the same model. A planner writes a step-by-step plan. An executor solves the task following the plan. "
              "A reviewer checks the answer. If the reviewer rejects it, the executor revises, for up to five rounds. "
              "This plan, execute, review pattern is common in multi-agent frameworks. And it works: on all twelve of our "
              "settings the team is the most accurate, 2 to 21 points above a direct answer.")
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
    box(s, 1330, 260, 462, 260, ORANGEBG, RULE, [big("+2 to +21", ORANGE, 36),
                                                say("points over a direct answer. Best on 12 / 12 settings.", 17)], margin=26)

    # team: why it helps
    s = slide("Why use a team? It is more accurate. Grey is a direct answer, blue a single agent that reasons, orange the team. "
              "On every one of our twelve settings the team is the most accurate: 2 to 21 points above the direct answer, "
              "and up to 9 points above the single agent. The gains are largest on hard reasoning, BBEH.")
    header(s, "Background", "Teams are more accurate")
    picture(s, "c_accuracy.png", 128, 260, 1150, 700)
    callout(s, 1330, 330, 462, 260, "+2 to +21", "points over a direct answer", ORANGE, ORANGEBG)
    callout(s, 1330, 630, 462, 260, "12 / 12", "settings where the team is best, up to 9 points over a single agent", ORANGE, ORANGEBG)

    # the catch: cost and harm
    s = slide("The accuracy comes at a price, in two ways. Left: the team uses 1.3 to 6.3 times the tokens of a single agent, "
              "because planner, executor and reviewer each add calls, for up to five rounds. Right: against the single agent, "
              "green bars are wrong answers the team fixes, red bars are right answers it breaks, up to 9 percent of tasks. "
              "So the team is better on balance, but it is expensive and not uniformly better.")
    header(s, "The catch", "Teams are costly, and sometimes harmful")
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
              "call the full team. All three use the same model. The ladder itself is simple. A task stops when a score "
              "is above a threshold. So the real question, and the rest of the talk, is: which score, and above all, which threshold?")
    header(s, "The idea", "Climb a ladder: call the team only when needed")
    box(s, 128, 255, 1664, 80, NAVY, paras=[dict(t="Stop if score > threshold.  The key question: which threshold?", size=22, bold=True, color=ORANGE_D)],
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
              "And if no rule passes the check, we simply always call the team.")
    header(s, "How to climb", "Deciding when to climb: four questions", size=34)
    qs = [("1", "Signal", "Does the cheap answer look right?", "confidence and agreement"),
          ("2", "Rule", "Where do we draw the line?", "a threshold on the score"),
          ("3", "Check", "Is that line safe?", "a betting test: the core"),
          ("4", "Depth", "Look one rung up or two?", "let the data choose")]
    for i, (k, t, q, a) in enumerate(qs):
        x = 128 + i * 426
        box(s, x, 290, 386, 540, ORANGEBG if k == "3" else BLUEBG, ORANGE if k == "3" else RULE, lw=3 if k == "3" else 1, paras=[dict(t=k, size=36, bold=True, font=SERIF, color=ORANGE, after=4),
                                               dict(t=t, size=24, bold=True, color=NAVY, after=12),
                                               dict(t=q, size=18, color=BODY, after=16),
                                               dict(t="→ " + a, size=18, bold=True, color=ORANGE)], margin=28, anchor="t")
    text(s, 128, 870, 1664, 60, [say("No rule passes the check?  →  always call the team.  Worst case: no savings, never lost accuracy.", 20, NAVY, bold=True)])

    # 1 signal: why these two
    s = slide("Question one: what tells us a cheap answer is good enough? A good signal must tell right answers from wrong ones, "
              "and it must cost far less than the team, or there is nothing to save. We use two. "
              "The first is confidence: how probable the model finds its own answer. A model that knows the answer tends to put "
              "its probability on one answer. An unsure model spreads it over several. It is free: it comes with the answer. "
              "The second is agreement: run the direct answer and the single agent independently and see whether they match. "
              "There is one right answer but many wrong ones, so two independent tries rarely land on the same wrong answer. "
              "It costs one more cheap call, and that call's answer is the one we keep. It needs no log-probabilities.")
    header(s, "Step 1 · the score", "Why confidence, and why agreement?")
    text(s, 128, 232, 1664, 40, [say("A good signal tells right from wrong, and costs far less than the team.", 20, NAVY, bold=True)])
    for c, (head, sub) in enumerate((("Confidence", "Is the model sure of its answer?"),
                                     ("Agreement", "Do two cheap answers match?"))):
        x = 128 + c * 864
        box(s, x, 300, 800, 670, PAPER, RULE)
        text(s, x + 36, 320, 730, 100, [dict(t=head, size=28, bold=True, font=SERIF, color=NAVY, after=2), say(sub, 18, MUTED)])
        text(s, x + 36, 448, 730, 30, [say("Example: “When did World War II end?”", 15, MUTED)])
    # confidence: sure vs unsure
    cases = [("Sure", [("1945", 0.96)], GREEN, ""),
             ("Unsure", [("1944", 0.38), ("1945", 0.33), ("1946", 0.29)], RED, "")]
    for r, (lab, bars, col, verdict) in enumerate(cases):
        y = 505 + r * 105
        text(s, 164, y, 160, 40, [dict(t=lab, size=18, bold=True, color=NAVY)])
        for j, (ans, pr) in enumerate(bars):
            yy = y + j * 38
            text(s, 330, yy, 80, 34, [say(ans, 16)])
            box(s, 410, yy + 6, 300 * pr, 24, col, shape=MSO_SHAPE.RECTANGLE)
            text(s, 420 + 300 * pr, yy, 70, 34, [say(f"{pr:.2f}", 15, MUTED)])
    text(s, 164, 770, 730, 100, [say("A model that knows tends to put its probability on **one** answer. An unsure one spreads it.", 18)])
    text(s, 164, 900, 730, 50, [say("Cost: free, it comes with the answer.", 18, GREEN, bold=True)])
    # agreement: match vs mismatch
    text(s, 1060, 490, 190, 30, [say("direct answer", 15, MUTED, align="c")])
    text(s, 1270, 490, 190, 30, [say("single agent", 15, MUTED, align="c")])
    for r, (u, v, col, verdict) in enumerate((("1945", "1945", GREEN, "agree: likely right"),
                                              ("1944", "1946", RED, "disagree: doubtful"))):
        y = 530 + r * 105
        for j, ans in enumerate((u, v)):
            box(s, 1060 + j * 210, y, 190, 85, BLUEBG, col, [dict(t=ans, size=22, bold=True, color=NAVY, align="c")], margin=4, lw=2)
        text(s, 1490, y + 25, 290, 40, [say(verdict, 16, col, bold=True)])
    text(s, 1028, 770, 730, 100, [say("One right answer, many wrong ones: two tries **rarely** match on a wrong one.", 18)])
    text(s, 1028, 900, 730, 50, [say("Cost: one more cheap call, no log-probs.", 18, GREEN, bold=True)])

    # 1 signal: evidence
    s = slide("Do they work? Left: for each setting, the direct answer's accuracy on its more confident half of tasks against "
              "its less confident half. On most settings the confident half is right much more often. On three Gemma settings "
              "it is backwards: the model is confidently wrong. Right: the single agent's accuracy when the two cheap answers agree "
              "against when they disagree. Agreement separates on most settings, including the three where confidence fails, "
              "for example 90 against 34 percent on Gemma FRAMES. It is weak on gpt-oss-20b BBEH, where both answers are mostly wrong. So the two signals cover each other. "
              "What does CLIMB do when confidence is backwards? Three things. First, option B ranks tasks by agreement "
              "first, so confidence only breaks ties. On Gemma FRAMES and MuSiQue, CLIMB picks option B in every split. "
              "Second, the check catches it: a backwards score puts the losses first, so the test fails early and that "
              "threshold is never deployed. Third, in the worst case CLIMB calls the team. On these three Gemma settings it "
              "uses 69 to 96 percent of the team's tokens at a 5 point tolerance: it saves less, but the promise still holds.")
    header(s, "Step 1 · the score", "Do they work? Mostly, and they cover each other")
    for c, (img, head) in enumerate((("c_sig_conf.png", "Confidence: direct answer accuracy"),
                                     ("c_sig_agree.png", "Agreement: single agent accuracy"))):
        x = 128 + c * 864
        text(s, x, 245, 800, 40, [dict(t=head, size=22, bold=True, color=NAVY)])
        picture(s, img, x, 318, 800, 370)
    text(s, 128, 695, 800, 40, [say("Mostly helps. [[c=B03A2E|Backwards on 3 (Gemma).]]", 18)])
    text(s, 992, 695, 800, 40, [say("Mostly helps, [[c=2F6B34|including those 3.]]", 18)])
    text(s, 128, 748, 1664, 40, [say("When confidence is backwards, CLIMB still holds:", 19, NAVY, bold=True)])
    fixes = [("1  Agreement first", "Option B ranks by agreement. Confidence only breaks ties."),
             ("2  The check catches it", "A backwards score fails the test early. It is not deployed."),
             ("3  Worst case: the team", "Less saving, no lost accuracy. Gemma: 69–96% of tokens.")]
    for i, (t, d) in enumerate(fixes):
        x = 128 + i * 566
        box(s, x, 812, 532, 158, NAVY, paras=[dict(t=t, size=19, bold=True, color=ORANGE_D, after=4), say(d, 15, LIGHT)],
            margin=18)

    # 2 rule: guessing fails
    s = slide("Question two: where do we draw the line? A rule stops the task when its score is above a threshold. "
              "The natural shortcut is to pick the threshold by hand. Such rules can look fine on average "
              "but lose a lot on some settings. A fixed confidence threshold with the same average cost as CLIMB "
              "loses up to 16 points. Always using the single agent loses up to 9. CLIMB at a 5 point tolerance loses at most 2.8. "
              "And nothing at deployment tells you which setting is the bad one.")
    header(s, "How to climb · rule", "Where to draw the line? Guessing can go badly")
    picture(s, "c_maxloss.png", 128, 260, 1150, 700)
    callout(s, 1330, 360, 462, 300, "16 points", "lost by a hand-picked threshold on one setting", RED, REDBG)
    text(s, 1330, 700, 462, 200, [say("Nothing at deployment says which setting will go wrong.", 18, MUTED)])

    # 3 check: the promise
    s = slide("How do we replace guessing with a promise? You pick only the tolerance, say 5 points, and the data picks the "
              "threshold. Here is the workflow. Take the labelled tasks and split them in two. "
              "On the calibration half, run the betting test on every candidate threshold, from strict to loose, and stop at "
              "the first failure. The thresholds before that failure are certified. Deploy the cheapest certified one. "
              "If none passes, always call the team. "
              "Then check on the test half, which calibration never saw: did the deployed threshold keep the loss within 5 points, "
              "and how many tokens did it save? In our experiments we repeat this split 200 times. "
              "In deployment, you only need the calibration half: the test half is simply your future tasks. "
              "Next, how the betting test works.")
    header(s, "Step 2 · setting the threshold", "Replace guessing with a promise: calibrate, then test")
    text(s, 128, 226, 1664, 40, [say("You pick only ε, e.g. “lose at most 5 points”.  The data picks the threshold.", 18, NAVY, bold=True)])
    T, H, B = 310, 250, 680
    box(s, 128, T, 240, B + 200 - T, BAND, NAVY, [dict(t="Labelled tasks", size=18, bold=True, color=NAVY, align="c", after=8),
                                                  say("split in two at random", 14, MUTED, align="c")], margin=12, lw=2)
    box(s, 430, T + 40, 250, 170, BLUEBG, BLUE, [dict(t="Calibration half", size=17, bold=True, color=NAVY, align="c")], margin=10, lw=2)
    box(s, 430, B + 15, 250, 170, BLUEBG, BLUE, [dict(t="Test half", size=17, bold=True, color=NAVY, align="c", after=4),
                                                say("unseen in calibration", 13, MUTED, align="c")], margin=10, lw=2)
    arrow(s, 376, T + 105, 46, 40)
    arrow(s, 376, B + 80, 46, 40)
    arrow(s, 688, T + 105, 46, 40)
    # betting test on each threshold
    box(s, 742, T, 560, H, NAVY, ORANGE, lw=3)
    text(s, 762, T + 16, 520, 36, [dict(t="Betting test on each threshold", size=17, bold=True, color=LIGHT, align="c")])
    for i, (sym, c) in enumerate((("✓", GREEN), ("✓", GREEN), ("✓", GREEN), ("✗", RED), ("–", MUTED))):
        box(s, 802 + i * 92, T + 72, 76, 76, GREENBG if c == GREEN else REDBG if c == RED else PAPER, c,
            [dict(t=sym, size=22, bold=True, color=c, align="c")], margin=2, lw=2)
    text(s, 762, T + 168, 520, 60, [say("strict → loose, stop at the first failure", 14, SOFT, align="c")])
    arrow(s, 1310, T + 105, 56, 40)
    box(s, 1374, T, 418, H, ORANGEBG, ORANGE, [say("Certified: before the first failure", 14, NAVY, align="c", after=8),
                                              dict(t="★ Deploy the cheapest", size=17, bold=True, color=ORANGE, align="c", after=8),
                                              say("None passes? Always the team.", 13, MUTED, align="c")], margin=14, lw=2.5)
    # test half: check the promise
    arrow(s, 1558, T + H + 10, 50, B - T - H - 20, ORANGE, MSO_SHAPE.DOWN_ARROW)
    arrow(s, 688, B + 80, 678, 40, RULE)
    text(s, 742, B + 30, 560, 36, [say("apply the ★ threshold to unseen tasks", 14, MUTED, align="c")])
    box(s, 1374, B, 418, 200, GREENBG, GREEN, [dict(t="Check the promise", size=19, bold=True, color=GREEN, align="c", after=6),
                                              say("Loss ≤ ε?  Tokens saved?", 15, NAVY, align="c")], margin=14, lw=2.5)
    text(s, 128, 915, 1664, 60, [say("In our experiments: 200 random splits.  In deployment: only the calibration half, the test half is your future tasks.", 15, MUTED, align="c")])

    # 3 check: the rules of the bet
    s = slide("Here is how CLIMB checks a threshold: the betting test. Imagine a gambler who bets that a threshold is safe. "
              "He starts with one dollar. We reveal the labelled tasks one by one, and compare the threshold's answer with the team's on each. "
              "If both give the same result, which is most tasks, his money grows a little, times 1.03. "
              "If the threshold stops on a wrong cheap answer the team got right, he loses a lot, about half. "
              "If it keeps a right cheap answer the team broke, he wins a lot, times 1.5. "
              "If his money ever reaches 20 dollars, the threshold passes. If it never does, it fails.")
    header(s, "Step 2 · setting the threshold · the betting test", "The betting test: start with $1, bet the threshold is safe")
    text(s, 128, 250, 1664, 40, [say("Reveal the labelled tasks one by one. Each task multiplies the money:", 20, NAVY, bold=True)])
    cases = [("Cheap wrong, team right", "× 0.53", "lose a lot", REDBG, RED),
             ("Same result", "× 1.03", "win a little (most tasks)", BAND, NAVY),
             ("Cheap right, team wrong", "× 1.53", "win a lot", GREENBG, GREEN)]
    for i, (t, m, d, f, c) in enumerate(cases):
        x = 128 + i * 575
        box(s, x, 320, 490, 340, f, c, [dict(t=t, size=19, bold=True, color=NAVY, align="c", after=10),
                                        dict(t=m, size=44, bold=True, font=SERIF, color=c, align="c", after=8),
                                        dict(t=d, size=20, bold=True, color=c, align="c")], margin=16, lw=2.5)
    box(s, 128, 700, 1664, 130, NAVY, paras=[dict(t="Money ever reaches $20  →  pass.     Never reaches it  →  fail.",
                                                 size=24, bold=True, font=SERIF, color=LIGHT, align="c")], margin=16)
    text(s, 128, 860, 1664, 80, [say("Multiplier = 1 + λ (ε − loss), loss = 1, 0 or −1. Shown for ε = 5 points and bet size λ = 0.5.", 17, MUTED)])

    # 3 check: good rules grow, bad rules shrink
    s = slide("Why does this tell safe thresholds from unsafe ones? Average the multiplier over tasks: it is one plus the bet size "
              "times the tolerance minus the threshold's true loss. A safe threshold loses less than the tolerance, so on average "
              "each task multiplies the money by more than one, and it grows. An unsafe one loses more, so each task "
              "multiplies it by less than one, and it shrinks. On the chart, three real thresholds from gpt-oss-120b on MuSiQue: "
              "the green one loses half a point and gets rich fast, the blue one loses 2 points and passes more slowly, "
              "the red one loses 6 points, more than the 5 allowed, and goes broke. The bet size is set from the earlier "
              "tasks only, never from the task being revealed.")
    header(s, "Step 2 · setting the threshold · the betting test", "Safe thresholds get rich, unsafe ones go broke")
    picture(s, "c_wealth.png", 128, 250, 1100, 700)
    box(s, 1290, 260, 502, 220, NAVY, paras=[dict(t="Average multiplier", size=18, bold=True, color=ORANGE_D, align="c", after=4),
                                             dict(t="1 + λ (ε − μ)", size=30, bold=True, font=SERIF, color=LIGHT, align="c", after=4),
                                             say("μ: the threshold’s true average loss", 15, SOFT, align="c")], margin=16)
    box(s, 1290, 500, 502, 200, GREENBG, GREEN, [dict(t="Safe: μ < ε", size=22, bold=True, color=GREEN, after=6),
                                                say("average > 1: money grows, passes", 17)], margin=22, lw=2)
    box(s, 1290, 720, 502, 200, REDBG, RED, [dict(t="Unsafe: μ > ε", size=22, bold=True, color=RED, after=6),
                                            say("average < 1: money shrinks, fails", 17)], margin=22, lw=2)
    text(s, 1290, 935, 502, 50, [say("Bet size λ: set from earlier tasks only.", 15, MUTED)])

    # 3 check: why $20
    s = slide("Why should reaching 20 convince us? Suppose the threshold is actually unsafe, so it loses more than the tolerance. "
              "Then every bet loses money on average, so the gambler's average wealth stays at most one dollar. "
              "His money can never go negative. If he reached 20 dollars more than 1 time in 20, those cases alone would "
              "push his average above one dollar. That is impossible. So an unsafe threshold passes at most 5 percent of the time. "
              "Twenty is just one over five percent. Stopping as soon as he reaches 20 is covered too: freeze the money "
              "there and the same argument holds. We check two families of rules, each at 5 percent, so 90 percent overall. "
              "A pass means we accept the threshold and are wrong at most 5 percent of the time, like a driving test: "
              "passing gets you a licence, it does not prove you can drive.")
    header(s, "Step 2 · setting the threshold · the betting test", "Why $20? An unsafe threshold rarely gets there")
    chain = [("Unsafe threshold", "each bet loses on average", REDBG, RED),
             ("Average ≤ $1", "starts at $1, never grows on average", BAND, NAVY),
             ("Never below $0", "a lost answer at most halves it", BAND, NAVY)]
    for i, (t, d, f, c) in enumerate(chain):
        x = 128 + i * 575
        box(s, x, 280, 490, 220, f, c, [dict(t=t, size=26, bold=True, color=c, align="c", after=10),
                                        dict(t=d, size=18, color=BODY, align="c")], margin=22, lw=2.5)
        if i < 2:
            arrow(s, x + 500, 365, 66, 50)
    box(s, 128, 550, 1664, 200, NAVY, paras=[
        dict(t="1  ≥  average wealth  ≥  20 × P(reach $20)", size=36, bold=True, font=SERIF, color=LIGHT, align="c", after=10),
        dict(t="so  [[c=66B2F0|P(an unsafe threshold passes)  ≤  1/20  =  5%]]", size=28, bold=True, color=LIGHT, align="c")], margin=20)
    box(s, 128, 790, 800, 180, PAPER, RULE, [
        say("**100 worlds, $100 in total**", 20, NAVY, after=6),
        say("Each pass holds $20, so at most 5 of 100 pass. $100 line → 1%.", 18, MUTED)], margin=24)
    box(s, 992, 790, 800, 180, PAPER, RULE, [
        say("**A pass: accept, wrong ≤ 5% of the time**", 20, NAVY, after=6),
        say("Like a driving test, not proof you can drive.", 18, MUTED)], margin=24)

    # 3 check: what a pass means
    s = slide("How should we read a pass? A bad rule reaches 20 dollars at most 5 percent of the time. So we agree: "
              "reach 20, and we accept the rule. That decision is wrong at most 5 percent of the time. "
              "Two readings are wrong. A pass does not prove the rule is good: a bad rule stays bad, we were just unlucky. "
              "And it does not mean 5 percent of passed rules are bad. It is like a driving test: if bad drivers pass at most "
              "5 percent of the time, passing gets you a licence, it does not prove you can drive. "
              "We run two such tests, one per family, each at 5 percent, so the promise holds with 90 percent probability.")
    header(s, "Step 2 · setting the threshold · the betting test", "What a pass means, and what it does not")
    box(s, 128, 270, 1664, 170, NAVY, paras=[
        dict(t="A bad rule reaches $20 at most 5% of the time.", size=28, bold=True, font=SERIF, color=LIGHT, align="c", after=8),
        dict(t="So: reach $20 → accept the rule.  This is wrong at most 5% of the time.", size=22, bold=True, color=ORANGE_D, align="c")], margin=18)
    reads = [("Right", "“We accept it, and are wrong at most 5% of the time.”", GREENBG, GREEN),
             ("Wrong", "“Reaching $20 proves the rule is good.”", REDBG, RED),
             ("Wrong", "“5% of the rules that pass are bad.”", REDBG, RED)]
    for i, (t, d, f, col) in enumerate(reads):
        x = 128 + i * 575
        box(s, x, 490, 490, 230, f, col, [dict(t=t, size=24, bold=True, color=col, align="c", after=10),
                                          dict(t=d, size=17, color=NAVY, align="c")], margin=18, lw=2.5)
    box(s, 128, 760, 1664, 200, PAPER, RULE, [
        say("**Like a driving test.** Bad drivers pass at most 5% of the time. Passing gets you a licence, it does not prove you can drive.", 19, NAVY, after=8),
        say("Two families, 5% each  →  the promise holds with probability ≥ 90%.", 18, MUTED)], margin=26)

    # 3 check: many rules, safe to risky
    s = slide("Putting it together, this is how CLIMB sets the threshold. Each dot is one candidate threshold. On the left, "
              "a strict threshold: almost every task goes to the team. Moving right, the threshold gets looser, more tasks stop "
              "early, cheaper but riskier. We run the betting test on each, from strict to loose, and stop at the first failure. "
              "The threshold we deploy is the loosest one that passed, the star, at 57 percent of the team's tokens. Nobody picked it "
              "by hand: it comes from your tolerance and your data. On another model or task, the same steps give a different threshold. "
              "Stopping at the first failure means testing many thresholds adds no extra risk: to pass an unsafe one, the first "
              "unsafe one in the line would have to pass, and that happens at most 5 percent of the time. "
              "The same runs for the two-rung family, and CLIMB keeps the cheaper one.")
    header(s, "Step 2 · setting the threshold", "Putting it together: how CLIMB sets the threshold")
    picture(s, "c_sweep.png", 128, 250, 1664, 700)
    text(s, 128, 960, 1664, 40, [say("gpt-oss-120b on MuSiQue, threshold on the direct answer’s confidence, one calibration half. Two-rung thresholds are tested the same way, the cheaper is kept.", 15, MUTED)])

    # 3 check: why bet on both sides
    s = slide("One more detail of the bet. Skipping the team is not only a risk: sometimes the cheap answer is right and the "
              "team would have broken it. Usual tests count only the losses. Our bet also pays out on saved answers, so they "
              "offset lost ones. On the chart, every test is valid, but they certify very different savings: a worst-case bound "
              "certifies nothing, ignoring saved answers keeps 79 percent of tokens, CLIMB's bet 62.")
    header(s, "Step 2 · setting the threshold · the betting test", "Why bet on saved answers too? It saves more")
    picture(s, "c_tests.png", 128, 260, 1150, 640)
    callout(s, 1330, 300, 462, 280, "62%", "of the team’s tokens when saves count. Losses only: 79%.", GREEN, GREENBG)
    text(s, 1330, 620, 462, 200, [say("Teams break right answers on up to 9% of tasks. The bet gives the rule credit for them.", 18, BODY)])
    text(s, 128, 910, 1150, 40, [say("Tokens each valid test can certify at ε = 5. Lower is better.", 16, MUTED)])

    # 4 depth
    s = slide("Question four: how far up the ladder should we look before deciding? Option A stops after the direct answer "
              "when it is confident. Option B also runs the single agent and stops when the two agree. Which is cheaper depends "
              "on the model and the task, so CLIMB checks both on the calibration data and deploys the cheaper one that passes, "
              "splitting the error budget between them. If neither passes, it always calls the team.")
    header(s, "How to climb · depth", "One rung or two? The data decides")
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
    box(s, 128, 870, 1664, 100, NAVY, paras=[dict(t="Record every task once, then replay [[c=66B2F0|200 random splits]]: half to calibrate CLIMB, half to test it.",
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

    # results: promise kept and savings, together
    s = slide("Results on 3 models and 4 benchmarks: math, two open-book question answering sets and hard reasoning. "
              "We record every task once, then replay 200 random splits, half to calibrate, half to test. "
              "Left: the share of splits where CLIMB lost more than its tolerance. The guarantee allows 10 percent, "
              "we see at most 2. Right: the share of the team's tokens CLIMB uses. It saves 38 percent at a 5 point "
              "tolerance and 65 percent at 10. Savings are smallest on Gemma, where confidence points the wrong way, "
              "and the promise still holds there.")
    header(s, "Results · 3 models × 4 benchmarks, 200 random splits", "These thresholds keep the promise and save 38–65%")
    for c, (img, head) in enumerate((("c_viol.png", "Splits that broke the promise (%)"),
                                     ("c_savings.png", "Tokens used (% of the team’s)"))):
        x = 128 + c * 864
        text(s, x, 250, 800, 40, [dict(t=head, size=22, bold=True, color=NAVY)])
        picture(s, img, x, 310, 800, 420)
    callout(s, 128, 760, 800, 200, "≤ 2%", "of splits broke the promise. Allowed: 10%.", GREEN, GREENBG)
    callout(s, 992, 760, 800, 200, "38% / 65% saved", "allowing 5 / 10 points of loss", BLUE, BLUEBG)

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
              "The guarantee assumes calibration tasks look like deployment tasks. Thank you.")
    header(s, "Takeaways", "Call the team only when you must", size=36)
    cols = [("Teams are costly", "and sometimes break right answers."),
            ("Data-set threshold", "you set the loss, a betting test finds the cheapest safe threshold."),
            ("38–65% saved", "at 5 and 10 points, with the promise kept.")]
    for i, (t, d) in enumerate(cols):
        x = 128 + i * 565
        box(s, x, 330, 533, 6, ORANGE, shape=MSO_SHAPE.RECTANGLE)
        text(s, x, 360, 533, 260, [dict(t=t, size=28, bold=True, font=SERIF, color=NAVY, after=10),
                                   say(d, 20, BODY)])
    text(s, 128, 740, 1664, 80, [say("Assumes calibration tasks resemble deployment tasks. Recalibrating takes under a second.", 18, BODY)])

    # what this offers Amex
    s = slide("Finally, what this could offer Amex. First, token spend. CLIMB only needs a score and a few hundred labelled "
              "tasks, so it can sit in front of any multi-agent workflow. Closed APIs often hide log-probabilities. We can still get "
              "them: feed the closed model's answer to an open model and read off each token's log-probability, the way "
              "PPO's reference model scores tokens written by another policy. The agreement of two cheap calls needs no log-probabilities at all. The guarantee "
              "holds whatever score we plug in: a weaker score saves less, it does not break the promise. We have not "
              "tested these proxy log-probabilities yet, so treat that part as a proposal. Second, design insights. More agents is "
              "not always better: reviewers pass many wrong drafts and teams break some right answers. Route each task "
              "instead of fixing one design for all. Cheap signals such as agreement are strong. And any routing rule, "
              "from model cascades to human review, can be given the same kind of guarantee. Thank you.")
    header(s, "For Amex", "What this offers Amex")
    cols = [("1", "Fewer tokens, even on closed models"), ("2", "Lessons for complex agent systems")]
    for i, (k, t) in enumerate(cols):
        x = 128 + i * 852
        box(s, x, 262, 60, 60, ORANGE, paras=[dict(t=k, size=24, bold=True, color=LIGHT, align="c")], margin=0, shape=MSO_SHAPE.OVAL)
        text(s, x + 80, 262, 740, 60, [dict(t=t, size=20, bold=True, color=NAVY)], anchor="m")
    # column 1: closed model scored by an open model
    flow = [("Closed model", "writes the output"), ("Open model", "reads it → log-probs"), ("CLIMB", "uses them as S₀")]
    for j, (t, d) in enumerate(flow):
        x = 128 + j * 272
        box(s, x, 360, 232, 140, BLUEBG if j < 2 else ORANGEBG, ORANGE if j == 2 else RULE,
            [dict(t=t, size=16, bold=True, color=NAVY, align="c", after=2), dict(t=d, size=13, color=MUTED, align="c")], margin=6, lw=1.5)
        if j < 2:
            arrow(s, x + 238, 410, 28, 40)
    text(s, 128, 515, 790, 40, [say("Same idea as the reference model in PPO.", 14, ORANGE, bold=True)])
    text(s, 128, 595, 790, 330, [
        say("• Feed the closed model’s output to an open model to get its **log-probs**", 16, after=10),
        say("• **38–65% fewer tokens** at 5–10 points in our tests", 16, after=10),
        say("• Agreement of two cheap calls needs **no log-probs**", 16, after=10),
        say("• A weaker score saves less, the **promise still holds**", 16, after=10),
        say("Getting a closed model’s log-probs from an open model is a proposal, not yet tested.", 13, MUTED)])
    # column 2: design lessons
    tips = [("More agents is not always better", "Reviewers pass 23–58% of wrong drafts."),
            ("Route each task, not the whole system", "Escalate only the doubtful tasks."),
            ("Cheap signals are strong", "Two cheap answers agree → usually right."),
            ("Put a guarantee on any routing rule", "Cascades, tool calls, human review.")]
    for j, (t, d) in enumerate(tips):
        box(s, 980, 360 + j * 140, 812, 120, PAPER, RULE, [dict(t=t, size=17, bold=True, color=NAVY, after=4), say(d, 14, MUTED)], margin=18, lw=1)
    text(s, 980, 940, 812, 50, [dict(t="Thank you. Questions?", size=22, bold=True, font=SERIF, color=ORANGE, align="r")])

    s = slide("Backup slides for questions.")
    text(s, 128, 420, 1664, 120, [dict(t="Backup", size=56, bold=True, font=SERIF, color=NAVY)])
    text(s, 128, 580, 1664, 60, [say("Details for questions", 24, MUTED)])

    s = slide("In one slide. Agent teams are more accurate, but they cost several times the tokens and sometimes break right "
              "answers. CLIMB tries cheap answers first and calls the team only when a cheap answer looks unreliable. "
              "You set how much accuracy you can lose, and the data sets the threshold: a betting test on a few hundred labelled tasks "
              "certifies the cheapest threshold that keeps this promise, with 90 percent probability. Across three models and four benchmarks "
              "it saves 38 to 65 percent of tokens, and the promise broke in at most 2 percent of splits. "
              "For Amex, it can sit in front of any multi-agent workflow.")
    header(s, "Executive summary", "Call the team only when you must, with a guarantee")
    rows = [("Problem", "Agent teams are more accurate, but use up to **6.3×** the tokens of a single agent and break right answers on up to **9%** of tasks."),
            ("Idea", "Try cheap answers first. Call the team only when a cheap answer looks unreliable (**confidence**, **agreement**)."),
            ("Guarantee", "You set the accuracy you can lose. The **data sets the threshold**: a **betting test** certifies the cheapest one that keeps it, **90%** of the time."),
            ("Result", "**38–65% fewer tokens** on 3 models × 4 benchmarks. The promise broke in **≤ 2%** of splits (10% allowed)."),
            ("For Amex", "Sits in front of any multi-agent workflow. Lessons for designing complex agent systems.")]
    for i, (k, d) in enumerate(rows):
        y = 265 + i * 142
        hot = k == "Guarantee"
        box(s, 128, y, 300, 120, ORANGE if hot else NAVY, paras=[dict(t=k, size=24, bold=True, color=LIGHT, align="c")], margin=8)
        box(s, 448, y, 1344, 120, ORANGEBG if hot else PAPER, ORANGE if hot else RULE, [say(d, 21, NAVY)], margin=26, lw=2 if hot else 1)

    # step 2 opener: why the threshold cannot be fixed by hand
    s = slide("Now step two, the threshold. Once we have a score, where do we set the line? Two shortcuts come to mind, and both "
              "break the promise. Left: fix one confidence threshold for every setting, chosen so its average cost matches CLIMB. "
              "It is fine on most settings, but loses 8 to 16 points on four of them. A threshold that is safe on all four "
              "gpt-oss-20b benchmarks loses up to 16 points on the other two models. The right threshold depends on the model "
              "and the task. Right: tune the threshold on each setting's own data, picking the cheapest one whose loss on the "
              "sample is within 5 points. A threshold tuned to the edge on one sample tends to cross it on the next: it breaks "
              "the promise on 31 percent of splits. So we need a threshold set from each setting's data, with a guarantee.")
    header(s, "Step 2 · setting the threshold", "Two shortcuts, and both break the promise")
    for c, (img, head, cap) in enumerate((
            ("c_fixed.png", "One threshold for all settings",
             "Above ε on [[c=B03A2E|4 of 12 settings, up to 16 points.]]"),
            ("c_tuned.png", "Tuned on each setting’s data, no test",
             "Breaks the promise on [[c=B03A2E|31% of splits]] at ε = 5."))):
        x = 128 + c * 864
        text(s, x, 245, 800, 40, [dict(t=head, size=22, bold=True, color=NAVY)])
        picture(s, img, x, 300, 800, 430)
        text(s, x, 745, 800, 40, [say(cap, 18)])
    box(s, 128, 830, 1664, 130, NAVY, paras=[
        dict(t="The right threshold depends on the model and task.", size=24, bold=True, font=SERIF, color=LIGHT, align="c", after=4),
        dict(t="We need it set from each setting’s data, with a guarantee.", size=20, bold=True, color=ORANGE_D, align="c")], margin=14)

    # betting test: why this test
    s = slide("Why a betting test? We need a test that, on a few hundred labelled tasks, accepts an unsafe threshold at most "
              "5 percent of the time. Three things make this hard here. First, the loss is signed: a cheap answer can lose a "
              "right answer, but it can also save one the team breaks, and teams break up to 9 percent. Classic bounds such as "
              "Hoeffding have to cover the whole range from minus one to one and certify nothing. Ignoring the saves throws "
              "that credit away. Second, most tasks have loss zero, so the loss varies little. A good test should use that. "
              "Third, we test many thresholds, and splitting the 5 percent among them would make each test much stricter. "
              "The betting test handles all three. It bets on the signed loss, its bet size adapts to the variance it has seen, "
              "and in a strict-to-loose sequence each test can use the full 5 percent. "
              "The guarantee needs no model of the data, only that calibration tasks look like deployment tasks. "
              "On the chart, every test is valid, but CLIMB's test certifies a threshold at 62 percent of the team's tokens, "
              "against 79 to 100 for the others.")
    header(s, "Step 2 · setting the threshold · the betting test", "Why a betting test?")
    text(s, 128, 232, 1664, 40, [say("We need: accept an unsafe threshold at most 5% of the time, from a few hundred tasks.", 18, NAVY, bold=True)])
    needs = [("Saves count", "A cheap answer can also save one the team breaks (up to 9%). The bet credits it."),
             ("Uses low variance", "Most tasks lose 0. The bet size adapts to what it has seen."),
             ("Many thresholds, full 5%", "Tested strict → loose, each test keeps the whole 5%."),
             ("Any data", "No model of the score. Only: calibration looks like deployment.")]
    for i, (t, d) in enumerate(needs):
        y = 300 + i * 165
        box(s, 128, y, 660, 150, PAPER, RULE, [dict(t=t, size=18, bold=True, color=NAVY, after=4), say(d, 15, BODY)], margin=18)
    text(s, 830, 300, 962, 40, [dict(t="All valid. Which one saves the most?", size=18, bold=True, color=NAVY)])
    picture(s, "c_tests.png", 830, 350, 962, 420)
    callout(s, 830, 800, 962, 155, "62% vs 79–100%", "of the team’s tokens at ε = 5", GREEN, GREENBG)

    # betting test: how CLIMB runs it on one threshold
    s = slide("Here is exactly what CLIMB does for one threshold. Take the calibration tasks in random order and start with "
              "one dollar. For each task, the threshold either stops early and keeps the cheap answer, or escalates and uses "
              "the team's answer. The loss is the team's correctness minus ours: plus one if we lost a right answer, minus one "
              "if we saved one, zero otherwise. An escalated task always has loss zero. The bet size is set from the earlier "
              "tasks only, never from the current one, and is capped so one loss at most halves the money. "
              "Multiply the money by one plus bet size times tolerance minus loss. In the example, with bet size 0.5 and a "
              "5 point tolerance, task 3 loses a right answer and the money halves, task 4 saves one and it grows by half. "
              "If the money ever reaches 20 dollars, the threshold is certified and we move to the next, looser one. "
              "If it never does, we stop: no looser threshold is deployed. We run this for options A and B, each at 5 percent.")
    header(s, "Step 2 · setting the threshold · the betting test", "In CLIMB: testing one threshold, task by task")
    cols = [("Task", 100), ("Score > t?", 190), ("Our answer", 180), ("Team", 120), ("Loss ℓ", 170), ("× money", 150), ("Money", 150)]
    rows = [("1", "yes, stop", "✓", "✓", "0", "× 1.03", "$1.03"),
            ("2", "no → team", "team’s", "✓", "0", "× 1.03", "$1.05"),
            ("3", "yes, stop", "✗", "✓", "+1 lost", "× 0.53", "$0.55"),
            ("4", "yes, stop", "✓", "✗", "−1 saved", "× 1.53", "$0.84"),
            ("5", "yes, stop", "✓", "✓", "0", "× 1.03", "$0.86"),
            ("…", "", "", "", "", "", "≥ $20?")]
    y0 = 250
    x = 128
    for c, (h, w) in enumerate(cols):
        box(s, x, y0, w - 6, 56, NAVY, paras=[dict(t=h, size=15, bold=True, color=LIGHT, align="c")], margin=2, shape=MSO_SHAPE.RECTANGLE)
        for r, row in enumerate(rows):
            v = row[c]
            col = RED if ("lost" in v or v in ("✗", "× 0.53")) else GREEN if ("saved" in v or v == "× 1.53") else NAVY
            fill = REDBG if r == 2 else GREENBG if r == 3 else PAPER if r % 2 == 0 else BAND
            box(s, x, y0 + 64 + r * 80, w - 6, 74, fill, None, [dict(t=v, size=17, bold=c in (4, 6), color=col, align="c")],
                margin=2, shape=MSO_SHAPE.RECTANGLE)
        x += w
    text(s, 128, y0 + 64 + 6 * 80 + 6, 1074, 60, [say("ε = 5 points, bet size λ = 0.5.  Escalated tasks use the team’s answer: loss 0.", 15, MUTED)])
    box(s, 1240, 250, 552, 610, NAVY, paras=[
        dict(t="Each threshold, strict → loose", size=18, bold=True, color=ORANGE_D, after=10),
        say("**1**  Tasks in random order, $1.", 15, LIGHT, after=8),
        say("**2**  ℓ = team right − ours right.", 15, LIGHT, after=8),
        say("**3**  λ from **earlier tasks only**. A loss at most halves the money.", 15, LIGHT, after=8),
        say("**4**  Money × (1 + λ (ε − ℓ)).", 15, LIGHT, after=8),
        say("**5**  Ever ≥ $20: certified, next threshold. Never: stop.", 15, LIGHT)], margin=22, anchor="t")
    box(s, 128, 890, 1664, 80, ORANGEBG, ORANGE, [say("Options A and B are each tested this way at 5%, so both hold together with probability ≥ 90%.", 18, NAVY, bold=True, align="c")],
        margin=8, lw=2)

    # step 2 opener: why the rule is a threshold
    s = slide("Why is the rule a threshold, and not something more complex? Think of it as packing a bag. Stopping a task "
              "early saves the team's tokens on it, but risks the accuracy the team would have added, its gain. "
              "You can afford a total risk of epsilon. To save the most tokens, you stop first the tasks with the least gain "
              "per token saved, exactly like a knapsack. Now, if the gain falls as confidence rises, those are the most "
              "confident tasks. So the best rule stops every task above some confidence, and escalates the rest: a threshold. "
              "The paper proves this as a proposition. Does the gain fall with confidence? On the right, the gain from calling "
              "the team in each fifth of tasks, on the nine settings where confidence ranks answers the right way. "
              "It falls from 17 points on the least confident fifth to 4 to 6 on the confident ones, and it falls on eight "
              "of the nine. On Gemma's three backwards settings it does not, and there the guarantee still holds, the savings "
              "come from agreement. So theory gives the form of the rule. What it cannot give is where the line goes: "
              "that depends on the unknown gain, and the data has to decide.")
    header(s, "Step 2 · setting the threshold", "Why a threshold? It is the best rule")
    text(s, 128, 270, 820, 100, [say("Stop the tasks where the team adds **least**.", 22, NAVY)])
    text(s, 128, 400, 820, 100, [say("The team adds least where confidence is **high**.", 22, NAVY)])
    y = 620
    text(s, 128, y - 60, 820, 40, [say("So the best rule is a threshold:", 20, ORANGE, bold=True)])
    box(s, 128, y, 470, 100, ORANGEBG, ORANGE, [say("below t: call the team", 20, NAVY, bold=True, align="c")], margin=6, lw=2, shape=MSO_SHAPE.RECTANGLE)
    box(s, 598, y, 350, 100, GREENBG, GREEN, [say("above t: stop", 20, GREEN, bold=True, align="c")], margin=6, lw=2, shape=MSO_SHAPE.RECTANGLE)
    box(s, 592, y - 20, 12, 140, RED, shape=MSO_SHAPE.RECTANGLE)
    text(s, 560, y + 125, 80, 34, [say("t", 22, RED, bold=True, align="c")])
    text(s, 128, y + 125, 400, 34, [say("low confidence", 16, MUTED)])
    text(s, 748, y + 125, 200, 34, [say("high", 16, MUTED, align="r")])
    text(s, 1000, 240, 792, 40, [dict(t="Gain from the team, by confidence", size=20, bold=True, color=NAVY)])
    picture(s, "c_monotone.png", 1000, 310, 792, 528)
    box(s, 128, 870, 1664, 90, NAVY, paras=[say("Theory gives the **form**.  [[c=66B2F0|The data decides where t goes.]]", 22, LIGHT, align="c")], margin=10)

    # 15-minute order: main talk, then backup (numbers follow the build order above)
    main = [1, 27, 2, 4, 6, 8, 9, 31, 28, 11, 29, 12, 13, 14, 30, 16, 22, 24, 25]
    backup = [26, 3, 5, 7, 10, 23, 15, 17, 18, 19, 20, 21]
    assert sorted(main + backup) == list(range(1, len(prs.slides) + 1))
    ids = prs.slides._sldIdLst
    els = list(ids)
    for e in els:
        ids.remove(e)
    for k in main + backup:
        ids.append(els[k - 1])
    for i, s in enumerate(prs.slides):
        if i:
            footer(s, i + 1)

    out = HERE / "CLIMB_talk.pptx"
    prs.save(out)
    print(out, len(prs.slides))


if __name__ == "__main__":
    build()
