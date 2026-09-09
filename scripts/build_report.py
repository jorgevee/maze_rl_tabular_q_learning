"""Builds the project report as a Word document.

Run with the venv that has python-docx:
    /tmp/docxenv/bin/python scripts/build_report.py

Content lives here rather than in a separate markdown file so the document
stays reproducible from source control.
"""

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ACCENT = RGBColor(0x1F, 0x3B, 0x73)
MUTED = RGBColor(0x55, 0x5A, 0x63)

doc = Document()

# ---------- base styles ----------
normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(11)
normal.paragraph_format.space_after = Pt(8)
normal.paragraph_format.line_spacing = 1.15

for name, size, color, before, after in [
    ("Heading 1", 18, ACCENT, 18, 6),
    ("Heading 2", 14, ACCENT, 14, 4),
    ("Heading 3", 12, ACCENT, 10, 3),
]:
    st = doc.styles[name]
    st.font.name = "Calibri"
    st.font.size = Pt(size)
    st.font.color.rgb = color
    st.font.bold = True
    st.paragraph_format.space_before = Pt(before)
    st.paragraph_format.space_after = Pt(after)


def mono(text):
    """Fixed-width block for equations and code, lightly indented."""
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.3)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.line_spacing = 1.0
    run = p.add_run(text)
    run.font.name = "Consolas"
    run.font.size = Pt(9.5)
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), "Consolas")
    rfonts.set(qn("w:hAnsi"), "Consolas")
    return p


def para(text, style=None, italic=False, bold=False, size=None, color=None):
    p = doc.add_paragraph(style=style)
    run = p.add_run(text)
    run.italic = italic
    run.bold = bold
    if size:
        run.font.size = Pt(size)
    if color:
        run.font.color.rgb = color
    return p


def rich(parts):
    """Paragraph from (text, bold, italic, code) tuples."""
    p = doc.add_paragraph()
    for text, b, i, code in parts:
        run = p.add_run(text)
        run.bold = b
        run.italic = i
        if code:
            run.font.name = "Consolas"
            run.font.size = Pt(10)
            rpr = run._element.get_or_add_rPr()
            rfonts = rpr.find(qn("w:rFonts"))
            if rfonts is None:
                rfonts = OxmlElement("w:rFonts")
                rpr.append(rfonts)
            rfonts.set(qn("w:ascii"), "Consolas")
            rfonts.set(qn("w:hAnsi"), "Consolas")
    return p


def bullet(text, level=0):
    p = doc.add_paragraph(text, style="List Bullet")
    p.paragraph_format.left_indent = Inches(0.3 + 0.25 * level)
    p.paragraph_format.space_after = Pt(4)
    return p


def table(headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        run = hdr[i].paragraphs[0].add_run(h)
        run.bold = True
        run.font.size = Pt(10)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(val))
            run.font.size = Pt(10)
    if widths:
        for r in t.rows:
            for i, w in enumerate(widths):
                r.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return t


# ======================================================================
# Title
# ======================================================================
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run("Reinforcement Learning in a Maze")
run.bold = True
run.font.size = Pt(24)
run.font.color.rgb = ACCENT

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = sub.add_run("From Tabular Q-Learning to DQN to PPO — findings, method, and next steps")
run.font.size = Pt(12)
run.font.color.rgb = MUTED

meta = doc.add_paragraph()
meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = meta.add_run("Hand-written C, no machine-learning libraries  |  September 2026")
run.font.size = Pt(9.5)
run.font.color.rgb = MUTED
meta.paragraph_format.space_after = Pt(16)

# ======================================================================
doc.add_heading("1. What this project is", level=1)

para(
    "A teaching project that solves grid mazes with three reinforcement-learning "
    "algorithms, each implemented from scratch in C with no ML library: tabular "
    "Q-learning, a Deep Q-Network (DQN), and Proximal Policy Optimization (PPO). "
    "Every network, gradient, and optimizer step is hand-written and verified "
    "against finite-difference checks."
)
para(
    "The interesting question is not “can an agent solve one maze” — all three do "
    "that easily. It is whether an agent trained on some mazes can solve mazes it "
    "has never seen. That question turned out to be much harder, and much more "
    "instructive, than expected."
)

doc.add_heading("The environment", level=2)
bullet("A grid maze. The agent occupies one cell and picks one of four moves: up, right, down, left.")
bullet("Rewards: +100 for reaching the goal, −5 for walking into a wall, −1 for any other step.")
bullet("Episodes end at the goal or after 200 steps.")
bullet(
    "Generalization suite: 16 training mazes (10×10) plus 18 held-out mazes the agent "
    "never trains on — six 10×10, six 8×8, six 12×12 — so both new layouts and new "
    "sizes are tested. Every maze is validated with breadth-first search and its "
    "optimal route length recorded."
)

# ======================================================================
doc.add_heading("2. What we learned", level=1)

para(
    "These findings came in sequence, and several of them overturned an earlier "
    "conclusion. That progression is the substance of the project.", italic=True
)

doc.add_heading("2.1  A model can score 100% on training and have learned the wrong thing", level=2)
para(
    "All 16 training mazes sent the agent from the same corner to the opposite "
    "corner. That is not merely a simplification — it means all 16 tasks share one "
    "start-to-goal direction. A network that learns “generally head down and right, "
    "dodge whatever wall appears” scores well on all of them without ever needing to "
    "represent where the goal actually is."
)
para(
    "The 100%-on-training number that looked like clean evidence of capability was "
    "substantially this shortcut. When we removed it by randomizing the start and "
    "goal cell on the same wall layouts, training accuracy collapsed from 48/48 to "
    "1/48 for the convolutional model."
)
rich([
    ("The general lesson: ", True, False, False),
    ("a held-out evaluation only tests what actually varies between training and "
     "testing. If every training example and every test example shares an "
     "invariant, no amount of “held-out” framing will detect a model exploiting it.",
     False, False, False),
])

doc.add_heading("2.2  An architecture comparison was confounded by that shortcut", level=2)
para(
    "Early results showed the convolutional model generalizing far better than the "
    "plain MLP (22.2% vs 3.7% held-out), which we attributed to convolution’s "
    "spatial weight sharing being a better inductive bias. Once the shared "
    "corner-to-corner direction was removed, with architecture and wall-layout "
    "diversity held fixed, that entire advantage vanished (both at 3.7%)."
)
para(
    "The convolutional model had not been shown to plan better. It had been shown to "
    "exploit the shortcut more effectively than the MLP — a much weaker claim than "
    "the one originally written down."
)

doc.add_heading("2.3  More compute does not substitute for the right training data", level=2)
para(
    "Quadrupling the training budget (5,000 → 20,000 episodes) measurably improved "
    "how well the network fit its own training distribution, and moved held-out "
    "performance by exactly zero. Fixing the actual problem instead — biasing the "
    "sampled start/goal pairs toward the long routes the benchmark tests — beat the "
    "4×-budget run on every metric while using a quarter of the compute."
)
rich([
    ("The general lesson: ", True, False, False),
    ("when a metric is stuck, check whether the training distribution resembles what "
     "you are measured on before spending more compute. That check cost two numbers; "
     "the budget experiment cost half an hour.", False, False, False),
])

doc.add_heading("2.4  Capacity and generalization are separable causes", level=2)
para(
    "The MLP fit its training data about twice as well as the convolutional model. "
    "Two explanations competed: convolution’s inductive bias is a poor fit here, or "
    "the convolutional model simply had 38% fewer parameters. We built a "
    "parameter-matched convolutional model (21,809 vs the MLP’s 22,084, identical "
    "otherwise) to separate them."
)
para(
    "Matching capacity closed almost the entire training-fit gap — and moved held-out "
    "performance not at all. Capacity explained the fitting difference; something "
    "about the architecture itself, not its size, explained the transfer difference."
)

doc.add_heading("2.5  Reinforcement learning results are extremely noisy", level=2)
para(
    "One result was reported from 3 random seeds and looked like a clear improvement. "
    "Checking the seeds individually showed a single lucky seed carried almost the "
    "entire effect; the other two showed nothing. Re-running at 10 seeds cut the "
    "claimed effect nearly in half."
)
rich([
    ("The general lesson: ", True, False, False),
    ("an aggregate over a few seeds is not evidence of a trend until you have looked "
     "at the seeds it is made of. This project now treats 10 seeds as a floor and 20 "
     "as the standard for any comparison between close alternatives.",
     False, False, False),
])

doc.add_heading("2.6  Changing the algorithm mattered more than anything else we tried", level=2)
para(
    "Every experiment up to this point varied the data or the architecture while "
    "holding the algorithm fixed at DQN. Swapping in PPO — with the architecture "
    "(22,084 parameters either way), observation, environment, training distribution, "
    "and environment-step budget all held constant — roughly doubled held-out success."
)
table(
    ["Algorithm", "Parameters", "Training fit", "Held-out success"],
    [
        ["DQN, layout MLP", "22,084", "49.1%", "8.9%  (32/360)"],
        ["DQN, convolutional", "13,624", "25.6%", "5.0%  (9/180)"],
        ["DQN, wide convolutional", "21,809", "45.0%", "5.6%  (10/180)"],
        ["PPO, layout MLP", "22,084", "49.4%", "18.3%  (66/360)"],
    ],
    widths=[1.9, 1.1, 1.2, 1.6],
)
para(
    "Training fit came out essentially identical (49.4% vs 49.1%), so this is not a "
    "difference in how well each fit its data — it is a difference in what transfers. "
    "Over 20 seeds: PPO wins 12, DQN wins 4, 4 ties; paired t = 3.41 (df = 19), "
    "p ≈ 0.003; the gap survives dropping each side’s best seed. The effect "
    "strengthened when seeds went from 10 to 20, which is what a real effect does."
)
rich([
    ("Scope, stated plainly: ", True, False, False),
    ("one task family, one observation encoding, one architecture size, one "
     "hyperparameter setting per algorithm, no tuning on either side. This "
     "establishes that PPO transfers better ", False, False, False),
    ("in this setup", False, True, False),
    (" — not a general claim about on-policy versus off-policy methods.",
     False, False, False),
])


doc.add_heading("2.7  Each PPO component was tested by removing it", level=2)
para(
    "Once PPO worked, its components could not be observed doing anything. "
    "Disabling or varying each one in turn showed what each is actually for — "
    "and two of them turned out to be inert at the default settings."
)

para("Reusing each batch of experience is where the sample efficiency comes from.", bold=True)
table(
    ["Reuse passes (K)", "Steps to first solve", "Clip rate", "Wall clock"],
    [
        ["1  (no reuse)", "44,442", "0.000", "39 ms"],
        ["4  (default)", "13,312", "0.012", "119 ms"],
        ["10", "7,578", "0.039", "282 ms"],
        ["20", "6,963", "0.061", "550 ms"],
    ],
    widths=[1.6, 1.6, 1.1, 1.3],
)
para(
    "K = 1 is essentially vanilla policy gradient and needs 6.4x the environment "
    "steps of K = 20. The clip rate is exactly zero at K = 1, as it must be: on "
    "the only pass the ratio is identically 1, so clipping cannot bind."
)

para("Clipping does nothing until updates are aggressive — then it prevents total collapse.", bold=True)
para(
    "Disabling clipping at the default learning rate changes nothing at any K. "
    "Raising the learning rate reveals what it is for. Measuring whether the "
    "final trained policy still solves the maze:"
)
table(
    ["Learning rate", "Clipping on", "Clipping off", "Final entropy (off)"],
    [
        ["0.0003 (default)", "10 / 10", "10 / 10", "0.008"],
        ["0.003", "9 / 10", "2 / 10", "0.024"],
        ["0.01", "7 / 10", "0 / 10", "0.000"],
    ],
    widths=[1.5, 1.2, 1.2, 1.6],
)
para(
    "At the highest rate without clipping, final entropy is exactly 0.000 on all "
    "ten seeds — complete policy collapse. Seven of those seeds found the goal "
    "early and then destroyed their own policy: the unclipped objective kept "
    "raising the winning action’s probability during batch reuse until nothing "
    "else could be sampled, and a policy that has become deterministic on a wrong "
    "action cannot explore back out."
)

para("The GAE lambda answer on one maze is the opposite of the answer on the real task.", bold=True)
para(
    "On a single fixed maze, low lambda wins monotonically across eight values. "
    "On the generalization suite that reverses completely:"
)
table(
    ["lambda", "Train-fit", "Held-out"],
    [
        ["0", "0.6%", "0.0%"],
        ["0.5", "21.9%", "10.6%"],
        ["0.95 (default)", "46.2%", "18.3%"],
        ["1.0", "37.5%", "22.8%"],
    ],
    widths=[1.5, 1.5, 1.5],
)
para(
    "Lambda = 0 — the best setting on the single maze — learns essentially nothing "
    "on the real task. Low lambda means trusting the critic; on one fixed maze the "
    "critic can be accurate, but across sixteen mazes with randomized goals it is "
    "badly wrong early in training, and lambda = 0 leans entirely on it."
)

para("The entropy bonus is load-bearing, because exploration is the bottleneck.", bold=True)
table(
    ["Entropy coefficient", "Train-fit", "Held-out", "Seeds learning nothing"],
    [
        ["0", "20.0%", "8.9%", "5 of 10"],
        ["0.01 (default)", "46.2%", "18.3%", "1 of 10"],
        ["0.05", "61.9%", "20.0%", "0 of 10"],
    ],
    widths=[1.7, 1.2, 1.2, 1.7],
)
para(
    "Without the bonus, half the seeds finish having learned nothing at all. With "
    "goals randomized far from the start, a policy that collapses early stops "
    "finding the goal, never receives the +100 reward, and has nothing to learn "
    "from. This also explains an otherwise odd observation: more exploration "
    "pressure raises training fit rather than trading against it, because "
    "exploration — not exploitation — is the binding constraint."
)

doc.add_heading("2.8  How much of this have we overfit?", level=2)
para(
    "Every result above was scored against the same eighteen held-out mazes, "
    "generated from one fixed seed. No gradient ever touched them, but roughly "
    "twenty configurations have now been compared on them and the best ones kept, "
    "which makes the suite a validation set in practice rather than a test set."
)
para(
    "This is measurable. From the observed spread across seeds, a ten-seed held-out "
    "mean carries about 4.1 percentage points of standard error. Selecting the best "
    "of twenty equally-good configurations inflates the apparent winner by roughly "
    "7.7 percentage points on average — with no real difference between them at all."
)
rich([
    ("What survives this: ", True, False, False),
    ("the PPO-versus-DQN comparison (a 9.4-point gap, pre-specified as a single "
     "comparison, twenty seeds, p ≈ 0.003, strengthening as seeds were added), and "
     "the large qualitative results — lambda = 0 failing outright, clipping "
     "preventing collapse, half the seeds learning nothing without an entropy bonus.",
     False, False, False),
])
rich([
    ("What does not: ", True, False, False),
    ("any “new best” worth a few percentage points that was selected from a sweep. "
     "The 22.8% at lambda = 1.0 is exactly the shape of number this process "
     "manufactures for free, which is why it is not claimed over the default.",
     False, False, False),
])
para(
    "A second form of the same problem: the maze-suite seed has never been varied, "
    "so strictly these are statements about these particular thirty-four mazes."
)

# ======================================================================
doc.add_heading("3. How PPO works in this maze", level=1)

para(
    "PPO belongs to a different family than the other two learners. Tabular "
    "Q-learning and DQN are value-based: they learn how good each action is and "
    "derive behavior by taking the best one. PPO is policy-based: it learns the "
    "behavior directly."
)
mono(
    "DQN:  network -> Q(s,a) for each action  -> act = argmax  (+ epsilon noise)\n"
    "PPO:  network -> pi(a|s), a distribution -> act = sample from it"
)
para(
    "One immediate consequence: exploration is built in. An uncertain policy spreads "
    "probability over several actions and naturally tries them. There is no epsilon "
    "schedule anywhere in the PPO implementation."
)

doc.add_heading("3.1  What the agent sees", level=2)
para(
    "Both PPO networks receive the same observation the layout-aware DQN uses, so "
    "comparisons are not secretly about different inputs. For the agent’s current "
    "cell, we build a 13×13 window centered on the agent containing:"
)
bullet("a wall plane — 1 where a cell is a wall or outside the maze, 0 where it is open")
bullet("a goal plane — 1 at the goal cell if it falls inside the window, 0 elsewhere")
bullet("two extra numbers: the goal’s offset from the agent, in x and y, normalized")
mono("observation size = 2 planes × 13 × 13  +  2 offsets  =  340 values")
para(
    "The window is agent-centered rather than absolute, which is what lets one "
    "network handle 8×8, 10×10, and 12×12 mazes with no change."
)

doc.add_heading("3.2  Two networks", level=2)
para("PPO uses an actor and a critic, kept as separate networks:")
mono(
    "actor:   340 inputs -> 64 hidden (ReLU) -> 4 logits      22,084 parameters\n"
    "critic:  340 inputs -> 64 hidden (ReLU) -> 1 value       21,889 parameters"
)
para(
    "The actor chooses actions. The critic estimates how good a state is, and is used "
    "only to judge whether an action did better or worse than expected. The actor’s "
    "size matches the layout-aware DQN exactly."
)

doc.add_heading("3.3  Turning logits into a move", level=2)
para(
    "The actor emits four numbers, one per direction. A softmax converts them into "
    "probabilities, and the agent samples from that distribution:"
)
mono(
    "pi(a|s) = exp(z_a) / sum over b of exp(z_b)\n"
    "\n"
    "Example, at some cell:\n"
    "  logits  z  = [ 2.1,  0.4, -1.0,  0.2 ]      (up, right, down, left)\n"
    "  pi(a|s)    = [ 0.71, 0.13, 0.03, 0.11 ]\n"
    "\n"
    "  -> usually goes up, but still tries right or left sometimes.\n"
    "     That residual probability IS the exploration."
)

doc.add_heading("3.4  Collecting experience", level=2)
para(
    "PPO is on-policy: it must learn from data its current policy generated, so there "
    "is no replay buffer. It runs the maze for a fixed batch of 2,048 steps, spanning "
    "several episodes, recording at each step:"
)
mono(
    "state, action taken, log pi(action|state), reward, and the critic's value V(s)"
)
para(
    "After a few passes of learning, the whole batch is thrown away. That is the "
    "central trade against DQN, which replays each transition many times: on the "
    "single maze, PPO needed about 2.9× more environment steps than DQN to first "
    "solve it."
)

doc.add_heading("3.5  Was that move better than expected? (the advantage)", level=2)
para(
    "The critic predicts the value of a state. The advantage measures how much better "
    "the action actually taken turned out to be. The one-step version:"
)
mono(
    "delta_t  =  r_t  +  gamma * V(s_next)  −  V(s_t)\n"
    "\n"
    "  gamma = 0.95 (discount)\n"
    "  V(s_next) = 0 when the step reached the goal, since nothing follows it"
)
para("Worked example, one step before the goal:")
mono(
    "  critic thought this state was worth   V(s_t)   = 60\n"
    "  the agent stepped onto the goal:      r_t      = +100,  V(s_next) = 0\n"
    "\n"
    "  delta = 100 + 0.95*0 − 60 = +40      -> much better than expected,\n"
    "                                          make this action more likely"
)
para(
    "Using only the one-step version is stable but leans entirely on the critic being "
    "right. Adding up all future rewards instead is unbiased but noisy. GAE "
    "(Generalized Advantage Estimation) blends the two with a parameter lambda, "
    "computed backwards over the batch:"
)
mono(
    "A_t  =  delta_t  +  gamma * lambda * A_(t+1)          lambda = 0.95\n"
    "\n"
    "  lambda = 0  ->  A_t = delta_t              (trust the critic; low noise, biased)\n"
    "  lambda = 1  ->  A_t = actual return − V(s_t)  (trust reality; unbiased, noisy)\n"
    "\n"
    "  A_t resets to 0 at an episode boundary — credit must not flow\n"
    "  backwards across the end of an episode."
)
para(
    "The critic’s training target falls out of the same quantity: target = A_t + V(s_t). "
    "Advantages are then normalized across the batch to keep gradient sizes stable."
)

doc.add_heading("3.6  The clipped objective — the heart of PPO", level=2)
para(
    "PPO reuses each batch for four passes of gradient updates. That is what makes it "
    "reasonably sample-efficient, but it creates a problem: after the first pass, the "
    "policy has changed, so the data was collected by a policy that no longer exists. "
    "The importance ratio measures that drift:"
)
mono(
    "ratio_t  =  pi_new(a_t|s_t) / pi_old(a_t|s_t)\n"
    "\n"
    "  ratio = 1.0   policy unchanged for this action\n"
    "  ratio = 1.5   the action is now 50% more likely than when it was collected\n"
    "  ratio = 0.5   half as likely"
)
para(
    "A naive objective would push a good action’s probability up without limit within "
    "a single batch, which destabilizes training. PPO takes the pessimistic of the "
    "raw and clipped versions:"
)
mono(
    "L_clip  =  min(  ratio * A,\n"
    "                 clip(ratio, 1−eps, 1+eps) * A  )        eps = 0.2"
)
para("What that minimum actually does:")
table(
    ["Situation", "Ratio", "Result"],
    [
        ["Good action (A > 0)", "above 1.2", "Clipped. No gradient — already likelier, stop."],
        ["Good action (A > 0)", "below 0.8", "Not clipped. Gradient flows — recover it."],
        ["Bad action (A < 0)", "below 0.8", "Clipped. No gradient — already rarer, stop."],
        ["Bad action (A < 0)", "above 1.2", "Not clipped. Gradient flows — push it down."],
    ],
    widths=[1.7, 1.0, 3.1],
)
para(
    "So clipping only ever removes the incentive to keep moving in a direction "
    "already moved too far. It never blocks a correction back toward the old policy. "
    "In the code this is literally “if the clipped branch is selected, contribute no "
    "policy gradient,” and a self-test verifies exactly that."
)

doc.add_heading("3.7  Keeping the policy from going deaf", level=2)
para(
    "Sampling alone does not guarantee lasting exploration: a policy can collapse to "
    "near-certainty early and stop discovering anything. An entropy bonus resists that."
)
mono(
    "H(s)  =  − sum over a of  pi(a|s) * log pi(a|s)\n"
    "\n"
    "  H = log(4) ≈ 1.386   all four moves equally likely (maximum uncertainty)\n"
    "  H → 0                one move has all the probability (fully committed)"
)

doc.add_heading("3.8  Putting it together", level=2)
mono(
    "policy_loss  =  −( mean of L_clip  +  0.01 * mean of entropy )\n"
    "value_loss   =    mean of Huber( target − V(s) )\n"
    "\n"
    "repeat until the step budget is spent:\n"
    "    collect 2,048 steps in the mazes using the current policy\n"
    "    compute advantages (GAE), then normalize them\n"
    "    for 4 passes:\n"
    "        shuffle, and for each minibatch of 64:\n"
    "            Adam update on the actor  using policy_loss\n"
    "            Adam update on the critic using value_loss\n"
    "    discard the batch"
)

doc.add_heading("3.9  Why each piece exists", level=2)
table(
    ["Component", "Problem it solves"],
    [
        ["Stochastic policy", "Exploration, without an external epsilon schedule"],
        ["Critic and advantages", "Reduces gradient noise versus using raw returns"],
        ["GAE lambda", "Dials the bias/noise tradeoff in the advantage estimate"],
        ["Importance ratio", "Lets one batch be reused for several gradient passes"],
        ["Clipping", "Stops that reuse from moving the policy too far off-batch"],
        ["Entropy bonus", "Stops premature collapse to a deterministic policy"],
        ["Advantage normalization", "Keeps gradient scale stable across batches"],
    ],
    widths=[2.0, 4.0],
)

# ======================================================================
doc.add_heading("4. Plan going forward", level=1)

doc.add_heading("4.1  Validate on fresh maze suites (highest priority)", level=2)
para(
    "Section 2.8 is the reason this comes first. Every number in this document "
    "was measured on one fixed set of held-out mazes, and roughly twenty "
    "configurations have been compared on it. Before any further tuning, the "
    "headline comparisons should be re-run on two or three freshly generated "
    "suites."
)
bullet("Add a --suite-seed flag; the maze generator already takes a seed, it is simply hardcoded today.")
bullet("Re-run PPO versus DQN, and the lambda and entropy sweeps, on each new suite.")
bullet(
    "Conclusions that hold across independent maze draws are properties of the "
    "task. Conclusions that move are properties of these thirty-four mazes. "
    "That distinction is currently unknown for every result here."
)
para(
    "This is cheap relative to its value: roughly forty minutes per suite for the "
    "PPO side. It should precede adopting any tuning change, including the "
    "higher entropy coefficient that currently looks attractive."
)

doc.add_heading("4.2  Remaining PPO questions", level=2)
bullet(
    "Test whether PPO's advantage over DQN is substantially the entropy bonus. "
    "At an entropy coefficient of zero PPO scores 8.9% held-out, which is exactly "
    "the DQN figure. If that holds at twenty seeds it materially changes the "
    "interpretation of the headline result."
)
bullet("Wire the procedural-maze training distribution into PPO; only random-goals is supported so far.")
bullet("Build a convolutional PPO variant, mirroring the DQN architecture comparison.")
bullet("Test a shared actor-critic trunk against the current separate networks.")
bullet(
    "Tune hyperparameters for both PPO and DQN before treating the transfer gap "
    "as a settled property of either algorithm — neither side is tuned today."
)

doc.add_heading("4.3  Move to a 3D maze", level=2)
para(
    "The natural next environment. raylib already supports 3D rendering, so the "
    "visualization side is largely free, and the RL side changes in specific, "
    "interesting ways:"
)
table(
    ["Aspect", "2D today", "3D version"],
    [
        ["Actions", "4 (up/right/down/left)",
         "6 with vertical movement, or 3–4 first-person (forward, turn, look)"],
        ["State space", "100 cells (10×10)",
         "1,000 cells for a 10×10×10 grid — 10× larger"],
        ["Observation", "13×13 top-down crop",
         "13×13×13 volume crop, or a first-person depth/ray view"],
        ["Observability", "Full (the crop sees walls directly)",
         "Optionally partial — a first-person agent cannot see behind walls"],
    ],
    widths=[1.2, 1.9, 2.9],
)
para(
    "Two design decisions matter more than the rest, and they should be made "
    "deliberately rather than by default:"
)
rich([
    ("Overhead volume versus first-person view. ", True, False, False),
    ("Keeping an agent-centered 3D volume is the direct extension of what works now "
     "and keeps the problem fully observable. Switching to a first-person view makes "
     "the maze partially observable, which is a genuinely harder and more realistic "
     "problem — and one where a recurrent policy or frame stacking starts to matter. "
     "These are different projects; the volume version is the safer next step.",
     False, False, False),
])
rich([
    ("Cost growth. ", True, False, False),
    ("A 13×13×13 volume is 2,197 cells per plane against 169 today — roughly 13× the "
     "input, so the first network layer grows accordingly and training slows in "
     "proportion. This is the point where the hand-written scalar C implementation "
     "may need attention: a convolutional encoder over the volume, a smaller crop "
     "radius, or simply accepting longer runs.",
     False, False, False),
])
para(
    "Everything already learned about methodology carries over unchanged, and should "
    "be applied from the start rather than rediscovered: avoid a fixed start-to-goal "
    "direction across training mazes, check that the training distribution resembles "
    "the evaluation distribution, track training fit alongside held-out success, and "
    "use at least 10 seeds before believing any comparison."
)

doc.add_heading("4.4  Suggested order", level=2)
bullet("Fresh maze suites, to find out which conclusions are real (section 4.1).")
bullet("The entropy-versus-DQN question, which is cheap and would sharpen the headline result.")
bullet("Procedural distribution and a convolutional variant for PPO, closing the open comparisons.")
bullet("A 3D maze with an agent-centered volume observation, fully observable — the direct extension.")
bullet(
    "Only then: first-person partial observability, which changes the problem "
    "class and probably needs memory in the policy."
)
para(
    "The four PPO component ablations (reuse passes, clipping, GAE lambda, "
    "entropy) are complete and are written up in section 2.7.", italic=True
)

# ======================================================================
doc.add_heading("Appendix: reproducing the results", level=1)
para("All results in this document come from the following commands.")
mono(
    "make test                    # all self-tests, including gradient checks\n"
    "make benchmark               # tabular vs DQN, single maze, 10 seeds\n"
    "make generalization          # DQN on the held-out suite\n"
    "make ppo                     # PPO vs DQN, single maze, budget-matched\n"
    "make ppo-generalization      # PPO on the held-out suite, 10 seeds\n"
    "\n"
    "# The 20-seed PPO vs DQN comparison in section 2.6:\n"
    "./maze_rl --ppo --generalize --random-goals --min-separation 10 \\\n"
    "          --steps 575000 --seeds 20 --seed 1 --csv ppo_gen.csv\n"
    "./maze_rl --generalization --episodes 5000 --seeds 20 --seed 1 \\\n"
    "          --random-goals --min-separation 10 --csv dqn_gen.csv"
)
para(
    "Full experiment logs live in EXPERIMENTS.md, methodology notes in "
    "lessons_learned.md, and the PPO design rationale in ppo_design.md.",
    italic=True,
)

doc.save("Maze_RL_Report.docx")
print("wrote Maze_RL_Report.docx")
