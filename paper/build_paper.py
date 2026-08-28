"""Build the research paper PDF from the results files.

    .venv\\Scripts\\python paper\\build_paper.py

Writes paper/label-efficient-injection-detection.pdf. Tables are read from results/
at build time; figures come from results/figures/. Fonts: DejaVu (bundled with matplotlib).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from detect.labeler import INSTRUCTION, SYSTEM  # noqa: E402

RES = ROOT / "results"
FIG = RES / "figures"
OUT = ROOT / "paper" / "label-efficient-injection-detection.pdf"
REPO = "https://github.com/fzee999/label-efficient-injection-detection"
TITLE_TEXT = "Label-Efficient Prompt-Injection Detection"
SUBTITLE = "Active Learning with an LLM as an Imperfect Labeler"
TEXT_W = 6.7 * inch

# ---------------------------------------------------------------- fonts
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for name, file in [
    ("Serif", "DejaVuSerif.ttf"), ("Serif-Bold", "DejaVuSerif-Bold.ttf"),
    ("Serif-Italic", "DejaVuSerif-Italic.ttf"), ("Serif-BoldItalic", "DejaVuSerif-BoldItalic.ttf"),
    ("Sans", "DejaVuSans.ttf"), ("Sans-Bold", "DejaVuSans-Bold.ttf"),
    ("Sans-Oblique", "DejaVuSans-Oblique.ttf"), ("Mono", "DejaVuSansMono.ttf"),
]:
    pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / file)))
pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-Bold",
                              italic="Serif-Italic", boldItalic="Serif-BoldItalic")
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold",
                              italic="Sans-Oblique", boldItalic="Sans-Bold")

# ---------------------------------------------------------------- styles
ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["Normal"], fontName="Serif", fontSize=10,
                      leading=14, alignment=TA_JUSTIFY, spaceAfter=6)
ABSTRACT = ParagraphStyle("abstract", parent=BODY, fontSize=9.5, leading=13,
                          leftIndent=28, rightIndent=28)
TITLE = ParagraphStyle("title", parent=ss["Title"], fontName="Sans-Bold", fontSize=17,
                       leading=22, alignment=TA_CENTER, spaceAfter=8)
AUTHOR = ParagraphStyle("author", parent=BODY, fontSize=10.5, alignment=TA_CENTER, spaceAfter=2)
META = ParagraphStyle("meta", parent=BODY, fontName="Serif-Italic", fontSize=9,
                      alignment=TA_CENTER, spaceAfter=14, textColor=colors.HexColor("#444444"))
SUB = ParagraphStyle("sub", parent=TITLE, fontName="Sans", fontSize=13, leading=17, spaceAfter=10)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Sans-Bold", fontSize=13,
                    leading=16, spaceBefore=14, spaceAfter=6, keepWithNext=1)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Sans-Bold", fontSize=11,
                    leading=14, spaceBefore=10, spaceAfter=4, keepWithNext=1)
BUL = ParagraphStyle("bullet", parent=BODY, leftIndent=16, bulletIndent=4, spaceAfter=3)
CAP = ParagraphStyle("caption", parent=BODY, fontName="Serif-Italic", fontSize=9,
                     leading=12, alignment=TA_CENTER, spaceBefore=4, spaceAfter=10)
CELL = ParagraphStyle("cell", parent=BODY, fontName="Sans", fontSize=8.2, leading=10,
                      alignment=0, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Sans-Bold")
CELLC = ParagraphStyle("cellc", parent=CELL, alignment=TA_CENTER)
MONO = ParagraphStyle("mono", parent=BODY, fontName="Mono", fontSize=7.8, leading=10,
                      leftIndent=10, backColor=colors.HexColor("#f6f6f6"), borderPadding=4,
                      spaceBefore=3, spaceAfter=8, alignment=0)
REF = ParagraphStyle("ref", parent=BODY, fontSize=9, leading=12, leftIndent=22,
                     firstLineIndent=-22, spaceAfter=3, alignment=0)

story: list = []
_pending: list = []   # a table waiting for its caption
_lead: list = []      # headings waiting for the block they introduce


def flush(*items):
    story.extend([*_lead, *items])
    _lead.clear()


def block(items: list):
    """Keep a figure or table together with its caption and any heading just before it."""
    story.append(KeepTogether([*_lead, *items]))
    _lead.clear()


def P(text: str, style=BODY):
    if style in (H1, H2):
        _lead.append(Paragraph(text, style))
    else:
        flush(Paragraph(text, style))


def bullets(items: list[str]):
    for it in items:
        flush(Paragraph(it, BUL, bulletText="•"))
    flush(Spacer(1, 3))


def table(data, col_widths, center_from=None, font_size=8.2):
    """Every cell wrapped in a Paragraph; first row bold; zebra rows."""
    rows = []
    for r_i, row in enumerate(data):
        cells = []
        for c_i, cell in enumerate(row):
            st = CELLB if r_i == 0 else (CELLC if center_from is not None and c_i >= center_from else CELL)
            cells.append(Paragraph(str(cell), st))
        rows.append(cells)
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    style = [
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    style += [("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f3f3f3")) for i in range(2, len(rows), 2)]
    t.setStyle(TableStyle(style))
    _pending.append(t)


def caption(text: str):
    cap = Paragraph(text, CAP)
    if sum(len(t._cellvalues) for t in _pending) > 20:   # long tables may split across pages
        flush(*_pending, cap)
    else:
        block([*_pending, cap])
    _pending.clear()


def figure(name: str, cap: str, width: float = TEXT_W):
    w, h = ImageReader(str(FIG / name)).getSize()
    block([Image(str(FIG / name), width=width, height=width * h / w), Paragraph(cap, CAP)])


def f3(v) -> str:
    return f"{float(v):.3f}"


def pct(v, digits=0) -> str:
    return f"{100 * float(v):.{digits}f}%"


def pval(p) -> str:
    p = float(p)
    return f"{p:.1e}" if p < 0.001 else f"{p:.3f}"


# ---------------------------------------------------------------- data
COMP = {lab: pd.read_csv(RES / f"strategy_comparison_{lab}.csv").set_index("strategy")
        for lab in ("perfect", "llm", "routed90")}
ACC = json.loads((RES / "labeler" / "accuracy.json").read_text())
TESTS = pd.read_csv(RES / "analysis" / "paired_tests.csv")
EQUIV = pd.read_csv(RES / "analysis" / "equivalence_routed_vs_dataset.csv")
SWEEP = pd.read_csv(RES / "threshold_sweep.csv")
TRANSFER = pd.read_csv(RES / "transfer" / "transfer_summary.csv").set_index("detector")
SPLITS = json.loads((RES / "split_report.json").read_text())
CEIL = json.loads((RES / "ceiling.json").read_text())["C1"]
LAB = pd.read_csv(ROOT / "data" / "ood" / "lab.csv", encoding="utf-8").set_index("id")
PER_TEXT = pd.read_csv(RES / "transfer" / "best_detector_per_lab_text.csv").set_index("id")
STRATS = ["random", "uncertainty", "diversity", "hybrid"]
LAB_NAMES = {"perfect": "dataset", "llm": "LLM only", "routed90": "routed 0.9"}

# ================================================================ TITLE
P(TITLE_TEXT, TITLE)
P(SUBTITLE, SUB)
P("Faraz Ahmad", AUTHOR)
P(f"Independent research<br/>Code, query logs, labeler outputs and results: {REPO}", META)

P("<b>Abstract.</b> A learned prompt-injection detector needs labeled examples, and labels are the "
  "expensive part. We study two ways to reduce that cost and how they interact: pool-based active "
  "learning, which lets the model choose which texts are labeled, and a local 8B language model used as "
  "a free but imperfect labeler. On a public injection dataset (8,084 pool texts, 2,060 test texts), with "
  "a logistic-regression detector on sentence embeddings, four query strategies and ten seeds per "
  "configuration, uncertainty sampling with correct labels reached 99% of the full-data F1 with a median "
  "of 60 labels against 260 for random sampling, and 600 actively chosen labels exceeded the model trained "
  "on all 8,084. The LLM labeler agreed with the dataset on 91.3% of texts and was well calibrated, but its "
  "errors were systematic, and with its labels the advantage of uncertainty sampling disappeared: the "
  "strategy concentrated the labeler's misses in its training set (36% wrong labels against 9% for random) "
  "and repeatedly queried near-copies of them. The pre-registered hypothesis that label noise hurts "
  "uncertainty sampling more than random sampling was supported (difference in learning-curve area 0.020, "
  "95% CI [0.013, 0.027], 10 of 10 seeds, Holm-adjusted p = 0.0004). Routing only the labeler's "
  "low-confidence texts to a human restored the advantage: at a confidence threshold of 0.8, reaching 99% "
  "of the full-data F1 took a median of 27 human labels. A hybrid uncertainty-diversity strategy was robust "
  "to label noise throughout. No detector transferred well to attacks from outside the training "
  "distribution; direct attacks disguised as ordinary requests were missed on every seed by every "
  "detector, including the one trained on all labels. Every deviation from the pre-registered plan is "
  "reported, and the code, query logs and the labeler's raw outputs are released.", ABSTRACT)

# ================================================================ 1 INTRODUCTION
P("1. Introduction", H1)
P("Prompt injection places instructions where a language model expects data, and the model follows them "
  "[14, 15, 16]. One common layer of defense is a classifier in front of the model that flags injection "
  "attempts in user input or retrieved content. Such a detector is a supervised model, and its cost is "
  "dominated by labels: a person reading each text and deciding whether it tries to take over an AI "
  "system. Two ideas promise to cut that cost. <b>Active learning</b> lets the current model choose the "
  "texts it would learn the most from, so that only those are labeled [1, 2]. <b>LLM labelers</b> replace "
  "the person with a language model that labels for free, with some error [9, 10].")
P("The two ideas pull against each other. Uncertainty sampling, the most common active-learning strategy, "
  "deliberately asks about the texts nearest the decision boundary, and those are the texts a noisy "
  "labeler is most likely to get wrong. The literature on imperfect oracles [5, 6, 7] treats labeler noise "
  "as a cost to be modelled; with an LLM labeler the noise is not random but systematic, tied to the kinds "
  "of text the LLM misreads. Whether active learning still saves labels under such a labeler, and how much "
  "human effort a mixed human-LLM labeler needs, are empirical questions this paper measures for "
  "injection detection.")
P("We fixed five hypotheses before running any experiment (Section 4.7) and report each against its "
  "pre-registered criterion, with every later addition declared (Section 8).")
P("<b>Contributions.</b>")
bullets([
    "A controlled measurement of label efficiency for embedding-based injection detection: four query "
    "strategies, eight labeler settings, ten seeds each, 320 runs, all with the same detector and protocol.",
    "Evidence of a feedback loop between uncertainty sampling and a systematically biased LLM labeler, with "
    "its mechanism measured: the direction of the label errors each strategy collects, and the near-copies "
    "it queries.",
    "A confidence-routed human-LLM labeler with every human label counted, mapping the trade-off between "
    "human effort and detector quality across six thresholds, and the number of human labels each "
    "combination needs to reach a near-ceiling detector.",
    "A transfer evaluation on attacks from outside the training data, showing that the forms of attack a "
    "detector misses do not depend on how many labels it was given or who gave them.",
])

# ================================================================ 2 RELATED WORK
P("2. Background and Related Work", H1)
P("<b>Active learning.</b> Settles [1] surveys pool-based active learning. Uncertainty sampling [2] "
  "queries the texts the model is least sure about; in binary classification, least-confidence, margin "
  "and entropy sampling select the same texts. Batch selection by uncertainty alone can pick redundant "
  "near-duplicates, which motivates diversity-based selection such as the core-set approach of Sener and "
  "Savarese [3], a greedy k-center cover of the pool, and hybrids such as BADGE [4] that combine "
  "uncertainty with diversity.")
P("<b>Imperfect labelers.</b> Donmez and Carbonell [5] study active learning with multiple imperfect "
  "oracles of different cost and reliability. Sheng, Provost and Ipeirotis [6] show when repeated noisy "
  "labels improve data quality, and Yan et al. [7] learn from crowds of labelers of varying expertise. "
  "Learning to defer [8] trains a model to decide when to hand a case to a human expert. Our routed labeler "
  "is the simplest member of this family: a fixed confidence threshold decides which texts the LLM keeps "
  "and which go to the human.")
P("<b>LLMs as annotators.</b> Wang et al. [9] use GPT-3 to cut labeling cost; Gilardi et al. [10] report "
  "that an LLM outperformed crowd workers on several text-annotation tasks; Wang et al. [11] route LLM "
  "labels to human verification by the LLM's own explanation and confidence. LLMs used as judges show "
  "position bias [12], and larger models are often reasonably calibrated about their own answers [13]. "
  "We observe both effects in our labeler.")
P("<b>Prompt injection.</b> Perez and Ribeiro [14] catalogued goal hijacking and prompt leaking; Greshake "
  "et al. [15] introduced indirect injection through retrieved content; Liu et al. [16] formalised attacks "
  "and defenses and benchmarked them. Public detection datasets include the safe-guard set [22], which we "
  "use for training and testing, and deepset's prompt-injection set [23], which we use for transfer.")

# ================================================================ 3 DATA
P("3. Data", H1)
pt = SPLITS["pool_and_test"]
P(f"<b>Pool and test.</b> The training split of safe-guard [22] ({pt['train_rows']:,} texts) is the labeling "
  f"pool and its test split ({pt['test_rows']:,}) is the test set; the test set is never used to train or "
  f"to choose what to label. Before any experiment we removed from the pool {pt['removed_in_test']} texts "
  f"that also appear in the test set, {pt['removed_conflicting_labels']} texts that appear with both labels, "
  f"and {pt['removed_duplicates']} exact duplicates, leaving {pt['pool_rows']:,}. Injections are "
  f"{SPLITS['label_balance']['pool_injection_pct']}% of the pool and "
  f"{SPLITS['label_balance']['test_injection_pct']}% of the test set. The dataset's license is not "
  "stated, so the repository stores only row numbers and labels and downloads the texts at a pinned "
  "revision with a checksum.")
P("<b>The dataset is easy, and slightly leaky.</b> For 98.9% of test texts the most similar pool text "
  "(by cosine similarity of embeddings) has the same label, against about 58% expected from the class "
  "balance. Some test texts are near-copies of pool texts, so we also report results on the 1,445 test "
  "texts (70%) whose nearest pool text is below cosine 0.90, the <i>far</i> subset; nearest-neighbour "
  "agreement there is still 98.4%. The pool itself is redundant: 225 pool texts have a near-copy "
  "(cosine ≥ 0.99) elsewhere in the pool.")
od, ol = SPLITS["ood_deepset"], SPLITS["ood_lab"]
P(f"<b>Outside sets.</b> Two sets measure transfer. The <i>lab</i> set has 56 texts from an earlier "
  f"companion project on defending a document-reading assistant for a fictional university: "
  f"{ol['direct']} direct attacks typed by the user, {ol['indirect']} indirect attacks (represented here by "
  f"the payload hidden in a document), {ol['leakage']} attempts to extract a secret, {ol['benign_question']} "
  f"benign questions and {ol['policy_doc']} benign policy documents. The <i>deepset</i> set is the test "
  f"split of deepset/prompt-injections [23] ({od['rows']} texts after removing one also in the pool; "
  f"{SPLITS['label_balance']['ood_deepset_injection_pct']}% injections); {od['lang']['de']} are German and "
  f"{od['lang']['es']} Spanish, and {od['roleplay']} are role-play requests that deepset labels as "
  "injections while the training data labels similar requests benign.")
table([
    ["Set", "Texts", "Injections", "Role"],
    ["Pool (safe-guard train, cleaned)", f"{pt['pool_rows']:,}", f"{SPLITS['label_balance']['pool_injection_pct']}%", "texts that can be queried and labeled"],
    ["Test (safe-guard test)", f"{pt['test_rows']:,}", f"{SPLITS['label_balance']['test_injection_pct']}%", "in-distribution evaluation"],
    ["Test, far subset", "1,445", "", "test texts with no pool text at cosine ≥ 0.90"],
    ["Lab (companion project)", "56", "45", "transfer, by attack type"],
    ["deepset test", f"{od['rows']}", f"{SPLITS['label_balance']['ood_deepset_injection_pct']}%", "transfer, multilingual, different definition"],
], [2.2 * inch, 0.7 * inch, 0.8 * inch, 3.0 * inch])
caption("Table 1. Data used in the study.")

# ================================================================ 4 METHOD
P("4. Method", H1)
P("4.1 Detector", H2)
P("Each text is embedded once by nomic-embed-text [17] (768 dimensions, unit length, with the model's "
  "<i>classification</i> task prefix), served locally by Ollama [24]. The detector is logistic regression "
  "[25] with balanced class weights and a 0.5 threshold. Regularisation was fixed at C = 1 before any "
  "active-learning run: five-fold cross-validation on the full pool preferred C = 10 by 0.4 points of F1, "
  "but small labeled sets need stronger regularisation, and one fixed value keeps every comparison fair. "
  f"Trained on all {pt['pool_rows']:,} pool labels the detector reaches test F1 {f3(CEIL['test']['f1'])} "
  f"(AUROC {f3(CEIL['test']['auroc'])}) and far-subset F1 {f3(CEIL['test_far']['f1'])}; these are the "
  "ceilings against which targets are set.")
P("4.2 Query strategies", H2)
bullets([
    "<b>Random</b>: the baseline.",
    "<b>Uncertainty</b>: the texts whose predicted injection probability is closest to 0.5 [2].",
    "<b>Diversity</b>: greedy k-center on cosine distance [3]; each pick is the pool text farthest from "
    "everything already labeled or picked.",
    "<b>Hybrid</b>: the 200 most uncertain texts (ten times the batch), then greedy k-center among them, "
    "in the spirit of uncertainty-plus-diversity methods [4].",
])
P("4.3 Labelers", H2)
P("<b>Dataset labels</b> stand in for a careful human. <b>LLM labels</b> come from Llama 3.1 8B Instruct "
  "[18] (4-bit, 8,192-token context) at temperature 0. The prompt (Appendix A) tells the model in the "
  "system message that the text is untrusted data, places the text between delimiters, gives the "
  "definition and instruction after it, and pre-fills the assistant turn with <i>Answer:</i> so that the "
  "next token is a letter. The injection probability is p = P(A) / (P(A) + P(B)) from the top-10 token "
  "probabilities, the label is injection if p ≥ 0.5, and the labeler's confidence is max(p, 1 − p).")
P("Three findings from building the labeler shaped this design. A first, naive prompt was itself "
  "injected: asked to classify a text that told it to ignore its instructions, it followed the text. "
  "Without the pre-filled reply, the first token was often not a letter. And the model prefers the letter "
  "B: with A meaning injection it flagged 45% of a 40-text development sample, with the letters swapped "
  "it flagged 90%, at nearly the same AUROC, a position bias of the kind reported for LLM judges [12]. "
  "The format was chosen on those 40 development texts, which are excluded from every labeler score. The "
  "whole pool was labeled once and the labels frozen, because temperature-0 decoding was not perfectly "
  "repeatable (relabeling 100 texts reproduced 99 labels).")
P("<b>Routed labels</b> keep the LLM's label when its confidence is at least a threshold τ and otherwise "
  "use the dataset label, counted as one human label. The pre-registered operating point is τ = 0.9, "
  "chosen from the labeler's calibration before any routed run; a sweep over τ ∈ {0.6, 0.7, 0.8, 0.9, "
  "0.95, 0.99} maps the trade-off. With LLM-only (no human) and dataset labels (all human) as the two "
  "ends, there are eight labeler settings.")
P("4.4 Protocol", H2)
P("Each run starts from 20 randomly chosen labeled texts (the same 20 for every strategy within a seed), "
  "adds a batch of 20 per round, and stops at 600 labels; the detector is retrained and evaluated after "
  "every round. Every combination of strategy and labeler setting is run with seeds 0 to 9: 32 "
  "configurations, 320 runs. Each run logs the texts it queried, so every final model can be rebuilt "
  "exactly from the log.")
P("4.5 Measures", H2)
bullets([
    "<b>F1 for the injection class</b> on the test set and on the far subset, after every round.",
    "<b>Curve area</b>: the area under the F1 curve from 20 to 600 labels, divided by 580, a single "
    "number for the whole learning curve.",
    "<b>Labels to target</b>: the first budget at which a run's test F1 reaches a target; reported as the "
    "median over seeds, with runs that never reach it counted as above 600. The pre-registered target is "
    "95% of the ceiling (0.934); 99% of the ceiling (0.973 on test, 0.938 on the far subset) was added "
    "later (Section 8).",
    "<b>Human labels to target</b>: for routed runs, the human labels used up to that first crossing.",
    "<b>Transfer</b>: recall on each outside set and attack type, and the share of benign texts flagged.",
])
P("4.6 Statistics", H2)
P("All comparisons are paired by seed. Six primary comparisons on curve area (Table 6) form one family, "
  "tested with paired t-tests, Holm-corrected together [19], with Wilcoxon signed-rank tests as a check; "
  "everything else is exploratory and uncorrected. Equivalence of routed and dataset labels is a "
  "two-one-sided-tests check [20]: the 90% confidence interval of the paired difference in F1 at 600 "
  "labels must lie within ±0.005.")
P("4.7 Pre-registration", H2)
P("The research questions, hypotheses, data, method and metrics were written down before any experiment "
  "ran [21]. The five hypotheses are scored in Table 7. Anything added afterwards is listed in Table 8 with "
  "its reason and whether it was fixed before or after seeing the data it affects.")

# ================================================================ 5 RESULTS
P("5. Results", H1)
P("5.1 With correct labels, label choice beats label count (H1)", H2)
figure("headline_curves.png", "Figure 1. Test F1 against labels used, mean of 10 seeds with 95% confidence "
       "bands, under the three main labelers. Dashed line: the detector trained on all 8,084 dataset labels.")
p, rt = COMP["perfect"], COMP["routed90"]
P("Every strategy, random included, reached the pre-registered 95% target at 40 labels, the second point "
  "of the protocol grid; on a finer grid random sampling needs a median of 25. The target was too easy for "
  "this dataset, and H1 is not supported as registered. At the stricter 99% target the strategies separate "
  f"clearly: uncertainty sampling needs a median of {p.loc['uncertainty','test_99']} labels, hybrid "
  f"{p.loc['hybrid','test_99']}, random {p.loc['random','test_99']}, and diversity never reaches it "
  f"(Table 2). Uncertainty and hybrid sampling end at F1 {f3(p.loc['uncertainty','f1_at_600'])} and "
  f"{f3(p.loc['hybrid','f1_at_600'])} with 600 labels, above the {f3(CEIL['test']['f1'])} of the model "
  f"trained on all {pt['pool_rows']:,}; on the far subset they reach {f3(p.loc['uncertainty','far_f1_at_600'])} "
  f"and {f3(p.loc['hybrid','far_f1_at_600'])} against {f3(CEIL['test_far']['f1'])}. Most of the pool is "
  "easy and redundant, and a labeled set concentrated near the boundary generalises better than all of it.")

rows = [["Strategy", "Labeler", "Curve area", "F1 at 600", "Far F1 at 600", "Labels to 99%", "Human labels to 99%", "Lab F1 at 600"]]
for lab, df in COMP.items():
    for s in STRATS:
        r = df.loc[s]
        reach = f"{r.test_99} ({int(r.test_99_reached)}/10)"
        human = {"perfect": reach, "llm": "0" if r.test_99_reached else "–",
                 "routed90": f"{r.test_99_human} ({int(r.test_99_human_reached)}/10)"}[lab]
        rows.append([s, LAB_NAMES[lab], f"{r.auc_mean:.4f} ± {r.auc_std:.4f}", f3(r.f1_at_600),
                     f3(r.far_f1_at_600), reach, human, f3(r.lab_f1_at_600)])
table(rows, [0.85 * inch, 0.8 * inch, 1.2 * inch, 0.55 * inch, 0.6 * inch, 0.9 * inch, 1.0 * inch, 0.6 * inch],
      center_from=2, font_size=7.8)
caption("Table 2. Results by strategy and labeler, mean of 10 seeds (curve area ± standard deviation; "
        "labels to target as a median, with the number of seeds that reach it). Under LLM labels, first "
        "crossings are noisy: hybrid crosses the 99% target once on 8 seeds although its mean F1 at 600 is "
        "below it.")
figure("curves_perfect.png", "Figure 2. All four strategies with dataset labels, on the test set and on the "
       "far subset (band: ±1 standard deviation over 10 seeds).")

P("5.2 The LLM labeler is accurate and calibrated, with systematic errors (H2)", H2)
c = ACC["confusion"]
P(f"On the {ACC['scored_texts']:,} scored pool texts the labeler agreed with the dataset on "
  f"{pct(ACC['accuracy'], 1)} (Cohen's κ = {ACC['cohen_kappa']:.2f}, AUROC {ACC['auroc']:.3f}), above the "
  f"80–90% range H2 predicted. It missed {c['fn']} injections and flagged {c['fp']} benign texts, an "
  "almost even split, so H2's second part, that most errors would be benign texts about instructions or "
  "rules, is not supported either. Its confidence is meaningful (Figure 3): accuracy rises from 59% in "
  "the least confident bin to 100% in the most confident, at or above the confidence stated in every bin. "
  "At τ = 0.9 the labeler keeps 62% of texts at 98.8% accuracy, which is what makes routing work.")
figure("labeler_calibration.png", "Figure 3. Labeler accuracy by its own confidence. Bars at or above 0.9 "
       "are kept by the routed labeler at τ = 0.9; the rest go to the human. Black ticks: the accuracy a "
       "perfectly calibrated labeler would have.", width=5.6 * inch)
tm, tf = ACC["tags_missed_injections"], ACC["tags_false_alarms"]
tag_rows = [["Text feature", "% of injections", "% of misses", "% of benign", "% of false alarms"]]
for key, name in [("override_phrase", "Override phrase (\"ignore\", \"disregard\")"),
                  ("sensitive_request", "Sensitive request (\"password\", \"confidential\")"),
                  ("completion_leak", "Completion leak (\"write a story that begins with…\") †"),
                  ("task_format", "Task format (\"In this task, you are given…\") †"),
                  ("persona", "Persona (\"you are\", \"act as\", \"from now on\")"),
                  ("mentions_instructions", "Mentions instructions or rules (H2's wording)")]:
    rows_ = [tm[key]["all_injections_pct"], tm[key]["missed_pct"], tf[key]["all_benign_pct"], tf[key]["false_alarm_pct"]]
    tag_rows.append([name] + [f"{v:g}" for v in rows_])
table(tag_rows, [3.0 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch, 1.0 * inch], center_from=1)
caption("Table 3. Where the labeler's errors fall. † Features defined after reading the disagreements "
        "(post hoc, descriptive only).")
P("The errors have a pattern (Table 3). Texts with an explicit override phrase are a quarter of all "
  "injections but only 5% of misses. Half of all misses are <i>completion leaks</i>, harmless-looking "
  "writing tasks that make the model reveal something (for example, a story that must begin with \"The "
  "instructions are irrelevant\"). Nearly half of all false alarms are task-format prompts from "
  "instruction datasets (\"In this task, you are given a social media post. Classify it as…\"). The "
  "labeler confuses giving an AI instructions with trying to override them, and misses attacks that do "
  "not look like attacks. Reading the 21 disagreements above confidence 0.95, 14 are clear labeler errors "
  "(including textbook jailbreaks called benign with near certainty) and 7 are debatable conventions of "
  "the dataset; none is clearly a dataset error.")

P("5.3 LLM labels erase the active-learning advantage (H3)", H2)
lm = COMP["llm"]
figure("feedback_loop.png", "Figure 4. Left: wrong labels among each strategy's 600, by direction. Right: "
       "texts in each labeled set with a near-copy in the same set. Means of 10 seeds.")
P(f"With LLM labels, uncertainty sampling's curve area fell by 0.041 and random sampling's by 0.021; the "
  "difference in loss, +0.020 (95% CI [0.013, 0.027]), held on all ten seeds (Holm-adjusted p = 0.0004), "
  "and H3 is supported. Over the whole curve, uncertainty sampling fell below random on 9 of 10 seeds, a "
  "difference that does not survive correction (Holm p = 0.061); at 600 labels it ended below random on "
  f"all ten ({f3(lm.loc['uncertainty','f1_at_600'])} against {f3(lm.loc['random','f1_at_600'])}, "
  "exploratory). The accurate summary is that LLM labels erase the advantage, not that they reverse it.")
P("Figure 4 shows the mechanism. Uncertainty sampling queries the borderline texts, and the LLM's misses "
  f"are borderline: {lm.loc['uncertainty','wrong_label_missed_injection']:.0f} of its 600 labels on average "
  f"are injections labeled benign, against {lm.loc['random','wrong_label_missed_injection']:.0f} for random, "
  "and 36% of all its labels are wrong against 9%. Those wrong labels keep the same region ambiguous, so "
  f"the strategy queries more of the same texts: its labeled set holds "
  f"{lm.loc['uncertainty','near_copies_in_labeled_set']:.0f} texts with a near-copy (cosine ≥ 0.95) "
  f"elsewhere in the set, against {p.loc['uncertainty','near_copies_in_labeled_set']:.0f} with dataset "
  "labels. The labeler's blind spot is not diluted, as it is for random sampling, but concentrated.")
P(f"The hybrid strategy is robust. Its k-center step never chose a near-copy (the "
  f"{lm.loc['hybrid','near_copies_in_labeled_set']:.1f} per set on average all sit in the random starting "
  "20), so its label errors are spread across the space rather than concentrated, and a linear model "
  f"tolerates spread-out noise far better. With LLM labels it ends at {f3(lm.loc['hybrid','f1_at_600'])} "
  f"against random's {f3(lm.loc['random','f1_at_600'])} (10 of 10 seeds at 600 labels, exploratory), and "
  "it has the best final F1 under every labeler. Diversity sampling alone is the weakest strategy in "
  "distribution: its labeled set is 93% benign, and its F1 with LLM labels falls as it labels more.")

P("5.4 Routing restores the advantage; the human share is the cost (H4)", H2)
eq90 = EQUIV[EQUIV.labeler == "routed90"].set_index("strategy")
P("At the pre-registered τ = 0.9, routed labels are statistically equivalent to dataset labels for all "
  "four strategies: every 90% confidence interval of the F1 difference at 600 labels lies within ±0.005 "
  "(Table 4). Uncertainty sampling's advantage over random returns (curve area +0.011, Holm p = 2.5e-5), "
  f"its wrong labels fall to {rt.loc['uncertainty','wrong_label_missed_injection'] + rt.loc['uncertainty','wrong_label_false_injection']:.0f} "
  f"and its near-copies to {rt.loc['uncertainty','near_copies_in_labeled_set']:.0f}, the level of random "
  "sampling. But the human supplies "
  f"{SWEEP[(SWEEP.tau == 0.9)].human_pct.min():.0f}–{SWEEP[(SWEEP.tau == 0.9)].human_pct.max():.0f}% of the "
  "labels, above the 30% H4 allowed, so H4 is not supported as registered: the accuracy half holds and the "
  "effort half fails. Active learning and routing compete for the same texts: the borderline texts "
  "uncertainty sampling wants are the ones the LLM is unsure about, so it sends 56% of its labels to the "
  "human against random's 37%.")
eq_rows = [["Strategy", "Threshold", "F1 difference at 600", "90% CI", "Equivalent (±0.005)"]]
for _, r in EQUIV.iterrows():
    eq_rows.append([r.strategy, r.labeler.replace("routed", "0."), f"{r.mean_diff:+.4f}",
                    f"[{r.ci90_low:+.4f}, {r.ci90_high:+.4f}]", "yes" if r.equivalent else "no"])
table(eq_rows, [1.2 * inch, 0.9 * inch, 1.4 * inch, 1.8 * inch, 1.3 * inch], center_from=1)
caption("Table 4. Routed minus dataset labels, test F1 at 600 labels, paired over 10 seeds.")
P("Counting labels the way they are paid for changes the picture (Figures 5 and 6). To reach 99% of the "
  "full-data F1, uncertainty sampling with routing needs a median of 41 human labels at τ = 0.9 and 27 at "
  "τ = 0.8, against 260 for random sampling with every label from a person and 60 for uncertainty "
  "sampling with every label from a person. Across thresholds, the hybrid strategy gives the best "
  "detector for every level of human effort: at τ = 0.7 it reaches F1 0.985 with 23% of its labels from "
  "the human, above random sampling with all of them (0.978). The full sweep is in Appendix B.")
figure("threshold_sweep.png", "Figure 5. Share of the 600 labels supplied by the human against test F1 at "
       "600, across routing thresholds (point labels). Left end: LLM labels only; right end: dataset labels "
       "only. Diversity, below 0.95 at every threshold, is omitted.", width=5.4 * inch)
figure("human_labels_to_target.png", "Figure 6. Human labels needed to reach 99% of the full-data test F1, "
       "median of 10 seeds. Light bars: labels supplied by the LLM.", width=5.6 * inch)
P(f"A labeler's blind spot survives in places the test set does not show. On the lab attacks, uncertainty "
  "sampling with LLM labels scores 0.305 lower F1 than random sampling (0 of 10 seeds better). At τ = 0.8, "
  "where its test F1 is already equivalent to dataset labels, it is still 0.058 lower; only at τ = 0.9 "
  "does the gap close. A threshold chosen on in-distribution F1 alone would hide this.")

P("5.5 Detectors do not transfer, and the hardest attacks are disguised ones (H5)", H2)
b = TRANSFER.loc["hybrid / routed90"]
P(f"The best detector, chosen by a rule fixed before computing transfer (highest mean test F1 at 600 "
  f"across all runs: hybrid with routing at 0.9), catches {pct(b.test_recall)} of test injections, "
  f"{pct(b.lab_attack_recall, 1)} of the lab attacks and {pct(b.deepset_recall)} of deepset's (Table 5). "
  "Lab recall is 20 points below test recall (18.8 to 21.5 across seeds), double the 10-point bar fixed "
  "in advance, so H5's first part is supported. Its second part, that payloads written as document text "
  "would be caught least, is not: they were caught "
  f"{pct(b.lab_indirect_recall)} of the time, against {pct(b.lab_leakage_recall)} of leakage attempts "
  f"and {pct(b.lab_direct_recall)} of direct attacks.")
figure("transfer_recall.png", "Figure 7. Recall by evaluation set and lab attack type for the best detector, "
       "two references and the full-data model, mean of 10 seeds.", width=5.6 * inch)
tr_rows = [["Measure", "Hybrid, routed 0.9 (best)", "Uncertainty, dataset", "Hybrid, LLM only", "All 8,084 dataset labels"]]
for col, name, fmt in [("test_recall", "Test recall", f3), ("test_far_recall", "Far-subset recall", f3),
                       ("lab_attack_recall", "Lab: all 45 attacks", f3), ("lab_direct_recall", "Lab: direct", f3),
                       ("lab_indirect_recall", "Lab: indirect (document payloads)", f3),
                       ("lab_leakage_recall", "Lab: leakage", f3),
                       ("lab_benign_questions_flagged", "Lab: benign questions flagged", pct),
                       ("lab_policy_docs_flagged", "Lab: policy documents flagged", pct),
                       ("lab_auroc", "Lab AUROC", f3), ("deepset_recall", "deepset recall", f3),
                       ("deepset_en_recall", "deepset recall, English", f3),
                       ("deepset_non_en_recall", "deepset recall, other languages", f3),
                       ("deepset_roleplay_recall", "deepset role-play items caught", f3),
                       ("deepset_auroc", "deepset AUROC", f3)]:
    tr_rows.append([name] + [fmt(TRANSFER.loc[d, col]) for d in
                             ["hybrid / routed90", "uncertainty / perfect", "hybrid / llm", "all 8,084 dataset labels"]])
table(tr_rows, [2.3 * inch, 1.15 * inch, 1.1 * inch, 1.0 * inch, 1.15 * inch], center_from=1)
caption("Table 5. Transfer to the outside sets, mean of 10 seeds (the full-data model is a single fit).")
P("The pattern is about the form of an attack, not its intent. Every lab document payload addresses the "
  "AI explicitly (\"IMPORTANT INSTRUCTION FOR ANY AI ASSISTANT…\"), which is the form of the training "
  "set's injections. The attacks the best detector missed on all ten seeds hide the instruction inside "
  "an ordinary request: a real question with an extra line, a translation exercise, a fill-in-the-blank, "
  "string concatenation, a fake notice, an urgent ticket (Appendix C). These seven direct attacks are "
  "missed on every seed by all four detectors in Table 5, including the one trained on every label, so "
  "neither more labels nor better-chosen labels fix them. On deepset, the ranking transfers (AUROC near "
  "0.9) but the threshold and the definition do not: every detector misses all four role-play items, "
  "which deepset calls injections and the training data does not, and recall on non-English texts is "
  "near zero. LLM labels hurt transfer most: hybrid with LLM labels catches 56% of the lab attacks, 23 "
  "points below the same strategy with routing.")

P("5.6 Statistical summary", H2)
prim = TESTS[TESTS.family == "primary"]
st_rows = [["Comparison (curve area)", "Mean diff.", "95% CI", "Seeds won", "t-test p", "Wilcoxon p", "Holm p"]]
for _, r in prim.iterrows():
    st_rows.append([escape(r.comparison).replace("-&gt;", "→"), f"{r.mean_diff:+.4f}", f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}]",
                    f"{int(r.wins)}/10", pval(r.t_p), pval(r.wilcoxon_p),
                    pval(r.holm_p) + ("" if r.significant else " (n.s.)")])
table(st_rows, [2.05 * inch, 0.7 * inch, 1.3 * inch, 0.55 * inch, 0.6 * inch, 0.75 * inch, 0.75 * inch],
      center_from=1, font_size=7.8)
caption("Table 6. The six primary comparisons, paired over 10 seeds and Holm-corrected together. With ten "
        "seeds the smallest possible two-sided Wilcoxon p is 0.002.")

# ================================================================ 6 DISCUSSION
P("6. Discussion", H1)
P("<b>Why an LLM labeler and uncertainty sampling interact badly.</b> A labeler that is 91% accurate "
  "would cost random sampling little: its errors are spread over the pool, and a regularised linear model "
  "averages them out. Uncertainty sampling removes that protection. Its queries are concentrated near the "
  "boundary, the LLM's errors are concentrated in the same place, and each wrong label keeps the region "
  "uncertain enough to be queried again. The imperfect-oracle literature models noise that is independent "
  "of the query [5, 6]; the case measured here, noise correlated with the very ambiguity active learning "
  "seeks, is the worst case for it. The practical consequence is that an LLM labeler should be judged not "
  "only by its accuracy but by where its errors lie relative to the detector's boundary.")
P("<b>Diversity as a guard.</b> The hybrid strategy's k-center step was not needed with clean labels "
  "(uncertainty sampling did slightly better over the whole curve), but it was decisive with noisy ones. "
  "Forbidding near-duplicates within the uncertain shortlist stops the loop without giving up the "
  "boundary focus. Diversity alone gives up that focus and is the weakest strategy in distribution; its "
  "good outside-set scores come from a shifted threshold, not better understanding, because a labeled set "
  "that is 93% benign makes balanced class weights flag readily.")
P("<b>Routing works because the labeler is calibrated.</b> Confidence-based routing is only as good as the "
  "confidence. Here the labeler's confidence tracked its accuracy closely, so a single threshold separated "
  "its reliable labels from the rest. Its most confident errors (21 above 0.95) still pass through, which "
  "is one reason the outside-set gap closes later than the test-set gap. A threshold should be chosen with "
  "an outside check, not on in-distribution F1 alone.")
P("<b>What this means for building a detector.</b> On this data, a practitioner with a budget of a few "
  "dozen human labels does best with a hybrid or uncertainty strategy and a calibrated LLM labeler whose "
  "unsure texts go to a person; an LLM labeler alone should not be paired with pure uncertainty sampling. "
  "None of these choices fixes transfer: the attacks that defeat a learned detector are the ones that do "
  "not look like attacks, and they need different data, not more of the same.")

# ================================================================ 7 HYPOTHESES
P("7. Hypotheses, Scored", H1)
table([
    ["#", "Pre-registered hypothesis", "Result", "Verdict"],
    ["H1", "Uncertainty sampling reaches 95% of full-data F1 with fewer than half the labels random needs",
     "All strategies reach 95% at 40 labels; at 99%, uncertainty 60 against random 260", "Not supported (target too easy)"],
    ["H2", "LLM agreement 80–90%; most errors benign texts about instructions or rules",
     "91.3% (κ 0.79); errors split 47% false alarms, 53% misses; false alarms are task prompts", "Not supported"],
    ["H3", "With LLM labels, uncertainty sampling loses more F1 than random sampling",
     "Loss difference +0.020 [0.013, 0.027], 10/10 seeds, Holm p 0.0004", "Supported"],
    ["H4", "Routing low-confidence texts reaches perfect-label F1 with under 30% human labels",
     "Equivalent F1 for all four strategies at τ = 0.9, but 31–56% human labels", "Not supported (effort)"],
    ["H5", "Outside recall well below in-distribution recall, lowest on document payloads",
     "Lab −20 points, deepset −88; document payloads caught 100%, direct attacks 51%", "Half supported"],
], [0.35 * inch, 2.45 * inch, 2.6 * inch, 1.3 * inch])
caption("Table 7. The five hypotheses fixed before any experiment, scored against their criteria.")

# ================================================================ 8 DEVIATIONS
P("8. Deviations from the Pre-registered Plan", H1)
P("The protocol never changed: 20 random starting labels, 20 per round, 600 in total, ten seeds, C = 1. "
  "Table 8 lists every addition and when it was made relative to the data it affects. Additions fixed "
  "before seeing their outcome can still test something; those made after can only describe.")
table([
    ["Addition", "Reason", "When"],
    ["Far subset (1,445 test texts with no pool text at cosine ≥ 0.90)", "The audit found near-copies across the split", "After the embedding audit, before any learning curve"],
    ["99% targets (test and far) and curve area 20–600", "Random sampling reached the 95% target almost at once", "After the random curve, before any active-learning run"],
    ["Labeler answer format (letters, pre-filled reply, A = injection)", "Chosen on 40 development texts, excluded from scoring", "Before labeling the pool"],
    ["H4 made precise: τ = 0.9, F1 within ±0.005, under 180 of 600 human labels", "The wording allowed several readings", "Before the first routed run"],
    ["Human labels to target", "Per-label human share hides total cost", "After the first routed result"],
    ["Test family: six primary comparisons, Holm-corrected; the rest exploratory", "The plan fixed hypotheses, not tests", "At analysis time, before any p-value"],
    ["Error features for completion leaks and task-format prompts", "Patterns seen in the disagreements", "After reading them; descriptive only"],
    ["Best-detector rule and H5's two criteria made precise", "Needed an objective rule", "Before computing transfer"],
], [2.6 * inch, 2.1 * inch, 2.0 * inch])
caption("Table 8. Additions to the pre-registered plan.")
P("One interpretation was corrected during analysis: an early reading that LLM labels <i>reverse</i> "
  "uncertainty sampling's advantage does not survive the multiple-test correction, and the paper states "
  "only that they erase it.")

# ================================================================ 9 LIMITATIONS
P("9. Limitations", H1)
bullets([
    "<b>One easy dataset.</b> Nearest-neighbour agreement of 98.9% means strategy differences live in the "
    "first few hundred labels. Harder, less templated data may change the size of every effect.",
    "<b>One of everything else:</b> one embedding model, one classifier with fixed regularisation, one "
    "labeling model and prompt, one batch size and budget.",
    "<b>The human is simulated by the dataset's labels</b>, which contain conflicting entries and debatable "
    "conventions.",
    "<b>Human cost is counted in labels, not time.</b> Routing sends the hardest texts to the person, so each "
    "routed label likely costs more than an average one; savings in time are probably smaller than savings "
    "in labels.",
    "<b>Small or mismatched outside sets.</b> The lab set has 45 attacks written by one author; deepset is a "
    "third German and uses a different definition of injection. Language tags are heuristic.",
    "<b>Ten seeds.</b> Enough for the main effects, not for small ones; the whole-curve comparisons under "
    "LLM labels did not survive correction.",
    "<b>The labeler is not perfectly repeatable</b> at temperature 0; its labels were frozen once and reused.",
])

# ================================================================ 10 CONCLUSION
P("10. Conclusion and Future Work", H1)
P("On this injection-detection task, choosing which texts to label mattered more than how many, but only "
  "with good labels. A free LLM labeler erased the benefit of uncertainty sampling by feeding it the "
  "labeler's own blind spot, and routing the labeler's unsure texts to a person restored it at a fraction "
  "of the human effort. A hybrid strategy that adds diversity to uncertainty was the robust choice under "
  "every labeler. Four directions follow: repeating the study on harder data where the ceiling is not "
  "reached in a few dozen labels; routing by disagreement between the labeler and the current detector, to "
  "catch confident labeler errors that a confidence threshold lets through; making uncertainty sampling "
  "noise-aware, for example by penalising near-copies of queried texts or re-querying labels the detector "
  "consistently contradicts; and measuring annotation time, so that cost is counted in minutes rather "
  "than labels.")

P("Note on tooling", H2)
P("This work was developed with AI assistance. Every reported result comes from running the code in the "
  "accompanying repository.")

# ================================================================ REFERENCES
P("References", H1)
for ref in [
    "[1] B. Settles. Active learning literature survey. Computer Sciences Technical Report 1648, University of Wisconsin–Madison, 2009.",
    "[2] D. D. Lewis and W. A. Gale. A sequential algorithm for training text classifiers. In <i>SIGIR</i>, 1994.",
    "[3] O. Sener and S. Savarese. Active learning for convolutional neural networks: A core-set approach. In <i>ICLR</i>, 2018.",
    "[4] J. T. Ash, C. Zhang, A. Krishnamurthy, J. Langford and A. Agarwal. Deep batch active learning by diverse, uncertain gradient lower bounds. In <i>ICLR</i>, 2020.",
    "[5] P. Donmez and J. G. Carbonell. Proactive learning: Cost-sensitive active learning with multiple imperfect oracles. In <i>CIKM</i>, 2008.",
    "[6] V. S. Sheng, F. Provost and P. G. Ipeirotis. Get another label? Improving data quality and data mining using multiple, noisy labelers. In <i>KDD</i>, 2008.",
    "[7] Y. Yan, R. Rosales, G. Fung and J. Dy. Active learning from crowds. In <i>ICML</i>, 2011.",
    "[8] H. Mozannar and D. Sontag. Consistent estimators for learning to defer to an expert. In <i>ICML</i>, 2020.",
    "[9] S. Wang, Y. Liu, Y. Xu, C. Zhu and M. Zeng. Want to reduce labeling cost? GPT-3 can help. In <i>Findings of EMNLP</i>, 2021.",
    "[10] F. Gilardi, M. Alizadeh and M. Kubli. ChatGPT outperforms crowd workers for text-annotation tasks. <i>Proceedings of the National Academy of Sciences</i> 120(30), 2023.",
    "[11] X. Wang, H. Kim, S. Rahman, K. Mitra and Z. Miao. Human-LLM collaborative annotation through effective verification of LLM labels. In <i>CHI</i>, 2024.",
    "[12] L. Zheng et al. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In <i>NeurIPS Datasets and Benchmarks</i>, 2023.",
    "[13] S. Kadavath et al. Language models (mostly) know what they know. arXiv:2207.05221, 2022.",
    "[14] F. Perez and I. Ribeiro. Ignore previous prompt: Attack techniques for language models. <i>NeurIPS ML Safety Workshop</i>, 2022.",
    "[15] K. Greshake, S. Abdelnabi, S. Mishra, C. Endres, T. Holz and M. Fritz. Not what you've signed up for: Compromising real-world LLM-integrated applications with indirect prompt injection. In <i>AISec</i>, 2023.",
    "[16] Y. Liu, Y. Jia, R. Geng, J. Jia and N. Z. Gong. Formalizing and benchmarking prompt injection attacks and defenses. In <i>USENIX Security</i>, 2024.",
    "[17] Z. Nussbaum, J. X. Morris, B. Duderstadt and A. Mulyar. Nomic Embed: Training a reproducible long context text embedder. arXiv:2402.01613, 2024.",
    "[18] Llama Team, AI at Meta. The Llama 3 herd of models. arXiv:2407.21783, 2024.",
    "[19] S. Holm. A simple sequentially rejective multiple test procedure. <i>Scandinavian Journal of Statistics</i> 6(2):65–70, 1979.",
    "[20] D. J. Schuirmann. A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. <i>Journal of Pharmacokinetics and Biopharmaceutics</i> 15(6):657–680, 1987.",
    "[21] B. A. Nosek, C. R. Ebersole, A. C. DeHaven and D. T. Mellor. The preregistration revolution. <i>Proceedings of the National Academy of Sciences</i> 115(11):2600–2606, 2018.",
    "[22] xTRam1. safe-guard-prompt-injection. Hugging Face dataset, revision a3a877d6.",
    "[23] deepset. prompt-injections. Hugging Face dataset, revision 4f61ecb0, Apache-2.0.",
    "[24] Ollama. https://ollama.com.",
    "[25] F. Pedregosa et al. Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research</i> 12:2825–2830, 2011.",
]:
    P(ref, REF)

# ================================================================ APPENDICES
flush(PageBreak())
P("Appendix A. The Labeler Prompt", H1)
P("System message:", BODY)
P(escape(SYSTEM), MONO)
P("User message ({text} is replaced by the text being labeled):", BODY)
P(escape(INSTRUCTION).replace("\n", "<br/>"), MONO)
P("Assistant turn, pre-filled: <i>Answer:</i>. One token is generated at temperature 0; the probabilities "
  "of A and B among the top 10 candidates give p.", BODY)

P("Appendix B. The Threshold Sweep", H1)
sw_rows = [["Strategy", "Labeler", "Human %", "Wrong labels", "F1 at 600", "Lab F1 at 600", "Human labels to 99% (seeds)"]]
for s in ["random", "uncertainty", "hybrid", "diversity"]:
    for _, r in SWEEP[SWEEP.strategy == s].iterrows():
        lab = "LLM only" if r.tau == 0.5 else ("dataset" if r.tau == 1.0 else f"τ = {r.tau:g}")
        sw_rows.append([s, lab, f"{r.human_pct:g}", f"{r.wrong_labels:g}", f"{r.f1_600:.4f}",
                        f"{r.lab_f1_600:.3f}", f"{r.human_to_99} ({int(r.reached_99)}/10)"])
table(sw_rows, [0.95 * inch, 0.8 * inch, 0.7 * inch, 0.85 * inch, 0.8 * inch, 0.9 * inch, 1.7 * inch],
      center_from=2, font_size=7.6)
caption("Table B1. Every strategy at every routing threshold, means of 10 seeds (human labels to target: "
        "median, with seeds reaching it).")

P("Appendix C. Attacks Every Seed of the Best Detector Missed", H1)
P("The lab attacks that hybrid sampling with routing at 0.9 failed to flag on all ten seeds, with the "
  "mean injection probability it assigned. The seven direct attacks were also missed on every seed by the "
  "other three detectors in Table 5.", BODY)
miss_rows = [["Id", "Type", "Text", "Mean p"]]
for i in ["D02", "D04", "D07", "D08", "D09", "D14", "D15", "L01", "L13"]:
    miss_rows.append([i, LAB.loc[i, "kind"], escape(" ".join(str(LAB.loc[i, "text"]).split())),
                      f"{PER_TEXT.loc[i, 'mean_score']:.2f}"])
table(miss_rows, [0.4 * inch, 0.6 * inch, 5.0 * inch, 0.6 * inch], font_size=7.8)
caption("Table C1. Missed attacks. \"I have been PWNED\" is the hijack marker of the companion project; "
        "L01 and L13 try to extract a secret staff override code.")

P("Appendix D. Reproduction and Evidence", H1)
P("Every number in this paper is computed from files in the repository. The LLM's label for every pool "
  "text is in <i>results/labeler/evidence.csv</i>: its raw one-token answer, the probabilities of A and B, "
  "the dataset's label, the routing decision at τ = 0.9, and the text's row in the pinned source dataset, "
  "so any label can be checked against its text; <i>scripts/evidence.py</i> rebuilds the file and confirms "
  "that it reproduces the labeler's published confusion matrix and coverage. The texts each run queried "
  "are in <i>results/queries/</i>, so "
  "every final model can be rebuilt; learning curves are in <i>results/curves/</i> and the test tables in "
  "<i>results/analysis/</i>. The repository README lists the commands that regenerate each result, and "
  "<i>paper/build_paper.py</i> rebuilds this document from them.")


# ---------------------------------------------------------------- page furniture
def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Sans", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawRightString(letter[0] - 0.9 * inch, 0.55 * inch, str(doc.page))
    canvas.drawString(0.9 * inch, 0.55 * inch, TITLE_TEXT)
    canvas.restoreState()


def main() -> int:
    doc = SimpleDocTemplate(
        str(OUT), pagesize=letter, invariant=1,   # fixed metadata timestamp
        leftMargin=0.9 * inch, rightMargin=0.9 * inch, topMargin=0.85 * inch, bottomMargin=0.85 * inch,
        title=f"{TITLE_TEXT}: {SUBTITLE}", author="Faraz Ahmad",
        subject="Active learning for prompt-injection detection with an LLM labeler",
        creator="paper/build_paper.py (reportlab)",
    )
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
