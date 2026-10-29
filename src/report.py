"""Generate the six-page faculty case study from actual saved evaluation outputs."""
import json
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                               Image, PageBreak, KeepTogether)
from src import ROOT
from src.attendance import percentage, safe_bunks_now
from src.subjects import sample_subjects

GREEN = colors.HexColor("#426B57")
INK = colors.HexColor("#242923")
PALE = colors.HexColor("#E8EEE8")
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleA", fontName="Helvetica-Bold", fontSize=28, leading=32, textColor=INK, spaceAfter=18))
styles.add(ParagraphStyle(name="SectionA", fontSize=19, leading=24, textColor=GREEN, spaceAfter=15, fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="BodyA", fontSize=10, leading=15, textColor=INK, spaceAfter=10))
styles.add(ParagraphStyle(name="SmallA", fontSize=8, leading=11, textColor=colors.HexColor("#73796F"), spaceAfter=8))


def p(text, style="BodyA"):
    return Paragraph(text, styles[style])


def table(rows, widths=None):
    wrapped = [[p(escape(str(cell)), "SmallA") for cell in row] for row in rows]
    result = Table(wrapped, colWidths=widths, hAlign="LEFT", repeatRows=1)
    result.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), PALE),
                               ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("LINEBELOW", (0, 0), (-1, -1), .4, colors.HexColor("#E5E7E1")),
                               ("TOPPADDING", (0, 0), (-1, -1), 7),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return result


def chart(name, width, height):
    return Image(str(ROOT / "outputs" / f"{name}.png"), width=width, height=height, kind="proportional")


def footer(canvas, doc):
    canvas.setStrokeColor(colors.HexColor("#E5E7E1"))
    canvas.line(48, 42, 547, 42)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREEN)
    canvas.drawString(48, 29, "ATTENDSENSE  /  Artificial Intelligence Case Study  /  VIT Vellore")
    canvas.drawRightString(547, 29, str(doc.page))


def architecture():
    drawing = Drawing(490, 128)
    for row, labels in enumerate([
        ("Semester JSON", "Exact attendance engine", "Semester / Planner"),
        ("Held-out synthetic past", "Certainty check -> ML", "Risk Analysis"),
        ("Synthetic generator", "Train / evaluate", "Insights / PDF")]):
        y = 90 - row * 40
        for col, label in enumerate(labels):
            x = col * 169
            drawing.add(Rect(x, y, 150, 29, rx=5, fillColor=PALE,
                             strokeColor=colors.HexColor("#E5E7E1")))
            drawing.add(String(x+75, y+11, label, fontName="Helvetica", fontSize=8,
                               textAnchor="middle", fillColor=INK))
            if col < 2:
                drawing.add(Line(x+152,y+14,x+165,y+14,strokeColor=GREEN))
                drawing.add(Polygon([x+165,y+14,x+161,y+17,x+161,y+11],fillColor=GREEN,strokeColor=GREEN))
    return drawing


def main():
    metrics = json.loads((ROOT / "models/metrics.json").read_text())
    selected = metrics["selected_model"]
    result = metrics["models"][selected]
    story = [p("ATTENDSENSE / ACADEMIC PLANNING", "SmallA"),
             p("Attendance intelligence.<br/>A clearer semester.", "TitleA"),
             p("1. Introduction", "SectionA"),
             p("Attendance management is a planning problem as well as a record-keeping task. A current percentage describes classes already conducted, but does not tell a student how future absences change eligibility. Two students with the same percentage can have different recovery options when the remaining class counts differ."),
             p("AttendSense brings subject-wise tracking, exact attendance planning, a what-if simulator and an educational shortage predictor into a local Streamlit application. The central question is practical: how many classes may be missed while still meeting a chosen attendance target?"),
             p("The distinction between certainty and uncertainty is explicit. Mathematical calculations determine attendance feasibility and absence allowances. Machine learning estimates final shortage only when several future outcomes remain possible. It does not replace an exact calculation."),
             p("Scope and demonstration context", "Heading3"),
             p("The editable demonstration contains 11 separate theory, lab and skills courses supplied from a VTOP snapshot for a final-year B.Tech student at VIT Vellore. Percentages are recalculated from counts. Remaining classes and past attendance trends were not supplied and are never fabricated. The default target of 75% is configurable per subject, rather than asserted as an institutional rule."),
             p("This is a course case study, not a university deployment. No login, scraping, cloud service or external language model is used. Student edits remain in a local JSON file; the ML dataset is generated separately."),
             table([["Exact planning", "Predictive experiment"],
                    ["Counts + target + known remaining classes", "Past attendance patterns + trained model"],
                    ["Deterministic, independently testable", "Synthetic-data estimates; real-world validity unknown"]], [245, 245]),
             Spacer(1, 15),
             p("Guiding principle: use mathematics for certainty, machine learning for uncertainty, and transparency for both.", "SmallA"),
             PageBreak(), p("2. Literature", "SectionA"),
             p("Traditional attendance monitoring records whether a student attends scheduled sessions. A percentage dashboard supports inspection but needs explicit future-class calculations to answer planning questions. This project treats those calculations as arithmetic, without claiming them as an ML innovation."),
             p("Educational data mining studies patterns in student records for tasks such as achievement and dropout prediction. The UCI Student Performance dataset includes demographic, school and grade information from two Portuguese schools [1]. Its documentation explicitly notes the relationship between earlier and final grades, illustrating why prediction-time feature availability matters. It is useful background, but does not provide the sequential future attendance events needed here."),
             p("The UCI Predict Students' Dropout and Academic Success dataset frames a different higher-education classification task [2]. It supplies context for student-risk prediction, rather than a direct attendance-shortage training set. Neither UCI dataset nor the user's semester records is included in AttendSense training."),
             p("Breiman's Random Forest method combines randomized tree predictors [3]. It motivates testing an ensemble capable of nonlinear relationships alongside an interpretable linear classifier. Model choice is empirical: Random Forest is a candidate, not a guaranteed winner. A DummyClassifier checks whether the learned models improve on an uninformative baseline."),
             p("3. Problem Statement / Objectives", "SectionA"),
             p("Given attended classes A, conducted classes C, remaining classes R and target fraction T, determine whether the final attendance target can be met, calculate the minimum additional attendance, and estimate shortage probability only if the final outcome is uncertain."),
             p("Objectives: maintain editable subject records; separate consecutive safe absences from semester-wide allowance; simulate final attendance interactively; compare reproducible ML baselines without target leakage; show evaluation evidence and limitations; and preserve local student data across application reruns."),
             p("The model predicts one binary outcome: 1 when simulated final attendance is strictly below the chosen threshold, and 0 otherwise. Equality meets the target. The project does not predict grades, dropout or disciplinary decisions."),
             PageBreak(), p("4. Methodology / Implementation", "SectionA"),
             p("Synthetic trajectories", "Heading3"),
             p(f"A fixed NumPy seed ({metrics['seed']}) generates {metrics['rows']:,} independent student-subject trajectories. Each has 35-100 total classes and a snapshot after at least 8 classes, before semester completion. Six equally represented behaviour families are consistent, irregular, improving, declining, occasional absence streaks and borderline attendance. Event outcomes are sampled sequentially from clipped attendance probabilities; labels are never assigned at random."),
             p("Targets are sampled from 65, 70, 75, 80 and 85%, with 75% most common. Weekly frequency is independently sampled from 1-5. Synthetic choices represent educational assumptions, not measured university behaviour. Both past and future event strings are retained for audit, but excluded from model features."),
             p("Feature engineering and leakage control", "Heading3"),
             p("At the snapshot, the system derives conducted and attended counts, current percentage, last-eight attendance rate, earlier historical rate, trailing absence streak and semester progress. Remaining classes, weekly frequency and threshold complete the ten-feature input. Historical rate uses all available past events if there are no earlier events outside the recent window. Future events determine final attendance and the shortage label only."),
             p("Preprocessing and evaluation", "Heading3"),
             p(f"A stratified 80/20 split reserves {metrics['test_rows']} trajectories for held-out testing. Five-fold stratified cross-validation operates solely on the {metrics['train_rows']} training rows. StandardScaler is inside each pipeline, preventing preprocessing leakage. Independent trajectories avoid repeated-student overlap across splits."),
             p("Candidates are a prior DummyClassifier, Logistic Regression (C=1, maximum 2,000 iterations), and Random Forest (240 trees, maximum depth 12, minimum leaf size 4). Mean training cross-validation F2 selects the model, weighting shortage recall more than precision while still penalizing false positives. A fixed 0.5 probability cutoff is used. No test-set tuning or probability calibration is performed."),
             p(f"The selected pipeline is {selected}. Held-out evaluation reports accuracy, precision, recall, F1, ROC-AUC and confusion matrix. Global permutation importance measures the reduction in test F1 over 10 shuffles; it is descriptive, not a local causal explanation."),
             p("Synthetic educational dataset, not actual VIT student records.", "SmallA"),
             PageBreak(), p("4. Methodology / Implementation (continued)", "SectionA"),
             p("Exact attendance engine", "Heading3"),
             table([["Quantity", "Definition"],
                    ["Current attendance", "100 x A / C (undefined if C = 0)"],
                    ["Required future attendance", "K = max(0, ceil(T x (C + R) - A))"],
                    ["Future absence allowance", "R - K, only when K <= R"],
                    ["Final attendance range", "100 x A/(C+R) to 100 x (A+R)/(C+R)"],
                    ["Scenario after X future attendances", "100 x (A+X)/(C+R), with 0 <= X <= R"],
                    ["Safe consecutive absences now", "max(0, floor(A/T) - C)"]], [176, 314]), Spacer(1, 12),
             p("Fraction-based comparisons prevent decimal rounding from changing equality decisions. All class counts must be nonnegative integers and A must not exceed C. Targets from 1-100% are accepted. Zero scheduled classes produce undefined attendance, not a manufactured 0%. If recovery is impossible, allowance is unavailable rather than a negative number."),
             p("Architecture and application workflow", "Heading3"),
             architecture(), Spacer(1, 8),
             p("Figure 1. Data flow: semester records are isolated from synthetic model training. All exact planning screens share the attendance engine.", "SmallA"),
             p("Subject edits are validated and written atomically. Deletion requires confirmation; resetting creates a timestamped backup. Unknown remaining class counts block projections. AI Risk Analysis uses the 300 held-out synthetic trajectories and their past attendance features, independently of personal semester records. The planner's scenario count is temporary until edited in My Semester."),
             p("Before inference, shortage probability becomes exactly 0 when even missing every remaining class meets the target, or exactly 1 when attending every remaining class cannot meet it. With no remaining classes, final status is exact. Otherwise the trained pipeline provides an estimate. Low (<30%), Moderate (30-<70%) and High (>=70%) are illustrative labels."),
             PageBreak(), p("5. Results", "SectionA"),
             p(f"Actual held-out evaluation: {metrics['test_rows']} synthetic trajectories", "Heading3")]
    rows = [["Model", "CV F2", "Accuracy", "Precision", "Recall", "F1", "AUC"]]
    for name, values in metrics["models"].items():
        rows.append([name] + [f"{values[k]:.3f}" for k in ["cv_f2", "accuracy", "precision", "recall", "f1", "roc_auc"]])
    story += [table(rows, [122, 55, 60, 65, 60, 60, 68]), Spacer(1, 10),
              p(f"Training CV selected {selected}. Its test accuracy is {result['accuracy']:.1%}, precision {result['precision']:.1%}, shortage recall {result['recall']:.1%}, F1 {result['f1']:.3f}, and ROC-AUC {result['roc_auc']:.3f}. These measure performance on simulated data from the same generator; they do not establish accuracy on real students."),
              Table([[chart("confusion_matrix", 240, 174), chart("model_comparison", 250, 174)]], colWidths=[245, 250]),
              p("Figure 2. Held-out confusion matrix (actual rows, predicted columns) and model precision/recall/F1 comparison. Figures are generated by the executed training pipeline.", "SmallA"),
              chart("feature_importance", 410, 230),
              p("Figure 3. Global permutation importance with shuffle variability. Correlated attendance features can share importance. Small or negative changes are not evidence of a protective causal effect.", "SmallA"),
              PageBreak(), p("5. Results (continued)", "SectionA"),
              p("Sample semester analysis", "Heading3")]
    sample_rows = [["Course", "Attended / conducted", "Exact %", "Safe bunks now"]]
    for s in sample_subjects():
        sample_rows.append([s["code"], f"{s['attended']} / {s['conducted']}", f"{percentage(s['attended'], s['conducted']):.2f}", safe_bunks_now(s["attended"], s["conducted"])])
    story += [table(sample_rows, [115, 160, 100, 115]), Spacer(1, 8),
              p("Table 2. Supplied demonstration snapshot at a configurable 75% target. Values are calculated from counts. Remaining classes are not supplied, so final projections and semester-wide absence allowances are deliberately unavailable. Safe bunks now is independent of the unknown semester horizon.", "SmallA"),
              p("Technical limitations and interpretation", "Heading3"),
              p("Synthetic behaviours omit health, timetable changes and many other influences. The generator's assumptions shape both training and testing, so a high internal score cannot support real-world claims. User-supplied trends may be inaccurate. Model probabilities are uncalibrated. Global feature importance cannot explain an individual prediction causally. Fixed risk categories are illustrative. Future work should use consented real longitudinal data, calibration and external validation before advising institutional decisions."),
              p("References", "Heading3"),
              p('[1] Cortez, P. and Silva, A. (2008). Using data mining to predict secondary school student performance. Dataset: UCI Student Performance. <link href="https://archive.ics.uci.edu/dataset/320/student+performance" color="#426B57">archive.ics.uci.edu/dataset/320/student+performance</link>', "SmallA"),
              p('[2] Martins, M. V. et al. (2021). Early prediction of student\'s performance in higher education: a case study. UCI Predict Students\' Dropout and Academic Success. <link href="https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success" color="#426B57">UCI dataset 697</link>. Supporting context; neither UCI dataset is training data.', "SmallA"),
              p('[3] Breiman, L. (2001). Random Forests. Machine Learning, 45, 5-32. <link href="https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf" color="#426B57">Original author manuscript</link>. References verified 29 October 2026.', "SmallA")]
    directory = ROOT / "reports"
    directory.mkdir(exist_ok=True)
    output = directory / "AttendSense_Case_Study.pdf"
    doc = SimpleDocTemplate(str(output), pagesize=(595.28, 841.89), leftMargin=48, rightMargin=48,
                            topMargin=45, bottomMargin=55, title="AttendSense - Attendance Intelligence Case Study", author="AttendSense")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Generated {output}")


if __name__ == "__main__":
    main()
