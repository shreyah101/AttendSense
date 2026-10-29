"""AttendSense local student planner. Run with streamlit run app.py."""
import json
from uuid import uuid4
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from src import ROOT
from src.attendance import percentage, plan, projected, safe_bunks_now, meets_target, certain_risk
from src.subjects import load_subjects, save_subjects, sample_subjects, PATH
from src.predict import predict
from src.synthetic import load_examples

st.set_page_config(page_title="AttendSense | Your semester", page_icon="◫", layout="wide")
st.markdown("""<style>
:root{--green:#426B57;--ink:#242923}
.stApp{background:#F7F6F2;color:var(--ink);font-family:'DM Sans','Segoe UI',sans-serif}
header[data-testid="stHeader"]{background:#F7F6F2}
.block-container{max-width:1280px;padding-top:2.3rem;padding-bottom:3rem}
h1,h2,h3{font-family:'DM Sans','Segoe UI',sans-serif;letter-spacing:-.035em;color:#242923}
h1{font-size:2.65rem!important;font-weight:600!important}
[data-testid="stSidebar"]{background:#E8EEE8;border-right:1px solid #E5E7E1}
[data-testid="stMetric"]{background:white;border:1px solid #E5E7E1;border-radius:14px;padding:20px}
[data-testid="stMetricValue"]{font-family:'DM Mono',Consolas,monospace;color:#426B57;font-size:1.85rem}
[data-testid="stVerticalBlockBorderWrapper"]>div{border-color:#E5E7E1!important;border-radius:14px!important}
.eyebrow{font-size:12px;letter-spacing:.16em;color:#73796F;text-transform:uppercase;margin-bottom:12px}
.subtle{color:#73796F;font-size:15px}.rule{border-top:1px solid #E5E7E1;margin:20px 0}
button{border-radius:9px!important} [data-testid="stForm"]{background:#FFFFFF}
@media(max-width:650px){h1{font-size:2rem!important}.block-container{padding:4rem 1rem 1rem}[data-testid="stMetric"]{padding:12px}}
</style>""", unsafe_allow_html=True)


def fmt(value):
    return "Not set" if value is None else f"{value:.2f}%"


def commit(records):
    try:
        save_subjects(records)
    except (ValueError, OSError, KeyError) as exc:
        st.error(str(exc))
        return
    st.session_state["notice"] = "Semester records saved locally."
    st.rerun()


def selector(records, key):
    return records[st.selectbox("Course", range(len(records)),
                              format_func=lambda i: f"{records[i]['code']} · {records[i]['name']}", key=key)]


def planning_metrics(subject):
    a, c, r, t = [subject[k] for k in ("attended", "conducted", "remaining", "threshold")]
    cols = st.columns(3)
    cols[0].metric("Safe bunks now", safe_bunks_now(a, c, t))
    if r is None:
        cols[1].metric("Future absence allowance", "Not set")
        cols[2].metric("Required future classes", "Not set")
        st.caption("Enter remaining classes to calculate semester outcomes. Safe bunks now uses only your current counts.")
    else:
        p = plan(a, c, r, t)
        cols[1].metric("Future absence allowance", p.allowance if p.feasible else "Impossible")
        cols[2].metric("Required future classes", p.required)
        st.caption(f"Final attendance range: {fmt(p.minimum)} to {fmt(p.maximum)}. "
                   + (f"Attend at least {p.required} of the remaining {r} classes." if p.feasible else "Even attending every remaining class cannot reach the target."))
    st.caption("Safe bunks now = consecutive misses while staying at target immediately afterward. Future allowance assumes sufficient attendance in the other remaining classes.")


def edit_subject(subject, records):
    with st.expander("Edit this course", expanded=False):
        with st.form(f"edit_{subject['id']}"):
            left, right = st.columns(2)
            code = left.text_input("Course code", subject["code"])
            name = right.text_input("Course name", subject["name"])
            kind = left.selectbox("Course type", ["Theory", "Lab", "Skills"], index=["Theory", "Lab", "Skills"].index(subject["type"]))
            t = right.number_input("Target (%)", 1.0, 100.0, float(subject["threshold"]), step=.5)
            a = left.number_input("Attended classes", 0, value=subject["attended"], step=1)
            c = right.number_input("Conducted classes", 0, value=subject["conducted"], step=1)
            has_remaining = st.checkbox("Remaining classes are known", subject["remaining"] is not None)
            r = st.number_input("Remaining classes", 0, value=subject["remaining"] or 0, step=1)
            if st.form_submit_button("Save course", type="primary"):
                replacement = dict(subject, code=code.strip(), name=name.strip(), type=kind, attended=a,
                                   conducted=c, threshold=t, remaining=r if has_remaining else None)
                commit([replacement if s["id"] == subject["id"] else s for s in records])
        confirmed = st.checkbox("I confirm deletion of this course", key=f"confirm_{subject['id']}")
        if st.button("Delete course", disabled=not confirmed, key=f"delete_{subject['id']}"):
            commit([s for s in records if s["id"] != subject["id"]])


def manage(records):
    with st.expander("Add a course"):
        with st.form("add"):
            code = st.text_input("New course code")
            name = st.text_input("New course name")
            kind = st.selectbox("New course type", ["Theory", "Lab", "Skills"])
            cols = st.columns(3)
            a = cols[0].number_input("Classes attended", 0, step=1)
            c = cols[1].number_input("Classes conducted", 0, step=1)
            t = cols[2].number_input("Attendance target (%)", 1.0, 100.0, 75.0)
            known = st.checkbox("Set remaining classes")
            r = st.number_input("Classes remaining", 0, step=1)
            if st.form_submit_button("Add course"):
                commit(records + [dict(id=uuid4().hex, code=code.strip(), name=name.strip(), type=kind,
                                       attended=a, conducted=c, threshold=t, remaining=r if known else None)])
    with st.expander("Backup and reset"):
        st.download_button("Download semester backup", json.dumps(records, indent=2), "semester_backup.json", "application/json")
        confirm = st.checkbox("Replace all records with the 11 sample courses (current file will be backed up)")
        if st.button("Reset to sample records", disabled=not confirm):
            try:
                from datetime import datetime
                backup = PATH.with_name(f"semester_subjects.{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.bak")
                backup.write_bytes(PATH.read_bytes())
                commit(sample_subjects())
            except OSError as exc:
                st.error(f"Reset cancelled: backup failed. {exc}")


def semester(records):
    st.title("Your semester, at a glance.")
    st.markdown('<p class="subtle">Track your classes, protect your attendance, and stay ahead of the semester.</p>', unsafe_allow_html=True)
    if records:
        a, c = sum(s["attended"] for s in records), sum(s["conducted"] for s in records)
        cols = st.columns(3)
        cols[0].metric("Courses this semester", len(records))
        cols[1].metric("Class-weighted attendance", fmt(percentage(a, c)))
        cols[2].metric("Below individual target", sum(meets_target(s["attended"], s["conducted"], s["threshold"]) is False for s in records))
        st.subheader("The course notebook")
        rows = []
        for s in records:
            pct = percentage(s["attended"], s["conducted"])
            status = meets_target(s["attended"], s["conducted"], s["threshold"])
            rows.append({"Code":s["code"], "Course":s["name"], "Type":s["type"], "Attended":s["attended"],
                         "Conducted":s["conducted"], "Attendance (%)":pct, "Target (%)":s["threshold"],
                         "Remaining":str(s["remaining"]) if s["remaining"] is not None else "Not set",
                         "Status":"No classes yet" if status is None else "At target" if status else "Below target"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            "Attendance (%)":st.column_config.ProgressColumn(format="%.2f%%", min_value=0, max_value=100)}, height=425)
        ranked = sorted((s for s in records if s["conducted"]), key=lambda s: abs(percentage(s["attended"], s["conducted"])-s["threshold"]))
        if ranked:
            st.caption("Closest to their targets: " + " · ".join(f"{s['code']} ({percentage(s['attended'], s['conducted'])-s['threshold']:+.2f} percentage points)" for s in ranked[:3]))
        st.subheader("How many classes can you still miss?")
        subject = selector(records, "semester_course")
        st.caption(f"{subject['type']} / {subject['attended']} attended of {subject['conducted']} conducted / target {subject['threshold']:g}%")
        planning_metrics(subject)
        edit_subject(subject, records)
    else:
        st.info("Your notebook is empty. Add a course below.")
    manage(records)


def risk():
    st.title("Explore synthetic attendance risk.")
    st.caption("Run the trained model on simulated student-subject records from the held-out test set.")
    st.info("Synthetic educational dataset, not actual VIT student records. Predictions have not been validated on actual university student records.")
    try:
        data = load_examples()
    except (OSError, ValueError, KeyError) as exc:
        st.error(str(exc))
        return
    examples = data[data.partition == "Held-out test"]
    st.write(f"**{len(data):,} synthetic trajectories** · {len(data)-len(examples):,} training records · **{len(examples)} held-out examples** available here.")
    st.caption("These examples were excluded from model fitting. Your sample courses are used only in My Semester and the deterministic Bunk Planner.")
    behaviour = st.selectbox("Simulated behaviour", ["All behaviours"] + sorted(examples.behaviour.unique().tolist()))
    if behaviour != "All behaviours":
        examples = examples[examples.behaviour == behaviour]
    # Put uncertain outcomes first so the initial example demonstrates actual ML.
    examples = examples.assign(certain=[certain_risk(int(row.attended), int(row.conducted), int(row.remaining), float(row.threshold)) is not None for row in examples.itertuples()])
    examples = examples.sort_values(["certain", "trajectory_id"])
    identity = st.selectbox("Synthetic student-subject record", examples.trajectory_id.tolist(),
                            format_func=lambda i: f"SIM-{i:04d} · {examples.loc[examples.trajectory_id == i, 'behaviour'].iloc[0]}")
    row = examples.loc[examples.trajectory_id == identity].iloc[0]
    a, c, r, t = int(row.attended), int(row.conducted), int(row.remaining), float(row.threshold)
    cols = st.columns(4)
    cols[0].metric("Current attendance", fmt(percentage(a, c)))
    cols[1].metric("Attended / conducted", f"{a} / {c}")
    cols[2].metric("Remaining classes", r)
    cols[3].metric("Target", f"{t:g}%")
    planning_metrics(dict(attended=a, conducted=c, remaining=r, threshold=t))
    st.subheader("Simulated past attendance")
    cols = st.columns(4)
    cols[0].metric("Recent attendance", f"{row.recent_rate:.1%}")
    cols[1].metric("Earlier historical rate", f"{row.historical_rate:.1%}")
    cols[2].metric("Consecutive absences", int(row.consecutive_absences))
    cols[3].metric("Classes per week", int(row.weekly_frequency))
    st.caption("Rates and absence streaks come from the simulated past sequence. Recent = last 8 classes; historical = earlier classes, or all past classes if no earlier window exists.")
    with st.expander("Inspect the past class sequence"):
        st.write(" · ".join("Present" if event == "1" else "Absent" for event in row.past_events))
    try:
        result = predict(a, c, r, t, recent_rate=float(row.recent_rate),
                         historical_rate=float(row.historical_rate),
                         consecutive_absences=int(row.consecutive_absences), weekly_frequency=int(row.weekly_frequency))
    except (ValueError, OSError) as exc:
        st.error(str(exc))
        return
    exact = result.source == "Mathematical certainty"
    st.metric("Exact shortage risk" if exact else "Model-estimated final shortage probability", f"{result.probability:.1%}")
    st.write(f"**{result.level} risk** · " + (result.source if exact else "Trained model using this synthetic record's past attendance features"))
    if exact:
        (st.success if result.probability == 0 else st.warning)("The final target is guaranteed even if all remaining classes are missed." if result.probability == 0 else "Even attending every remaining class cannot reach the final target.")
    else:
        st.caption("Illustrative categories: Low <30%; Moderate 30–<70%; High ≥70%. Model probabilities are uncalibrated estimates, not guarantees.")
    st.write(f"This simulated record has a {row.recent_rate:.1%} recent attendance rate versus {row.historical_rate:.1%} earlier, with {int(row.consecutive_absences)} trailing absences. These describe the inputs, not a causal explanation of this individual prediction.")
    p = plan(a, c, r, t)
    st.write(f"For this example, attend at least **{p.required} of {r}** future classes to reach the target." if p.feasible else "For this example, recovery to the target is mathematically impossible with the scheduled classes.")
    with st.expander("Reveal the simulated final outcome (evaluation only)"):
        st.metric("Simulated final attendance", fmt(float(row.final_percentage)))
        st.write("Actual simulated outcome: **" + ("Shortage" if int(row.shortage) else "No shortage") + "**")
        st.caption("Future events, final attendance and the final shortage label are shown only for comparison. They are never passed to the predictor. The saved test metrics evaluate the raw model before mathematical safeguards.")


def planner(records):
    st.title("What if you miss the next class?")
    st.caption("A little planning, a clearer semester. This screen uses exact mathematics.")
    subject = selector(records, "planner_course")
    a, c, t = [subject[k] for k in ("attended", "conducted", "threshold")]
    r = st.number_input("Remaining classes for this scenario", min_value=0, value=subject["remaining"], step=1, key=f"scenario_r_{subject['id']}")
    st.caption("Scenario inputs stay in this planner. Save remaining classes in My Semester to update the permanent record.")
    if r is None:
        st.info("Remaining classes: Not set. Enter a count to explore outcomes.")
        return
    planning_metrics(dict(subject, remaining=r))
    x = st.slider("Planned attended classes", 0, r, r, key=f"attend_{subject['id']}_{r}") if r else 0
    cols = st.columns(3)
    cols[0].metric("Planned missed classes", r-x)
    cols[1].metric("Projected final attendance", fmt(projected(a, c, r, x)))
    cols[2].metric("Course target", f"{t:g}%")
    status = meets_target(a+x, c+r, t)
    if status is None:
        st.info("No classes are scheduled; attendance is undefined.")
        return
    (st.success if status else st.warning)("Your scenario reaches the target." if status else "Your scenario finishes below the target.")
    # Cap plotted samples for large inputs while the calculations remain exact.
    import numpy as np
    xs = sorted(set(np.linspace(0, r, min(r+1, 501), dtype=int).tolist() + [x]))
    ys = [projected(a, c, r, int(i)) for i in xs]
    fig, ax = plt.subplots(figsize=(9, 3.5), facecolor="#F7F6F2")
    ax.set_facecolor("#F7F6F2")
    colors = ["#426B57" if meets_target(a+int(i), c+r, t) else "#BA6659" for i in xs]
    ax.plot(xs, ys, color="#73796F", linewidth=1)
    ax.scatter(xs, ys, c=colors, s=16)
    ax.axhline(t, color="#D7A04B", linestyle="--", label=f"Target {t:g}%")
    ax.scatter([x], [projected(a, c, r, x)], s=100, facecolor="#242923", zorder=5, label="Your scenario")
    ax.set(xlabel="Future classes attended", ylabel="Final attendance (%)", ylim=(0, 103))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def insights():
    st.title("Behind the prediction.")
    st.caption("An educational experiment, with its workings on the page.")
    path = ROOT / "models/metrics.json"
    if not path.exists():
        st.warning("Run python -m src.generate_data and python -m src.train first.")
        return
    m = json.loads(path.read_text())
    st.info("Synthetic educational dataset, not actual VIT student records.")
    st.write(f"**{m['rows']:,} independent trajectories** · {m['train_rows']:,} training / {m['test_rows']:,} held-out test · stratified 80/20 split · seed {m['seed']}")
    st.write(f"Selected model: **{m['selected_model']}**. {m['selection']}.")
    with st.expander("Browse the synthetic dataset", expanded=False):
        try:
            data = load_examples()
            partition = st.selectbox("Dataset partition", ["All records", "Training", "Held-out test"])
            visible = data if partition == "All records" else data[data.partition == partition]
            st.caption(f"Showing {len(visible):,} synthetic records. AI Risk Analysis predicts on held-out examples only.")
            st.dataframe(visible.drop(columns=["past_events", "future_events"]), hide_index=True, width="stretch")
            st.download_button("Download synthetic dataset", data.to_csv(index=False), "synthetic_attendance.csv", "text/csv")
        except (OSError, ValueError, KeyError) as exc:
            st.error(str(exc))
    metric = m["models"][m["selected_model"]]
    cols = st.columns(5)
    for col, key in zip(cols, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
        col.metric(key.replace("_", " ").title(), f"{metric[key]:.3f}")
    st.dataframe(pd.DataFrame(m["models"]).T.drop(columns="confusion_matrix"), width="stretch")
    cols = st.columns(2)
    for col, name in zip(cols, ["confusion_matrix", "model_comparison"]):
        col.image(str(ROOT / "outputs" / f"{name}.png"), width="stretch")
    st.image(str(ROOT / "outputs/feature_importance.png"), width="stretch")
    st.caption("Permutation importance measures global held-out F1 changes, averaged over 10 shuffles; correlated features may share importance. It is not an individual or causal explanation.")
    with st.expander("Dataset distribution and limitations", expanded=True):
        st.image(str(ROOT / "outputs/class_distribution.png"), width="stretch")
        st.write("Past class sequences produce counts, last-eight attendance, earlier attendance and trailing absence streaks. Future events determine labels but never enter model inputs. Each row is an independent trajectory. Scaling is fitted inside training folds. Test metrics describe the raw model; deployment applies mathematical overrides separately.")
        st.write("Synthetic behaviours simplify real life. Probabilities are uncalibrated estimates, with no real-student validation or institutional endorsement. Weekly frequency is sampled independently and is not a causal risk factor. Targets outside 65–85%, very early snapshots and unusual timetables are outside the simulated domain.")
        st.markdown("Literature context: [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance), [UCI Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success), [Breiman: Random Forests](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf). These are references, not training data.")
    report = ROOT / "reports/AttendSense_Case_Study.pdf"
    if report.exists():
        st.download_button("Download faculty case study", report.read_bytes(), report.name, "application/pdf")


with st.sidebar:
    st.markdown("## AttendSense")
    st.caption("THE SEMESTER NOTEBOOK")
    screen = st.radio("Navigate", ["My Semester", "AI Risk Analysis", "Bunk Planner", "Model Insights"], label_visibility="collapsed")
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.caption("VIT Vellore · Editable semester sample")
    st.caption("75% is a configurable default. Set the actual requirement for each course.")
    st.caption("Local records · Offline planning")
st.markdown('<div class="eyebrow">AttendSense / Academic planning</div>', unsafe_allow_html=True)
if "notice" in st.session_state:
    st.success(st.session_state.pop("notice"))
if screen == "AI Risk Analysis":
    risk()
elif screen == "Model Insights":
    insights()
else:
    try:
        records = load_subjects()
    except (ValueError, OSError, KeyError, TypeError) as exc:
        st.error(f"Cannot read semester records. The file was preserved: {exc}")
        st.stop()
    if screen == "My Semester":
        semester(records)
    elif not records:
        st.info("Add a course in My Semester to use the planner.")
    else:
        planner(records)
