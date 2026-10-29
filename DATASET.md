# Dataset provenance and assumptions

**Synthetic educational dataset, not actual VIT student records.**

`python -m src.generate_data` writes 1,500 independent student-subject trajectories
using NumPy's generator with seed 42. No semester JSON or external dataset is read.
Each trajectory is a new identity with one snapshot; there are no shared students
across records. Stratified row splitting is therefore appropriate.

## Generative assumptions

Each simulated semester has 35–100 total classes. The snapshot follows 8 or more
classes and precedes the final class. There are 250 trajectories per behaviour:

| Behaviour | Assumption |
|---|---|
| Consistent | Base attendance probability uniformly 0.84–0.98 |
| Irregular | Base 0.55–0.89 with per-event Gaussian noise, SD 0.17 |
| Improving | Base 0.55–0.89 with positive semester slope 0.12–0.35 |
| Declining | Same base with negative slope of that magnitude |
| Streaks | Base 0.55–0.89; 7% per-event chance of 2–4 low-probability events |
| Borderline | Base within ±0.06 of the selected threshold |

For improving/declining patterns, the slope multiplies `(event_index/total - 0.5)`.
Streak shocks reduce probability by 0.5 and can restart before completing. All event
probabilities are clipped to 0.05–0.99. Events are sampled in sequence; outcomes,
not probabilities, determine counts. Future events follow the same latent process.
No records are filtered or relabelled to improve model scores.

Thresholds 65/70/75/80/85% have probabilities 0.08/0.12/0.60/0.12/0.08. Weekly
frequency is independently uniform over 1–5 and does not influence the generated
outcome. These are explicit modelling assumptions, not observed institutional facts.

## Features and audit columns

The feature allowlist is `src.generate_data.FEATURES`:

| Feature | Snapshot-time definition |
|---|---|
| conducted, attended | Length and sum of past events |
| current_percentage | 100 × attended / conducted |
| remaining | Scheduled total minus conducted |
| recent_rate | Mean of last min(8, conducted) past events, fraction 0–1 |
| historical_rate | Mean of past events before recent window; all past if empty |
| consecutive_absences | Number of trailing zeros in the past sequence |
| semester_progress | conducted / (conducted + remaining) |
| weekly_frequency | Scheduled classes per week |
| threshold | Required percentage |

Audit columns are `trajectory_id`, `behaviour`, `past_events`, `future_events`,
`final_attended`, `final_percentage`, and `shortage`. They **never enter training**.
When loading CSV audit strings, use `dtype={"past_events": str, "future_events": str}`
to preserve leading zeros. The target is exactly
`100 * final_attended < threshold * total_classes`; equality is not a shortage.

## Evaluation protocol

An 80/20 stratified split (seed 42) gives 1,200 training and 300 test rows.
Five-fold stratified training cross-validation compares prior DummyClassifier,
scaled Logistic Regression, and scaled Random Forest. Scaling is inside pipelines.
The model with the highest mean CV F2 is selected before inspecting test results.
F2 prioritizes shortage recall while retaining a precision penalty. Its advantage
over other candidates can be small relative to fold variability; it is not proof of
statistical superiority. There is no search over seeds or test-set tuning.

AI Risk Analysis runs predictions on the original 300 held-out examples, with past
features populated from this dataset. It never reads personal semester attendance.
The CSV hash is checked against saved training metadata before reconstructing the
original split. Model Insights exposes all 1,500 records with partition labels;
future outcomes are available for evaluation only and excluded from inference.

Held-out test metrics evaluate the raw model at a fixed 0.5 cutoff, separately from
the mathematical safeguards used at inference. Permutation importance uses test F1,
10 shuffles, and seed 42. It does not select or tune the model. Outputs include the
dataset SHA-256, sklearn version, features, CV variability and confusion matrices.

Probabilities are uncalibrated model estimates. The 30%/70% risk labels are
illustrative. Synthetic train/test performance does not establish real-world
predictive value. Counts, targets and timelines outside the generator's domain
require caution. No real-student, external or temporal validation has occurred.

## Supporting literature (not training sources)

- [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance)
- [UCI Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- [Breiman (2001), Random Forests, author manuscript](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)

These address related educational prediction or classifier methodology, rather
than supplying the required future class attendance sequences. Verified 29 October 2026.
