"""Training-only selection by cross-validated F2; one untouched test evaluation."""
import json
import hashlib
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             fbeta_score, roc_auc_score, confusion_matrix, make_scorer)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src import ROOT
from src.generate_data import FEATURES, SEED

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold", "figure.facecolor": "#F7F6F2"})
GREEN, RED = "#426B57", "#BA6659"


def train() -> dict:
    path = ROOT / "data" / "synthetic_attendance.csv"
    df = pd.read_csv(path)
    x_train, x_test, y_train, y_test = train_test_split(df[FEATURES], df.shortage, test_size=.2,
                                                      stratify=df.shortage, random_state=SEED)
    candidates = {
        "Dummy": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=240, min_samples_leaf=4,
                                                max_depth=12, random_state=SEED, n_jobs=1),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    fitted, results = {}, {}
    for name, estimator in candidates.items():
        pipe = Pipeline([("scale", StandardScaler()), ("model", estimator)])
        scores = cross_validate(pipe, x_train, y_train, cv=cv,
                                scoring={"f2": make_scorer(fbeta_score, beta=2, zero_division=0),
                                         "precision": "precision", "recall": "recall"})
        results[name] = {"cv_f2": float(scores["test_f2"].mean()),
                         "cv_f2_std": float(scores["test_f2"].std()),
                         "cv_precision": float(scores["test_precision"].mean()),
                         "cv_recall": float(scores["test_recall"].mean())}
        fitted[name] = pipe.fit(x_train, y_train)
    selected = max(results, key=lambda name: results[name]["cv_f2"])
    for name, pipe in fitted.items():
        labels = pipe.predict(x_test)
        probabilities = pipe.predict_proba(x_test)[:, 1]
        results[name].update(accuracy=float(accuracy_score(y_test, labels)),
                             precision=float(precision_score(y_test, labels, zero_division=0)),
                             recall=float(recall_score(y_test, labels, zero_division=0)),
                             f1=float(f1_score(y_test, labels, zero_division=0)),
                             roc_auc=float(roc_auc_score(y_test, probabilities)),
                             confusion_matrix=confusion_matrix(y_test, labels, labels=[0, 1]).tolist())
    model = fitted[selected]
    importance = permutation_importance(model, x_test, y_test, n_repeats=10,
                                       random_state=SEED, scoring="f1")
    metrics = dict(selected_model=selected, seed=SEED, rows=len(df), train_rows=len(x_train),
                   test_rows=len(x_test), features=FEATURES, sklearn_version=sklearn.__version__,
                   dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   selection="Highest mean training 5-fold F2 (recall weighted); fixed probability cutoff 0.5",
                   models=results, shortage_counts=df.shortage.value_counts().sort_index().to_dict(),
                   importance={feature: {"mean": float(mean), "std": float(std)}
                               for feature, mean, std in zip(FEATURES, importance.importances_mean, importance.importances_std)})
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "outputs").mkdir(exist_ok=True)
    joblib.dump(dict(model=model, features=FEATURES, selected_model=selected), ROOT / "models/attendance_model.joblib")
    (ROOT / "models/metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(6, 4))
    cm = np.array(results[selected]["confusion_matrix"])
    ax.imshow(cm, cmap="Greens")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center", color="white" if value > cm.max()/2 else GREEN, fontsize=18)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["No shortage", "Shortage"],
           yticklabels=["No shortage", "Shortage"], xlabel="Predicted", ylabel="Actual", title=f"{selected} | held-out test")
    save(fig, "confusion_matrix")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    pd.DataFrame(results).T[["precision", "recall", "f1"]].plot.bar(ax=ax, color=[GREEN, "#D7A04B", RED], rot=0)
    ax.set(ylim=(0, 1.08), ylabel="Held-out score", title="Model comparison")
    ax.legend(loc="lower right")
    save(fig, "model_comparison")
    order = np.argsort(importance.importances_mean)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(np.array(FEATURES)[order], importance.importances_mean[order], xerr=importance.importances_std[order], color=GREEN)
    ax.set(xlabel="Decrease in held-out F1 after permutation", title="Global permutation importance")
    save(fig, "feature_importance")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(["No shortage", "Shortage"], df.shortage.value_counts().reindex([0, 1]), color=[GREEN, RED])
    ax.set(ylabel="Synthetic trajectories", title="Simulated outcome distribution")
    save(fig, "class_distribution")
    print(json.dumps({"selected": selected, "test": results[selected]}, indent=2))
    return metrics


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / f"{name}.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    train()
