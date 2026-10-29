"""Load auditable synthetic examples with the original train/test membership."""
import hashlib
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from src import ROOT


def load_examples() -> pd.DataFrame:
    """Verify dataset provenance before exposing held-out prediction examples."""
    path = ROOT / "data/synthetic_attendance.csv"
    metrics = json.loads((ROOT / "models/metrics.json").read_text(encoding="utf-8"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != metrics["dataset_sha256"]:
        raise ValueError("Synthetic data changed since training. Run python -m src.train before exploring predictions.")
    data = pd.read_csv(path, dtype={"past_events": str, "future_events": str})
    train_ids, test_ids = train_test_split(data.index, test_size=.2, stratify=data.shortage,
                                         random_state=metrics["seed"])
    data["partition"] = "Training"
    data.loc[test_ids, "partition"] = "Held-out test"
    return data
