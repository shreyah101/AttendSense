"""Probability estimates with exact mathematical safeguards."""
from dataclasses import dataclass
import joblib
import pandas as pd
from src import ROOT
from src.attendance import certain_risk, percentage, validate
from src.generate_data import FEATURES


@dataclass(frozen=True)
class Prediction:
    probability: float
    source: str
    level: str


def predict(attended: int, conducted: int, remaining: int, threshold: float = 75,
            *, recent_rate: float | None = None, historical_rate: float | None = None,
            consecutive_absences: int | None = None, weekly_frequency: int | None = None,
            artifact=None) -> Prediction:
    validate(attended, conducted, remaining, threshold)
    probability = certain_risk(attended, conducted, remaining, threshold)
    source = "Mathematical certainty"
    if probability is None:
        if conducted == 0:
            raise ValueError("Past attendance is unavailable; record conducted classes before requesting ML.")
        if recent_rate is None or historical_rate is None or consecutive_absences is None or weekly_frequency is None:
            raise ValueError("Provide past attendance rates, absence streak and weekly frequency. No trends are inferred.")
        if not (0 <= recent_rate <= 1 and 0 <= historical_rate <= 1):
            raise ValueError("Past attendance rates must be between 0 and 1.")
        if isinstance(consecutive_absences, bool) or not isinstance(consecutive_absences, int) or not 0 <= consecutive_absences <= conducted-attended:
            raise ValueError("Absence streak cannot exceed missed classes.")
        if not isinstance(weekly_frequency, int) or isinstance(weekly_frequency, bool) or not 1 <= weekly_frequency <= 7:
            raise ValueError("Weekly frequency must be an integer from 1 to 7.")
        artifact = artifact if artifact is not None else joblib.load(ROOT / "models/attendance_model.joblib")
        values = [conducted, attended, percentage(attended, conducted), remaining, recent_rate,
                  historical_rate, consecutive_absences, conducted/(conducted+remaining), weekly_frequency, threshold]
        probability = float(artifact["model"].predict_proba(pd.DataFrame([values], columns=FEATURES))[0, 1])
        source = "Model estimate from past attendance features"
    level = "Low" if probability < .3 else "Moderate" if probability < .7 else "High"
    return Prediction(probability, source, level)
