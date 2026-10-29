import json
import joblib
import numpy as np
import pandas as pd
import pytest
from src import ROOT
from src.generate_data import generate, FEATURES
from src.predict import predict
from src.synthetic import load_examples


def test_reproducibility():
    pd.testing.assert_frame_equal(generate(50), generate(50))
    assert not generate(50).equals(generate(50,43))


def test_synthetic_prediction_examples_are_held_out():
    from sklearn.model_selection import train_test_split
    data = load_examples()
    train, test = train_test_split(data.index, test_size=.2, stratify=data.shortage, random_state=42)
    assert set(data[data.partition == "Held-out test"].index) == set(test)
    assert not set(train) & set(test)
    assert len(test) == 300 and len(train) == 1200
    assert all(len(row.past_events) == row.conducted for row in data.itertuples())


def test_dataset_consistency():
    df = generate()
    assert len(df) == 1500
    assert df.trajectory_id.is_unique
    assert .2 < df.shortage.mean() < .8
    for row in df.itertuples():
        past = [int(e) for e in row.past_events]
        future = [int(e) for e in row.future_events]
        assert sum(past) == row.attended and len(past) == row.conducted
        assert len(future) == row.remaining
        assert row.current_percentage == pytest.approx(100*sum(past)/len(past))
        assert row.recent_rate == np.mean(past[-8:])
        assert row.historical_rate == np.mean(past[:-8] or past)
        assert row.semester_progress == row.conducted/(row.conducted+row.remaining)
        trailing = len(row.past_events)-len(row.past_events.rstrip('0'))
        assert trailing == row.consecutive_absences
        assert row.final_attended == sum(past+future)
        assert row.shortage == int(100*sum(past+future) < row.threshold*len(past+future))
    assert not {'shortage','future_events','past_events','final_attended','final_percentage','behaviour','trajectory_id'} & set(FEATURES)


def test_artifact_and_probability():
    artifact = joblib.load(ROOT/'models/attendance_model.joblib')
    assert artifact['features'] == FEATURES
    for rate in [0,.5,1]:
        result = predict(35,43,20,recent_rate=rate,historical_rate=.8,
                         consecutive_absences=0,weekly_frequency=3,artifact=artifact)
        assert 0 <= result.probability <= 1
        assert 'Model estimate' in result.source
    metrics = json.loads((ROOT/'models/metrics.json').read_text())
    assert metrics['selected_model'] == max(metrics['models'],key=lambda name:metrics['models'][name]['cv_f2'])
    for values in metrics['models'].values():
        assert sum(map(sum,values['confusion_matrix'])) == metrics['test_rows']


def test_overrides_without_trends_or_model():
    assert predict(4,4,1,artifact={}).probability == 0
    assert predict(0,4,1,artifact={}).probability == 1
    assert predict(3,4,0,artifact={}).probability == 0
    assert predict(2,4,0,artifact={}).probability == 1
    with pytest.raises(ValueError):
        predict(3,4,4)
    with pytest.raises(ValueError):
        predict(0,0,0)
    with pytest.raises(ValueError):
        predict(3,4,4,recent_rate=2,historical_rate=.8,consecutive_absences=0,weekly_frequency=3)
