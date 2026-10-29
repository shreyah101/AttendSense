"""Real Streamlit reruns with isolated semester storage."""
from unittest.mock import patch
from uuid import uuid4
from src import ROOT
# Streamlit creates unused scratch storage even for from_file tests. On managed
# Windows hosts its owner-only temp ACL can reject shutdown cleanup. Create this
# scratch folder with inherited workspace permissions instead. Cleanup stays on.
def workspace_scratch(*args, **kwargs):
    path = ROOT / "tmp" / "streamlit" / uuid4().hex
    path.mkdir(parents=True)
    return str(path)


with patch("tempfile.mkdtemp", workspace_scratch):
    from streamlit.testing.v1 import AppTest
import src.subjects as subjects


def widget(items, label):
    return next(item for item in items if item.label == label)


def setup_app(monkeypatch, tmp_path):
    path = tmp_path / "semester.json"
    original_load, original_save = subjects.load_subjects, subjects.save_subjects
    monkeypatch.setattr(subjects, "PATH", path)
    monkeypatch.setattr(subjects, "load_subjects", lambda: original_load(path))
    monkeypatch.setattr(subjects, "save_subjects", lambda records, destination=path: original_save(records, destination))
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run(), path


def test_edit_add_delete_reset(monkeypatch, tmp_path):
    app, path = setup_app(monkeypatch, tmp_path)
    assert not app.exception
    assert len(subjects.load_subjects()) == 11
    widget(app.number_input, "Attended classes").set_value(36)
    widget(app.checkbox, "Remaining classes are known").check()
    widget(app.number_input, "Remaining classes").set_value(10)
    widget(app.button, "Save course").click().run()
    assert not app.exception
    assert subjects.load_subjects()[0]['attended'] == 36
    assert subjects.load_subjects()[0]['remaining'] == 10
    app.run()
    assert widget(app.number_input, "Attended classes").value == 36
    widget(app.text_input, "New course code").set_value("TEST101")
    widget(app.text_input, "New course name").set_value("Test course")
    widget(app.button, "Add course").click().run()
    assert len(subjects.load_subjects()) == 12
    app.selectbox[0].set_value(11).run()
    assert widget(app.button, "Delete course").disabled
    widget(app.checkbox, "I confirm deletion of this course").check().run()
    widget(app.button, "Delete course").click().run()
    assert not app.exception
    assert len(subjects.load_subjects()) == 11
    widget(app.checkbox, "Replace all records with the 11 sample courses (current file will be backed up)").check().run()
    widget(app.button, "Reset to sample records").click().run()
    assert not app.exception
    assert subjects.load_subjects() == subjects.sample_subjects()
    assert len(list(path.parent.glob("*.bak"))) == 1


def test_risk_and_planner(monkeypatch, tmp_path):
    app, path = setup_app(monkeypatch, tmp_path)
    widget(app.checkbox, "Remaining classes are known").check()
    widget(app.number_input, "Remaining classes").set_value(10)
    widget(app.button, "Save course").click().run()
    app.sidebar.radio[0].set_value("AI Risk Analysis").run()
    assert not app.exception
    assert widget(app.selectbox, "Synthetic student-subject record").options[0].startswith("SIM-")
    assert any(m.label == "Model-estimated final shortage probability" for m in app.metric)
    assert not app.number_input
    widget(app.selectbox, "Simulated behaviour").set_value("declining").run()
    assert not app.exception
    app.sidebar.radio[0].set_value("Bunk Planner").run()
    app.slider[0].set_value(5).run()
    assert not app.exception
    assert any(m.value == '75.47%' for m in app.metric)
    app.number_input[0].set_value(0).run()
    assert not app.exception and not app.slider
    app.sidebar.radio[0].set_value("Model Insights").run()
    assert not app.exception
    assert len(app.metric) == 5
    assert len(app.dataframe[0].value) == 1500


def test_synthetic_risk_independent_of_semester(monkeypatch, tmp_path):
    app, path = setup_app(monkeypatch, tmp_path)
    before = path.read_bytes()
    # Synthetic analysis must not read personal records, even if that file is invalid.
    def forbidden():
        raise AssertionError("Personal semester data reached synthetic risk analysis")
    monkeypatch.setattr(subjects, "load_subjects", forbidden)
    app.sidebar.radio[0].set_value("AI Risk Analysis").run()
    assert not app.exception
    assert any(m.label == "Model-estimated final shortage probability" for m in app.metric)
    assert path.read_bytes() == before
