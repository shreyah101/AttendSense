import json
import pytest
from src.subjects import sample_subjects, load_subjects, save_subjects, SAMPLE


def test_sample_and_persistence(tmp_path):
    path = tmp_path / "subjects.json"
    records = load_subjects(path)
    assert len(records) == 11
    assert [(s['code'],s['attended'],s['conducted']) for s in records] == [(s[0],s[2],s[3]) for s in SAMPLE]
    assert all(s['remaining'] is None for s in records)
    for s in records:
        s['attended'] = 0
        s['remaining'] = 12
        s['threshold'] = 80
    save_subjects(records,path)
    assert load_subjects(path) == records
    save_subjects(records[:-1],path)
    assert len(load_subjects(path)) == 10
    save_subjects([],path)
    assert load_subjects(path) == []


def test_validation_preserves_file(tmp_path):
    path = tmp_path/'subjects.json'
    data = load_subjects(path)
    original = path.read_bytes()
    data[0]['attended'] = 999
    with pytest.raises(ValueError):
        save_subjects(data,path)
    assert path.read_bytes() == original
    data = sample_subjects()
    data[0]['code'] = data[1]['code']
    with pytest.raises(ValueError):
        save_subjects(data,path)


def test_damaged_json_not_reset(tmp_path):
    path = tmp_path/'subjects.json'
    path.write_text('broken data')
    with pytest.raises(ValueError):
        load_subjects(path)
    assert path.read_text() == 'broken data'
