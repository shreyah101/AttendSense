"""Validated, atomic local persistence. Existing or damaged data is never reset silently."""
import json
import os
from pathlib import Path
from uuid import uuid4
from src import ROOT
from src.attendance import validate

PATH = ROOT / "data" / "semester_subjects.json"
SAMPLE = [
    ("BCSE302L", "Database Systems - Theory", 35, 43, "Theory"),
    ("BCSE302P", "Database Systems - Lab", 24, 30, "Lab"),
    ("BCSE303L", "Operating Systems - Theory", 35, 43, "Theory"),
    ("BCSE303P", "Operating Systems - Lab", 20, 26, "Lab"),
    ("BCSE306L", "Artificial Intelligence", 34, 42, "Theory"),
    ("BCSE307L", "Compiler Design - Theory", 35, 44, "Theory"),
    ("BCSE307P", "Compiler Design - Lab", 20, 26, "Lab"),
    ("BCSE308L", "Computer Networks - Theory", 34, 41, "Theory"),
    ("BCSE308P", "Computer Networks - Lab", 26, 30, "Lab"),
    ("BCSE355L", "Cloud Architecture Design", 33, 41, "Theory"),
    ("BSTS301P", "Advanced Competitive Coding I", 34, 41, "Skills"),
]


def sample_subjects() -> list[dict]:
    return [dict(id=code, code=code, name=name, attended=a, conducted=c, type=kind,
                 remaining=None, threshold=75.0) for code, name, a, c, kind in SAMPLE]


def validate_subject(subject: dict) -> None:
    validate(subject["attended"], subject["conducted"],
             0 if subject["remaining"] is None else subject["remaining"], subject["threshold"])
    if any(not isinstance(subject.get(k), str) or not subject[k].strip() for k in ("id", "code", "name")):
        raise ValueError("ID, course code and name are required.")
    if subject["type"] not in ("Theory", "Lab", "Skills"):
        raise ValueError("Choose Theory, Lab or Skills.")


def save_subjects(subjects: list[dict], path: Path = PATH) -> None:
    if not isinstance(subjects, list):
        raise ValueError("Subjects must be a list.")
    for subject in subjects:
        validate_subject(subject)
    for key in ("id", "code"):
        if len({s[key].strip().casefold() for s in subjects}) != len(subjects):
            raise ValueError(f"Duplicate subject {key}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(subjects, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_subjects(path: Path = PATH) -> list[dict]:
    if not path.exists():
        save_subjects(sample_subjects(), path)
    subjects = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(subjects, list):
        raise ValueError("Invalid subject file; restore your backup or explicitly reset it.")
    for subject in subjects:
        validate_subject(subject)
    for key in ("id", "code"):
        if len({s[key].strip().casefold() for s in subjects}) != len(subjects):
            raise ValueError(f"Duplicate subject {key} in stored data.")
    return subjects
