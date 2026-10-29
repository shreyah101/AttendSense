"""Keep test scratch files inside the workspace on managed Windows hosts."""
from pathlib import Path
from uuid import uuid4
import shutil
import pytest
from src import ROOT


@pytest.fixture
def tmp_path():
    path = ROOT / "tmp" / "tests" / uuid4().hex
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path)
