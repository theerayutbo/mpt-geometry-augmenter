import os
from pathlib import Path

import pytest

DEFAULT_MPT_ROOT = Path(__file__).resolve().parents[2] / "ext" / "MPT-Calculator"


@pytest.fixture(scope="session")
def mpt_root() -> Path:
    root = Path(os.environ.get("MPT_ROOT", DEFAULT_MPT_ROOT))
    if not (root / "main.py").exists():
        pytest.skip(f"MPT-Calculator not found at {root} (set MPT_ROOT)")
    return root
