"""Paths supplied by run.py; importing never reads participant data."""
import os
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
WORKSPACE = Path(os.environ.get("TRUST_AV_WORKSPACE", REPO/"work")).expanduser().resolve()
DATA_DIR = Path(os.environ.get("TRUST_AV_DATA_DIR", WORKSPACE/"inputs/derived")).expanduser().resolve()
QUESTIONNAIRE = Path(os.environ.get("TRUST_AV_QUESTIONNAIRE", WORKSPACE/"inputs/questionnaire.xlsx")).expanduser().resolve()
def relative_label(path, root=WORKSPACE):
    path=Path(path).resolve()
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
