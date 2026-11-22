"""Resolve local inputs and experiment output directories."""
import os
import gzip
import json
from functools import lru_cache
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
DATA = Path(os.environ.get("STIR_DATA", "data/sherlock")).resolve()
OUTPUT = Path(os.environ.get("STIR_OUTPUT", "results")).resolve()
CACHE = OUTPUT / "cache"


@lru_cache(maxsize=1)
def archived_outcomes():
    with gzip.open(SCRIPTS.parent / "reference_results.json.gz", "rt", encoding="utf-8") as stream:
        return json.load(stream)


def trial_records(scenario):
    return archived_outcomes()[f"inject/inject_{scenario}.json"]
