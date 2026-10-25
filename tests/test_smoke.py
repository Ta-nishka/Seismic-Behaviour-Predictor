"""Smoke tests: every dashboard section must render without raising.
Run from the repo root:  pytest -q
"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _from_repo_root(monkeypatch):
    monkeypatch.chdir(ROOT)      # the app reads data/ and results/ relative to the repo root

SECTIONS = ["Catalog", "Graph", "Embeddings", "Stations", "Models", "Predict", "About"]


def run(section, **state):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.session_state["nav"] = section
    for k, v in state.items():
        at.session_state[k] = v
    return at.run()


@pytest.mark.parametrize("section", SECTIONS)
def test_section_renders(section):
    assert not run(section).exception


def test_outlook_tool():
    at = run("Predict")
    at.radio[0].set_value("Regional activity outlook").run()
    assert not at.exception


def test_empty_filter_is_handled():
    at = run("Catalog", f_mag=(7.05, 7.1), f_depth=["deep"])
    assert not at.exception
