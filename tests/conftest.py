"""Shared test setup: the cleaned assessment data, loaded once per test run.

The spreadsheets live in data/ (gitignored). Tests that use these fixtures are skipped when they aren't there.
"""

import os
from pathlib import Path

import pytest

# Never send test runs to LangSmith, even when .env turns tracing on: graph.py's load_dotenv() doesn't override a
# variable that's already set, so this wins.
os.environ["LANGSMITH_TRACING"] = "false"

from indurent_matching.data import load_marketed, load_portfolio

DATA = Path(__file__).resolve().parents[1] / "data"
PORTFOLIO = DATA / "Current Portfolio.xlsx"
MARKETED = DATA / "Marketed Warehouses.xlsx"


@pytest.fixture(scope="session")
def portfolio():
    """The cleaned portfolio (one row per unit), or skip the test if the file isn't present."""
    if not PORTFOLIO.exists():
        pytest.skip("assessment data not present")
    return load_portfolio(PORTFOLIO)


@pytest.fixture(scope="session")
def marketed():
    """The cleaned marketed listings (M1-M5), or skip the test if the file isn't present."""
    if not MARKETED.exists():
        pytest.skip("assessment data not present")
    return load_marketed(MARKETED)
