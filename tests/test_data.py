"""Tests for indurent_matching.data.

Unit tests use small made-up frames. Tests marked ``real_data`` read the assessment files in data/
(gitignored) and are skipped when those files aren't present.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from indurent_matching.data import add_estate_structure, load_marketed, load_portfolio, parse_number, snake_case

DATA = Path(__file__).resolve().parents[1] / "data"
PORTFOLIO = DATA / "Current Portfolio.xlsx"
MARKETED = DATA / "Marketed Warehouses.xlsx"
real_data = pytest.mark.skipif(not PORTFOLIO.exists(), reason="assessment data not present")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (12, 12.0),
        (7.05, 7.05),
        ("12", 12.0),
        (" 12.5 ", 12.5),
        ("-1.5", -1.5),  # a negative number, not a range
        ("20-28", 24.0),
        ("25 - 33", 29.0),
        (0, 0.0),  # zeros are kept; each column's step decides what they mean
    ],
)
def test_parse_number_values(value, expected):
    assert parse_number(value) == expected


@pytest.mark.parametrize("value", ["X", "-", "TBC", "", None, np.nan, True, False, "abc"])
def test_parse_number_missing(value):
    assert np.isnan(parse_number(value))


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Property Id", "property_id"),
        ("Car Parking Spaces #", "car_parking_spaces"),
        ("Min. Eaves m", "min_eaves_m"),
        ("Doors #", "doors"),
        ("EPC Rating", "epc_rating"),
    ],
)
def test_snake_case(name, expected):
    assert snake_case(name) == expected


def test_estate_structure():
    df = pd.DataFrame({"industrial_estate_name": ["A", "A", "A", "B"]})
    out = add_estate_structure(df)
    assert out["estate_units"].tolist() == [3, 3, 3, 1]
    assert out["is_single_unit"].tolist() == [False, False, False, True]


@real_data
def test_load_portfolio():
    df = load_portfolio(PORTFOLIO)
    assert len(df) == 1250
    assert df.index.is_unique and df.index.name == "property_id"
    assert df["industrial_estate_name"].nunique() == 263
    estates = df.groupby("industrial_estate_name")["estate_units"].first()
    assert (estates == 1).sum() == 125  # EDA section 1
    assert df["size_sqm"].dtype == float and df["doors"].dtype == float


@real_data
def test_load_marketed():
    df = load_marketed(MARKETED)
    assert df.index.tolist() == ["M1", "M2", "M3", "M4", "M5"]
    assert "property_id" not in df.columns
    assert df.loc["M3", "industrial_estate_name"] == "Spitfire Park"
