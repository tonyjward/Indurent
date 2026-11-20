"""Tests for indurent_matching.data.

Unit tests use small made-up frames. Tests taking the ``portfolio`` / ``marketed`` fixtures (tests/conftest.py)
use the assessment files in data/ and are skipped when those files aren't present.
"""

import numpy as np
import pandas as pd
import pytest

from indurent_matching.data import (
    add_estate_structure,
    clean_build_year,
    clean_eaves,
    clean_size,
    parse_number,
    snake_case,
)


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


def test_clean_size():
    df = pd.DataFrame({"size_sqm": [262.0, 499.9, 500.0, 5_000.0, 600_000.0, 700_000.0, np.nan]})
    df["quality_notes"] = [[] for _ in range(len(df))]
    out = clean_size(df)
    assert "size_sqft" in out.columns and "size_sqm" not in out.columns
    assert out["size_sqft"].iloc[3] == 5_000.0  # values unchanged, only renamed
    assert out["size_sqft_doubtful"].tolist() == [True, True, False, False, False, False, False]
    assert out["quality_notes"].iloc[0] == [
        "size 262 sq ft: below Indurent's smallest unit band (500 sq ft), so doubtful"
    ]
    assert out["quality_notes"].iloc[5] == ["size 700,000 sq ft: very large (over 600,000 sq ft); rare but plausible"]
    assert out["quality_notes"].iloc[4] == []  # 600,000 exactly is not "over"
    assert not out["size_sqft_doubtful"].iloc[5]  # very large is a note, not a flag


def test_clean_build_year():
    df = pd.DataFrame({"build_year": [1995.0, np.nan]})
    df["quality_notes"] = [[] for _ in range(len(df))]
    out = clean_build_year(df)
    assert out["build_year"].iloc[0] == 1995.0 and np.isnan(out["build_year"].iloc[1])  # not filled
    assert out["quality_notes"].tolist() == [[], ["no build year, so age can't be compared"]]


def test_build_year_counts(portfolio, marketed):
    assert portfolio["build_year"].isna().sum() == 16  # EDA 2.3
    assert portfolio["build_year"].dropna().between(1939, 2022).all()
    assert "no build year, so age can't be compared" in marketed.loc["M2", "quality_notes"]  # Tech Hub


def test_clean_eaves_zero_is_missing_each_side_kept():
    df = pd.DataFrame({"min_eaves_m": [0.0, 6.0, 0.0, 5.0], "max_eaves_m": [0.0, 0.0, 8.0, 7.0]})
    out = clean_eaves(df)
    assert out["min_eaves_m"].isna().tolist() == [True, False, True, False]
    assert out["max_eaves_m"].isna().tolist() == [True, True, False, False]
    assert out.loc[1, "min_eaves_m"] == 6.0  # recorded side kept when the other is zero


def test_eaves_after_cleaning(portfolio):
    assert not (portfolio["min_eaves_m"] > portfolio["max_eaves_m"]).any()  # no swaps needed (EDA 2.5)
    assert not (portfolio[["min_eaves_m", "max_eaves_m"]] == 0).any().any()
    both = portfolio["min_eaves_m"].notna() & portfolio["max_eaves_m"].notna()
    assert both.sum() == 277  # units with both recorded (EDA 2.5)


def test_load_portfolio(portfolio):
    df = portfolio
    assert len(df) == 1250
    assert df.index.is_unique and df.index.name == "property_id"
    assert df["industrial_estate_name"].nunique() == 263
    estates = df.groupby("industrial_estate_name")["estate_units"].first()
    assert (estates == 1).sum() == 125  # EDA section 1
    assert df["size_sqft"].dtype == float and df["doors"].dtype == float
    assert round(df["size_sqft"].sum() / 1e6, 1) == 30.6  # EDA 2.1: matches Indurent's 28-40m sq ft
    assert df["size_sqft_doubtful"].sum() == 10
    assert sum(any("very large" in n for n in notes) for notes in df["quality_notes"]) == 5


def test_load_marketed(marketed):
    df = marketed
    assert df.index.tolist() == ["M1", "M2", "M3", "M4", "M5"]
    assert "property_id" not in df.columns
    assert df.loc["M3", "industrial_estate_name"] == "Spitfire Park"
