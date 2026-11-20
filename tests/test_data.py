"""Tests for indurent_matching.data.

Unit tests use small made-up frames. Tests taking the ``portfolio`` / ``marketed`` fixtures (tests/conftest.py)
use the assessment files in data/ and are skipped when those files aren't present.
"""

import numpy as np
import pandas as pd
import pytest

from indurent_matching.data import (
    add_estate_structure,
    apply_known_issues,
    clean_build_year,
    clean_doors,
    clean_eaves,
    clean_epc,
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


def _doors_frame(estates, doors, size=10_000.0):
    df = add_estate_structure(pd.DataFrame({"industrial_estate_name": estates, "doors": doors, "size_sqft": size}))
    df["quality_notes"] = [[] for _ in range(len(df))]
    return df


def test_clean_doors_estate_total_vs_unit_count():
    out = clean_doors(
        _doors_frame(
            ["A", "A", "A", "B", "C", "C", "D"],  # A: copied total; B: single unit; C: differing counts; D: zero
            [6.0, 6.0, 6.0, 4.0, 3.0, 9.0, 0.0],
        )
    )
    assert out["doors_describe_unit"].tolist() == [False, False, False, True, True, True, True]
    assert out["doors_per_unit_estimate"].iloc[0] == 2.0  # 6 doors / 3 units
    assert out["doors_per_unit_estimate"].iloc[3:].isna().all()
    assert np.isnan(out["doors"].iloc[6])  # zero means not recorded
    assert out["quality_notes"].iloc[0] == [
        "doors: 6 is an estate total across 3 units (about 2.0 per unit), not this unit's own count"
    ]
    assert out["quality_notes"].iloc[4] == []


def test_clean_doors_density_check():
    # B: 4 doors on 1,000 sq ft (250 per door) is too dense; A: an estate total is never checked; C: plausible;
    # D: 3 doors on 1,000 sq ft is dense too, but a count of 3 or fewer is never flagged.
    df = _doors_frame(["A", "A", "B", "C", "D"], [40.0, 40.0, 4.0, 4.0, 3.0])
    df["size_sqft"] = [1_000.0, 1_000.0, 1_000.0, 40_000.0, 1_000.0]
    out = clean_doors(df)
    assert out["doors_doubtful"].tolist() == [False, False, True, False, False]
    assert out["quality_notes"].iloc[2] == [
        "doors: 4 on 1,000 sq ft (about 250 sq ft per door) is implausibly dense, so doubtful"
    ]
    assert "_sqft_per_door" not in out.columns


def test_clean_doors_listings_count_as_their_own():
    df = pd.DataFrame({"doors": [2.0, 0.0], "size_sqft": [20_000.0, 20_000.0]})
    df["quality_notes"] = [[] for _ in range(len(df))]
    out = clean_doors(df)
    assert out["doors_describe_unit"].all()
    assert np.isnan(out["doors"].iloc[1])


def test_doors_after_cleaning(portfolio, marketed):
    estate_total = ~portfolio["doors_describe_unit"]
    assert portfolio.loc[estate_total, "industrial_estate_name"].nunique() == 30  # EDA 2.6: 30 of 32 estates
    assert portfolio.loc[portfolio["industrial_estate_name"] == "Ruby Arc 89", "doors_per_unit_estimate"].iloc[
        0
    ] == pytest.approx(27 / 21)
    assert portfolio.loc[portfolio["industrial_estate_name"] == "Crimson Quay 31", "doors_describe_unit"].all()
    assert not (portfolio["doors"] == 0).any()
    assert marketed["doors_describe_unit"].all()
    assert marketed["doors_doubtful"].tolist() == [False, False, False, True, False]  # M4: 16 doors on 617 sq ft
    flagged = portfolio.loc[portfolio["doors_doubtful"], "industrial_estate_name"]
    assert "Opal Forge 42" in flagged.values and "Granite Summit 84" in flagged.values
    assert "Granite Depot 69" not in flagged.values  # 3 doors on ~4,000 sq ft: plausible (EDA 2.6)


def test_apply_known_issues():
    # Units 1 and 2 are in the register; unit 2 has no door count, so only its eaves are flagged. Unit 3 is outside
    # the register but already flagged by an earlier step, which must be kept.
    df = pd.DataFrame(
        {"property_id": [1, 2, 3], "doors": [5.0, np.nan, 5.0], "max_eaves_m": [10.0, 8.5, 10.0]},
    )
    df["doors_doubtful"] = [False, False, True]
    df["quality_notes"] = [[] for _ in range(len(df))]
    issues = [{"property_ids": [1, 2], "fields": ["doors", "max_eaves_m"], "reason": "a test reason"}]
    out = apply_known_issues(df, issues)
    assert out["doors_doubtful"].tolist() == [True, False, True]
    assert out["max_eaves_m_doubtful"].tolist() == [True, True, False]
    assert out["quality_notes"].iloc[0] == ["doors 5, max_eaves_m 10: a test reason, so doubtful"]  # one note
    assert out["quality_notes"].iloc[1] == ["max_eaves_m 8.5: a test reason, so doubtful"]
    assert out["quality_notes"].iloc[2] == []


def test_clean_epc():
    out = clean_epc(pd.DataFrame({"epc_rating": ["A", False, np.nan, "G"]}))
    assert out["epc_rating"].isna().tolist() == [False, True, True, False]
    assert out["epc_rating"].dropna().tolist() == ["A", "G"]


def test_known_issues_after_cleaning(portfolio, marketed):
    block = portfolio["property_id"].between(1173, 1187)
    mews = portfolio["industrial_estate_name"] == "Granite Mews 59"
    for field in ["min_eaves_m", "max_eaves_m", "yard_depth_m"]:
        flagged = portfolio[f"{field}_doubtful"]
        assert flagged.equals(portfolio[field].notna() & (block | (mews & (field != "yard_depth_m"))))
    assert portfolio.loc[block, "yard_depth_m_doubtful"].sum() == 11  # EDA 2.7: 11 of the 34 yard depths
    assert portfolio["build_year_doubtful"].sum() == 5 and portfolio.loc[mews, "build_year_doubtful"].all()
    assert portfolio.loc[block & portfolio["doors"].notna(), "doors_doubtful"].all()
    doubtful = [c for c in marketed.columns if c.endswith("_doubtful")]
    assert set(doubtful) == {c for c in portfolio.columns if c.endswith("_doubtful")}  # same columns in both files
    assert not marketed[[c for c in doubtful if c != "doors_doubtful"]].any().any()  # only M4's doors, from step 6
    assert set(portfolio["epc_rating"].dropna()) == set("ABCDE")
    assert portfolio["epc_rating"].isna().sum() == 326  # 319 blank + 7 FALSE


def test_column_order(portfolio, marketed):
    assert portfolio.columns[0] == "property_id" and marketed.columns[0] == "listing_id"
    assert portfolio.columns[-1] == "quality_notes" and marketed.columns[-1] == "quality_notes"
    # Same columns in the same order, apart from the id and the portfolio-only estate structure.
    shared = [c for c in portfolio.columns if c not in {"property_id", "estate_units", "is_single_unit"}]
    assert shared == list(marketed.columns[1:])


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
