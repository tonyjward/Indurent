"""Tests for indurent_matching.geo and the location step in data.py."""

import numpy as np
import pandas as pd
import pytest

from indurent_matching.geo import assess_locations, haversine_miles, nearest_city

# Made-up points: central London (on land), central Birmingham, mid-Channel, and one without coordinates.
POINTS = pd.DataFrame(
    {
        "latitude": [51.5074, 51.5074, 52.4862, 50.0, np.nan],
        "longitude": [-0.1278, -0.1278, -1.8904, -2.0, np.nan],
        "region": ["London", "Scotland", "Midlands", "South West", "London"],
    },
    index=["london_ok", "london_mislabelled", "birmingham", "channel", "missing"],
)


def test_confidence_levels():
    out = assess_locations(POINTS)
    assert out["location_confidence"].to_dict() == {
        "london_ok": "high",
        "london_mislabelled": "medium",  # on land, but nowhere near Scotland
        "birmingham": "high",
        "channel": "unknown",  # far offshore
        "missing": "unknown",
    }
    assert out.loc["london_ok", "km_offshore"] == 0 and out.loc["london_ok", "km_to_region"] == 0
    assert out.loc["channel", "km_offshore"] > 50
    assert out.loc["london_mislabelled", "km_to_region"] > 400


def test_derived_region_only_for_trusted_locations():
    out = assess_locations(POINTS)
    assert out.loc["london_mislabelled", "derived_region"] == "London"  # where it actually is
    assert out.loc["channel", "derived_region"] is None
    assert out.loc["missing", "derived_region"] is None


def test_haversine_london_to_birmingham():
    assert haversine_miles(51.5074, -0.1278, 52.4862, -1.8904) == pytest.approx(101, abs=1)


def test_nearest_city_skips_untrusted_rows():
    usable = pd.Series([True, True, True, False, False], index=POINTS.index)
    out = nearest_city(POINTS, usable)
    assert out.loc["london_ok", "nearest_city"] == "London"
    assert out.loc["london_ok", "miles_to_city"] == pytest.approx(0, abs=0.01)
    assert out.loc["birmingham", "nearest_city"] == "Birmingham"
    assert out.loc[["channel", "missing"], "miles_to_city"].isna().all()


def test_portfolio_matches_eda(portfolio):
    df = portfolio
    assert df["location_confidence"].value_counts().to_dict() == {"medium": 652, "unknown": 358, "high": 240}
    bands = pd.cut(df["km_offshore"], [-1, 0, 1, 5, np.inf]).value_counts(sort=False).tolist()
    assert bands == [868, 24, 8, 270]  # on land, within 1 km, 1-5 km, 5 km+ (EDA 2.2)
    assert (df["miles_to_city"] <= 10).sum() == 50  # EDA section 5
    assert df["miles_to_city"].isna().sum() == 358
    london = df[df["region"] == "London"]
    assert (london["derived_region"] == "London").sum() == 0  # no "London" unit is in London (EDA 2.2)


def test_marketed_locations(marketed):
    df = marketed
    assert df["location_confidence"].to_dict() == {
        "M1": "medium",
        "M2": "unknown",
        "M3": "high",
        "M4": "medium",
        "M5": "unknown",
    }
    assert df.loc["M1", "derived_region"] == "Wales"  # labelled North West
    assert any("27 km offshore" in n for n in df.loc["M5", "quality_notes"])  # Chancery Depot
    assert "no coordinates, so location can't be used" in df.loc["M2", "quality_notes"]  # Tech Hub
