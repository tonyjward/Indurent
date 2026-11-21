"""Tests for indurent_matching.similarity.

Unit tests use small made-up frames. Tests taking the ``portfolio`` / ``marketed`` fixtures (tests/conftest.py)
check the scores against the similarity notebooks and are skipped when the data files aren't present.
"""

import numpy as np
import pandas as pd
import pytest

from indurent_matching.similarity import (
    age_homogeneity,
    age_similarity,
    get_year_band,
    halving,
    location_context,
    location_homogeneity,
    location_similarity,
    sits_on_estate,
    usable,
)


def _units(build_years, doubtful=None):
    return pd.DataFrame(
        {
            "build_year": build_years,
            "build_year_doubtful": doubtful if doubtful is not None else [False] * len(build_years),
        }
    )


def _listing(build_year):
    return pd.Series({"build_year": build_year, "build_year_doubtful": False})


def test_halving():
    assert halving(np.array([0, 10, 20, -10]), 10).tolist() == [1.0, 0.5, 0.25, 0.5]  # direction doesn't matter
    assert np.isnan(halving(np.nan, 10))


def test_usable_on_a_column_and_on_one_row():
    df = _units([2000, 1990, np.nan], doubtful=[False, True, False])
    df.index = [11, 12, 13]
    out = usable(df, "build_year")
    assert out.index.tolist() == [11, 12, 13]  # keeps the portfolio's index
    assert out.iloc[0] == 2000 and out.iloc[1:].isna().all()  # doubtful and missing become NaN
    assert usable(df.loc[11], "build_year") == 2000  # one row gives one value
    assert np.isnan(usable(df.loc[12], "build_year"))


@pytest.mark.parametrize("year, start", [(1975, 1975), (1977, 1975), (1979, 1975), (1980, 1980), (2022, 2020)])
def test_get_year_band(year, start):
    assert get_year_band(year) == start


# Most frequent band 1970-74 has 3 units; 1980-84 has 1; the 1999 unit is doubtful, so 1995-99 counts as empty.
UNITS = _units([1970, 1971, 1973, 1981, 1999, np.nan], doubtful=[False, False, False, False, True, False])


@pytest.mark.parametrize(
    "year, expected",
    [(1972, 1.0), (1984, 1 / 3), (1997, 0.0), (1960, 0.0)],
)
def test_age_homogeneity(year, expected):
    assert age_homogeneity(_listing(year), UNITS) == pytest.approx(expected)


def test_age_homogeneity_missing_is_not_available():
    assert np.isnan(age_homogeneity(_listing(np.nan), UNITS))


def test_age_similarity():
    out = age_similarity(_listing(1971), UNITS)
    assert out.iloc[:4].round(3).tolist() == [0.933, 1.0, 0.871, 0.5]
    assert out.iloc[4:].isna().all()  # doubtful and missing units: not available
    assert age_similarity(_listing(np.nan), UNITS).isna().all()


def test_age_scores_match_notebook(portfolio, marketed):
    scores = {i: round(age_homogeneity(row, portfolio), 2) for i, row in marketed.iterrows()}
    assert {i: s for i, s in scores.items() if not np.isnan(s)} == {"M1": 0.86, "M3": 0.86, "M4": 0.86, "M5": 0.2}
    assert np.isnan(scores["M2"])  # no build year
    m1 = age_similarity(marketed.loc["M1"], portfolio)
    assert len(m1) == len(portfolio) and m1.isna().sum() == portfolio["build_year"].isna().sum()


# --- Location ----------------------------------------------------------------------------------------------------

MILES_PER_DEGREE_LAT = 69.09  # along a meridian, for the Earth radius used in geo.haversine_miles


def _place(name, lat, lon=0.0, confidence="high"):
    return {"industrial_estate_name": name, "latitude": lat, "longitude": lon, "location_confidence": confidence}


# Six estates in a line running north, about 6.9 miles apart (A to F), two units each; Z far away in Scotland;
# and one unit near A whose location isn't trusted.
LINE = pd.DataFrame(
    [_place(name, 52.0 + 0.1 * i) for i, name in enumerate("ABCDEF") for _ in range(2)]
    + [_place("Z", 56.0, -4.0), _place("Z", 56.0, -4.0), _place("Untrusted", 52.0, confidence="unknown")]
)
LINE["property_id"] = range(len(LINE))


def test_sits_on_estate():
    assert sits_on_estate(pd.Series(_place("listing", 52.0005)), LINE) == "A"
    beyond = 52.0 + 0.3 / MILES_PER_DEGREE_LAT  # 0.3 miles from A's units, beyond the 0.25-mile limit
    assert sits_on_estate(pd.Series(_place("listing", beyond, lon=-0.05)), LINE) is None
    assert sits_on_estate(pd.Series(_place("listing", 52.0, confidence="unknown")), LINE) is None


def test_location_homogeneity():
    # C sits in the middle of the line, so (like D) it's the best-placed estate: a listing on it scores 1.
    on_c = location_homogeneity(pd.Series(_place("listing", 52.2)), LINE)
    on_a = location_homogeneity(pd.Series(_place("listing", 52.0)), LINE)
    on_z = location_homogeneity(pd.Series(_place("listing", 56.0, -4.0)), LINE)
    assert on_c == pytest.approx(1.0)
    assert 0.5 < on_a < 1.0  # at the end of the line: its nearest estates are further away
    assert on_z < 0.05  # far from everything except the estate it sits on
    assert np.isnan(location_homogeneity(pd.Series(_place("listing", 52.0, confidence="unknown")), LINE))


def test_location_context():
    ctx = location_context(pd.Series(_place("listing", 52.0)), LINE)
    assert ctx["sits_on_estate"] == "A"
    assert ctx["nearest_other_estate"] == "B"  # A itself is left out
    assert ctx["nearest_other_estate_miles"] == pytest.approx(0.1 * MILES_PER_DEGREE_LAT, rel=0.01)
    assert 0 <= ctx["better_placed_than_pct"] <= 100
    unusable = location_context(pd.Series(_place("listing", 52.0, confidence="unknown")), LINE)
    assert all(value is None for value in unusable.values())


def test_location_similarity():
    units = pd.DataFrame(
        [
            _place("here", 52.0),
            _place("ten miles", 52.0 + 10 / MILES_PER_DEGREE_LAT),
            _place("x", 52.0, confidence="unknown"),
        ],
        index=[101, 102, 103],
    )
    out = location_similarity(pd.Series(_place("listing", 52.0)), units)
    assert out.index.tolist() == [101, 102, 103]
    assert out.loc[101] == pytest.approx(1.0)
    assert out.loc[102] == pytest.approx(0.5, abs=0.005)
    assert np.isnan(out.loc[103])  # untrusted unit location: not available
    assert location_similarity(pd.Series(_place("listing", 52.0, confidence="unknown")), units).isna().all()


def test_location_scores_match_notebook(portfolio, marketed):
    scores = {i: round(location_homogeneity(row, portfolio), 2) for i, row in marketed.iterrows()}
    assert {i: s for i, s in scores.items() if not np.isnan(s)} == {"M1": 0.23, "M3": 0.62, "M4": 0.73}
    assert np.isnan(scores["M2"]) and np.isnan(scores["M5"])  # no coordinates; offshore
    sits_on = {i: sits_on_estate(row, portfolio) for i, row in marketed.iterrows()}
    assert sits_on == {
        "M1": "Scarlet Pavilion 78",
        "M2": None,
        "M3": "Velvet Park 60",
        "M4": "Sable Mill 91",
        "M5": None,
    }
    m1 = location_context(marketed.loc["M1"], portfolio)
    assert round(m1["nearest_other_estate_miles"], 1) == 4.4 and round(m1["better_placed_than_pct"]) == 7
