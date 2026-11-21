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
    homogeneity_scores,
    location_context,
    location_homogeneity,
    location_similarity,
    most_similar,
    sits_on_estate,
    size_context,
    size_homogeneity,
    size_similarity,
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


# --- Physical (size) ---------------------------------------------------------------------------------------------


def _sized(sizes, doubtful=None, min_eaves=None, max_eaves=None):
    n = len(sizes)
    return pd.DataFrame(
        {
            "size_sqft": sizes,
            "size_sqft_doubtful": doubtful if doubtful is not None else [False] * n,
            "min_eaves_m": min_eaves if min_eaves is not None else [np.nan] * n,
            "max_eaves_m": max_eaves if max_eaves is not None else [np.nan] * n,
            "min_eaves_m_doubtful": [False] * n,
            "max_eaves_m_doubtful": [False] * n,
        }
    )


def _sized_listing(size, min_eaves=np.nan, max_eaves=np.nan):
    return _sized([size], min_eaves=[min_eaves], max_eaves=[max_eaves]).iloc[0]


# Five units around 4,000 sq ft, one at 40,000, and a doubtful 300 sq ft unit (left out).
SIZES = _sized([3_500, 3_800, 4_000, 4_200, 4_500, 40_000, 300], doubtful=[False] * 6 + [True])


def test_size_homogeneity():
    assert size_homogeneity(_sized_listing(4_000), SIZES) == pytest.approx(1.0)  # in the fullest window
    assert size_homogeneity(_sized_listing(40_000), SIZES) == pytest.approx(1 / 5)  # alone in its window
    assert size_homogeneity(_sized_listing(300), SIZES) == 0  # only the doubtful unit is near it, and it isn't counted
    assert np.isnan(size_homogeneity(_sized_listing(np.nan), SIZES))


def test_size_similarity():
    units = _sized([2_000, 4_000, 8_000, 1_000, 500], doubtful=[False] * 4 + [True])
    out = size_similarity(_sized_listing(2_000), units)
    assert out.iloc[:4].tolist() == pytest.approx([1.0, 0.5, 0.25, 0.5])  # same, 2x, 4x, half
    assert np.isnan(out.iloc[4])  # doubtful size: not available
    assert size_similarity(_sized_listing(np.nan), units).isna().all()


def test_size_context():
    # Three similar-sized units with eaves (one with max only) and one far bigger, which isn't a peer.
    units = _sized(
        [3_500, 4_000, 4_500, 40_000],
        min_eaves=[5.0, 6.0, np.nan, 12.0],
        max_eaves=[6.0, 7.0, 8.0, 14.0],
    )
    ctx = size_context(_sized_listing(4_000, min_eaves=9.0, max_eaves=10.0), units)
    assert ctx["eaves_m"] == 9.5  # average of min and max
    assert ctx["similar_size_units_with_eaves"] == 3
    assert ctx["their_median_eaves_m"] == 6.5  # 5.5, 6.5 and 8.0 (max only)
    assert ctx["eaves_higher_than_pct"] == 100  # tall for its size
    assert all(v is None for v in size_context(_sized_listing(4_000), units).values())  # no eaves: no context


def test_size_scores_match_notebook(portfolio, marketed):
    scores = {i: round(size_homogeneity(row, portfolio), 2) for i, row in marketed.iterrows()}
    assert scores == {"M1": 0.41, "M2": 0.38, "M3": 0.87, "M4": 0.17, "M5": 0.41}
    m4 = size_context(marketed.loc["M4"], portfolio)
    assert (
        m4["eaves_m"] == 7.0
        and m4["similar_size_units_with_eaves"] == 6
        and round(m4["their_median_eaves_m"], 1) == 4.5
    )
    assert size_context(marketed.loc["M1"], portfolio)["eaves_m"] is None  # M1 has no eaves
    assert size_similarity(marketed.loc["M3"], portfolio).isna().sum() == 10  # the 10 doubtful sizes


# --- Combining the blocks ----------------------------------------------------------------------------------------


def _unit(estate, lat, build_year, size, confidence="high"):
    return {
        "industrial_estate_name": estate,
        "unit_name": "Unit",
        "latitude": lat,
        "longitude": 0.0,
        "location_confidence": confidence,
        "build_year": build_year,
        "build_year_doubtful": False,
        "size_sqft": size,
        "size_sqft_doubtful": False,
    }


# The listing: built 2000, 4,000 sq ft, at latitude 52. Expected overall scores (equal weights) in the comments.
CANDIDATES = pd.DataFrame(
    [
        _unit("E1", 52.0, 2000, 4_000),  # 0: identical on all three -> 1.0
        _unit("E1", 52.0, 2010, 4_000),  # 1: 10 years newer -> age 0.5 -> (0.5 + 1 + 1) / 3 = 0.83
        _unit("E2", 52.0, 2010, 8_000),  # 2: 10 years newer, twice the size -> (0.5 + 1 + 0.5) / 3 = 0.67
        _unit("E3", 52.0, 2000, 4_000, confidence="unknown"),  # 3: location unknown, counts 0 -> 0.67 (tie with 2)
        _unit("E1", 52.0, np.nan, 2_000),  # 4: build year unknown (0), half the size -> (0 + 1 + 0.5) / 3 = 0.5
    ]
)
TARGET = pd.Series(_unit("listing", 52.0, 2000, 4_000))


def test_most_similar_ranking_and_missing_blocks():
    out = most_similar(TARGET, CANDIDATES)
    assert out.index.tolist() == [0, 1, 2, 3, 4]  # 2 and 3 tie at 0.67: Property Id order
    assert out["overall"].round(2).tolist() == [1.0, 0.83, 0.67, 0.67, 0.5]
    assert np.isnan(out.loc[3, "location"]) and np.isnan(out.loc[4, "age"])  # shown blank, counted as 0
    assert out.loc[0, "miles_from_listing"] == 0 and np.isnan(out.loc[3, "miles_from_listing"])


def test_most_similar_drops_blocks_the_listing_lacks():
    no_location = TARGET.copy()
    no_location["location_confidence"] = "unknown"
    out = most_similar(no_location, CANDIDATES)
    assert out["location"].isna().all() and out["miles_from_listing"].isna().all()
    assert out.loc[3, "overall"] == pytest.approx(1.0)  # (age 1 + size 1) / 2: location dropped, not 0
    nothing = no_location.copy()
    nothing["build_year"], nothing["size_sqft"] = np.nan, np.nan
    with pytest.raises(ValueError, match="no usable"):
        most_similar(nothing, CANDIDATES)


def test_most_similar_weights_cap_and_k():
    out = most_similar(TARGET, CANDIDATES, weights={"age": 0})  # ignore age: 0 and 1 tie at 1, then 2 and 4 at 0.75
    assert out.index.tolist() == [0, 1, 2, 4, 3]
    assert most_similar(TARGET, CANDIDATES, max_per_estate=1).index.tolist() == [0, 2, 3]  # one unit per estate
    assert len(most_similar(TARGET, CANDIDATES, k=2)) == 2
    with pytest.raises(ValueError, match="unknown blocks"):
        most_similar(TARGET, CANDIDATES, weights={"eaves": 1})


def test_most_similar_on_real_data(portfolio, marketed):
    m3 = most_similar(marketed.loc["M3"], portfolio)
    assert (m3["industrial_estate_name"] == "Velvet Park 60").all()  # no cap: all on the estate M3 sits on
    capped = most_similar(marketed.loc["M3"], portfolio, max_per_estate=2)
    assert len(capped) == 10 and capped["industrial_estate_name"].value_counts().max() == 2
    m2 = most_similar(marketed.loc["M2"], portfolio)
    assert m2[["age", "location"]].isna().all().all()  # M2 has no build year or location: size only
    assert (m2["overall"] == m2["physical"]).all()


def test_homogeneity_scores(portfolio, marketed):
    m4 = homogeneity_scores(marketed.loc["M4"], portfolio)
    assert {k: round(m4[k], 2) for k in ["age", "location", "physical"]} == {
        "age": 0.86,
        "location": 0.73,
        "physical": 0.17,
    }
    assert m4["location_context"]["sits_on_estate"] == "Sable Mill 91"
    assert m4["size_context"]["eaves_m"] == 7.0
    m2 = homogeneity_scores(marketed.loc["M2"], portfolio)
    assert np.isnan(m2["age"]) and np.isnan(m2["location"]) and round(m2["physical"], 2) == 0.38
