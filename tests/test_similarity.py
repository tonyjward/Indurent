"""Tests for indurent_matching.similarity.

Unit tests use small made-up frames. Tests taking the ``portfolio`` / ``marketed`` fixtures (tests/conftest.py)
check the scores against the similarity notebooks and are skipped when the data files aren't present.
"""

import numpy as np
import pandas as pd
import pytest

from indurent_matching.similarity import age_homogeneity, age_similarity, get_year_band, halving, usable


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
    assert len(m1) == len(portfolio) and m1.isna().sum() == portfolio["build_year"].isna().sum() + 5
