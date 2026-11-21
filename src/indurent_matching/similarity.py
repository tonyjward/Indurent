"""Similarity between a marketed listing and the portfolio, one block at a time (age, location, physical).

Two kinds of score, each between 0 and 1 (DESIGN.md section 2; evidence in notebooks 02-04):

* **Homogeneity** (prompt 2): how typical the listing is of the portfolio on one block. Each is scaled so the most
  typical value in the portfolio scores 1.
* **Pairwise similarity** (prompt 1): how similar the listing is to each portfolio unit on one block. Every block
  uses the same form, ``0.5 ** (gap / half_life)``: 1 for an identical value, halving with each half-life.

NaN means "not available": the listing (or, for pairwise similarity, the unit) has no usable value. Turning a
unit's NaN into 0 when the blocks are combined is a ranking decision, made where the blocks are combined.

Each function takes one listing (a row of ``load_marketed``) and the whole portfolio, and works on whole columns at
once (vectorised with numpy), not unit by unit.
"""

import numpy as np
import pandas as pd

from indurent_matching.config import AGE_BAND_YEARS, AGE_HALF_LIFE_YEARS


def halving(gap, half_life: float):
    """Similarity that is 1 at a gap of 0 and halves with every ``half_life`` of gap (NaN stays NaN)."""
    return 0.5 ** (np.abs(gap) / half_life)


def usable(df, field: str):
    """``field`` where it isn't flagged doubtful (``<field>_doubtful``), NaN where it is or is missing.

    Works on the whole portfolio (returns a column, keeping its index) or on one listing (returns a single value):
    the value is multiplied by 1 where it's trusted and by NaN where it's doubtful.
    """
    return df[field] * np.where(df[f"{field}_doubtful"], np.nan, 1)


# --- Age (notebook 02) -------------------------------------------------------------------------------------------


def get_year_band(year, width: int = AGE_BAND_YEARS):
    """First year of the band a build year falls in, e.g. 1977 -> 1975 for 5-year bands (works on a column too)."""
    return year // width * width


def age_homogeneity(listing: pd.Series, portfolio: pd.DataFrame) -> float:
    """How typical the listing's build year is of the portfolio, from 0 to 1 (prompt 2).

    Portfolio units are grouped into fixed build-year bands, ``AGE_BAND_YEARS`` wide (1970-74, 1975-79, ...). The
    score is the number of units in the listing's band divided by the number in the most frequent band. So a listing in
    the most frequent band scores 1, and one in a band with no portfolio units scores 0. For example, a listing built in
    1977 falls in 1975-79; if that band holds 109 units and the most frequent holds 127, it scores 109 / 127 = 0.86.

    Doubtful and missing portfolio build years aren't counted. Returns NaN ("not available") if the listing has no
    usable build year.
    """
    year = usable(listing, "build_year")
    if pd.isna(year):
        return np.nan
    units_per_band = get_year_band(usable(portfolio, "build_year").dropna()).value_counts()
    units_in_listing_band = units_per_band.get(get_year_band(year), 0)  # 0 if no portfolio units in that band
    return units_in_listing_band / units_per_band.max()


def age_similarity(listing: pd.Series, portfolio: pd.DataFrame) -> pd.Series:
    """How similar the listing's build year is to each portfolio unit's, from 0 to 1 (prompt 1).

    Built in the same year scores 1, and the score halves for every ``AGE_HALF_LIFE_YEARS`` years between them. With
    10-year halving, a listing built in 1977 scores 1 against a 1977 unit, 0.57 against a 1985 unit (8 years apart)
    and 0.25 against a 1997 unit (20 years apart). Older and newer count the same.

    Returns one score per portfolio unit, on the portfolio's index. A unit without a usable build year (missing or
    doubtful) gets NaN ("not available"), and so does every unit if the listing has no usable build year.
    """
    year = usable(listing, "build_year")
    return halving(usable(portfolio, "build_year") - year, AGE_HALF_LIFE_YEARS)
