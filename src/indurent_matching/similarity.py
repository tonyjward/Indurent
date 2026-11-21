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

from indurent_matching.config import (
    AGE_BAND_YEARS,
    AGE_HALF_LIFE_YEARS,
    LOCATION_HALF_LIFE_MILES,
    LOCATION_NEIGHBOUR_ESTATES,
    SITS_ON_ESTATE_MILES,
)
from indurent_matching.geo import haversine_miles


def halving(gap: float | pd.Series, half_life: float) -> float | pd.Series:
    """Similarity that is 1 at a gap of 0 and halves with every ``half_life`` of gap (NaN stays NaN)."""
    return 0.5 ** (np.abs(gap) / half_life)


def usable(df: pd.DataFrame | pd.Series, field: str) -> pd.Series | float:
    """``field`` where it isn't flagged doubtful (``<field>_doubtful``), NaN where it is or is missing.

    Works on the whole portfolio (returns a column, keeping its index) or on one listing (returns a single value):
    the value is multiplied by 1 where it's trusted and by NaN where it's doubtful.
    """
    return df[field] * np.where(df[f"{field}_doubtful"], np.nan, 1)


# --- Age (notebook 02) -------------------------------------------------------------------------------------------


def get_year_band(year: float | pd.Series, width: int = AGE_BAND_YEARS) -> float | pd.Series:
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


# --- Location (notebook 03) --------------------------------------------------------------------------------------


def has_usable_location(df: pd.DataFrame | pd.Series) -> pd.Series | bool:
    """True where the location can be trusted: high or medium confidence (EDA 2.2). Works on a frame or one row.

    Location quality is recorded as a confidence level rather than a ``_doubtful`` flag, so it has its own helper.
    """
    return df["location_confidence"] != "unknown"


def _miles_from(listing: pd.Series, points: pd.DataFrame) -> pd.Series:
    """Straight-line miles from the listing to each point (rows with ``latitude`` and ``longitude``)."""
    return haversine_miles(listing["latitude"], listing["longitude"], points["latitude"], points["longitude"])


def estate_centres(portfolio: pd.DataFrame) -> pd.DataFrame:
    """One point per estate: the average coordinates of its units with a usable location."""
    located = portfolio[has_usable_location(portfolio)]
    return located.groupby("industrial_estate_name")[["latitude", "longitude"]].mean()


def sits_on_estate(listing: pd.Series, portfolio: pd.DataFrame) -> str | None:
    """The portfolio estate the listing sits on, or None.

    A listing sits on an estate if its nearest portfolio unit (usable locations only) is within
    ``SITS_ON_ESTATE_MILES``. None if it's further from every unit, or has no usable location itself.
    """
    if not has_usable_location(listing):
        return None
    located = portfolio[has_usable_location(portfolio)]
    miles = _miles_from(listing, located)
    if miles.min() > SITS_ON_ESTATE_MILES:
        return None
    return located.loc[miles.idxmin(), "industrial_estate_name"]


def _mean_miles_to_nearest_estates(point: pd.Series, centres: pd.DataFrame, leave_out: str | None) -> float:
    """Mean miles from a point to its ``LOCATION_NEIGHBOUR_ESTATES`` nearest estates, leaving out ``leave_out``."""
    others = centres.drop(index=leave_out) if leave_out is not None else centres
    return _miles_from(point, others).nsmallest(LOCATION_NEIGHBOUR_ESTATES).mean()


def _portfolio_mean_miles(centres: pd.DataFrame) -> pd.Series:
    """For every portfolio estate, the mean miles to its nearest other estates (the yardstick for listings)."""
    return pd.Series({name: _mean_miles_to_nearest_estates(c, centres, name) for name, c in centres.iterrows()})


def location_homogeneity(listing: pd.Series, portfolio: pd.DataFrame) -> float:
    """How well placed the listing is among Indurent's estates, from 0 to 1 (prompt 2).

    Measures the mean miles from the listing to its ``LOCATION_NEIGHBOUR_ESTATES`` nearest portfolio estates, leaving
    out the estate it sits on (that's reported by ``location_context`` instead). The score is the same measure for
    the best-placed portfolio estate divided by the listing's, so the best-placed estate scores 1 and the score
    halves each time the distance doubles. For example, if the best-placed estate's nearest estates average 8.5 miles
    and a listing's average 36 miles, it scores 8.5 / 36 = 0.24. Capped at 1.

    Returns NaN ("not available") if the listing has no usable location.
    """
    if not has_usable_location(listing):
        return np.nan
    centres = estate_centres(portfolio)
    listing_miles = _mean_miles_to_nearest_estates(listing, centres, sits_on_estate(listing, portfolio))
    best_miles = _portfolio_mean_miles(centres).min()
    return min(1.0, best_miles / listing_miles)


def location_context(listing: pd.Series, portfolio: pd.DataFrame) -> dict:
    """Facts that explain the location score in an answer; none of them is scored.

    * ``sits_on_estate``: the portfolio estate the listing sits on, or None.
    * ``nearest_other_estate`` and ``nearest_other_estate_miles``: the closest other estate and its distance.
    * ``better_placed_than_pct``: the share of portfolio estates whose nearest estates are further away than the
      listing's (a percentile; not the score, because it stretches small differences).

    All values are None if the listing has no usable location.
    """
    keys = ["sits_on_estate", "nearest_other_estate", "nearest_other_estate_miles", "better_placed_than_pct"]
    if not has_usable_location(listing):
        return dict.fromkeys(keys)
    centres = estate_centres(portfolio)
    sits_on = sits_on_estate(listing, portfolio)
    others = centres.drop(index=sits_on) if sits_on is not None else centres
    miles = _miles_from(listing, others)
    listing_miles = _mean_miles_to_nearest_estates(listing, centres, sits_on)
    portfolio_miles = _portfolio_mean_miles(centres)
    return {
        "sits_on_estate": sits_on,
        "nearest_other_estate": miles.idxmin(),
        "nearest_other_estate_miles": miles.min(),
        "better_placed_than_pct": (portfolio_miles > listing_miles).mean() * 100,
    }


def location_similarity(listing: pd.Series, portfolio: pd.DataFrame) -> pd.Series:
    """How close the listing is to each portfolio unit, from 0 to 1 (prompt 1).

    The same spot scores 1, and the score halves for every ``LOCATION_HALF_LIFE_MILES`` miles between them: with
    10-mile halving, a unit 10 miles away scores 0.5 and one 20 miles away 0.25. Units on the estate the listing
    sits on score about 1.

    Returns one score per portfolio unit, on the portfolio's index. A unit without a usable location gets NaN
    ("not available"), and so does every unit if the listing has no usable location.
    """
    if not has_usable_location(listing):
        return pd.Series(np.nan, index=portfolio.index)
    miles = _miles_from(listing, portfolio).where(has_usable_location(portfolio))
    return halving(miles, LOCATION_HALF_LIFE_MILES)
