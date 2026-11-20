"""Load and clean the portfolio and marketed-listing spreadsheets.

Each step follows a section of ``notebooks/01_eda.ipynb``, where the evidence for every rule is shown.
The functions here only apply the decisions; they don't re-derive them.

Step 0: load the raw files, rename columns to snake_case, key the portfolio by its existing ``property_id`` and
give the marketed listings ids (M1-M5, since their Property Id column is empty), parse the numeric columns, and
describe the estate structure.

Step 1 (EDA 2.1): size is square feet; flag implausibly small units, note very large ones.

Step 2 (EDA 2.2, section 5): location confidence from the coastline and labelled-region checks, the region the
coordinates fall in, and the nearest major city (trusted locations only). See ``geo.py``.

Step 3 (EDA 2.3): build year is left missing where blank (no filling from the estate) and noted.

Step 4 (EDA 2.4): car parking needs no cleaning beyond step 0's parsing. It is an estate-level figure (the same on
every unit of an estate) that doesn't track unit size, so it is kept for filtering and context but not used for
similarity; that exclusion lives in the similarity code, not here.

Step 5 (EDA 2.5): zero eaves mean "not recorded" and become missing; each side is kept where recorded.

Quality is recorded on each row in two ways:

* ``<field>_doubtful`` (True/False): the value is kept but not trusted, so similarity skips it.
* ``quality_notes`` (list of str): plain-English reasons, passed to the AI so answers can explain them.
  A note without a flag is context only (e.g. "very large"); it doesn't remove the value from scoring.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from indurent_matching.config import MIN_PLAUSIBLE_SQFT, VERY_LARGE_SQFT
from indurent_matching.geo import assess_locations, nearest_city

# Columns parsed as numbers, under their snake_case names. size_sqm holds square feet (EDA 2.1); it is renamed
# to size_sqft in the size step, where that decision is made. region is the dataset's own label, which only
# agrees with the coordinates 26% of the time (EDA 2.2).
NUMERIC = [
    "latitude",
    "longitude",
    "car_parking_spaces",
    "size_sqm",
    "build_year",
    "yard_depth_m",
    "min_eaves_m",
    "max_eaves_m",
    "doors",
]

_RANGE = re.compile(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)")


def parse_number(value: object) -> float:
    """Convert one spreadsheet cell to a float.

    Numbers pass through. Text that is a number ("12", " -1.5 ") is converted. A range such as "20-28"
    becomes its midpoint (EDA 2.7). Anything else, including the placeholders seen in the data
    (``X``, ``-``, ``TBC``), blanks and booleans, becomes NaN.

    Zeros are returned as 0.0: whether zero means "not recorded" depends on the column, and is decided in
    that column's step.
    """
    if isinstance(value, bool) or value is None:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass
    match = _RANGE.fullmatch(text)
    if match:
        low, high = (float(g) for g in match.groups())
        return (low + high) / 2
    return np.nan


def snake_case(name: str) -> str:
    """Lower-case a column name, drop punctuation and replace spaces with underscores.

    "Car Parking Spaces #" -> "car_parking_spaces", "Min. Eaves m" -> "min_eaves_m".
    """
    return re.sub(r"\s+", "_", re.sub(r"[^a-z0-9 ]", "", name.lower()).strip())


def _read(path: Path) -> pd.DataFrame:
    """Read a spreadsheet with every cell kept as Excel holds it, and snake_case the column names."""
    df = pd.read_excel(path, dtype=object)
    df.columns = [snake_case(c) for c in df.columns]
    required = {"property_id", "industrial_estate_name", "unit_name", *NUMERIC}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    return df


def _parse_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Apply ``parse_number`` to every numeric column."""
    df = df.copy()
    for col in NUMERIC:
        df[col] = df[col].map(parse_number).astype(float)
    return df


def _add_note(df: pd.DataFrame, mask: pd.Series, template: str) -> None:
    """Append a note to ``quality_notes`` for every row where ``mask`` is True (in place).

    ``template`` is formatted with that row's values, e.g. "size {size_sqft:,.0f} sq ft: ...".
    """
    for idx in df.index[mask]:
        df.at[idx, "quality_notes"] = [*df.at[idx, "quality_notes"], template.format(**df.loc[idx])]


def clean_size(df: pd.DataFrame) -> pd.DataFrame:
    """Step 1 (EDA 2.1): read size as square feet, flag implausibly small units, note very large ones.

    The column is labelled sqm, but its total (30.6m) matches Indurent's published 28-40m sq ft and eaves
    heights fit the sq ft reading, so the values are kept and the column renamed to ``size_sqft``. Thresholds are
    in ``config.py``.
    """
    df = df.rename(columns={"size_sqm": "size_sqft"})
    small = df["size_sqft"] < MIN_PLAUSIBLE_SQFT
    df["size_sqft_doubtful"] = small
    _add_note(
        df,
        small,
        f"size {{size_sqft:,.0f}} sq ft: below Indurent's smallest unit band ({MIN_PLAUSIBLE_SQFT:,} sq ft), so doubtful",
    )
    _add_note(
        df,
        df["size_sqft"] > VERY_LARGE_SQFT,
        f"size {{size_sqft:,.0f}} sq ft: very large (over {VERY_LARGE_SQFT:,} sq ft); rare but plausible",
    )
    return df


def clean_location(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2 (EDA 2.2, section 5): judge how far each row's coordinates can be trusted.

    Adds ``km_offshore``, ``km_to_region``, ``location_confidence`` (high / medium / unknown), ``derived_region``,
    ``nearest_city`` and ``miles_to_city``. Coordinates are kept as recorded; ``location_confidence`` plays the
    role of a doubtful flag, so there's no ``latitude_doubtful``. Unknown locations get no city distance.
    """
    df = pd.concat([df, assess_locations(df)], axis=1)
    df = pd.concat([df, nearest_city(df, df["location_confidence"] != "unknown")], axis=1)

    no_coords = df["latitude"].isna() | df["longitude"].isna()
    offshore = df["location_confidence"].eq("unknown") & ~no_coords
    outside = df["location_confidence"].eq("medium") & df["km_to_region"].notna()
    no_label = df["location_confidence"].eq("medium") & df["km_to_region"].isna()
    _add_note(df, no_coords, "no coordinates, so location can't be used")
    _add_note(df, offshore, "coordinates are {km_offshore:.0f} km offshore, so location can't be used")
    _add_note(
        df,
        outside,
        "location medium confidence: {km_to_region:.0f} km outside labelled region ({region}), "
        "falls in {derived_region}",
    )
    _add_note(df, no_label, "no region label to check the coordinates against, so location is medium confidence")
    return df


def clean_build_year(df: pd.DataFrame) -> pd.DataFrame:
    """Step 3 (EDA 2.3): note missing build years; leave them missing.

    Units in the same estate differ by about 30 years (median), so filling a blank from the estate would be a
    guess. A unit without a build year simply has no age comparison.
    """
    df = df.copy()
    _add_note(df, df["build_year"].isna(), "no build year, so age can't be compared")
    return df


def clean_eaves(df: pd.DataFrame) -> pd.DataFrame:
    """Step 5 (EDA 2.5): treat zero eaves as "not recorded".

    About 30% of each column is zero, and no actual warehouse has 0 m eaves. Min and max are cleaned separately, so a
    unit with only one side recorded keeps it. Once zeros are missing, no unit has min above max (the 49 raw cases
    were all max = 0), so nothing is swapped.

    Eaves are copied across most multi-unit estates, but unlike doors they stay a feature for every unit: units
    built together plausibly share a height, whereas one door count across units of very different sizes doesn't.
    Missing eaves get no note; similarity reports which features each match rested on.
    """
    df = df.copy()
    for col in ["min_eaves_m", "max_eaves_m"]:
        df[col] = df[col].mask(df[col] == 0)
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning steps shared by both files, in EDA order."""
    df = _parse_numeric(df)
    df["quality_notes"] = [
        [] for _ in range(len(df))
    ]  # starts an empty list placeholder for every row, later steps append to it using _add_note
    df = clean_size(df)
    df = clean_location(df)
    df = clean_build_year(df)
    df = clean_eaves(df)
    return df


def add_estate_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``estate_units`` (units in the row's estate) and ``is_single_unit``.

    Each row is one unit; an estate can hold one unit or many (EDA section 1). Values copied across a
    multi-unit estate are estate-level facts, which later steps need to know.
    """
    df = df.copy()
    df["estate_units"] = df.groupby("industrial_estate_name")["industrial_estate_name"].transform("size").astype(int)
    df["is_single_unit"] = df["estate_units"] == 1
    return df


def load_portfolio(path: str | Path) -> pd.DataFrame:
    """Load the portfolio file: one row per unit, keyed by the integer ``property_id``."""
    df = _clean(_read(Path(path)))
    if df["property_id"].isna().any() or not df["property_id"].is_unique:
        raise ValueError("portfolio property_id must be present and unique")
    df["property_id"] = df["property_id"].astype(int)
    return add_estate_structure(df).set_index("property_id", drop=False)


def load_marketed(path: str | Path) -> pd.DataFrame:
    """Load the marketed-listings file, keyed by ``listing_id`` ("M1", "M2", ... in file order).

    The file has an empty Property Id column, so listings get their own ids. Estate structure is not
    added: a listing's estate isn't in the portfolio, so the file can't say how many units it has.
    """
    df = _clean(_read(Path(path))).drop(columns="property_id")
    df.insert(0, "listing_id", [f"M{n}" for n in range(1, len(df) + 1)])
    return df.set_index("listing_id", drop=False)
