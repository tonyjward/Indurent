"""Load and clean the portfolio and marketed-listing spreadsheets.

Each step follows a section of ``notebooks/01_eda.ipynb``, where the evidence for every rule is shown.
The functions here only apply the decisions; they don't re-derive them.

Step 0 (this file so far): load the raw files, rename columns to snake_case, key the portfolio by its existing
``property_id`` and give the marketed listings ids (M1-M5, since their Property Id column is empty), parse the
numeric columns, and describe the estate structure.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

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
    df = _parse_numeric(_read(Path(path)))
    if df["property_id"].isna().any() or not df["property_id"].is_unique:
        raise ValueError("portfolio property_id must be present and unique")
    df["property_id"] = df["property_id"].astype(int)
    return add_estate_structure(df).set_index("property_id", drop=False)


def load_marketed(path: str | Path) -> pd.DataFrame:
    """Load the marketed-listings file, keyed by ``listing_id`` ("M1", "M2", ... in file order).

    The file has an empty Property Id column, so listings get their own ids. Estate structure is not
    added: a listing's estate isn't in the portfolio, so the file can't say how many units it has.
    """
    df = _parse_numeric(_read(Path(path))).drop(columns="property_id")
    df.insert(0, "listing_id", [f"M{n}" for n in range(1, len(df) + 1)])
    return df.set_index("listing_id", drop=False)
