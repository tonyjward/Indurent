"""The tools the AI calls: plain Python functions over the cleaned data, with no LangGraph, model or API key.

Each tool takes simple arguments a model can produce (``listing="M3"``, ``k=10``, a list of filter conditions) and
returns a small JSON-friendly dictionary: a one-line ``summary``, the rows that matter (Property Ids, names, scores,
key facts) and ``caveats`` the answer should pass on. The model never receives the whole table.

The filtered set of portfolio units (``candidates``, a list of Property Ids; None means the whole portfolio) is
passed in and returned, not held here: ``graph.py`` keeps it in the conversation state. That keeps these functions
predictable, and usable from a notebook or the app without an AI.

Every tool takes the data as its first argument (``Data``); ``graph.py`` fills that in, so the model only sees the
other arguments. Bad arguments raise ``ToolError`` with a message written for the model to read and correct.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from indurent_matching.similarity import (
    BLOCKS,
    homogeneity_scores,
    miles_to_units,
    most_similar,
    sits_on_estate,
    usable,
)


class ToolError(ValueError):
    """A tool was called with arguments it can't use. The message tells the model what would work."""


@dataclass(frozen=True)
class Data:
    """The cleaned portfolio and marketed listings, as returned by ``load_portfolio`` and ``load_marketed``."""

    portfolio: pd.DataFrame
    marketed: pd.DataFrame


# --- Helpers -----------------------------------------------------------------------------------------------------


def _value(x, digits: int | None = None):
    """A JSON-friendly version of one value: NaN becomes None, numpy numbers become Python numbers."""
    if x is None or (isinstance(x, float | np.floating) and np.isnan(x)):
        return None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, float | np.floating):
        if digits == 0:
            return round(float(x))  # years, sizes: whole numbers (round with no digits gives an int)
        return round(float(x), digits) if digits is not None else float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _resolve_listing(data: Data, listing: str) -> pd.Series:
    """Find one marketed listing by id ("M3") or by name ("Spitfire Park", "Spitfire Park, Unit 42")."""
    marketed = data.marketed
    options = ", ".join(f"{i} ({row['industrial_estate_name']}, {row['unit_name']})" for i, row in marketed.iterrows())
    wanted = str(listing).strip().lower()
    by_id = marketed[marketed["listing_id"].str.lower() == wanted]
    if len(by_id) == 1:
        return by_id.iloc[0]
    full_name = (marketed["industrial_estate_name"] + ", " + marketed["unit_name"]).str.lower()
    by_name = marketed[(full_name == wanted) | (marketed["industrial_estate_name"].str.lower() == wanted)]
    if len(by_name) == 1:
        return by_name.iloc[0]
    partial = marketed[full_name.str.contains(wanted, regex=False)] if wanted else marketed.iloc[0:0]
    if len(partial) == 1:
        return partial.iloc[0]
    if len(partial) > 1:
        matches = ", ".join(
            f"{i} ({row['industrial_estate_name']}, {row['unit_name']})" for i, row in partial.iterrows()
        )
        raise ToolError(f"'{listing}' matches more than one listing; use one of: {matches}")
    raise ToolError(f"No marketed listing matches '{listing}'. The listings are: {options}")


def _matched_on(listing: pd.Series) -> dict:
    """Which blocks a listing can be compared on, and why any can't."""
    blocks, missing = [], {}
    if pd.notna(usable(listing, "build_year")):
        blocks.append("age")
    else:
        missing["age"] = "no build year"
    if listing["location_confidence"] != "unknown":
        blocks.append("location")
    elif pd.isna(listing["latitude"]) or pd.isna(listing["longitude"]):
        missing["location"] = "no coordinates"
    else:
        missing["location"] = "coordinates are offshore, so can't be trusted"
    if pd.notna(usable(listing, "size_sqft")):
        blocks.append("physical")
    else:
        missing["physical"] = "no usable size"
    return {"matched_on": blocks, "not_available": missing}


def _listing_facts(listing: pd.Series) -> dict:
    return {
        "listing_id": listing["listing_id"],
        "estate": listing["industrial_estate_name"],
        "unit": listing["unit_name"],
        "region_label": _value(listing["region"]),
        "size_sqft": _value(listing["size_sqft"], 0),
        "build_year": _value(listing["build_year"], 0),
        "location_confidence": listing["location_confidence"],
        "nearest_city": _value(listing["nearest_city"]),
        "miles_to_city": _value(listing["miles_to_city"], 1),
        "epc_rating": _value(listing["epc_rating"]),
        **_matched_on(listing),
    }


def _candidate_rows(data: Data, candidates: list[int] | None) -> pd.DataFrame:
    """The portfolio rows in the candidate set (the whole portfolio when ``candidates`` is None)."""
    if candidates is None:
        return data.portfolio
    unknown = sorted(set(candidates) - set(data.portfolio.index))
    if unknown:
        raise ToolError(f"Unknown Property Ids in candidates: {unknown[:10]}")
    return data.portfolio.loc[sorted(set(candidates))]


# --- Listings ----------------------------------------------------------------------------------------------------


def list_listings(data: Data) -> dict:
    """List the marketed listings that can be compared with the portfolio, with what each can be matched on."""
    listings = [_listing_facts(row) for _, row in data.marketed.iterrows()]
    return {
        "summary": f"{len(listings)} marketed listings: {', '.join(x['listing_id'] for x in listings)}",
        "listings": listings,
    }


def get_listing(data: Data, listing: str) -> dict:
    """One marketed listing's facts and data-quality notes, found by id ("M3") or name ("Spitfire Park").

    Use this to resolve which property the user means. If a request says "the marketed property" and doesn't say
    which, ask, or answer for every listing; never pick one silently.
    """
    row = _resolve_listing(data, listing)
    facts = _listing_facts(row)
    facts["sits_on_estate"] = sits_on_estate(row, data.portfolio)
    facts["quality_notes"] = list(row["quality_notes"])
    return {"summary": f"{facts['listing_id']}: {facts['estate']}, {facts['unit']}", **facts}


# --- Filtering ---------------------------------------------------------------------------------------------------

# Fields the model may filter on, with how to read each one. Doubtful values count as unknown.
FILTER_FIELDS = {
    "miles_to_city": "number",  # miles to the nearest major city; unknown where the location isn't trusted
    "nearest_city": "text",
    "size_sqft": "number",
    "build_year": "number",
    "derived_region": "text",  # the region the coordinates fall in (the raw region label is unreliable)
    "location_confidence": "text",  # high / medium / unknown
    "epc_rating": "text",  # A-G; unreliable in this data (EDA 2.9), so answers should say so
    "estate_units": "number",  # units in the unit's estate
    "car_parking_spaces": "number",  # an estate-level figure
}
OPERATORS = {"number": {"<=", ">=", "<", ">", "==", "in", "between"}, "text": {"==", "in"}}


def _field_values(portfolio: pd.DataFrame, field: str) -> pd.Series:
    """A field's values for filtering, with doubtful values treated as unknown (NaN)."""
    if f"{field}_doubtful" in portfolio:
        return usable(portfolio, field)
    return portfolio[field]


def _passes(values: pd.Series, op: str, value) -> pd.Series:
    """True where a known value meets the condition (unknown values give False; they're handled separately)."""
    if op == "in":
        return values.isin(value)
    if op == "between":
        low, high = value
        return values.between(low, high)
    return {
        "<=": values <= value,
        ">=": values >= value,
        "<": values < value,
        ">": values > value,
        "==": values == value,
    }[op]


def _check_condition(condition: dict) -> tuple[str, str, object]:
    field, op, value = condition.get("field"), condition.get("op"), condition.get("value")
    if field not in FILTER_FIELDS:
        raise ToolError(f"Can't filter on '{field}'. Fields: {sorted(FILTER_FIELDS)}")
    kind = FILTER_FIELDS[field]
    if op not in OPERATORS[kind]:
        raise ToolError(f"'{op}' doesn't apply to {field} ({kind}). Use one of: {sorted(OPERATORS[kind])}")
    if op == "in" and not isinstance(value, list):
        raise ToolError(f"'in' needs a list of values for {field}")
    if op == "between" and not (isinstance(value, list) and len(value) == 2):
        raise ToolError(f"'between' needs [low, high] for {field}")
    return field, op, value


def _describe(field: str, op: str, value) -> str:
    if op == "between":
        return f"{field} between {value[0]} and {value[1]}"
    if op == "in":
        return f"{field} in {value}"
    return f"{field} {op} {value}"


def filter_rows(data: Data, conditions: list[dict], include_unknown: bool, candidates: list[int] | None = None) -> dict:
    """Narrow the portfolio units to those meeting every condition, and return the new candidate set.

    Each condition is ``{"field": ..., "op": ..., "value": ...}``; all must hold. Ops are ``<=``, ``>=``, ``<``,
    ``>``, ``==``, ``in`` (a list) and ``between`` ([low, high], inclusive); text fields take ``==`` and ``in`` only.

    ``include_unknown`` is required: it decides what happens to units with no usable value for a condition's field
    (for distances, units whose location can't be trusted). True keeps them, False drops them; the summary always
    reports how many there were. **Default to False** (exclude them) unless the user asks to include them, or only
    asks to remove units *known* to fail the condition. Always say in the answer how many were left out, and offer
    to include them.

    Direction matters for wording like the brief's prompt 3: "excluding any property more than 10 miles from a
    major city" means **keep** ``miles_to_city <= 10``. Prompt 3 then asks for the closest properties: pass the
    returned ``candidates`` to ``find_nearest``.

    Starts from ``candidates`` (None: the whole portfolio), so filters can be applied one after another.
    """
    if not isinstance(include_unknown, bool):
        raise ToolError("include_unknown must be true or false: decide what happens to units with no value")
    if not conditions:
        raise ToolError("Give at least one condition")
    checked = [_check_condition(c) for c in conditions]
    rows = _candidate_rows(data, candidates)
    keep = pd.Series(True, index=rows.index)
    unknown = pd.Series(False, index=rows.index)
    for field, op, value in checked:
        values = _field_values(rows, field)
        missing = values.isna()
        unknown |= missing
        keep &= _passes(values, op, value) | (missing & include_unknown)
    kept = rows.index[keep]
    described = " and ".join(_describe(*c) for c in checked)
    fields = sorted({field for field, _, _ in checked})
    summary = (
        f"Kept {len(kept):,} of {len(rows):,} units ({described}). "
        f"{int(unknown.sum()):,} had no usable value for {', '.join(fields)} and were "
        f"{'kept' if include_unknown else 'excluded'}."
    )
    return {
        "summary": summary,
        "candidates": [int(i) for i in kept],
        "kept": len(kept),
        "started_with": len(rows),
        "unknown": int(unknown.sum()),
        "unknown_included": include_unknown,
        "conditions": [{"field": f, "op": o, "value": v} for f, o, v in checked],
    }


# --- Prompt 1: most similar --------------------------------------------------------------------------------------


def _match_row(property_id, row: pd.Series, portfolio: pd.DataFrame) -> dict:
    doubtful = [
        f
        for f in ("size_sqft", "build_year", "min_eaves_m", "max_eaves_m", "doors")
        if portfolio.at[property_id, f"{f}_doubtful"]
    ]
    return {
        "property_id": int(property_id),
        "estate": row["industrial_estate_name"],
        "unit": row["unit_name"],
        "overall": _value(row["overall"], 3),
        **{block: _value(row[block], 3) for block in BLOCKS},
        "build_year": _value(row["build_year"], 0),
        "size_sqft": _value(row["size_sqft"], 0),
        "miles_from_listing": _value(row["miles_from_listing"], 1),
        "doubtful_fields": doubtful,
    }


def find_similar(
    data: Data,
    listing: str,
    k: int = 10,
    weights: dict[str, float] | None = None,
    max_per_estate: int | None = None,
    candidates: list[int] | None = None,
) -> dict:
    """The k portfolio units most similar to a marketed listing, best first (the brief's prompt 1).

    Similarity is scored separately on age, location and physical (size), each 0-1, and averaged with equal weights
    unless ``weights`` says otherwise (only when the user asks for one dimension to count more). Blocks the listing
    can't be compared on are dropped; a unit's unknown block counts as 0 and is shown as null.

    There's no cap on units per estate: units on the estate a listing sits on are genuinely among the most similar.
    Set ``max_per_estate`` only when the user asks for a spread (e.g. "similar units on other estates").

    Searches ``candidates`` (None: the whole portfolio), e.g. the set returned by ``filter_rows``.

    For the *nearest* or *closest* properties on the map (the brief's prompt 3), use ``find_nearest`` instead.
    """
    if not isinstance(k, int) or not 1 <= k <= 50:
        raise ToolError("k must be a whole number from 1 to 50")
    row = _resolve_listing(data, listing)
    pool = _candidate_rows(data, candidates)
    if pool.empty:
        raise ToolError("The candidate set is empty: there's nothing to compare against")
    try:
        result = most_similar(row, pool, k=k, weights=weights, max_per_estate=max_per_estate)
    except ValueError as error:
        raise ToolError(str(error)) from error

    matches = [_match_row(pid, r, data.portfolio) for pid, r in result.iterrows()]
    blocks = _matched_on(row)
    caveats = []
    if blocks["not_available"]:
        missing = "; ".join(f"{block} ({reason})" for block, reason in blocks["not_available"].items())
        caveats.append(f"Matched on {' and '.join(blocks['matched_on'])} only. Not available: {missing}.")
    estate = sits_on_estate(row, data.portfolio)
    on_estate = int((result["industrial_estate_name"] == estate).sum()) if estate else 0
    if estate:
        caveats.append(
            f"The listing sits on {estate}, an estate Indurent already owns; {on_estate} of the "
            f"{len(result)} matches are on it."
        )
    if "location" in blocks["matched_on"]:
        unknown = int(result["location"].isna().sum())
        if unknown:
            caveats.append(f"{unknown} match(es) have no usable location (counted as 0 on location).")
    if blocks["matched_on"] == ["physical"] and len(result):
        everything = most_similar(row, pool, k=len(pool))
        threshold = round(result["overall"].iloc[-1], 2)
        ties = int((everything["overall"].round(2) >= threshold).sum())
        if ties > len(result):
            caveats.append(
                f"{ties} units score {threshold:.2f} or more on size alone, so the list is one of several "
                "equally good ones."
            )
    if candidates is not None:
        caveats.append(f"Searched a filtered set of {len(pool):,} units, not the whole portfolio.")
    if weights:
        caveats.append(f"Weights changed from equal: {weights}.")
    if max_per_estate is not None:
        caveats.append(f"At most {max_per_estate} unit(s) per estate, as requested.")
    return {
        "summary": f"Top {len(matches)} matches for {row['listing_id']} ({row['industrial_estate_name']}, "
        f"{row['unit_name']}) from {len(pool):,} units",
        "listing": _listing_facts(row),
        "matches": matches,
        "caveats": caveats,
    }


# --- Prompt 3: nearest on the map ---------------------------------------------------------------------------------


def find_nearest(data: Data, listing: str, k: int = 10, candidates: list[int] | None = None) -> dict:
    """The k portfolio units nearest to a marketed listing on the map, by straight-line miles (the brief's prompt 3).

    "Closest properties" means nearest on the map, not most similar (use ``find_similar`` for that). For prompt 3,
    first narrow the portfolio with ``filter_rows`` (keep ``miles_to_city <= 10``) and pass its ``candidates``.

    Units without a usable location can't be placed, so they're left out, and the caveats say how many. A listing
    without a usable location (no coordinates, or offshore) can't be used at all: that's an error, with the reason.
    """
    if not isinstance(k, int) or not 1 <= k <= 50:
        raise ToolError("k must be a whole number from 1 to 50")
    row = _resolve_listing(data, listing)
    blocks = _matched_on(row)
    if "location" not in blocks["matched_on"]:
        raise ToolError(
            f"{row['listing_id']} can't be placed on the map: {blocks['not_available']['location']}. "
            "Nearest properties can't be found for it; say so in the answer."
        )
    pool = _candidate_rows(data, candidates)
    miles = miles_to_units(row, pool)
    placed = miles.dropna().sort_index().sort_values(kind="stable")  # ties: Property Id order
    nearest = placed.head(k)
    matches = [
        {
            "property_id": int(pid),
            "estate": pool.at[pid, "industrial_estate_name"],
            "unit": pool.at[pid, "unit_name"],
            "miles_from_listing": _value(m, 1),
            "build_year": _value(pool.at[pid, "build_year"], 0),
            "size_sqft": _value(pool.at[pid, "size_sqft"], 0),
            "nearest_city": _value(pool.at[pid, "nearest_city"]),
            "miles_to_city": _value(pool.at[pid, "miles_to_city"], 1),
        }
        for pid, m in nearest.items()
    ]
    caveats = []
    unplaced = int(miles.isna().sum())
    if unplaced:
        caveats.append(
            f"{unplaced:,} of the {len(pool):,} units searched have no usable location, so couldn't be ranked."
        )
    estate = sits_on_estate(row, data.portfolio)
    if estate:
        on_estate = sum(m["estate"] == estate for m in matches)
        caveats.append(
            f"The listing sits on {estate}, an estate Indurent already owns; {on_estate} of the "
            f"{len(matches)} nearest are on it."
        )
    if candidates is not None:
        caveats.append(f"Searched a filtered set of {len(pool):,} units, not the whole portfolio.")
    return {
        "summary": f"The {len(matches)} units nearest to {row['listing_id']} ({row['industrial_estate_name']}, "
        f"{row['unit_name']}) on the map, from {len(placed):,} units with a usable location",
        "listing": _listing_facts(row),
        "matches": matches,
        "caveats": caveats,
    }


# --- Prompt 2: homogeneity ---------------------------------------------------------------------------------------

HOMOGENEITY_MEANING = (
    "Homogeneity here means how typical the listing is of Indurent's portfolio, scored 0-1 on each dimension, "
    "where 1 is the most typical value in the portfolio. 'Correlation score' in the request is read this way."
)


def get_homogeneity_scores(data: Data, listing: str) -> dict:
    """How typical a marketed listing is of the portfolio, scored separately on age, location and physical (prompt 2).

    Each score is 0-1, where 1 is the most typical value in the portfolio; null means not available, with the reason
    in ``not_available``. ``location_context`` and ``size_context`` explain the scores and aren't scored. State the
    interpretation (``meaning``) in the answer.
    """
    row = _resolve_listing(data, listing)
    result = homogeneity_scores(row, data.portfolio)
    blocks = _matched_on(row)
    return {
        "summary": f"Homogeneity scores for {row['listing_id']} ({row['industrial_estate_name']}, {row['unit_name']})",
        "meaning": HOMOGENEITY_MEANING,
        "scores": {block: _value(result[block], 2) for block in BLOCKS},
        "not_available": blocks["not_available"],
        "location_context": {key: _value(v, 1) for key, v in result["location_context"].items()},
        "size_context": {key: _value(v, 1) for key, v in result["size_context"].items()},
        "listing": _listing_facts(row),
    }
