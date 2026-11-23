"""The tools the AI calls: plain Python functions over the cleaned data, with no LangGraph, model or API key.

Each tool takes simple arguments a model can produce (`listing="M3"`, `k=10`, a list of filter conditions) and
returns a small JSON-friendly dictionary: a one-line `summary`, the rows that matter (Property Ids, names, scores,
key facts) and `caveats` the answer should pass on. The model never receives the whole table.

The filtered set of portfolio units (`candidates`, a list of Property Ids; None means the whole portfolio) is
passed in and returned, not held here: `graph.py` keeps it in the conversation state. That keeps these functions
predictable, and usable from a notebook or the app without an AI.

Every tool takes the data as its first argument (`Data`); `graph.py` fills that in, so the model only sees the
other arguments. Bad arguments raise `ToolError` with a message written for the model to read and correct.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from indurent_matching.config import SITS_ON_ESTATE_MILES
from indurent_matching.similarity import (
    DIMENSIONS,
    has_usable_location,
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
    """The cleaned portfolio and marketed listings, as returned by `load_portfolio` and `load_marketed`."""

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
    """Which dimensions a listing can be compared on, and why any can't."""
    dimensions, missing = [], {}
    if pd.notna(usable(listing, "build_year")):
        dimensions.append("age")
    else:
        missing["age"] = "no build year"
    if listing["location_confidence"] != "unknown":
        dimensions.append("location")
    elif pd.isna(listing["latitude"]) or pd.isna(listing["longitude"]):
        missing["location"] = "no coordinates"
    else:
        missing["location"] = "coordinates are offshore, so can't be trusted"
    if pd.notna(usable(listing, "size_sqft")):
        dimensions.append("physical")
    else:
        missing["physical"] = "no usable size"
    return {"matched_on": dimensions, "not_available": missing}


def _listing_facts(listing: pd.Series) -> dict:
    """A listing's facts as the model sees them, with its data-quality notes.

    The notes go wherever the facts go (list_listings, every search), so the model never has to work out a
    data-quality problem for itself: in an eval it did, from a region label and a nearest city, and got it wrong.
    """
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
        "quality_notes": list(listing["quality_notes"]),
    }


def _distance_facts(units: pd.DataFrame) -> str:
    """How far an estate's units are from a major city, and how many have no usable location, as a sentence."""
    located = units.dropna(subset=["miles_to_city"])
    parts = []
    if len(located):
        nearest, furthest = round(located["miles_to_city"].min(), 1), round(located["miles_to_city"].max(), 1)
        miles = f"{nearest}" if nearest == furthest else f"{nearest}-{furthest}"
        city = located["nearest_city"].mode().iloc[0]
        parts.append(f"{len(located)} are {miles} miles from {city} (the nearest major city)")
    unknown = len(units) - len(located)
    if unknown:
        parts.append(f"{unknown} {'has' if unknown == 1 else 'have'} no usable location")
    return f"Of its {len(units)} units, {'; '.join(parts)}."


def _owned_estate_caveat(
    estate: str, on_list: int, listed: int, pool: pd.DataFrame, portfolio: pd.DataFrame, filtered: bool, noun: str
) -> str:
    """The caveat for a listing on an estate Indurent owns: how many of the results are on it.

    When a filtered search finds none on it, it also says why: the filter removed the estate's units (with their
    distance from a city and how many have no location, so the model can see which applies), or they were searched
    and ranked lower. Without these facts, the model guessed the reason (evals 2026-11-23).
    """
    caveat = (
        f"The listing sits on {estate}, an estate Indurent already owns (at least one of its units is within "
        f"{SITS_ON_ESTATE_MILES} miles of the listing); {on_list} of the {listed} {noun} are on it."
    )
    if filtered and on_list == 0:
        in_pool = int((pool["industrial_estate_name"] == estate).sum())
        if in_pool == 0:
            units = portfolio[portfolio["industrial_estate_name"] == estate]
            caveat += f" None of {estate}'s units are in the filtered set: the filter removed them. "
            caveat += _distance_facts(units)
        else:
            caveat += f" {in_pool} of its units are in the filtered set but rank below these."
    return caveat


def _candidate_rows(data: Data, candidates: list[int] | None) -> pd.DataFrame:
    """The portfolio rows in the candidate set (the whole portfolio when `candidates` is None)."""
    if candidates is None:
        return data.portfolio
    unknown = sorted(set(candidates) - set(data.portfolio.index))
    if unknown:
        raise ToolError(f"Unknown Property Ids in candidates: {unknown[:10]}")
    return data.portfolio.loc[sorted(set(candidates))]


# --- Listings ----------------------------------------------------------------------------------------------------


def list_listings(data: Data) -> dict:
    """List the marketed listings that can be compared with the portfolio, with what each can be matched on.

    Args:
        data: the cleaned portfolio and listings. Filled in by the agent; the model doesn't supply it.

    Returns:
        `summary`, and `listings`: one entry per listing with its facts, `matched_on` (the dimensions it can be
        compared on), `not_available` (the dimensions it can't, with the reason) and `quality_notes`.
    """
    listings = [_listing_facts(row) for _, row in data.marketed.iterrows()]
    summary = f"{len(listings)} marketed listings: {', '.join(x['listing_id'] for x in listings)}."

    # Spell out how many can't be compared on every dimension, so the model quotes the count rather than working it out.
    limited = []
    for x in listings:
        if x["not_available"]:
            limited.append(f"{x['listing_id']} (no {' or '.join(x['not_available'])})")
    if limited:
        summary += f" {len(limited)} can't be compared on every dimension: {', '.join(limited)}."
    else:
        summary += " All can be compared on every dimension."
    return {"summary": summary, "listings": listings}


def _estate_status(row: pd.Series, portfolio: pd.DataFrame) -> str:
    """The portfolio estate a listing sits on, in words that can't be misread.

    A blank (null) would mean two things: the listing isn't on an Indurent estate, or it has no usable location
    so nobody can tell. The model read the second as the first in an eval, so each case is spelled out.
    """
    if not has_usable_location(row):
        return f"unknown: {row['listing_id']} has no usable location"
    estate = sits_on_estate(row, portfolio)
    if estate is None:
        return "none: it isn't on an estate Indurent owns"
    return estate


def get_listing(data: Data, listing: str) -> dict:
    """One marketed listing's facts and data-quality notes, found by id ("M3") or name ("Spitfire Park").

    Use this to resolve which property the user means. If a request says "the marketed property" and doesn't say
    which, ask, or answer for every listing; never pick one silently.

    Args:
        data: the cleaned portfolio and listings. Filled in by the agent; the model doesn't supply it.
        listing: the listing's id ("M3") or name ("Spitfire Park", "Spitfire Park, Unit 42"). Case doesn't matter,
            and part of a name works if it matches only one listing.

    Returns:
        `summary`, the listing's facts, `matched_on` and `not_available`, `sits_on_estate` (the portfolio
        estate it sits on, "none: ..." or "unknown: ..." when it has no usable location) and `quality_notes`.

    Raises:
        ToolError: no listing matches, or more than one does. The message lists the options.
    """
    row = _resolve_listing(data, listing)
    facts = _listing_facts(row)
    facts["sits_on_estate"] = _estate_status(row, data.portfolio)
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
# Why a field's value can be unknown, so the summary says it and the model doesn't have to work it out.
UNKNOWN_BECAUSE = {
    "miles_to_city": "miles_to_city is unknown where a unit has no usable location",
    "nearest_city": "nearest_city is unknown where a unit has no usable location",
    "derived_region": "derived_region is unknown where a unit has no usable location",
    "size_sqft": "size_sqft is unknown where none is recorded or it's doubtful",
    "build_year": "build_year is unknown where none is recorded",
    "epc_rating": "epc_rating is unknown where none is recorded",
    "car_parking_spaces": "car_parking_spaces is unknown where none is recorded",
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

    Each condition is `{"field": ..., "op": ..., "value": ...}`; all must hold. Ops are `<=`, `>=`, `<`,
    `>`, `==`, `in` (a list) and `between` ([low, high], inclusive); text fields take `==` and `in` only.

    `include_unknown` is required: it decides what happens to units with no usable value for a condition's field
    (for distances, units whose location can't be trusted). True keeps them, False drops them; the summary always
    reports how many there were. **Default to False** (exclude them) unless the user asks to include them, or only
    asks to remove units *known* to fail the condition. Always say in the answer how many were left out, and offer
    to include them.

    Direction matters for wording like the brief's prompt 3: "excluding any property more than 10 miles from a
    major city" means **keep** `miles_to_city <= 10`. Prompt 3 then asks for the closest properties: pass the
    returned `candidates` to `find_nearest`.

    Starts from `candidates` (None: the whole portfolio), so filters can be applied one after another.

    Args:
        data: the cleaned portfolio and listings. Filled in by the agent; the model doesn't supply it.
        conditions: a list of `{"field": ..., "op": ..., "value": ...}`, all of which must hold.
            `field` is one of: miles_to_city, nearest_city, size_sqft, build_year, derived_region,
            location_confidence, epc_rating, estate_units, car_parking_spaces. `op` is `<=`, `>=`, `<`,
            `>`, `==`, `in` or `between` (text fields: `==` and `in` only). `value` is a number or
            text; a list for `in`; `[low, high]` for `between`.
        include_unknown: required. True keeps units with no usable value for a condition's field; False drops
            them (the default policy, above).
        candidates: the Property Ids to start from, e.g. those returned by a previous `filter_rows`. None (the
            default) starts from the whole portfolio.

    Returns:
        `summary`; `candidates` (the Property Ids kept, to pass to the next tool); `kept`,
        `started_with`, `unknown` and `unknown_included`; and `conditions` as applied.

    Raises:
        ToolError: an unknown field, an operator that doesn't apply to the field, a malformed value,
            `include_unknown` not true or false, or no conditions.
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
    reasons = [UNKNOWN_BECAUSE[field] for field in fields if field in UNKNOWN_BECAUSE]
    if unknown.any() and reasons:
        summary += f" ({'; '.join(reasons)}.)"
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


DOUBTFUL_FIELDS = ("size_sqft", "build_year", "min_eaves_m", "max_eaves_m", "doors")


def _as_matches(result: pd.DataFrame, portfolio: pd.DataFrame) -> list[dict]:
    """Turn the ranked table from `most_similar` into one JSON-friendly dictionary per unit, in rank order."""
    matches = []
    for property_id, row in result.iterrows():
        flags = portfolio.loc[property_id]
        matches.append(
            {
                "property_id": int(property_id),
                "estate": row["industrial_estate_name"],
                "unit": row["unit_name"],
                "overall": _value(row["overall"], 3),
                "age": _value(row["age"], 3),
                "location": _value(row["location"], 3),
                "physical": _value(row["physical"], 3),
                "build_year": _value(row["build_year"], 0),
                "size_sqft": _value(row["size_sqft"], 0),
                "miles_from_listing": _value(row["miles_from_listing"], 1),
                # Fields recorded for this unit but flagged doubtful, so an answer can mention them.
                "doubtful_fields": [field for field in DOUBTFUL_FIELDS if flags[f"{field}_doubtful"]],
            }
        )
    return matches


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
    unless `weights` says otherwise (only when the user asks for one dimension to count more). Dimensions the listing
    can't be compared on are dropped; a unit's unknown dimension counts as 0 and is shown as null.

    There's no cap on units per estate: units on the estate a listing sits on are genuinely among the most similar.
    Set `max_per_estate` only when the user asks for a spread (e.g. "similar units on other estates").

    Searches `candidates` (None: the whole portfolio), e.g. the set returned by `filter_rows`.

    For the *nearest* or *closest* properties on the map (the brief's prompt 3), use `find_nearest` instead.

    Args:
        data: the cleaned portfolio and listings. Filled in by the agent; the model doesn't supply it.
        listing: the listing's id ("M3") or name ("Spitfire Park", "Spitfire Park, Unit 42"). Case doesn't matter,
            and part of a name works if it matches only one listing.
        k: how many units to return, from 1 to 50 (default 10).
        weights: relative weights for "age", "location" and "physical", e.g. `{"location": 2}`; dimensions not named
            keep weight 1. Only when the user asks for one dimension to count more.
        max_per_estate: the most matches allowed from any one estate. Only when the user asks for a spread.
        candidates: the Property Ids to search, e.g. the `candidates` returned by `filter_rows`. None (the
            default) searches the whole portfolio.

    Returns:
        `summary`; `listing` (its facts); `matches`, best first, each with `property_id`, `estate`,
        `unit`, `overall` and the `age`, `location` and `physical` scores (null: unknown for that unit),
        `build_year`, `size_sqft`, `miles_from_listing` and `doubtful_fields`; and `caveats` to pass on.

    Raises:
        ToolError: `k` out of range, an unknown listing or Property Id, an empty candidate set, an unknown dimension
            in `weights`, or a listing with nothing to compare.
    """
    # 1. Check the arguments and find the listing and the units to search.
    if not isinstance(k, int) or not 1 <= k <= 50:
        raise ToolError("k must be a whole number from 1 to 50")
    row = _resolve_listing(data, listing)  # "M3" or "Spitfire Park" -> the listing's row
    pool = _candidate_rows(data, candidates)  # the filtered set, or the whole portfolio
    if pool.empty:
        raise ToolError("The candidate set is empty: there's nothing to compare against")

    # 2. Rank the units (the scoring itself is in similarity.most_similar).
    try:
        result = most_similar(row, pool, k=k, weights=weights, max_per_estate=max_per_estate)
    except ValueError as error:  # e.g. an unknown dimension in weights: pass the message on to the model
        raise ToolError(str(error)) from error
    matches = _as_matches(result, data.portfolio)  # the ranked table -> one dictionary per unit

    # 3. Caveats: anything the answer should tell the user about how these matches were found.
    dimensions = _matched_on(row)
    caveats = []

    # The listing can't be compared on every dimension (e.g. M2 has no build year or location).
    if dimensions["not_available"]:
        missing = "; ".join(f"{dimension} ({reason})" for dimension, reason in dimensions["not_available"].items())
        caveats.append(f"Matched on {' and '.join(dimensions['matched_on'])} only. Not available: {missing}.")

    # The listing sits on an estate Indurent owns: say so, and how many matches are on it.
    estate = sits_on_estate(row, data.portfolio)
    if estate:
        on_estate = int((result["industrial_estate_name"] == estate).sum())
        filtered = candidates is not None
        caveats.append(_owned_estate_caveat(estate, on_estate, len(result), pool, data.portfolio, filtered, "matches"))

    # Some matches have no usable location, so they scored 0 on it.
    if "location" in dimensions["matched_on"]:
        unknown = int(result["location"].isna().sum())
        if unknown:
            caveats.append(f"{unknown} match(es) have no usable location (counted as 0 on location).")

    # Matched on size alone: many units can score about the same, so the list is one of several.
    if dimensions["matched_on"] == ["physical"] and len(result):
        everything = most_similar(row, pool, k=len(pool))
        threshold = round(result["overall"].iloc[-1], 2)  # the last match's score, to 2 decimal places
        ties = int((everything["overall"].round(2) >= threshold).sum())
        if ties > len(result):
            caveats.append(
                f"{ties} units score {threshold:.2f} or more on size alone, so the list is one of several "
                "equally good ones."
            )

    # Anything that changed the search from the default.
    if candidates is not None:
        caveats.append(f"Searched a filtered set of {len(pool):,} units, not the whole portfolio.")
    if weights:
        caveats.append(f"Weights changed from equal: {weights}.")
    if max_per_estate is not None:
        caveats.append(f"At most {max_per_estate} unit(s) per estate, as requested.")

    # 4. The result the model receives. The summary says how the scores were combined, so the answer can say it.
    names = [{"physical": "physical (size)"}.get(dimension, dimension) for dimension in dimensions["matched_on"]]
    if len(names) == 1:
        scored_on = f"scored on {names[0]} alone"
    elif weights:
        scored_on = f"scored on {', '.join(names[:-1])} and {names[-1]}, weighted as requested: {weights}"
    else:
        scored_on = f"scored on {', '.join(names[:-1])} and {names[-1]}, weighted equally"
    return {
        "summary": f"Top {len(matches)} matches for {row['listing_id']} ({row['industrial_estate_name']}, "
        f"{row['unit_name']}) from {len(pool):,} units, {scored_on}",
        "listing": _listing_facts(row),
        "matches": matches,
        "caveats": caveats,
    }


# --- Prompt 2: homogeneity ---------------------------------------------------------------------------------------

HOMOGENEITY_MEANING = (
    "Homogeneity here means how typical the listing is of Indurent's portfolio, scored 0-1 on each dimension, "
    "where 1 is the most typical value in the portfolio. 'Correlation score' in the request is read this way."
)


def _percent(share: float) -> float:
    return round(100 * float(share), 1)


def _position(row: pd.Series, portfolio: pd.DataFrame, dimensions: list[str]) -> dict:
    """Where the listing's build year and size sit in the portfolio: the share of units older/newer, smaller/larger.

    A homogeneity score says how atypical a listing is, not which way (a low age score could mean old or new), so
    this gives the direction. Only for the dimensions the listing can be compared on.
    """
    position = {}
    if "age" in dimensions:
        years = usable(portfolio, "build_year").dropna()
        year = row["build_year"]
        position["age"] = {
            "build_year": _value(year, 0),
            "pct_of_units_older": _percent((years < year).mean()),
            "pct_of_units_newer": _percent((years > year).mean()),
        }
    if "physical" in dimensions:
        sizes = usable(portfolio, "size_sqft").dropna()
        size = row["size_sqft"]
        position["physical"] = {
            "size_sqft": _value(size, 0),
            "pct_of_units_smaller": _percent((sizes < size).mean()),
            "pct_of_units_larger": _percent((sizes > size).mean()),
        }
    return position


def get_homogeneity_scores(data: Data, listing: str) -> dict:
    """How typical a marketed listing is of the portfolio, scored separately on age, location and physical (prompt 2).

    Each score is 0-1, where 1 is the most typical value in the portfolio; null means not available, with the reason
    in `not_available`. A low score says the listing is atypical, not which way: `position` gives the direction
    (the share of portfolio units older/newer and smaller/larger than the listing). `location_context` and
    `size_context` explain the scores and aren't scored. State the interpretation (`meaning`) in the answer.

    Args:
        data: the cleaned portfolio and listings. Filled in by the agent; the model doesn't supply it.
        listing: the listing's id ("M3") or name ("Spitfire Park", "Spitfire Park, Unit 42"). Case doesn't matter,
            and part of a name works if it matches only one listing.

    Returns:
        `summary`; `meaning` (the interpretation to state); `scores` for `age`, `location` and
        `physical` (0-1, or null when not available); `not_available` (the reasons); `position`;
        `location_context`; `size_context`; and `listing` (its facts).

    Raises:
        ToolError: no listing matches, or more than one does.
    """
    row = _resolve_listing(data, listing)
    result = homogeneity_scores(row, data.portfolio)
    dimensions = _matched_on(row)
    location_context = {key: _value(v, 1) for key, v in result["location_context"].items()}
    location_context["sits_on_estate"] = _estate_status(row, data.portfolio)  # never a blank that reads as "no"
    return {
        "summary": f"Homogeneity scores for {row['listing_id']} ({row['industrial_estate_name']}, {row['unit_name']})",
        "meaning": HOMOGENEITY_MEANING,
        "scores": {dimension: _value(result[dimension], 2) for dimension in DIMENSIONS},
        "not_available": dimensions["not_available"],
        "position": _position(row, data.portfolio, dimensions["matched_on"]),
        "location_context": location_context,
        "size_context": {key: _value(v, 1) for key, v in result["size_context"].items()},
        "listing": _listing_facts(row),
    }


# --- Prompt 3: nearest on the map ---------------------------------------------------------------------------------


def _as_nearest(nearest: pd.Series, pool: pd.DataFrame) -> list[dict]:
    """Turn the nearest units (miles, by Property Id) into one JSON-friendly dictionary per unit, nearest first."""
    matches = []
    for property_id, miles in nearest.items():
        unit = pool.loc[property_id]
        matches.append(
            {
                "property_id": int(property_id),
                "estate": unit["industrial_estate_name"],
                "unit": unit["unit_name"],
                "miles_from_listing": _value(miles, 1),
                "build_year": _value(unit["build_year"], 0),
                "size_sqft": _value(unit["size_sqft"], 0),
                "nearest_city": _value(unit["nearest_city"]),
                "miles_to_city": _value(unit["miles_to_city"], 1),
            }
        )
    return matches


def find_nearest(data: Data, listing: str, k: int = 10, candidates: list[int] | None = None) -> dict:
    """The k portfolio units nearest to a marketed listing on the map, by straight-line miles (the brief's prompt 3).

    "Closest properties" means nearest on the map, not most similar (use `find_similar` for that). For prompt 3,
    first narrow the portfolio with `filter_rows` (keep `miles_to_city <= 10`) and pass its `candidates`.

    Units without a usable location can't be placed, so they're left out, and the caveats say how many. A listing
    without a usable location (no coordinates, or offshore) can't be used at all: that's an error, with the reason.

    Args:
        data: the cleaned portfolio and listings. Filled in by the agent; the model doesn't supply it.
        listing: the listing's id ("M3") or name ("Spitfire Park", "Spitfire Park, Unit 42"). Case doesn't matter,
            and part of a name works if it matches only one listing.
        k: how many units to return, from 1 to 50 (default 10).
        candidates: the Property Ids to search, e.g. the `candidates` returned by `filter_rows`. None (the
            default) searches the whole portfolio.

    Returns:
        `summary`; `listing` (its facts); `matches`, nearest first, each with `property_id`, `estate`,
        `unit`, `miles_from_listing`, `build_year`, `size_sqft`, `nearest_city` and `miles_to_city`;
        and `caveats` to pass on.

    Raises:
        ToolError: `k` out of range, an unknown listing or Property Id, or a listing with no usable location.
    """
    # 1. Check the arguments, find the listing, and make sure it can be placed on the map.
    if not isinstance(k, int) or not 1 <= k <= 50:
        raise ToolError("k must be a whole number from 1 to 50")
    row = _resolve_listing(data, listing)  # "M3" or "Spitfire Park" -> the listing's row
    dimensions = _matched_on(row)
    if "location" not in dimensions["matched_on"]:  # no coordinates, or offshore
        raise ToolError(
            f"{row['listing_id']} can't be placed on the map: {dimensions['not_available']['location']}. "
            "Nearest properties can't be found for it; say so in the answer."
        )
    pool = _candidate_rows(data, candidates)  # the filtered set, or the whole portfolio

    # 2. Rank the units by distance. Units with no usable location have no distance, so they're left out.
    miles = miles_to_units(row, pool)
    placed = miles.dropna().sort_index().sort_values(kind="stable")  # nearest first; ties in Property Id order
    matches = _as_nearest(placed.head(k), pool)  # the k nearest -> one dictionary per unit

    # 3. Caveats: anything the answer should tell the user about how these were found.
    caveats = []

    # Units that couldn't be ranked because their location isn't usable.
    unplaced = int(miles.isna().sum())
    if unplaced:
        caveats.append(
            f"{unplaced:,} of the {len(pool):,} units searched have no usable location, so couldn't be ranked. "
            "Keeping or dropping them can't change this list."
        )

    # The last unit listed is as near as units left out (at the 1 decimal place shown), so the list is one of
    # several equally near ones. Not "listed in Property Id order": units that show the same rounded distance are
    # still in order of their exact distance, and the model repeated the claim (eval 2026-11-23).
    if len(matches) == k and len(placed) > k:
        last = round(float(placed.iloc[k - 1]), 1)
        as_near = int((placed.round(1) <= last).sum())
        if as_near > k:
            caveats.append(
                f"{as_near} units are {last} miles or nearer (to 1 decimal place), so these {k} aren't the only "
                "ones that near. Units that show the same distance are listed in order of their exact distance."
            )

    # The listing sits on an estate Indurent owns: say so, and how many of the nearest are on it.
    estate = sits_on_estate(row, data.portfolio)
    if estate:
        on_estate = sum(match["estate"] == estate for match in matches)
        filtered = candidates is not None
        caveats.append(_owned_estate_caveat(estate, on_estate, len(matches), pool, data.portfolio, filtered, "nearest"))

    # The search was over a filtered set, not the whole portfolio.
    if candidates is not None:
        caveats.append(f"Searched a filtered set of {len(pool):,} units, not the whole portfolio.")

    # 4. The result the model receives.
    return {
        "summary": f"The {len(matches)} units nearest to {row['listing_id']} ({row['industrial_estate_name']}, "
        f"{row['unit_name']}) on the map, from {len(placed):,} units with a usable location",
        "listing": _listing_facts(row),
        "matches": matches,
        "caveats": caveats,
    }
