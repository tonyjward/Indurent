"""Tests for indurent_matching.tools.

The filter tests use a small made-up portfolio; the others use the assessment files (via the `portfolio` and
`marketed` fixtures in tests/conftest.py) and are skipped when those aren't present.
"""

import json

import numpy as np
import pandas as pd
import pytest

from indurent_matching.tools import (
    Data,
    ToolError,
    filter_rows,
    find_nearest,
    find_similar,
    get_homogeneity_scores,
    get_listing,
    list_listings,
)


@pytest.fixture(scope="module")
def data(portfolio, marketed):
    return Data(portfolio, marketed)


# --- filter_rows on a made-up portfolio --------------------------------------------------------------------------

UNITS = pd.DataFrame(
    {
        "miles_to_city": [2.0, 8.0, 15.0, np.nan],
        "size_sqft": [1_000.0, 5_000.0, 20_000.0, 300.0],
        "size_sqft_doubtful": [False, False, False, True],
        "derived_region": ["London", "Midlands", "Midlands", None],
    },
    index=[11, 12, 13, 14],
)
SMALL = Data(portfolio=UNITS, marketed=pd.DataFrame())


def test_filter_unknowns_are_an_explicit_choice():
    near = [{"field": "miles_to_city", "op": "<=", "value": 10}]
    dropped = filter_rows(SMALL, near, include_unknown=False)
    kept = filter_rows(SMALL, near, include_unknown=True)
    assert dropped["candidates"] == [11, 12] and dropped["unknown"] == 1
    assert kept["candidates"] == [11, 12, 14]
    assert "1 had no usable value for miles_to_city and were excluded" in dropped["summary"]
    assert "(miles_to_city is unknown where a unit has no usable location.)" in dropped["summary"]  # says why


def test_filter_doubtful_values_count_as_unknown():
    out = filter_rows(SMALL, [{"field": "size_sqft", "op": "<", "value": 2_000}], include_unknown=False)
    assert out["candidates"] == [11]  # unit 14's 300 sq ft is doubtful, so it's unknown, not "small"
    assert out["unknown"] == 1


def test_filter_operators_and_chaining():
    midlands = filter_rows(SMALL, [{"field": "derived_region", "op": "==", "value": "Midlands"}], include_unknown=False)
    assert midlands["candidates"] == [12, 13]
    between = [{"field": "size_sqft", "op": "between", "value": [4_000, 30_000]}]
    assert filter_rows(SMALL, between, include_unknown=False)["candidates"] == [12, 13]
    chained = filter_rows(SMALL, [{"field": "miles_to_city", "op": "<", "value": 10}], False, midlands["candidates"])
    assert chained["candidates"] == [12] and chained["started_with"] == 2
    both = [
        {"field": "derived_region", "op": "in", "value": ["London", "Midlands"]},
        {"field": "size_sqft", "op": ">", "value": 2_000},
    ]
    assert filter_rows(SMALL, both, include_unknown=False)["candidates"] == [12, 13]  # all conditions must hold


@pytest.mark.parametrize(
    "conditions, include_unknown, message",
    [
        ([{"field": "eaves", "op": "<", "value": 5}], False, "Can't filter on"),
        ([{"field": "derived_region", "op": "<", "value": "M"}], False, "doesn't apply"),
        ([{"field": "size_sqft", "op": "between", "value": 5}], False, "between"),
        ([{"field": "size_sqft", "op": "<", "value": 5}], None, "include_unknown"),
        ([], False, "at least one condition"),
    ],
)
def test_filter_rejects_bad_arguments(conditions, include_unknown, message):
    with pytest.raises(ToolError, match=message):
        filter_rows(SMALL, conditions, include_unknown)


# --- Listings ------------------------------------------------------------------------------------------------------


def test_list_listings(data):
    out = list_listings(data)
    assert [x["listing_id"] for x in out["listings"]] == ["M1", "M2", "M3", "M4", "M5"]
    m2 = out["listings"][1]
    assert m2["matched_on"] == ["physical"] and set(m2["not_available"]) == {"age", "location"}
    # The data-quality notes come with the listing, so the model doesn't work out its own (eval 2026-11-23).
    assert "falls in Wales" in out["listings"][0]["quality_notes"][0]
    assert out["summary"] == (
        "5 marketed listings: M1, M2, M3, M4, M5. "
        "2 can't be compared on every dimension: M2 (no age or location), M5 (no location)."
    )


@pytest.mark.parametrize("query", ["M3", "m3", "Spitfire Park", "spitfire park, unit 42", "Spitfire"])
def test_get_listing_by_id_or_name(data, query):
    out = get_listing(data, query)
    assert out["listing_id"] == "M3" and out["sits_on_estate"] == "Velvet Park 60"
    assert out["build_year"] == 1976 and isinstance(out["build_year"], int)


def test_get_listing_says_when_the_estate_is_unknown(data):
    # M2 has no coordinates: which estate it's on is unknown, which a blank would hide (eval 2026-11-23).
    assert get_listing(data, "M2")["sits_on_estate"] == "unknown: M2 has no usable location"


def test_get_listing_unknown_or_ambiguous(data):
    with pytest.raises(ToolError, match="No marketed listing matches"):
        get_listing(data, "M9")
    with pytest.raises(ToolError, match="more than one listing") as error:
        get_listing(data, "Unit 1")  # Unit 10 and Unit 14
    assert "M1" in str(error.value) and "M4" in str(error.value) and "M3" not in str(error.value)


def test_results_are_json_friendly(data):
    for out in [
        list_listings(data),
        get_listing(data, "M5"),
        find_similar(data, "M2"),
        find_nearest(data, "M1"),
        get_homogeneity_scores(data, "M2"),
    ]:
        json.dumps(out)  # no numpy types or NaN


# --- Prompts 1 to 3 ------------------------------------------------------------------------------------------------


def test_find_similar_prompt_1(data):
    out = find_similar(data, "M3")
    assert len(out["matches"]) == 10 and {m["estate"] for m in out["matches"]} == {"Velvet Park 60"}
    assert any("sits on Velvet Park 60" in c and "10 of the 10" in c for c in out["caveats"])
    assert out["summary"].endswith("scored on age, location and physical (size), weighted equally")
    assert find_similar(data, "M5")["summary"].endswith("scored on age and physical (size), weighted equally")
    m2 = find_similar(data, "M2")
    assert m2["summary"].endswith("scored on physical (size) alone")
    assert any(c.startswith("Matched on physical only") for c in m2["caveats"])
    assert any("on size alone" in c for c in m2["caveats"])  # many units tie
    assert all(m["age"] is None and m["location"] is None for m in m2["matches"])


def test_find_similar_options(data):
    capped = find_similar(data, "M3", max_per_estate=2)
    assert max(pd.Series([m["estate"] for m in capped["matches"]]).value_counts()) == 2
    assert any("At most 2" in c for c in capped["caveats"])
    weighted = find_similar(data, "M3", weights={"location": 2})
    assert weighted["summary"].endswith("weighted as requested: {'location': 2}")
    with pytest.raises(ToolError, match="k must be"):
        find_similar(data, "M3", k=0)
    with pytest.raises(ToolError, match="unknown dimensions"):
        find_similar(data, "M3", weights={"eaves": 1})


def test_find_similar_on_a_filtered_set(data):
    near = filter_rows(data, [{"field": "miles_to_city", "op": "<=", "value": 10}], include_unknown=False)
    out = find_similar(data, "M3", candidates=near["candidates"])
    assert {m["property_id"] for m in out["matches"]} <= set(near["candidates"])
    assert any("filtered set of 50" in c for c in out["caveats"])
    with pytest.raises(ToolError, match="Unknown Property Ids"):
        find_similar(data, "M3", candidates=[999_999])


def test_prompt_3_filter_then_find_nearest(data):
    near = filter_rows(data, [{"field": "miles_to_city", "op": "<=", "value": 10}], include_unknown=False)
    assert near["kept"] == 50 and near["unknown"] == 358  # EDA section 5
    out = find_nearest(data, "M3", candidates=near["candidates"])
    miles = [m["miles_from_listing"] for m in out["matches"]]
    assert miles == sorted(miles) and out["matches"][0]["estate"] == "Jade Bridge 6"  # 9.1 miles
    assert {m["property_id"] for m in out["matches"]} <= set(near["candidates"])
    assert any("0 of the 10 nearest are on it" in c for c in out["caveats"])  # Velvet Park 60 is 14 miles from a city
    # ...and the caveat says why, so the model doesn't guess (eval 2026-11-23).
    assert any(
        "None of Velvet Park 60's units are in the filtered set: the filter removed them" in c for c in out["caveats"]
    )


def test_find_nearest_whole_portfolio_and_unplaceable_listings(data):
    out = find_nearest(data, "M3")
    assert {m["estate"] for m in out["matches"]} == {"Velvet Park 60"}  # the estate it sits on
    assert any("358 of the 1,250 units" in c for c in out["caveats"])  # unknown locations can't be ranked
    assert any("Keeping or dropping them can't change this list" in c for c in out["caveats"])
    # Velvet Park 60 has more units at 0.0 miles than the 10 listed: say so (eval 2026-11-23).
    assert "Units that show the same distance are listed in order of their exact distance" in " ".join(out["caveats"])
    assert "25 units are 0.0 miles or nearer (to 1 decimal place), so these 10 aren't the only ones that near" in (
        " ".join(out["caveats"])
    )
    assert "within 0.25 miles of the listing" in " ".join(out["caveats"])  # what "sits on" means
    for listing, reason in [("M2", "no coordinates"), ("M5", "offshore")]:
        with pytest.raises(ToolError, match=reason):
            find_nearest(data, listing)


def test_get_homogeneity_scores_prompt_2(data):
    m3 = get_homogeneity_scores(data, "Spitfire Park")
    assert m3["scores"] == {"age": 0.86, "location": 0.62, "physical": 0.87}
    assert m3["location_context"]["sits_on_estate"] == "Velvet Park 60"
    assert "how typical" in m3["meaning"]
    m5 = get_homogeneity_scores(data, "M5")
    assert m5["scores"]["location"] is None and "offshore" in m5["not_available"]["location"]
    assert m5["location_context"]["sits_on_estate"] == "unknown: M5 has no usable location"
    # The direction a score can't give (eval 2026-11-23): M5 is older than almost every unit, M4 smaller.
    assert m5["position"]["age"] == {"build_year": 1966, "pct_of_units_older": 2.0, "pct_of_units_newer": 97.3}
    m4 = get_homogeneity_scores(data, "M4")["position"]["physical"]
    assert m4 == {"size_sqft": 617, "pct_of_units_smaller": 1.3, "pct_of_units_larger": 98.7}
    assert list(get_homogeneity_scores(data, "M2")["position"]) == ["physical"]  # no build year, so no age


def test_owned_estate_caveat_says_why_none_are_listed():
    from indurent_matching.tools import _owned_estate_caveat

    portfolio = pd.DataFrame(
        {
            "industrial_estate_name": ["Velvet Park 60", "Velvet Park 60", "Velvet Park 60", "Jade Bridge 6"],
            "miles_to_city": [13.9, 14.0, None, 7.1],
            "nearest_city": ["Manchester", "Manchester", None, "Manchester"],
        }
    )
    without = portfolio[portfolio["industrial_estate_name"] != "Velvet Park 60"]
    removed = _owned_estate_caveat("Velvet Park 60", 0, 10, without, portfolio, filtered=True, noun="nearest")
    # The facts to see why (eval 2026-11-23): too far from a city, or no location at all.
    assert removed.endswith(
        "None of Velvet Park 60's units are in the filtered set: the filter removed them. Of its 3 units, "
        "2 are 13.9-14.0 miles from Manchester (the nearest major city); 1 has no usable location."
    )
    ranked_lower = _owned_estate_caveat("Velvet Park 60", 0, 10, portfolio, portfolio, filtered=True, noun="nearest")
    assert ranked_lower.endswith("3 of its units are in the filtered set but rank below these.")
    unfiltered = _owned_estate_caveat("Velvet Park 60", 0, 10, portfolio, portfolio, filtered=False, noun="nearest")
    assert unfiltered.endswith("0 of the 10 nearest are on it.")  # no reason needed without a filter
