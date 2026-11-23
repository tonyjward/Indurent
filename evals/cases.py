"""The eval cases: the brief's three prompts, variations on them, and questions the agent should decline.

Each case says which tool calls it expects (in order), which tools it mustn't use, and what a good answer does
(for the judge). See `indurent_matching.evals` for how they're checked.
"""

from indurent_matching.chat import BRIEF_PROMPTS
from indurent_matching.evals import Call, Case, Mentions, NoFigures, Ranking, Scores

WITHIN_10_MILES = [{"field": "miles_to_city", "op": "<=", "value": 10}]
AT_LEAST_10000_SQFT = [{"field": "size_sqft", "op": ">=", "value": 10000}]
EPC_A_TO_C = [{"field": "epc_rating", "op": "in", "value": ["A", "B", "C"]}]
LISTINGS = ["M1", "M2", "M3", "M4", "M5"]
LOCATED = ["M1", "M3", "M4"]  # the listings with a usable location (M2 has no coordinates, M5's are offshore)


def names(*accepted):
    """An argument check for a listing: true if the value names one of these (by id or estate, any case)."""

    def check(value) -> bool:
        return any(name.lower() in str(value).lower() for name in accepted)

    return check


def large_units(conditions) -> bool:
    """True if the conditions keep units of 10,000 sq ft or more (>= or >, either way)."""
    return any(c["field"] == "size_sqft" and c["op"] in (">=", ">") and c["value"] in (10000, 9999) for c in conditions)


def epc_c_or_better(conditions) -> bool:
    """True if the conditions keep EPC ratings A to C (A+ may be included too: a real rating, absent from this data)."""
    for c in conditions:
        is_epc_list = c["field"] == "epc_rating" and c["op"] == "in"
        if is_epc_list and {"A", "B", "C"} <= set(c["value"]) <= {"A+", "A", "B", "C"}:
            return True
    return False


def location_doubled(weights) -> bool:
    """True if location counts twice as much as each other dimension (e.g. {"location": 2})."""
    if not isinstance(weights, dict):
        return False
    others = [weights.get(dimension, 1) for dimension in ("age", "physical")]
    return all(weights.get("location", 1) == 2 * other for other in others)


CASES = [
    # --- The brief's prompts, word for word ---
    Case(
        name="brief_1_most_similar",
        questions=[BRIEF_PROMPTS["prompt_1"]],
        expect=[],
        forbid=["find_nearest", "filter_rows"],
        behaviour="The request doesn't say which of the five listings it means. A good answer either asks which "
        "listing, or gives the most similar units for every listing; it never picks one listing silently.",
        # If it answers rather than asking, each listing it answers for must have its true top 10, in order.
        answer_checks=[Ranking(listing, "find_similar", at_least=10, required=False) for listing in LISTINGS],
    ),
    Case(
        name="brief_2_homogeneity",
        questions=[BRIEF_PROMPTS["prompt_2"]],
        expect=[Call("get_homogeneity_scores")],
        forbid=["filter_rows"],
        behaviour="Gives homogeneity scores for each listing (or asks which), with physical, location and age "
        "scored separately, and says how 'correlation score for homogeneity' was interpreted.",
        answer_checks=[Scores(listing) for listing in LISTINGS],
    ),
    Case(
        name="brief_3_closest_near_city",
        questions=[BRIEF_PROMPTS["prompt_3"]],
        expect=[
            Call("filter_rows", {"conditions": WITHIN_10_MILES, "include_unknown": False, "refine": False}),
            Call("find_nearest", {"use_filter": True}),
        ],
        forbid=["find_similar"],
        behaviour="Gives the nearest units on the map within 10 miles of a major city (for each listing that can "
        "be located, or asks which). Says how many units were left out because their location is unknown. It may "
        "offer to include them, or explain that units with no location can't be ranked by distance; either is "
        "right for a nearest-on-the-map search.",
        answer_checks=[
            *[
                Ranking(listing, "find_nearest", filters=[(WITHIN_10_MILES, False)], required=False)
                for listing in LOCATED
            ],
            Mentions([50, 358], "units kept by the filter, and left out as unknown"),
        ],
    ),
    # --- Variations ---
    Case(
        name="similar_named_listing",
        questions=["Find the 10 units most similar to M3."],
        expect=[Call("find_similar", {"listing": names("M3", "Spitfire"), "use_filter": False})],
        forbid=["find_nearest"],
        behaviour="Lists 10 units most similar to M3, says M3 sits on an estate Indurent already owns, and that "
        "age, location and size were weighted equally.",
        answer_checks=[Ranking("M3", "find_similar", at_least=10)],
    ),
    Case(
        name="similar_size_only",
        questions=["Which portfolio units are most like M2?"],
        expect=[Call("find_similar", {"listing": names("M2", "Tech Hub")})],
        forbid=["find_nearest"],
        behaviour="Says M2 can only be compared on size (no build year or coordinates), and that many units score "
        "about the same, so the list is one of several equally good ones.",
        answer_checks=[Ranking("M2", "find_similar")],
    ),
    Case(
        name="nearest_by_name",
        questions=["Which of our units are closest to Spitfire Park?"],
        expect=[Call("find_nearest", {"listing": names("M3", "Spitfire"), "use_filter": False})],
        forbid=["find_similar"],
        behaviour="Lists the portfolio units nearest to the Spitfire Park listing (M3) on the map, with distances.",
        answer_checks=[Ranking("M3", "find_nearest")],
    ),
    Case(
        name="weighted_location",
        questions=["Find the 5 units most similar to M1, with location counting twice as much as the others."],
        expect=[Call("find_similar", {"listing": names("M1", "Cherry"), "k": 5, "weights": location_doubled})],
        forbid=["find_nearest"],
        behaviour="Lists 5 units most similar to M1 with location weighted double, and says the weights were "
        "changed from equal as asked.",
        answer_checks=[Ranking("M1", "find_similar", weights={"location": 2}, at_least=5)],
    ),
    Case(
        name="refine_filter",
        questions=[
            "Which units are within 10 miles of a major city?",
            "Now keep only those of 10,000 sq ft or more, and find the ones nearest to M3.",
        ],
        expect=[
            Call("filter_rows", {"conditions": WITHIN_10_MILES, "refine": False}),
            Call("filter_rows", {"conditions": large_units, "refine": True}),
            Call("find_nearest", {"listing": names("M3", "Spitfire"), "use_filter": True}),
        ],
        forbid=[],
        behaviour="Narrows the earlier filter to units of 10,000 sq ft or more (keeping the 10-mile condition) and "
        "lists the nearest of those to M3, saying how many units the filter kept.",
        answer_checks=[
            Ranking("M3", "find_nearest", filters=[(WITHIN_10_MILES, False), (AT_LEAST_10000_SQFT, False)]),
            Mentions([17], "units the refined filter kept"),
        ],
    ),
    Case(
        name="include_unknown_when_asked",
        questions=[
            (
                "Find the units nearest to M3, dropping only those known to be more than 10 miles from a major "
                "city. Keep any whose distance is unknown."
            )
        ],
        expect=[
            Call("filter_rows", {"conditions": WITHIN_10_MILES, "include_unknown": True}),
            Call("find_nearest", {"listing": names("M3", "Spitfire"), "use_filter": True}),
        ],
        forbid=[],
        behaviour="Keeps units with an unknown distance as asked, says how many of those there are, and lists "
        "the nearest units to M3 from the filtered set.",
        answer_checks=[
            Ranking("M3", "find_nearest", filters=[(WITHIN_10_MILES, True)]),
            Mentions([358], "units kept because their distance is unknown"),
        ],
    ),
    Case(
        name="epc_filter",
        questions=["Find the units most similar to M3 that have an EPC rating of C or better."],
        expect=[
            Call("filter_rows", {"conditions": epc_c_or_better}),
            Call("find_similar", {"listing": names("M3", "Spitfire"), "use_filter": True}),
        ],
        forbid=[],
        behaviour="Filters to EPC A-C and lists the most similar units, and warns that the portfolio's EPC ratings "
        "look unreliable, so the result should be treated with caution.",
        answer_checks=[Ranking("M3", "find_similar", filters=[(EPC_A_TO_C, False)])],
    ),
    # --- Out of scope: the data can't answer these ---
    Case(
        name="scope_rent",
        questions=["What rent per sq ft should we offer for M3?"],
        expect=[],
        forbid=[],
        behaviour="Says the data has no rents or prices, so it can't suggest a figure, and gives no rent figure. "
        "It may offer what it can do instead (e.g. similar units).",
        answer_checks=[NoFigures(r"£\s?\d[\d,.]*|\d[\d,.]*\s*(?:per|/)\s*sq\.?\s?ft", "a rent figure")],
    ),
    Case(
        name="scope_guess_missing_value",
        questions=["M2 has no build year recorded. What's your best guess for it?"],
        expect=[],
        forbid=[],
        behaviour="Doesn't estimate or guess a build year: says it isn't recorded, so M2 can't be compared on age. "
        "No year is suggested for M2.",
        answer_checks=[NoFigures(r"\b(?:18|19|20)\d\d\b", "a year, which could only be a guess")],
    ),
]
