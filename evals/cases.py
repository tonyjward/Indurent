"""The eval cases: the brief's three prompts, variations on them, and questions the agent should decline.

Each case says which tool calls it expects (in order), which tools it mustn't use, and what a good answer does
(for the judge). See `indurent_matching.evals` for how they're checked.
"""

from indurent_matching.chat import BRIEF_PROMPTS
from indurent_matching.evals import Call, Case

WITHIN_10_MILES = [{"field": "miles_to_city", "op": "<=", "value": 10}]


def names(*accepted):
    """An argument check for a listing: true if the value names one of these (by id or estate, any case)."""

    def check(value) -> bool:
        return any(name.lower() in str(value).lower() for name in accepted)

    return check


def large_units(conditions) -> bool:
    """True if the conditions keep units of 10,000 sq ft or more (>= or >, either way)."""
    return any(c["field"] == "size_sqft" and c["op"] in (">=", ">") and c["value"] in (10000, 9999) for c in conditions)


def epc_c_or_better(conditions) -> bool:
    """True if the conditions keep EPC ratings A to C."""
    return any(
        c["field"] == "epc_rating" and c["op"] == "in" and set(c["value"]) == {"A", "B", "C"} for c in conditions
    )


def location_doubled(weights) -> bool:
    """True if location counts twice as much as each other block (e.g. {"location": 2})."""
    if not isinstance(weights, dict):
        return False
    others = [weights.get(block, 1) for block in ("age", "physical")]
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
    ),
    Case(
        name="brief_2_homogeneity",
        questions=[BRIEF_PROMPTS["prompt_2"]],
        expect=[Call("get_homogeneity_scores")],
        forbid=["filter_rows"],
        behaviour="Gives homogeneity scores for each listing (or asks which), with physical, location and age "
        "scored separately, and says how 'correlation score for homogeneity' was interpreted.",
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
        "be located, or asks which). Says how many units were left out because their location is unknown, and "
        "offers to include them.",
    ),
    # --- Variations ---
    Case(
        name="similar_named_listing",
        questions=["Find the 10 units most similar to M3."],
        expect=[Call("find_similar", {"listing": names("M3", "Spitfire"), "use_filter": False})],
        forbid=["find_nearest"],
        behaviour="Lists 10 units most similar to M3, says M3 sits on an estate Indurent already owns, and that "
        "age, location and size were weighted equally.",
    ),
    Case(
        name="similar_size_only",
        questions=["Which portfolio units are most like M2?"],
        expect=[Call("find_similar", {"listing": names("M2", "Tech Hub")})],
        forbid=["find_nearest"],
        behaviour="Says M2 can only be compared on size (no build year or coordinates), and that many units score "
        "about the same, so the list is one of several equally good ones.",
    ),
    Case(
        name="nearest_by_name",
        questions=["Which of our units are closest to Spitfire Park?"],
        expect=[Call("find_nearest", {"listing": names("M3", "Spitfire"), "use_filter": False})],
        forbid=["find_similar"],
        behaviour="Lists the portfolio units nearest to the Spitfire Park listing (M3) on the map, with distances.",
    ),
    Case(
        name="weighted_location",
        questions=["Find the 5 units most similar to M1, with location counting twice as much as the others."],
        expect=[Call("find_similar", {"listing": names("M1", "Cherry"), "k": 5, "weights": location_doubled})],
        forbid=["find_nearest"],
        behaviour="Lists 5 units most similar to M1 with location weighted double, and says the weights were "
        "changed from equal as asked.",
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
    ),
    # --- Out of scope: the data can't answer these ---
    Case(
        name="scope_rent",
        questions=["What rent per sq ft should we offer for M3?"],
        expect=[],
        forbid=[],
        behaviour="Says the data has no rents or prices, so it can't suggest a figure, and gives no rent figure. "
        "It may offer what it can do instead (e.g. similar units).",
    ),
    Case(
        name="scope_guess_missing_value",
        questions=["M2 has no build year recorded. What's your best guess for it?"],
        expect=[],
        forbid=[],
        behaviour="Doesn't estimate or guess a build year: says it isn't recorded, so M2 can't be compared on age. "
        "No year is suggested for M2.",
    ),
]
