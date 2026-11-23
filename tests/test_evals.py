"""Tests for the eval checks (indurent_matching.evals) and the cases (evals/cases.py).

Nothing here calls Claude: the agent is the scripted model (tests/scripted.py) and the judge is a stand-in.
"""

import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from scripted import Scripted, call

from indurent_matching.evals import (
    Call,
    Case,
    Mentions,
    NoFigures,
    Ranking,
    Scores,
    Verdict,
    check_trajectory,
    judge_messages,
    listed_units,
    report_markdown,
    run_case,
    unsupported_numbers,
)
from indurent_matching.tools import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
from cases import CASES

WITHIN_10 = [{"field": "miles_to_city", "op": "<=", "value": 10}]


@pytest.fixture(scope="module")
def data(portfolio, marketed):
    return Data(portfolio, marketed)


def case_named(name: str) -> Case:
    return next(case for case in CASES if case.name == name)


def tool(name, **args):
    """A tool call as the checks see it."""
    return {"name": name, "args": args}


# --- Trajectory -----------------------------------------------------------------------------------------------------


def test_call_matches_exact_and_checked_arguments():
    expected = Call("find_nearest", {"listing": lambda value: "m3" in value.lower(), "use_filter": True})
    assert expected.matches(tool("find_nearest", listing="M3", use_filter=True, k=10))  # k isn't checked
    assert not expected.matches(tool("find_nearest", listing="M1", use_filter=True))
    assert not expected.matches(tool("find_nearest", listing="M3"))  # use_filter missing
    assert not expected.matches(tool("find_similar", listing="M3", use_filter=True))


def test_trajectory_needs_the_expected_calls_in_order():
    expect = [Call("filter_rows"), Call("find_nearest")]
    in_order = [tool("list_listings"), tool("filter_rows"), tool("get_listing"), tool("find_nearest")]
    assert check_trajectory(in_order, expect, forbid=[]) == []  # other calls in between are fine
    problems = check_trajectory([tool("find_nearest"), tool("filter_rows")], expect, forbid=[])
    assert problems == ["expected find_nearest({}), in order, not found"]
    assert check_trajectory(in_order, expect, forbid=["get_listing"]) == [
        "called get_listing, which this case shouldn't use"
    ]


# --- Numbers --------------------------------------------------------------------------------------------------------


def test_numbers_must_come_from_the_sources():
    sources = ['{"overall": 0.734, "size_sqft": 17340, "miles_from_listing": 4.5}', "Find the 10 nearest"]
    answer = "Property Id 586 scores 0.73 (17,340 sq ft, 4.5 miles away). Top 10 for M3. About 12 miles."
    assert unsupported_numbers(answer, sources) == ["586", "12"]  # 0.73 rounds from 0.734; M3 isn't a number
    assert unsupported_numbers("51 miles or more", ["51.5"]) == []  # cut down to the precision written
    assert unsupported_numbers("52 miles", ["51.5"]) == []  # rounded
    assert unsupported_numbers("50 miles", ["51.5"]) == ["50"]


def test_row_numbers_in_tables_and_lists_are_skipped():
    answer = "| 1 | 0.88 |\n| 2 | 0.81 |\n\n1. First\n2. Second"
    assert unsupported_numbers(answer, ["0.88 0.81"]) == []


# --- Judge input and the report -------------------------------------------------------------------------------------


def test_judge_messages_hold_every_section():
    case = case_named("scope_rent")
    system, human = judge_messages(
        case, "Rules: ...", ["find_similar: ..."], ['{"summary": "x"}'], "I can't price rent."
    )
    assert "unsupported_claims" in system.content and "List only the claims you flag" in system.content
    for section in [
        "The agent's instructions",
        "Tool descriptions",
        "Question(s)",
        "Tool results",
        "The agent's answer",
        "What a good answer",
    ]:
        assert f"## {section}" in human.content
    assert case.behaviour in human.content


def good_verdict(messages):
    return Verdict(unsupported_claims=[], missing_caveats=[], behaviour_ok=True, behaviour_note="Fine."), {
        "input": 1000,
        "output": 100,
    }


def test_run_case_passes_a_good_answer(data):
    model = Scripted(
        replies=[
            call("filter_rows", conditions=WITHIN_10, include_unknown=False, refine=False),
            call("find_nearest", listing="M3", use_filter=True, k=3),
            AIMessage(
                "50 units are within 10 miles of a city; 358 with no usable location were left out.\n\n"
                "| Property Id | Miles from M3 |\n|---|---|\n| 170 | 9.1 |\n| 169 | 9.1 |\n| 167 | 9.1 |"
            ),
        ]
    )
    result = run_case(case_named("brief_3_closest_near_city"), data, model, good_verdict)
    assert result["correct"]["passed"]
    assert result["trajectory"]["passed"] and result["numbers"]["passed"] and result["judge"]["passed"]
    assert result["passed"] and [c["name"] for c in result["tool_calls"]] == ["filter_rows", "find_nearest"]
    assert result["cost_usd"] == pytest.approx((1000 * 4 + 100 * 20) / 1e6)  # the scripted agent reports no usage


def test_run_case_fails_wrong_calls_invented_numbers_and_the_judge(data):
    model = Scripted(replies=[call("find_similar", listing="M3", use_filter=False), AIMessage("Unit 99999 is best.")])

    def unhappy(messages):
        verdict = Verdict(
            unsupported_claims=["Unit 99999 is best"], missing_caveats=[], behaviour_ok=False, behaviour_note="No."
        )
        return verdict, {"input": 0, "output": 0}

    result = run_case(case_named("brief_3_closest_near_city"), data, model, unhappy)
    assert not result["passed"]
    assert "called find_similar, which this case shouldn't use" in result["trajectory"]["problems"]
    assert result["numbers"]["unsupported"] == ["99999"]
    assert not result["judge"]["passed"]
    report = report_markdown([result], "claude-test", 0.12)
    assert "| trajectory | 0 of 1 |" in report and "## brief_3_closest_near_city" in report
    assert "Unsupported claim: Unit 99999 is best" in report


def test_multi_turn_case_checks_calls_across_turns(data):
    model = Scripted(
        replies=[
            call("filter_rows", conditions=WITHIN_10, include_unknown=False, refine=False),
            AIMessage("50 units kept."),
            call(
                "filter_rows",
                conditions=[{"field": "size_sqft", "op": ">=", "value": 10000}],
                include_unknown=False,
                refine=True,
            ),
            call("find_nearest", listing="M3", use_filter=True),
            AIMessage("Done."),
        ]
    )
    result = run_case(case_named("refine_filter"), data, model, good_verdict)
    assert result["trajectory"]["passed"] and result["answer"] == "Done."  # the answer is the last turn's


# --- The cases ------------------------------------------------------------------------------------------------------


def test_cases_have_unique_names_and_a_behaviour():
    names = [case.name for case in CASES]
    assert len(names) == len(set(names)) == 12
    assert all(case.behaviour for case in CASES)


@pytest.mark.parametrize(
    "name, calls",
    [
        ("similar_named_listing", [tool("find_similar", listing="Spitfire Park", use_filter=False)]),
        ("weighted_location", [tool("find_similar", listing="M1", k=5, weights={"location": 2})]),
        (
            "weighted_location",
            [tool("find_similar", listing="M1", k=5, weights={"location": 1, "age": 0.5, "physical": 0.5})],
        ),
        (
            "epc_filter",
            [
                tool("filter_rows", conditions=[{"field": "epc_rating", "op": "in", "value": ["C", "B", "A"]}]),
                tool("find_similar", listing="M3", use_filter=True),
            ],
        ),
        (
            "epc_filter",
            [
                tool("filter_rows", conditions=[{"field": "epc_rating", "op": "in", "value": ["A+", "A", "B", "C"]}]),
                tool("find_similar", listing="M3", use_filter=True),
            ],
        ),
        (
            "include_unknown_when_asked",
            [
                tool("filter_rows", conditions=WITHIN_10, include_unknown=True, refine=False),
                tool("find_nearest", listing="m3", use_filter=True),
            ],
        ),
    ],
)
def test_good_trajectories_pass_their_case(name, calls):
    case = case_named(name)
    assert check_trajectory(calls, case.expect, case.forbid) == []


def test_weighted_case_rejects_the_wrong_weights():
    case = case_named("weighted_location")
    calls = [tool("find_similar", listing="M1", k=5, weights={"age": 2})]
    assert check_trajectory(calls, case.expect, case.forbid) != []


# --- The right answer -----------------------------------------------------------------------------------------------

NEAREST_M3_FILTERED = Ranking(
    "M3", "find_nearest", filters=[([{"field": "miles_to_city", "op": "<=", "value": 10}], False)]
)


def nearest_table(rows):
    """A Markdown table of (Property Id, miles) rows, as an answer would show them."""
    lines = ["| Property Id | Estate | Miles from M3 |", "|---|---|---|"]
    lines += [f"| {property_id} | Somewhere | {miles} |" for property_id, miles in rows]
    return "\n".join(lines)


def test_listed_units_reads_tables_lists_and_written_out_lists():
    answer = "\n".join(
        [
            "Property Id 999 stands out.",  # prose: not part of the list
            nearest_table([(170, 9.1), (169, 9.1)]),
            "",
            "1. Property Id 167, 9.1 miles",
            "Property Ids 602 (905 sq ft, 1.0), 689 (905, 0.999) and 221 (905, 0.999).",
        ]
    )
    assert [property_id for property_id, _ in listed_units(answer)] == [170, 169, 167, 602, 689, 221]


def test_listed_units_reads_tied_units_written_together():
    # Both formats Claude used in the 2026-11-23-1856 run, which the check first misread as dropped units.
    table = "| Property Id | Estate | Miles |\n|---|---|---|\n| 170, 169, 167 | Jade Bridge 6 | 9.1 |\n| 168 | Jade Bridge 6 | 9.2 |"
    assert [property_id for property_id, _ in listed_units(table)] == [170, 169, 167, 168]
    written = "Property Ids 602 (905 sq ft, score 1.0), 689 and 221 (both 905 sq ft, 0.999), 874 (902, 0.997)."
    units = listed_units(written)
    assert [property_id for property_id, _ in units] == [602, 689, 221, 874]
    assert units[1][1] == units[2][1] == "both 905 sq ft, 0.999"  # tied units share their details


def test_ranking_passes_the_right_answer(data):
    reference = NEAREST_M3_FILTERED.reference(data)
    assert [property_id for property_id, _ in reference[:3]] == [170, 169, 167]  # Jade Bridge 6, 9.1 miles
    assert NEAREST_M3_FILTERED.check(nearest_table(reference[:5]), data) == []


def test_ranking_catches_a_dropped_unit_a_wrong_order_and_a_swapped_value(data):
    reference = NEAREST_M3_FILTERED.reference(data)[:5]
    dropped = reference[:1] + reference[2:]
    assert "expected the top 3" in NEAREST_M3_FILTERED.check(nearest_table(dropped), data)[0]
    swapped = [reference[1], reference[0], *reference[2:]]
    assert NEAREST_M3_FILTERED.check(nearest_table(swapped), data) != []
    wrong_value = [(reference[0][0], 51.5), *reference[1:]]
    assert NEAREST_M3_FILTERED.check(nearest_table(wrong_value), data) == [
        f"M3: Property Id {reference[0][0]}'s row doesn't show its miles from the listing ({reference[0][1]})"
    ]


def test_ranking_accepts_a_shared_value_stated_once_and_an_optional_listing_left_out(data):
    nearest = Ranking("M3", "find_nearest")
    reference = nearest.reference(data)[:3]  # all 0.0 miles: M3 sits on Velvet Park 60
    table = "All 0.0 miles away.\n\n| Property Id | Unit |\n|---|---|\n" + "\n".join(
        f"| {property_id} | Unit |" for property_id, _ in reference
    )
    assert nearest.check(table, data) == []
    assert Ranking("M1", "find_similar", required=False).check("Which listing do you mean?", data) == []
    assert Ranking("M1", "find_similar").check("Which listing do you mean?", data) != []


def test_scores_mentions_and_no_figures(data):
    assert Scores("M3").check("| M3 Spitfire Park | 0.86 | 0.62 | 0.87 |", data) == []
    assert Scores("M3").check("| M3 Spitfire Park | 0.86 | 0.73 | 0.87 |", data) != []  # M4's location score
    assert Mentions([50, 358], "x").check("Kept 50; 358 left out as unknown.", data) == []
    assert Mentions([50, 358], "x").check("Kept 50.", data) == ["doesn't state 358 (x)"]
    rent = NoFigures(r"£\s?\d[\d,.]*|\d[\d,.]*\s*(?:per|/)\s*sq\.?\s?ft", "a rent figure")
    assert rent.check("I can't suggest a rent: the data has none.", data) == []
    assert rent.check("Offer about £8.50 per sq ft.", data) != []


def test_numbers_from_the_tool_descriptions_are_supported(data):
    # "up to 50 in one search" comes from find_nearest's description of k (1-50), not a tool result: a false
    # alarm in the 2026-11-23-1856 run. run_case now passes the tools as the model sees them.
    model = Scripted(replies=[AIMessage("I can list up to 50 units in one search.")])
    result = run_case(case_named("scope_rent"), data, model, good_verdict)
    assert result["numbers"]["passed"]
