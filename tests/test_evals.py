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
    Verdict,
    check_trajectory,
    judge_messages,
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


def test_row_numbers_in_tables_and_lists_are_skipped():
    answer = "| 1 | 0.88 |\n| 2 | 0.81 |\n\n1. First\n2. Second"
    assert unsupported_numbers(answer, ["0.88 0.81"]) == []


# --- Judge input and the report -------------------------------------------------------------------------------------


def test_judge_messages_hold_every_section():
    case = case_named("scope_rent")
    system, human = judge_messages(case, ["find_similar: ..."], ['{"summary": "x"}'], "I can't price rent.")
    assert "unsupported_claims" in system.content
    for section in ["Tool descriptions", "Question(s)", "Tool results", "The agent's answer", "What a good answer"]:
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
            AIMessage("Nearest within 10 miles of a city: Property Ids 170, 169 and 167."),
        ]
    )
    result = run_case(case_named("brief_3_closest_near_city"), data, model, good_verdict)
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
