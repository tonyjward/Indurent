"""Tests for indurent_matching.graph, with a scripted chat model: no Claude calls, no API key.

The scripted model replies with the next message in its script each time it's called, so a test lays out the tool
calls and the final answer, and LangGraph runs the real tools and checks in between.
"""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from scripted import Scripted, call

from indurent_matching.graph import build_graph
from indurent_matching.tools import Data


def run(data, *replies, question="A question"):
    model = Scripted(replies=list(replies))
    out = build_graph(data, model=model).invoke({"messages": [HumanMessage(question)]})
    return out, model


@pytest.fixture(scope="module")
def data(portfolio, marketed):
    return Data(portfolio, marketed)


def tool_results(out):
    return [
        json.loads(m.content) if m.content.startswith("{") else m.content
        for m in out["messages"]
        if isinstance(m, ToolMessage)
    ]


WITHIN_10 = [{"field": "miles_to_city", "op": "<=", "value": 10}]


def test_prompt_3_filter_flows_into_find_nearest(data):
    out, _ = run(
        data,
        call("filter_rows", conditions=WITHIN_10, include_unknown=False, refine=False),
        call("find_nearest", listing="M3", use_filter=True, k=3),
        AIMessage(
            "Nearest within 10 miles of a city: Property Ids 170, 169 and 167 (Jade Bridge 6). "
            "358 units with no usable location were left out."
        ),
    )
    assert len(out["candidates"]) == 50  # the filter is held in the state
    filtered, nearest = tool_results(out)
    assert "candidates" not in filtered and filtered["kept"] == 50  # the ids stay in the state, not the prompt
    assert [m["property_id"] for m in nearest["matches"]] == [170, 169, 167]
    assert out["messages"][-1].content.startswith("Nearest within 10 miles")  # the model's answer ends the turn


def test_search_without_use_filter_ignores_the_filter(data):
    out, _ = run(
        data,
        call("filter_rows", conditions=WITHIN_10, include_unknown=False, refine=False),
        call("find_nearest", listing="M3", use_filter=False, k=3),
        AIMessage("358 left out. Property Id 848 is nearest."),
    )
    nearest = tool_results(out)[1]
    assert {m["estate"] for m in nearest["matches"]} == {"Velvet Park 60"}  # the whole portfolio


def test_use_filter_without_a_filter_is_an_error_for_the_model(data):
    out, _ = run(data, call("find_similar", listing="M3", use_filter=True), AIMessage("I need a filter first."))
    assert tool_results(out)[0].startswith("Error: use_filter is true but no filter is in place")


EPC_A = [{"field": "epc_rating", "op": "==", "value": "A"}]


def test_refine_true_narrows_the_current_filter(data):
    # "Within 10 miles of a major city", then "now only those with an EPC rating of A".
    out, _ = run(
        data,
        call("filter_rows", conditions=WITHIN_10, include_unknown=False, refine=False),
        call("filter_rows", conditions=EPC_A, include_unknown=False, refine=True),
        AIMessage("11 units; 358 and 9 were unknown."),
    )
    first, second = tool_results(out)
    assert first["kept"] == 50
    assert second["started_with"] == 50 and second["kept"] == 11  # built on the first filter
    assert len(out["candidates"]) == 11


def test_refine_false_replaces_the_current_filter(data):
    out, _ = run(
        data,
        call("filter_rows", conditions=WITHIN_10, include_unknown=False, refine=False),
        call("filter_rows", conditions=EPC_A, include_unknown=False, refine=False),
        AIMessage("276 units; 326 were unknown."),
    )
    second = tool_results(out)[1]
    assert second["started_with"] == 1_250 and second["kept"] == 276  # the 10-mile filter is gone
    assert len(out["candidates"]) == 276


def test_reset_filters(data):
    out, _ = run(
        data,
        call("filter_rows", conditions=WITHIN_10, include_unknown=False, refine=False),
        call("reset_filters"),
        AIMessage("Cleared; 358 were unknown."),
    )
    assert out["candidates"] is None


def test_tool_errors_come_back_as_messages(data):
    out, _ = run(data, call("get_listing", listing="M9"), AIMessage("There's no M9."))
    assert tool_results(out)[0].startswith("Error: No marketed listing matches 'M9'")
