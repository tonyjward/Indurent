"""Tests for indurent_matching.chat: streaming the answer and reading back the agent's tool calls.

They run the real graph and tools with a scripted chat model (tests/scripted.py), so no Claude calls.
"""

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver
from scripted import Scripted, call

from indurent_matching import chat
from indurent_matching.graph import build_graph
from indurent_matching.tools import Data


@pytest.fixture(scope="module")
def data(portfolio, marketed):
    return Data(portfolio, marketed)


def conversation(data, *replies):
    """A graph with memory, a scripted model and its config, ready for several questions."""
    graph = build_graph(data, model=Scripted(replies=list(replies)), checkpointer=InMemorySaver())
    return graph, chat.thread_config("test")


def ask(graph, config, question="A question"):
    """Ask a question; return the streamed answer and the conversation's messages after it."""
    answer = "".join(chat.stream_text(graph, question, config))
    return answer, graph.get_state(config).values["messages"]


def test_text_of_keeps_only_text_blocks():
    content = [
        {"type": "thinking", "thinking": "hmm"},
        {"type": "text", "text": "Hello"},
        {"type": "tool_use", "name": "find_nearest", "input": {}},
        {"type": "text", "text": " there"},
    ]
    assert chat.text_of(content) == "Hello there"
    assert chat.text_of("plain") == "plain"
    assert chat.text_of(AIMessageChunk(content=[{"type": "text", "text": "x", "index": 0}]).content) == "x"


def test_stream_text_yields_the_answer_and_not_tool_results(data):
    graph, config = conversation(data, call("list_listings"), AIMessage("There are five listings."))
    answer, _ = ask(graph, config)
    assert answer == "There are five listings."  # the list_listings JSON isn't streamed


def test_text_before_a_tool_call_is_separated_from_the_answer(data):
    graph, config = conversation(
        data, call("list_listings", content="Let me check the listings."), AIMessage("There are five.")
    )
    answer, _ = ask(graph, config)
    assert answer == "Let me check the listings.\n\nThere are five."


def test_tool_calls_lists_the_latest_question_only_with_outcomes(data):
    graph, config = conversation(
        data,
        call("list_listings"),
        AIMessage("Five listings."),
        call("get_listing", listing="M9"),
        call("find_nearest", listing="M3", use_filter=False, k=3),
        AIMessage("Done."),
    )
    ask(graph, config)
    _, messages = ask(graph, config, "Second question")
    calls = chat.tool_calls(messages)
    assert [c["name"] for c in calls] == ["get_listing", "find_nearest"]  # not the first question's list_listings
    assert calls[0]["outcome"].startswith("Error")  # there's no M9
    assert calls[1]["args"] == {"listing": "M3", "use_filter": False, "k": 3}
    assert "M3" in calls[1]["outcome"]  # the result's summary
