"""Tests for the web app (app/app.py), through Streamlit's AppTest: the real page, no browser.

It runs the real graph and tools with a scripted chat model (tests/scripted.py), so no Claude calls.
They need the assessment spreadsheets, like the other real-data tests.
"""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from scripted import Scripted, call
from streamlit.testing.v1 import AppTest

from indurent_matching import chat

APP = str(Path(__file__).resolve().parents[1] / "app" / "app.py")
WITHIN_10 = [{"field": "miles_to_city", "op": "<=", "value": 10}]
PROMPT_3_REPLIES = [
    call("filter_rows", conditions=WITHIN_10, include_unknown=False, refine=False),
    call("find_nearest", listing="M3", use_filter=True, k=3),
    AIMessage("Nearest within 10 miles of a city: Property Ids 170, 169 and 167."),
]


@pytest.fixture(autouse=True)
def needs_data(portfolio, marketed):
    """Skip these tests when the spreadsheets aren't in data/ (the app loads them itself)."""


def start(*replies, **session):
    """Open the app with a scripted model and any session values set beforehand."""
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["chat_model"] = Scripted(replies=list(replies))
    for key, value in session.items():
        at.session_state[key] = value
    return at.run()


def replies(at):
    return [m for m in at.chat_message if m.name == "assistant"]


def sidebar_text(at):
    return " ".join(c.value for c in at.sidebar.caption) + " ".join(m.value for m in at.sidebar.markdown)


def test_opens_with_a_greeting_the_prompt_buttons_and_the_listings():
    at = start()
    assert not at.exception
    assert len(replies(at)) == 1 and "Every number comes from" in replies(at)[0].markdown[0].value
    labels = [b.label for b in at.button]
    assert {"10 most similar units", "Homogeneity scores", "Closest, within 10 miles of a city"} <= set(labels)
    text = sidebar_text(at)
    for listing_id in ["M1", "M2", "M3", "M4", "M5"]:
        assert f"**{listing_id}**" in text
    m2 = next(m for m in at.sidebar.markdown if m.value.startswith("**M2**"))
    assert "Can't compare on location: no coordinates" in m2.help  # the details show on hover


def test_a_prompt_button_asks_that_prompt_and_shows_the_tool_calls():
    at = start(*PROMPT_3_REPLIES)
    next(b for b in at.button if b.label == "Closest, within 10 miles of a city").click().run()
    assert not at.exception
    user = [m for m in at.chat_message if m.name == "user"]
    assert user[0].markdown[0].value == chat.BRIEF_PROMPTS["prompt_3"]  # the brief's wording, not the label
    answer = replies(at)[-1]
    assert answer.markdown[0].value.startswith("Nearest within 10 miles")
    assert answer.expander[0].label == "How this was worked out (2 tool calls)"
    assert "find_nearest" in " ".join(m.value for m in answer.expander[0].markdown)
    assert "Closest, within 10 miles of a city" not in [b.label for b in at.button]  # only on a new conversation


def test_new_chat_clears_the_conversation():
    at = start(*PROMPT_3_REPLIES)
    at.chat_input[0].set_value("Closest within 10 miles of a city to M3?").run()
    next(b for b in at.button if b.label == "New chat").click().run()
    assert len(replies(at)) == 1  # just the greeting
    assert at.session_state.graph.get_state(chat.thread_config(at.session_state.thread_id)).values == {}  # fresh memory


def test_a_model_error_is_shown_in_the_chat():
    at = start(RuntimeError("overloaded"))
    at.chat_input[0].set_value("Hello").run()
    assert not at.exception
    assert replies(at)[-1].markdown[0].value == "Sorry, something went wrong: overloaded"
