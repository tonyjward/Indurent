"""Indurent matching: the web app. One conversation with the agent, like Claude or ChatGPT.

    uv run streamlit run app/app.py

The answer streams in as it's written. Under each answer, "How this was worked out" lists the tool calls behind it,
so every number can be traced to a tool. The sidebar lists the marketed listings. Needs ANTHROPIC_API_KEY in .env.
"""

import json
import os
import uuid
from pathlib import Path

import streamlit as st
from langgraph.checkpoint.memory import InMemorySaver

from indurent_matching import chat
from indurent_matching import tools as T
from indurent_matching.data import load_marketed, load_portfolio
from indurent_matching.graph import build_graph

DATA_DIR = Path(os.environ.get("INDURENT_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))

# Button labels for the brief's prompts (the full prompt shows when you hover).
PROMPT_LABELS = {
    "prompt_1": "10 most similar units",
    "prompt_2": "Homogeneity scores",
    "prompt_3": "Closest, within 10 miles of a city",
}

st.set_page_config(page_title="Indurent matching", page_icon="🏭", layout="wide")
# A chat column a little wider than Streamlit's default, and wide tables scroll sideways rather than being cut off.
st.markdown(
    """<style>
.block-container {max-width: 1100px; padding-left: 1rem; padding-right: 1rem;}
[data-testid="stMarkdownContainer"] table {display: block; overflow-x: auto; max-width: 100%; font-size: 0.9rem;}
</style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_data() -> T.Data:
    """The cleaned portfolio and listings, loaded once for the whole app rather than once per visitor."""
    return T.Data(
        load_portfolio(DATA_DIR / "Current Portfolio.xlsx"),
        load_marketed(DATA_DIR / "Marketed Warehouses.xlsx"),
    )


state = st.session_state
data = load_data()


def greeting() -> str:
    return (
        f"Hi! I compare the {len(data.marketed)} newly marketed warehouses (listed on the left) with Indurent's "
        f"portfolio of {len(data.portfolio):,} units. For a listing, I can:\n\n"
        f"- find the most similar units in the portfolio, comparing age, location and size\n"
        f"- score how typical the listing is of the portfolio, for age, location and size separately\n"
        f"- find the nearest units on the map, for example only those within 10 miles of a major city\n\n"
        f"Every number comes from the matching tools: open **How this was worked out** under an answer to see "
        f"the steps behind it.\n\n"
        f"Try one of the brief's example prompts below, or ask your own."
    )


def new_conversation() -> None:
    """Start a fresh conversation: new agent memory, so no earlier messages or filter.

    `chat_model` is only set by the tests (a scripted model); otherwise build_graph uses Claude.
    """
    state.thread_id = str(uuid.uuid4())
    state.messages = [{"role": "assistant", "text": greeting(), "tools": []}]
    state.graph = build_graph(data, model=state.get("chat_model"), checkpointer=InMemorySaver())


def show_tool_calls(calls: list[dict]) -> None:
    """The tool calls behind an answer, folded away under the answer."""
    if not calls:
        return
    with st.expander(f"How this was worked out ({len(calls)} tool call{'s' if len(calls) != 1 else ''})"):
        for call in calls:
            st.markdown(f"`{call['name']}({json.dumps(call['args'])})`")
            if call["outcome"]:
                st.caption(call["outcome"])


def show_listing(listing: dict) -> None:
    """One marketed listing in the sidebar, on one line. Its facts, and what it can't be compared on, show on hover."""
    year = listing["build_year"] if listing["build_year"] is not None else "year unknown"
    details = f"{listing['size_sqft']:,} sq ft · {year} · {listing['region_label'] or 'region unknown'}"
    for block, reason in listing["not_available"].items():
        details += f"  \nCan't compare on {block}: {reason}"
    st.markdown(f"**{listing['listing_id']}** {listing['estate']}, {listing['unit']}", help=details)


if "graph" not in state:
    new_conversation()
config = chat.thread_config(state.thread_id)

with st.sidebar:
    if st.button("New chat", width="stretch"):
        new_conversation()
        st.rerun()
    st.markdown("#### Marketed listings")
    for listing in T.list_listings(data)["listings"]:
        show_listing(listing)

for message in state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])
        show_tool_calls(message["tools"])

clicked = None
if len(state.messages) == 1:  # a new conversation: offer the brief's prompts
    columns = st.columns(len(PROMPT_LABELS))
    for column, (key, label) in zip(columns, PROMPT_LABELS.items()):
        if column.button(label, help=chat.BRIEF_PROMPTS[key], width="stretch"):
            clicked = chat.BRIEF_PROMPTS[key]

question = st.chat_input("Ask about the marketed listings") or clicked

if question:
    state.messages.append({"role": "user", "text": question, "tools": []})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(chat.stream_text(state.graph, question, config))
            calls = chat.tool_calls(state.graph.get_state(config).values["messages"])
        except Exception as error:  # noqa: BLE001 -- shown in the chat rather than crashing the page
            answer = f"Sorry, something went wrong: {error}"
            calls = []
    state.messages.append({"role": "assistant", "text": answer, "tools": calls})
    st.rerun()
