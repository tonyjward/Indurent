"""The agent: a LangGraph graph that gives the tools in `tools.py` to Claude.

```
START -> agent <-> tools -> END
```

* **agent:** Claude, with the system prompt and the tools. It turns the request into tool calls, then writes the
  answer from the tool results. It computes nothing itself.
* **tools:** runs the tool calls. The tools wrap `tools.py`; the data is filled in here, so the model only sees the
  arguments it should choose.

The graph state holds the messages and the filtered set from the last `filter_rows` (`candidates`, None for the whole
portfolio). The search tools read `candidates` only when the model sets `use_filter=true`, so a filter never leaks
into a search by accident.

`build_graph(data)` makes the graph. Tests pass a scripted model; the app passes a checkpointer for conversation
memory; LangGraph Studio uses `studio.py`, which builds it without one (Studio supplies its own).

A production version would add a validation step after the agent (DESIGN.md, production notes): check that every
cited Property Id came from a tool, and that a filter's unknown count is reported, sending a failing answer back.
"""

import json
import os
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import InjectedState, ToolNode, tools_condition
from langgraph.types import Command
from pydantic import BaseModel

from indurent_matching import tools as T

load_dotenv()
MODEL = os.environ.get("CHAT_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("CHAT_EFFORT", "medium")  # Claude Opus 5.5's default; set explicitly so it's visible

SYSTEM_PROMPT = """You are an analyst for Indurent's Acquisitions team. You compare newly marketed warehouse listings \
with Indurent's existing portfolio of industrial units, using the tools provided.

Rules:
- Every number, score, distance and Property Id in your answer must come from a tool result in this conversation. \
Never estimate or fill in a missing value.
- Cite portfolio units as "Property Id 869" (or "Property Ids 869, 865 and 857").
- If a request doesn't say which marketed listing it means and there's more than one, ask, or answer for every \
listing. Never pick one silently.
- Lead with what limits the answer: blocks a listing can't be compared on (e.g. no coordinates), and data-quality \
notes. Pass on the caveats the tools return.
- "Similar" means find_similar, scored on age, location and physical (size). "Closest" or "nearest" means nearest \
on the map: use find_nearest.
- A "correlation score for homogeneity" means how typical the listing is of the portfolio: use \
get_homogeneity_scores, give each dimension separately, and state that interpretation.
- When filtering, exclude units with unknown values (include_unknown=false) unless the user asks to include them, \
or only asks to remove units known to fail the condition. Always say how many units were left out as unknown, \
and offer to include them.
- "Excluding any property more than 10 miles from a major city" means keep miles_to_city <= 10.
- If a filter uses epc_rating, say that the portfolio's EPC ratings look unreliable (newer buildings are mostly \
rated E), so the result should be treated with caution.
- Use refine=true when the user adds to the current filter ("now only those with..."), and refine=false when \
they start a new one.
- After filter_rows, call the search tool with use_filter=true to search the filtered set. Use \
use_filter=false for a search that shouldn't be filtered. Call reset_filters when the user starts a new question \
that shouldn't keep an earlier filter.
- Don't change the weights or set max_per_estate unless the user asks.
- If a listing sits on an estate Indurent already owns, say so: it's the most useful single location fact.
- Be concise: a short summary, then the units as a list or table with the facts that explain the ranking."""


class State(MessagesState):
    """The messages, and the filtered set from the last filter_rows (None: the whole portfolio)."""

    candidates: list[int] | None


class Condition(BaseModel):
    """One filter condition: a field, an operator and a value."""

    field: Literal[
        "miles_to_city",
        "nearest_city",
        "size_sqft",
        "build_year",
        "derived_region",
        "location_confidence",
        "epc_rating",
        "estate_units",
        "car_parking_spaces",
    ]
    op: Literal["<=", ">=", "<", ">", "==", "in", "between"]
    value: float | str | list[float | str]


def _result(content: dict) -> str:
    """Turn a tool result into the JSON text Claude receives.

    The functions in tools.py return Python dictionaries. A tool's result reaches Claude as text, so this turns the
    dictionary into JSON before handing it back.
    """
    return json.dumps(content)


def _filtered(state: dict, use_filter: bool) -> list[int] | None:
    """Work out which units a search should look at.

    If use_filter is false, it returns None, which means search the whole portfolio. If use_filter is true, it
    returns the filtered set of property ids from the state. If use_filter is true but no filter has been run, it
    raises an error telling Claude to run filter_rows first, rather than quietly searching the whole portfolio and
    giving an answer that claims to be filtered when it isn't.

    find_similar and find_nearest both use this, so the rule lives in one place.
    """
    if not use_filter:
        return None
    if state.get("candidates") is None:
        raise T.ToolError("use_filter is true but no filter is in place: call filter_rows first, or set it false")
    return state["candidates"]


def make_tools(data: T.Data) -> list:
    """The tools the model can call: the functions in tools.py with the data filled in, plus reset_filters."""

    @tool(parse_docstring=True)
    def list_listings() -> str:
        """List the marketed listings that can be compared with the portfolio, with what each can be matched on."""
        return _result(T.list_listings(data))

    @tool(parse_docstring=True)
    def get_listing(listing: str) -> str:
        """One marketed listing's facts and data-quality notes. Use it to resolve which property the user means.

        Args:
            listing: the listing's id ("M3") or name ("Spitfire Park", "Spitfire Park, Unit 42").
        """
        try:
            return _result(T.get_listing(data, listing))
        except T.ToolError as error:
            return f"Error: {error}"

    @tool(parse_docstring=True)
    def filter_rows(
        conditions: list[Condition],
        include_unknown: bool,
        refine: bool,
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Narrow the portfolio to the units meeting every condition. The search tools use it with use_filter=true.

        Ops: <=, >=, <, >, ==, in (a list), between ([low, high], inclusive); text fields take == and in only.
        "Excluding any property more than 10 miles from a major city" means keep miles_to_city <= 10.

        Args:
            conditions: the conditions, all of which must hold.
            include_unknown: true keeps units with no usable value for a condition's field; false drops them.
                Use false unless the user asks to include them; always say how many were left out.
            refine: true narrows the current filter further; false starts again from the whole portfolio.

        Notes:
            Injected state allows the tool to read the graph state. When refine is true we need the current
            filtered set, and LangGraph supplies it directly from the state, so we don't rely on Claude to pass
            the property ids across, which it might get wrong.

            Writing the new filtered set back is different. Normally a tool just returns a dict, but that only
            goes back to Claude as a message and doesn't change the state. So instead we return a Command, which
            tells LangGraph to update the candidates state with the filtered property ids.

            Because we return a Command, LangGraph no longer adds the tool's message for us, so we also update
            the messages state with a ToolMessage. Claude's API needs every tool call to be answered with the same
            id it was called with, which is why the tool takes tool_call_id (LangGraph fills this in, like state).
        """
        start = state.get("candidates") if refine else None
        try:
            result = T.filter_rows(data, [c.model_dump() for c in conditions], include_unknown, start)
        except T.ToolError as error:
            return Command(update={"messages": [ToolMessage(f"Error: {error}", tool_call_id=tool_call_id)]})
        shown = {key: value for key, value in result.items() if key != "candidates"}  # the ids stay in the state
        return Command(
            update={
                "candidates": result["candidates"],
                "messages": [ToolMessage(_result(shown), tool_call_id=tool_call_id)],
            }
        )

    @tool  # no Args section: its only argument is filled in by LangGraph
    def reset_filters(tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
        """Clear the filter, so searches use the whole portfolio again."""
        return Command(
            update={"candidates": None, "messages": [ToolMessage("Filter cleared.", tool_call_id=tool_call_id)]}
        )

    @tool(parse_docstring=True)
    def find_similar(
        listing: str,
        use_filter: bool,
        state: Annotated[dict, InjectedState],
        k: int = 10,
        weights: dict[str, float] | None = None,
        max_per_estate: int | None = None,
    ) -> str:
        """The k portfolio units most similar to a listing, scored on age, location and physical (size). Prompt 1.

        For the nearest or closest properties on the map, use find_nearest instead.

        Args:
            listing: the listing's id ("M3") or name.
            use_filter: true searches the set from filter_rows; false searches the whole portfolio.
            k: how many units to return, 1-50.
            weights: relative weights for "age", "location", "physical", e.g. {"location": 2}. Only if the user
                asks for one dimension to count more.
            max_per_estate: the most matches from any one estate. Only if the user asks for a spread.
        """
        try:
            return _result(T.find_similar(data, listing, k, weights, max_per_estate, _filtered(state, use_filter)))
        except T.ToolError as error:
            return f"Error: {error}"

    @tool(parse_docstring=True)
    def get_homogeneity_scores(listing: str) -> str:
        """How typical a listing is of the portfolio, scored separately on age, location and physical. Prompt 2.

        A low score says the listing is atypical, not which way: use `position` for that (e.g. older or smaller
        than most units).

        Args:
            listing: the listing's id ("M3") or name.
        """
        try:
            return _result(T.get_homogeneity_scores(data, listing))
        except T.ToolError as error:
            return f"Error: {error}"

    @tool(parse_docstring=True)
    def find_nearest(listing: str, use_filter: bool, state: Annotated[dict, InjectedState], k: int = 10) -> str:
        """The k portfolio units nearest to a listing on the map, by straight-line miles. Prompt 3 ("closest").

        Args:
            listing: the listing's id ("M3") or name.
            use_filter: true searches the set from filter_rows; false searches the whole portfolio.
            k: how many units to return, 1-50.
        """
        try:
            return _result(T.find_nearest(data, listing, k, _filtered(state, use_filter)))
        except T.ToolError as error:
            return f"Error: {error}"

    return [list_listings, get_listing, filter_rows, reset_filters, find_similar, get_homogeneity_scores, find_nearest]


# --- The graph ---------------------------------------------------------------------------------------------------


def default_model(effort: str | None = None) -> ChatAnthropic:
    """Claude, as configured by CHAT_MODEL and CHAT_EFFORT (key from ANTHROPIC_API_KEY or .env).

    `effort` overrides CHAT_EFFORT (the eval judge uses "low"). If ANTHROPIC_WORKSPACE_ID is set (a key that isn't
    scoped to one workspace), it's sent as a header so the requests go to that workspace, as in the fpl chat app.
    """
    workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    return ChatAnthropic(
        model=MODEL,
        max_tokens=16000,
        output_config={"effort": effort or EFFORT},
        default_headers={"anthropic-workspace-id": workspace_id} if workspace_id else None,
    )


def build_graph(data: T.Data, model=None, checkpointer=None):
    """The compiled graph. `model`: a LangChain chat model (tests pass a scripted one; default Claude).
    `checkpointer`: conversation memory for the app (Studio supplies its own, so leave it None there)."""
    tool_list = make_tools(data)
    llm = (model or default_model()).bind_tools(tool_list)

    def agent(state: State) -> dict:
        return {"messages": [llm.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])]}

    graph = StateGraph(State)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tool_list))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # tool calls -> tools; a plain answer -> END
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)
