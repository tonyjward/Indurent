"""Helpers between the agent and the web app: send a question, stream the answer, and read back what the agent did.

They're kept out of `app/` so they can be tested without Streamlit, and the brief prompts script uses them too.
"""

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# The brief's three example prompts, word for word.
BRIEF_PROMPTS = {
    "prompt_1": "Find the 10 most similar properties in the estate to the newly marketed property.",
    "prompt_2": "Provide a correlation score for the homogeneity of the marketed property(ies) with the rest of the "
    "estate, scoring each by physical characteristics, location, and age separately.",
    "prompt_3": "Find the closest properties to the marketed property, after excluding any property in the portfolio "
    "that is more than 10 miles from a major city.",
}

# The most graph steps one question may take (each model call and each round of tool calls is a step). The brief's
# prompts take up to about 15; this stops a confused agent going round in circles.
RECURSION_LIMIT = 40


def thread_config(thread_id: str) -> dict:
    """The config that tells the graph which conversation a question belongs to."""
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}


def text_of(content) -> str:
    """The text of a model message.

    Claude's content is either a plain string or a list of blocks (text, thinking, tool calls). Only the text
    blocks are part of the answer, so the rest are dropped.
    """
    if isinstance(content, str):
        return content
    pieces = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            pieces.append(block.get("text", ""))
    return "".join(pieces)


def stream_text(graph, question: str, config: dict):
    """Ask the agent a question and yield its answer as it's written, for Streamlit's `st.write_stream`.

    Only text from the agent node is passed on (not tool results). If Claude says something before calling a tool
    and then carries on after, the two model calls are separated by a blank line.
    """
    last_id = None
    said_something = False
    for chunk, meta in graph.stream({"messages": [HumanMessage(question)]}, config, stream_mode="messages"):
        if meta.get("langgraph_node") != "agent" or not isinstance(chunk, AIMessage):
            continue
        piece = text_of(chunk.content)
        if not piece:
            continue
        if said_something and chunk.id != last_id:
            yield "\n\n"
        last_id = chunk.id
        said_something = True
        yield piece


def _results_by_call(messages: list) -> dict:
    """Each tool result, keyed by the id of the tool call it answers."""
    results = {}
    for message in messages:
        if isinstance(message, ToolMessage):
            results[message.tool_call_id] = str(message.content)
    return results


def _outcome(result: str | None) -> str:
    """One line on what a tool call returned: its error, else the result's summary."""
    if result is None:
        return "no result"
    if result.startswith("Error"):
        return result
    try:
        parsed = json.loads(result)
    except json.JSONDecodeError:
        return result  # a plain-text result, e.g. "Filter cleared."
    return parsed.get("summary", "")


def tool_calls(messages: list) -> list[dict]:
    """The tool calls the agent made answering the latest question: name, arguments and outcome.

    The latest question is the last human message; everything after it is the agent's work on it.
    """
    start = 0
    for position, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            start = position
    turn = messages[start:]
    results = _results_by_call(turn)
    calls = []
    for message in turn:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                calls.append({"name": call["name"], "args": call["args"], "outcome": _outcome(results.get(call["id"]))})
    return calls
