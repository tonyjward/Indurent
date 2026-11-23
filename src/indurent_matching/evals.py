"""Evals for the agent: run a question through it and check what it did and said.

Four checks per case, cheapest first:

* **Trajectory (code):** did it call the right tools, in the right order, with the right key arguments? Each case
  lists the calls it expects; other calls in between are fine. A case can also list tools it must not call.
* **Numbers (code):** every number in the answer must appear in the tool results or the question. This catches
  invented scores, distances, counts and Property Ids. It's the check a production validation step would run.
* **Judge (LLM):** Claude reads the tool results and the answer and lists any claim the results don't support
  (such as an invented link between two units, which the number check can't see because both Ids are real), any
  caveat the answer leaves out, and whether the answer behaved as the case expects (e.g. asked which listing, or
  declined to estimate).
* **Right answer (code):** the answer reports the right result. Each case has a reference answer worked out by
  calling the tools directly with the right arguments (the tools' own tests check those results). The units the
  answer lists must be the reference ranking, in order, each with its own value on its row; required counts must
  appear; and an out-of-scope answer must contain no invented figure. The other checks can't see a dropped unit,
  a wrong order, or one unit's distance shown against another.

`run_case` runs one case and applies the checks; `evals/run_evals.py` runs them all against Claude and writes a
report. The tests run these functions with a scripted model and judge, so they cost nothing.
"""

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from indurent_matching import chat
from indurent_matching import tools as T
from indurent_matching.graph import SYSTEM_PROMPT, build_graph, make_tools

# --- Cases ---------------------------------------------------------------------------------------------------------


@dataclass
class Call:
    """A tool call a case expects: the tool's name, and the arguments that matter.

    Each argument is either the exact value expected, or a function that takes the value given and returns True
    if it's acceptable (for arguments with more than one right answer, e.g. "M3" or "Spitfire Park").
    Arguments not listed aren't checked.
    """

    name: str
    args: dict = field(default_factory=dict)

    def matches(self, call: dict) -> bool:
        if call["name"] != self.name:
            return False
        for key, expected in self.args.items():
            if key not in call["args"]:
                return False
            given = call["args"][key]
            if callable(expected):
                if not expected(given):
                    return False
            elif given != expected:
                return False
        return True

    def __str__(self) -> str:
        shown = {key: ("<check>" if callable(value) else value) for key, value in self.args.items()}
        return f"{self.name}({json.dumps(shown)})"


@dataclass
class Case:
    """One eval case.

    Args:
        name: a short id for the report.
        questions: what the user says, one entry per turn (most cases have one).
        expect: the tool calls expected, in order, across the whole conversation.
        forbid: tools that must not be called.
        behaviour: what a good answer does, in a sentence, for the judge.
        answer_checks: the reference answer, as checks (`Ranking`, `Scores`, `Mentions`, `NoFigures`).
    """

    name: str
    questions: list[str]
    expect: list[Call]
    forbid: list[str]
    behaviour: str
    answer_checks: list = field(default_factory=list)


# --- Check 1: trajectory --------------------------------------------------------------------------------------------


def check_trajectory(calls: list[dict], expect: list[Call], forbid: list[str]) -> list[str]:
    """What's wrong with the tool calls, as a list of problems (empty: it passed).

    The expected calls must appear in the order given; other calls may come before, between or after them.
    """
    problems = []
    position = 0
    for expected in expect:
        while position < len(calls) and not expected.matches(calls[position]):
            position += 1
        if position == len(calls):
            problems.append(f"expected {expected}, in order, not found")
            break
        position += 1
    for call in calls:
        if call["name"] in forbid:
            problems.append(f"called {call['name']}, which this case shouldn't use")
    return problems


# --- Check 2: numbers -----------------------------------------------------------------------------------------------

# A number: digits with optional thousands commas and decimals. Not when it's part of a word or id ("M3", "GW7").
NUMBER = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w])|(?<![\w.])\d+(?:\.\d+)?(?![\w])")
# Row numbers in a table's first column ("| 1 |") and numbered list items ("1. "): positions, not facts.
ROW_NUMBER = re.compile(r"^\s*\|\s*\d+\s*\||^\s*\d+\.\s", re.MULTILINE)


def numbers_in(text: str) -> list[str]:
    """The numbers written in a piece of text, as written (e.g. "1,250", "0.88")."""
    return NUMBER.findall(text)


def _value(number: str) -> float:
    return float(number.replace(",", ""))


def _decimals(number: str) -> int:
    return len(number.split(".")[1]) if "." in number else 0


def _rounds_or_cuts_to(source: float, value: float, decimals: int) -> bool:
    """True if `source` rounded, or cut down, to `decimals` places gives `value`."""
    scale = 10**decimals
    return round(source, decimals) == value or math.floor(source * scale) / scale == value


def unsupported_numbers(answer: str, sources: list[str]) -> list[str]:
    """Numbers in the answer that don't appear in any source (the tool results and the questions).

    A number counts as supported if a source has it, or has a value that rounds or cuts down to it at the
    answer's precision (0.73 for a tool's 0.734; "51 miles or more" for 51.5). Row numbers in tables and numbered
    lists are skipped.
    """
    known = []
    for source in sources:
        known += [_value(number) for number in numbers_in(source)]
    unsupported = []
    for number in numbers_in(ROW_NUMBER.sub("|", answer)):
        value, decimals = _value(number), _decimals(number)
        if not any(_rounds_or_cuts_to(source, value, decimals) for source in known):
            unsupported.append(number)
    return sorted(set(unsupported), key=unsupported.index)


# --- Check 3: the judge ---------------------------------------------------------------------------------------------

JUDGE_PROMPT = """You check answers written by an analyst agent for a property company. The agent answers \
questions about newly marketed warehouse listings and the company's portfolio of units, using tools. The tool \
results are its only source of facts about the data; the tool descriptions say how its scores are calculated, \
and its instructions (system prompt) give rules it must follow, such as warnings to give. Statements that come \
from the instructions or tool descriptions are supported.

You get the agent's instructions, the tool descriptions, the user's question(s), the tool results, the agent's \
answer, and what a good answer does for this case. Report:

1. unsupported_claims: statements in the answer about the data that the tool results don't support. Include \
invented numbers, units, estates, relationships between units (e.g. saying two units are neighbours when nothing \
says so), and wrong explanations of why a unit ranks where it does. Quote or closely paraphrase each one. Don't \
include: descriptions of the method that agree with the tool descriptions, offers to do more, restating the \
question, or rounding of numbers that are in the results. List only the claims you flag: don't list claims you \
checked and found supported.
2. missing_caveats: limits or warnings in the tool results that the answer should pass on but doesn't (e.g. a \
listing can't be compared on location, units left out because their value is unknown). Leave empty if none.
3. behaviour_ok: whether the answer does what the case expects. behaviour_note: one sentence on why."""


class Verdict(BaseModel):
    """The judge's verdict on one answer."""

    unsupported_claims: list[str] = Field(description="Claims about the data the tool results don't support.")
    missing_caveats: list[str] = Field(description="Caveats from the tool results the answer leaves out.")
    behaviour_ok: bool = Field(description="Whether the answer does what the case expects.")
    behaviour_note: str = Field(description="One sentence on why.")


def judge_messages(
    case: Case, instructions: str, tool_descriptions: list[str], tool_results: list[str], answer: str
) -> list:
    """The messages the judge is sent for one answer. `instructions` is the agent's system prompt."""
    sections = [
        "## The agent's instructions\n\n" + instructions,
        "## Tool descriptions\n\n" + "\n\n".join(tool_descriptions),
        "## Question(s)\n\n" + "\n\n".join(case.questions),
        "## Tool results\n\n" + "\n\n".join(tool_results),
        "## The agent's answer\n\n" + answer,
        "## What a good answer does\n\n" + case.behaviour,
    ]
    return [SystemMessage(JUDGE_PROMPT), HumanMessage("\n\n".join(sections))]


def claude_judge(model=None) -> Callable:
    """A judge function backed by Claude (low effort), using Claude's structured outputs for the verdict.

    The function returns the verdict and the judge call's token usage.
    """
    if model is None:
        from indurent_matching.graph import default_model

        model = default_model(effort="low")
    structured = model.with_structured_output(Verdict, method="json_schema", include_raw=True)

    def judge(messages: list) -> tuple[Verdict, dict]:
        out = structured.invoke(messages)
        if out["parsing_error"] is not None:
            raise out["parsing_error"]
        return out["parsed"], usage_of([out["raw"]])

    return judge


# --- Check 4: the right answer --------------------------------------------------------------------------------------

PROPERTY_ID = re.compile(r"Property Ids?\s*\**\s*(\d+)")
# An item in a written-out list: "Property Ids 602 (905 sq ft, score 1.0), 689 (905, 0.999) and ...".
LISTED_IN_TEXT = re.compile(r"(\d+)\s*\(([^)]*)\)")
LIST_ITEM = re.compile(r"^\s*(?:\d+[.)]|[-*])\s")


def listed_units(answer: str) -> list[tuple[int, str]]:
    """The units an answer lists, in order: (Property Id, the table row or list item it's on).

    Units come from table rows (the Property Id column, or "Property Id N" in the row), list items that name a
    Property Id, and sentences that write a list out ("Property Ids 602 (905 sq ft, 1.0), 689 (...), ..."; the
    brackets stand in for the row). Other prose mentions are left out: they're often a unit picked out of the
    list, not the list.
    """
    units = []
    id_column = None
    for line in answer.splitlines():
        text = line.strip()
        if not text.startswith("|"):
            id_column = None  # a table ends at the first line that isn't a row
            written_out = LISTED_IN_TEXT.findall(line)
            if "Property Ids" in line and len(written_out) >= 3:
                units += [(int(property_id), details) for property_id, details in written_out]
            elif LIST_ITEM.match(line) and PROPERTY_ID.search(line):
                units.append((int(PROPERTY_ID.search(line).group(1)), line))
            continue
        cells = [cell.strip().strip("*").strip() for cell in text.strip("|").split("|")]
        headers = [cell.lower() for cell in cells]
        if "property id" in headers or "id" in headers:
            id_column = headers.index("property id") if "property id" in headers else headers.index("id")
            continue
        if set(text) <= set("|-: "):
            continue  # the line under a table's header
        if id_column is not None and id_column < len(cells) and re.search(r"\d+", cells[id_column]):
            units.append((int(re.search(r"\d+", cells[id_column]).group()), line))
        elif PROPERTY_ID.search(line):
            units.append((int(PROPERTY_ID.search(line).group(1)), line))
    return units


def _shows(text: str, value: float) -> bool:
    """True if the text has a number that is `value` rounded or cut down to the precision written."""
    return any(_rounds_or_cuts_to(value, _value(number), _decimals(number)) for number in numbers_in(text))


def _matching_run(listed: list[int], expected: list[int]) -> int:
    """How many of the listed Ids, from the first, match the expected Ids in order."""
    length = 0
    for listed_id, expected_id in zip(listed, expected):
        if listed_id != expected_id:
            break
        length += 1
    return length


@dataclass
class Ranking:
    """The answer lists a listing's ranked units correctly: the top ones, in order, each with its own value.

    Args:
        listing: the listing ("M3").
        tool: "find_similar" (value: the overall score) or "find_nearest" (value: miles from the listing).
        filters: the filters to apply first, in order, as (conditions, include_unknown).
        weights: for find_similar, if the case asks for them.
        at_least: how many units the answer must list, from the top (e.g. 10 for "the 10 most similar").
        required: False when the answer may leave this listing out (e.g. it asks which listing is meant).
    """

    listing: str
    tool: str
    filters: list = field(default_factory=list)
    weights: dict | None = None
    at_least: int = 3
    required: bool = True

    def reference(self, data) -> list[tuple[int, float]]:
        """The right ranking, from the tools: (Property Id, value), best first."""
        candidates = None
        for conditions, include_unknown in self.filters:
            candidates = T.filter_rows(data, conditions, include_unknown, candidates)["candidates"]
        if self.tool == "find_similar":
            out = T.find_similar(data, self.listing, k=50, weights=self.weights, candidates=candidates)
            return [(m["property_id"], m["overall"]) for m in out["matches"]]
        out = T.find_nearest(data, self.listing, k=50, candidates=candidates)
        return [(m["property_id"], m["miles_from_listing"]) for m in out["matches"]]

    def check(self, answer: str, data) -> list[str]:
        reference = self.reference(data)
        expected = [property_id for property_id, _ in reference]
        listed = listed_units(answer)
        ids = [property_id for property_id, _ in listed]
        if expected[0] not in ids:
            if self.required:
                return [f"{self.listing}: the top unit, Property Id {expected[0]}, isn't listed"]
            return []

        # The longest run of the reference, in order, starting wherever the answer lists the top unit (it may list
        # it more than once, e.g. in a summary table and again in the full one).
        start, run = 0, 0
        for position, property_id in enumerate(ids):
            if property_id != expected[0]:
                continue
            length = _matching_run(ids[position:], expected)
            if length > run:
                start, run = position, length

        problems = []
        needed = min(self.at_least, len(expected))
        if run < needed:
            problems.append(
                f"{self.listing}: expected the top {needed} as Property Ids {expected[:needed]}, in that order; "
                f"the answer lists {ids[start : start + needed]}"
            )
        what = "overall score" if self.tool == "find_similar" else "miles from the listing"
        values = dict(reference)
        shown = listed[start : start + run]
        # A value shared by every unit listed may be stated once ("all 0.0 miles away") rather than on each row.
        shared = {values[property_id] for property_id, _ in shown}
        if len(shared) == 1 and None not in shared and _shows(answer, shared.pop()):
            return problems
        for property_id, row in shown:
            value = values[property_id]
            if value is not None and not _shows(row, value):
                problems.append(f"{self.listing}: Property Id {property_id}'s row doesn't show its {what} ({value})")
        return problems


@dataclass
class Scores:
    """The answer gives a listing's homogeneity scores (all it has), close to where it names the listing."""

    listing: str
    required: bool = False

    def check(self, answer: str, data) -> list[str]:
        scores = T.get_homogeneity_scores(data, self.listing)["scores"]
        wanted = {dimension: value for dimension, value in scores.items() if value is not None}
        lines = answer.splitlines()
        mentions = [n for n, line in enumerate(lines) if re.search(rf"\b{self.listing}\b", line)]
        if not mentions:
            return [f"{self.listing}: not mentioned"] if self.required else []
        for n in mentions:
            nearby = " ".join(lines[n : n + 8])  # a table row, or a short section under the listing's name
            if all(_shows(nearby, value) for value in wanted.values()):
                return []
        return [f"{self.listing}: its scores {wanted} aren't all shown with it"]


@dataclass
class Mentions:
    """The answer states these counts (e.g. how many units a filter kept and left out)."""

    numbers: list[float]
    why: str

    def check(self, answer: str, data) -> list[str]:
        missing = [number for number in self.numbers if not _shows(answer, number)]
        if missing:
            return [f"doesn't state {', '.join(str(n) for n in missing)} ({self.why})"]
        return []


@dataclass
class NoFigures:
    """The answer contains no figure of this kind (e.g. no rent for an out-of-scope rent question)."""

    pattern: str
    why: str

    def check(self, answer: str, data) -> list[str]:
        found = re.findall(self.pattern, answer)
        if found:
            return [f"gives {', '.join(found)} ({self.why})"]
        return []


def check_answer(case: Case, answer: str, data) -> list[str]:
    """What's wrong with the answer against the case's reference answer (empty: it's right)."""
    problems = []
    for answer_check in case.answer_checks:
        problems += answer_check.check(answer, data)
    return problems


# --- Running a case -------------------------------------------------------------------------------------------------

# Claude Opus 5.5's price per million tokens, in US dollars (the agent and the judge both use it).
PRICE_PER_MTOK = {"input": 4.00, "output": 20.00}


def cost_usd(tokens: dict) -> float:
    """The cost of some token usage ({"input": ..., "output": ...})."""
    return sum(tokens[kind] * PRICE_PER_MTOK[kind] / 1e6 for kind in PRICE_PER_MTOK)


def usage_of(messages: list) -> dict:
    """Input and output tokens across the model messages in a list."""
    tokens = {"input": 0, "output": 0}
    for message in messages:
        if isinstance(message, AIMessage):
            usage = message.usage_metadata or {}
            tokens["input"] += usage.get("input_tokens", 0)
            tokens["output"] += usage.get("output_tokens", 0)
    return tokens


def all_tool_calls(messages: list) -> list[dict]:
    """Every tool call in the conversation, in order: name and arguments."""
    calls = []
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                calls.append({"name": call["name"], "args": call["args"]})
    return calls


def last_answer(messages: list) -> str:
    """Everything the agent said after the last question, as the user saw it (text before tool calls included)."""
    start = max(position for position, message in enumerate(messages) if isinstance(message, HumanMessage))
    pieces = [chat.text_of(m.content) for m in messages[start:] if isinstance(m, AIMessage)]
    return "\n\n".join(piece for piece in pieces if piece)


def run_case(case: Case, data, model, judge: Callable) -> dict:
    """Run one case through a fresh conversation and apply the three checks.

    Args:
        case: the case.
        data: the cleaned portfolio and listings (`tools.Data`).
        model: the chat model for the agent (Claude, or a scripted one in the tests).
        judge: takes the judge messages and returns (Verdict, token usage); see `claude_judge`.

    Returns:
        The case name, the tool calls, the answer, each check's result, `passed` (all three checks passed)
        and the agent's and the judge's token usage.
    """
    graph = build_graph(data, model=model, checkpointer=InMemorySaver())
    config = chat.thread_config(f"eval-{case.name}")
    for question in case.questions:
        graph.invoke({"messages": [HumanMessage(question)]}, config)
    messages = graph.get_state(config).values["messages"]

    calls = all_tool_calls(messages)
    answer = last_answer(messages)
    tool_results = [str(m.content) for m in messages if isinstance(m, ToolMessage)]

    trajectory = check_trajectory(calls, case.expect, case.forbid)
    numbers = unsupported_numbers(answer, tool_results + case.questions)
    descriptions = [f"{t.name}: {t.description}" for t in make_tools(data)]
    verdict, judge_tokens = judge(judge_messages(case, SYSTEM_PROMPT, descriptions, tool_results, answer))
    judge_ok = not verdict.unsupported_claims and verdict.behaviour_ok
    correct = check_answer(case, answer, data)

    return {
        "case": case.name,
        "questions": case.questions,
        "tool_calls": calls,
        "answer": answer,
        "trajectory": {"passed": not trajectory, "problems": trajectory},
        "numbers": {"passed": not numbers, "unsupported": numbers},
        "judge": {"passed": judge_ok, **verdict.model_dump()},
        "correct": {"passed": not correct, "problems": correct},
        "passed": not trajectory and not numbers and judge_ok and not correct,
        "tokens": {"agent": usage_of(messages), "judge": judge_tokens},
        "cost_usd": round(cost_usd(usage_of(messages)) + cost_usd(judge_tokens), 4),
    }


# --- The report -----------------------------------------------------------------------------------------------------


def _mark(passed: bool) -> str:
    return "pass" if passed else "FAIL"


def report_markdown(results: list[dict], model: str, cost_usd: float) -> str:
    """A readable report: pass rates per check, a table of cases, then the details of each failure."""
    checks = ["trajectory", "numbers", "judge", "correct"]
    lines = [f"# Eval report: {model}", "", f"{len(results)} runs, ${cost_usd:.2f}.", ""]
    lines += ["| Check | Passed |", "|---|---|"]
    for check in [*checks, "overall"]:
        passed = sum(1 for r in results if (r["passed"] if check == "overall" else r[check]["passed"]))
        lines.append(f"| {check} | {passed} of {len(results)} |")
    lines += [
        "",
        "| Case | Trajectory | Numbers | Judge | Right answer | Missing caveats |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        marks = " | ".join(_mark(r[check]["passed"]) for check in checks)
        lines.append(f"| {r['case']} | {marks} | {len(r['judge']['missing_caveats'])} |")

    for r in results:
        if r["passed"] and not r["judge"]["missing_caveats"]:
            continue
        lines += ["", f"## {r['case']}", ""]
        lines += [f"> {question}" for question in r["questions"]]
        lines += ["", "Tool calls:", ""]
        lines += [f"- `{c['name']}({json.dumps(c['args'])})`" for c in r["tool_calls"]] or ["- none"]
        lines.append("")
        lines += [f"- Trajectory: {problem}" for problem in r["trajectory"]["problems"]]
        if r["numbers"]["unsupported"]:
            lines.append(f"- Numbers not in any tool result: {', '.join(r['numbers']['unsupported'])}")
        lines += [f"- Unsupported claim: {claim}" for claim in r["judge"]["unsupported_claims"]]
        lines += [f"- Wrong answer: {problem}" for problem in r["correct"]["problems"]]
        lines += [f"- Missing caveat: {caveat}" for caveat in r["judge"]["missing_caveats"]]
        lines.append(f"- Behaviour {_mark(r['judge']['behaviour_ok'])}: {r['judge']['behaviour_note']}")
        lines += ["", "<details><summary>Answer</summary>", "", r["answer"], "", "</details>"]
    return "\n".join(lines) + "\n"
