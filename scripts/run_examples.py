"""Run example conversations through the agent (real Claude calls) and save each one to outputs/ as Markdown.

The brief's three prompts, word for word, each in its own file; then conversations that show what else the agent
can do: refining a filter over several turns, the choice of what to do with unknown values, listings with missing
data, weights and a spread across estates, the EPC warning, and questions it should decline.

Each file shows every turn: the question, the tool calls Claude made (name, arguments and a one-line outcome), and
its answer, with the tokens and cost.

    uv run python scripts/run_examples.py                       # all of them (about $1.50)
    uv run python scripts/run_examples.py prompt_3_closest_near_city refining_filters

Needs ANTHROPIC_API_KEY in .env. Stops before starting an example once the spend reaches COST_CAP_USD.
"""

import json
import sys
from datetime import date
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from indurent_matching import chat
from indurent_matching.data import load_marketed, load_portfolio
from indurent_matching.evals import cost_usd, last_answer, usage_of
from indurent_matching.graph import MODEL, build_graph
from indurent_matching.tools import Data

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
COST_CAP_USD = 3.00

# Each example: a file name, a title, what it shows (one or two sentences, for the reader), and the user's turns.
EXAMPLES = [
    {
        "name": "prompt_1_most_similar",
        "title": "Prompt 1: the most similar properties",
        "shows": "The brief's first prompt, word for word. It doesn't say which listing, so the agent has to ask "
        "or answer for all five, and say what each listing can be compared on.",
        "turns": [chat.BRIEF_PROMPTS["prompt_1"]],
    },
    {
        "name": "prompt_2_homogeneity",
        "title": "Prompt 2: homogeneity scores",
        "shows": "The brief's second prompt, word for word: a score for each dimension, separately, with the "
        "reading of 'correlation score for homogeneity' stated.",
        "turns": [chat.BRIEF_PROMPTS["prompt_2"]],
    },
    {
        "name": "prompt_3_closest_near_city",
        "title": "Prompt 3: the closest properties near a major city",
        "shows": "The brief's third prompt, word for word: a filter (keep units within 10 miles of a major city), "
        "then the nearest units on the map from what's left, with the units of unknown location counted.",
        "turns": [chat.BRIEF_PROMPTS["prompt_3"]],
    },
    {
        "name": "refining_filters",
        "title": "Refining a filter over several turns",
        "shows": "The filtered set is held in the conversation, so each follow-up narrows it further (refine), a "
        "search can run on the result, and a new question starts again from the whole portfolio.",
        "turns": [
            "Which of our units are within 20 miles of a major city?",
            "Now keep only those of 10,000 sq ft or more.",
            "And only those built in 1990 or later. Which of those are nearest to Spitfire Park?",
            "New question: which units in the whole portfolio are most similar to Stable Lane?",
        ],
    },
    {
        "name": "unknown_values",
        "title": "What to do with unknown values",
        "shows": "By default a filter leaves out units whose value is unknown, and the answer says how many. Asked to, "
        "the agent keeps them. M5 is compared on age and size only (its coordinates are offshore), so units with no "
        "location can still match it well: keeping them changes the list.",
        "turns": [
            "Find the 10 units most similar to Chancery Depot (M5), keeping only units within 20 miles of a major city.",
            "Now include the units whose location is unknown as well.",
        ],
    },
    {
        "name": "listing_with_missing_data",
        "title": "Listings with missing data",
        "shows": "M2 has no build year or coordinates, so it can only be compared on size, and many units tie. M5's "
        "coordinates are offshore, so it can't be placed on the map at all.",
        "turns": [
            "Which portfolio units are most like Tech Hub, Unit 8?",
            "Which of our units are nearest to M5 on the map?",
        ],
    },
    {
        "name": "weights_and_spread",
        "title": "Weights, and a spread across estates",
        "shows": "Similarity weights age, location and size equally unless asked; here location counts double. Then "
        "a cap of 2 units per estate, for a spread rather than one estate's units.",
        "turns": [
            "Find the 10 units most similar to M1, with location counting twice as much as age and size.",
            "Same again, but at most 2 units from any one estate.",
        ],
    },
    {
        "name": "epc_caution",
        "title": "A filter on EPC rating, with its warning",
        "shows": "The portfolio's EPC ratings look unreliable (newer buildings are mostly rated E), so the agent "
        "applies the filter as asked but warns that the result should be treated with caution.",
        "turns": ["Find the units most similar to M3 that have an EPC rating of C or better."],
    },
    {
        "name": "out_of_scope",
        "title": "Questions the data can't answer",
        "shows": "The agent never estimates a value or fills in missing data: asked for a rent figure, or a guess at "
        "a build year that isn't recorded, it says what the data doesn't hold and offers what it can do.",
        "turns": [
            "What rent per sq ft should we offer for M3?",
            "M2 has no build year recorded. What's your best guess for it?",
        ],
    },
]


def run_example(graph, example: dict) -> dict:
    """Run one example's turns in a fresh conversation; return each turn's tool calls and answer, and the cost."""
    config = chat.thread_config(f"example-{example['name']}")
    turns = []
    for question in example["turns"]:
        graph.invoke({"messages": [HumanMessage(question)]}, config)
        messages = graph.get_state(config).values["messages"]
        turns.append({"question": question, "tool_calls": chat.tool_calls(messages), "answer": last_answer(messages)})
    tokens = usage_of(graph.get_state(config).values["messages"])
    return {**example, "turns": turns, "tokens": tokens, "cost_usd": round(cost_usd(tokens), 4)}


def markdown(result: dict) -> str:
    """One example as a readable Markdown page."""
    lines = [f"# {result['title']}", "", result["shows"], ""]
    for number, turn in enumerate(result["turns"], start=1):
        if len(result["turns"]) > 1:
            lines += [f"## Turn {number}", ""]
        lines += [f"> **User:** {turn['question']}", "", "**Tool calls:**", ""]
        for call in turn["tool_calls"]:
            lines.append(f"- `{call['name']}({json.dumps(call['args'])})`")
            if call["outcome"]:
                lines.append(f"  - {call['outcome']}")
        if not turn["tool_calls"]:
            lines.append("- none")
        lines += ["", "**Answer:**", "", turn["answer"], ""]
    tokens = result["tokens"]
    lines.append(
        f"*{MODEL}, {date.today():%d %B %Y}. Tokens: {tokens['input']:,} in, {tokens['output']:,} out; "
        f"${result['cost_usd']:.2f}.*"
    )
    return "\n".join(lines) + "\n"


def index(results: list[dict]) -> str:
    """outputs/README.md: what each file shows."""
    intro = (
        f"Real conversations with the agent ({MODEL}, {date.today():%d %B %Y}), made by "
        "`scripts/run_examples.py`. Each file shows the questions, the tool calls Claude made (with their "
        "arguments and a one-line outcome) and its answers."
    )
    lines = [
        "# Example outputs",
        "",
        intro,
        "",
        "| File | What it shows |",
        "|---|---|",
    ]
    for result in results:
        lines.append(f"| [`{result['name']}.md`]({result['name']}.md) | {result['shows']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    chosen = sys.argv[1:]
    unknown = set(chosen) - {example["name"] for example in EXAMPLES}
    if unknown:
        sys.exit(f"Unknown examples: {sorted(unknown)}")
    data = Data(
        load_portfolio(ROOT / "data" / "Current Portfolio.xlsx"),
        load_marketed(ROOT / "data" / "Marketed Warehouses.xlsx"),
    )
    graph = build_graph(data, checkpointer=InMemorySaver())
    OUT.mkdir(exist_ok=True)

    spent = 0.0
    for example in EXAMPLES:
        if chosen and example["name"] not in chosen:
            continue
        if spent >= COST_CAP_USD:
            print(f"Stopping before {example['name']}: ${spent:.2f} spent, cap ${COST_CAP_USD:.2f}")
            break
        result = run_example(graph, example)
        spent += result["cost_usd"]
        (OUT / f"{example['name']}.md").write_text(markdown(result))
        print(f"{example['name']}: {len(result['turns'])} turn(s), ${result['cost_usd']:.3f} (total ${spent:.2f})")

    (OUT / "README.md").write_text(index(EXAMPLES))
    print(f"Total ${spent:.2f}. Saved to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
