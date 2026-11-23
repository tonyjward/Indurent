"""Run the brief's three example prompts through the agent (real Claude calls) and save the answers.

Each prompt runs in a fresh conversation, word for word as the brief gives it. For each one the script records the
tool calls Claude made (name and arguments), the final answer, and the tokens used and their cost. Results go to
data/examples/ (gitignored: they contain portfolio data), as JSON and as a readable Markdown file.

    uv run python scripts/run_brief_prompts.py

Needs ANTHROPIC_API_KEY in .env. Stops before starting a prompt if the spend so far has reached COST_CAP_USD.
"""

import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from indurent_matching.data import load_marketed, load_portfolio
from indurent_matching.graph import MODEL, build_graph
from indurent_matching.tools import Data

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "examples"
COST_CAP_USD = 2.00
PRICE_PER_MTOK = {"input": 4.00, "output": 20.00}  # Claude Opus 5.5

PROMPTS = {
    "prompt_1": "Find the 10 most similar properties in the estate to the newly marketed property.",
    "prompt_2": "Provide a correlation score for the homogeneity of the marketed property(ies) with the rest of the "
    "estate, scoring each by physical characteristics, location, and age separately.",
    "prompt_3": "Find the closest properties to the marketed property, after excluding any property in the portfolio "
    "that is more than 10 miles from a major city.",
}


def text_of(message: AIMessage) -> str:
    """The text of a model message (its content may be a string or a list of blocks)."""
    if isinstance(message.content, str):
        return message.content
    return "".join(block.get("text", "") for block in message.content if isinstance(block, dict))


def summarise(messages: list) -> dict:
    """Tool calls, final answer and token usage from one conversation."""
    calls, tokens = [], {"input": 0, "output": 0}
    for message in messages:
        if isinstance(message, AIMessage):
            calls += [{"name": c["name"], "args": c["args"]} for c in message.tool_calls]
            usage = message.usage_metadata or {}
            tokens["input"] += usage.get("input_tokens", 0)
            tokens["output"] += usage.get("output_tokens", 0)
    errors = [m.content for m in messages if isinstance(m, ToolMessage) and str(m.content).startswith("Error")]
    cost = sum(tokens[kind] * PRICE_PER_MTOK[kind] / 1e6 for kind in tokens)
    return {
        "tool_calls": calls,
        "tool_errors": errors,
        "answer": text_of(messages[-1]),
        "tokens": tokens,
        "cost_usd": round(cost, 4),
    }


def main() -> None:
    data = Data(
        load_portfolio(ROOT / "data" / "Current Portfolio.xlsx"),
        load_marketed(ROOT / "data" / "Marketed Warehouses.xlsx"),
    )
    graph = build_graph(data)
    OUT.mkdir(parents=True, exist_ok=True)
    results, spent = {}, 0.0
    for name, prompt in PROMPTS.items():
        if spent >= COST_CAP_USD:
            print(f"Stopping before {name}: ${spent:.2f} spent, cap ${COST_CAP_USD:.2f}")
            break
        out = graph.invoke({"messages": [HumanMessage(prompt)]}, {"recursion_limit": 40})
        results[name] = {"prompt": prompt, **summarise(out["messages"])}
        spent += results[name]["cost_usd"]
        print(
            f"{name}: {len(results[name]['tool_calls'])} tool calls, {results[name]['tokens']}, "
            f"${results[name]['cost_usd']:.3f} (total ${spent:.3f})"
        )

    (OUT / "brief_prompts.json").write_text(json.dumps({"model": MODEL, "results": results}, indent=2))
    lines = [f"# Brief prompts: answers from {MODEL}", ""]
    for name, r in results.items():
        lines += [f"## {name.replace('_', ' ').title()}", "", f"> {r['prompt']}", "", "**Tool calls:**", ""]
        lines += [f"- `{c['name']}({json.dumps(c['args'])})`" for c in r["tool_calls"]] or ["- none"]
        lines += ["", "**Answer:**", "", r["answer"], "", f"*Tokens: {r['tokens']}; cost ${r['cost_usd']:.3f}*", ""]
    (OUT / "brief_prompts.md").write_text("\n".join(lines))
    print(f"Total ${spent:.3f}. Saved to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
