# Development notes

Working detail that the [README](README.md) leaves out: what the cleaned data carries, viewing the agent in
LangSmith Studio, deploying the app, and maintenance.

## The cleaned data

```python
from indurent_matching.data import load_marketed, load_portfolio

portfolio = load_portfolio("data/Current Portfolio.xlsx")  # one row per unit, keyed by property_id
listings = load_marketed("data/Marketed Warehouses.xlsx")  # keyed by listing_id M1-M5
```

Each row carries quality information:

- `<field>_doubtful` flags for size, build year, yard depth, min and max eaves, and doors: the value is kept but
  not used for matching.
- `location_confidence`: `high` (inside its labelled region), `medium` (on land but outside it) or `unknown`
  (no coordinates, or 1 km+ offshore). Unknown locations get no `nearest_city` / `miles_to_city`.
- `quality_notes`: plain-English reasons, e.g. "coordinates are 27 km offshore, so location can't be used".
- `doors_describe_unit`: False where the door count is an estate total copied onto every unit of the estate.

Flags come from general rules (run on every row, including new listings) and from a known-issues register in
`config.py`: findings tied to particular Property Ids in this file. **Review the register whenever the portfolio
data is refreshed**; its Property Ids may no longer apply.

## Keys (`.env`)

In the project root (gitignored, never committed):

```
ANTHROPIC_API_KEY=...        # Claude; each question costs roughly $0.10-0.15
ANTHROPIC_WORKSPACE_ID=...   # optional: only for a key that isn't scoped to one workspace
LANGSMITH_API_KEY=...        # free LangSmith account, for tracing and Studio
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com   # only for an EU LangSmith account
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=indurent
```

The tests switch tracing off, so they never create traces. Eval runs are best sent to their own project:
`LANGSMITH_PROJECT=indurent-evals uv run python evals/run_evals.py`.

## Viewing the graph in LangSmith Studio

[LangSmith Studio](https://docs.langchain.com/langsmith/studio) shows the agent's graph and lets you chat with it,
step by step: each model call, each tool call and its result, and the state (the messages and the filtered
`candidates`).

```bash
uv run langgraph dev
```

It reads `langgraph.json`, builds the graph from `studio.py` on the real data, and opens Studio in the browser.
Pick the `indurent` graph, add a human message under Input (e.g. one of the brief's prompts) and press Submit.
Restart the server after changing `.env`.

**If Studio says "Failed to fetch":** the browser is blocking the Studio web page from reaching the server on
localhost (Safari and Brave do this, and Chrome can too). Start the server with a tunnel instead, which gives it a
public `trycloudflare.com` address: `uv run langgraph dev --tunnel`.

**If it then says the domain "is not allowed":** the tunnel address changes each time the server starts, and
LangSmith only connects to domains on its list. Add the new `trycloudflare.com` domain under Advanced Settings in
Studio's connection dialog.

## Running the brief's prompts through the agent

`uv run python scripts/run_brief_prompts.py` asks the agent the three prompts, word for word, and saves its answers
and tool calls to `data/examples/` (gitignored; about $0.40, stops at a $2 cap).

## Deploying the app

The app runs on beast behind the existing Cloudflare Tunnel (DESIGN.md section 6). `~/repos/Indurent` on beast is
a git remote of this repo (`beast`); `data/*.xlsx` and `.env` are copied across by hand. To deploy a change:

```bash
git push beast main                          # from the laptop
sudo systemctl restart indurent-app          # on beast
```

`deploy/indurent-app.service` has the one-off install steps.

## Maintenance

`data/geo/uk_boundaries.geojson` (UK land outline and regions, from [Natural Earth](https://www.naturalearthdata.com/),
public domain) is committed, so there's nothing to build. To refresh it from Natural Earth:

```bash
uv run python scripts/build_boundaries.py
```
