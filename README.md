# Indurent: matching marketed warehouses to the portfolio

Python logic that compares marketed warehouse listings with Indurent's existing estate, using vectorised
similarity, retrieval over the portfolio, and an LLM to interpret requests and explain results.

- **How it works and why:** [`DESIGN.md`](DESIGN.md)
- **Data-quality evidence behind every cleaning rule:** [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb)

> Work in progress: data preparation, similarity, the tools and the agent are done; the web app follows.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 if needed).

1. **Install** the package and its dependencies into `.venv`:

   ```bash
   uv sync
   ```

2. **Add the assessment data** (not included in the repo). Copy the two spreadsheets into `data/`:

   ```
   data/Current Portfolio.xlsx
   data/Marketed Warehouses.xlsx
   ```

3. **Run the tests:**

   ```bash
   uv run pytest
   ```

   Tests that need the spreadsheets are skipped if they aren't in `data/`.

Run commands through `uv run` (or activate `.venv` first) so they use the project's environment, not a system or
Anaconda Python.

## Using it

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

To rerun the EDA notebook: `uv run jupyter lab`, then open `notebooks/01_eda.ipynb`.

## Project layout

```
src/indurent_matching/   the package
  config.py              thresholds, city list, known-issues register and similarity settings, each citing its evidence
  geo.py                 coastline and region checks, distances, nearest major city
  data.py                load and clean both spreadsheets
  similarity.py          the three similarity blocks (age, location, size) and how they combine
  tools.py               the functions the AI calls: plain Python, no LangGraph or model
  graph.py               the LangGraph agent: Claude with the tools
  studio.py              builds the graph on the real data for LangGraph Studio
notebooks/               01 EDA; 02-04 similarity by block; 05 prompt 1; 06 the similarity functions;
                         07 prompt 2; 08 the tools
tests/                   pytest suite (conftest.py loads the spreadsheets once for the real-data tests)
scripts/                 one-off maintenance scripts (see below)
data/geo/                UK boundaries (committed; the rest of data/ is gitignored)
langgraph.json           tells `langgraph dev` where the graph is
```

### How the modules fit together

Each module is listed under the module that uses it:

```
studio.py / the app            loads the data and builds the agent
├── data.py                    loads and cleans the spreadsheets
│   └── geo.py                 coastline and region checks, distances
└── graph.py                   the LangGraph agent: Claude with the tools
    └── tools.py               the functions the AI calls
        └── similarity.py      age, location and size scores
            └── geo.py
```

`config.py` (thresholds and settings) isn't shown: the cleaning, geography and similarity modules all read their
settings from it.

The two branches meet only at the top: `data.py` produces the cleaned tables, and the agent branch works on them.
Everything below `graph.py` is plain Python, so `tools.py` and `similarity.py` work without LangGraph or a model
(from a notebook, the tests or the app).

## Maintenance

`data/geo/uk_boundaries.geojson` (UK land outline and regions, from [Natural Earth](https://www.naturalearthdata.com/),
public domain) is committed, so there's nothing to build. To refresh it from Natural Earth:

```bash
uv run python scripts/build_boundaries.py
```
