# Indurent: matching marketed warehouses to the portfolio

Python logic that compares marketed warehouse listings with Indurent's existing estate, using vectorised
similarity, retrieval over the portfolio, and an LLM to interpret requests and explain results.

- **How it works and why:** [`DESIGN.md`](DESIGN.md)
- **Data-quality evidence behind every cleaning rule:** [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb)

> Work in progress: data preparation is being built step by step; matching, the AI layer and the app follow.

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

portfolio = load_portfolio("data/Current Portfolio.xlsx")    # one row per unit, keyed by property_id
listings = load_marketed("data/Marketed Warehouses.xlsx")    # keyed by listing_id M1-M5
```

Each row carries quality information:

- `<field>_doubtful` flags: the value is kept but not used for matching (so far `size_sqft_doubtful`).
- `location_confidence`: `high` (inside its labelled region), `medium` (on land but outside it) or `unknown`
  (no coordinates, or 1 km+ offshore). Unknown locations get no `nearest_city` / `miles_to_city`.
- `quality_notes`: plain-English reasons, e.g. "coordinates are 27 km offshore, so location can't be used".

To rerun the EDA notebook: `uv run jupyter lab`, then open `notebooks/01_eda.ipynb`.

## Project layout

```
src/indurent_matching/   the package
  data.py                load and clean both spreadsheets
  geo.py                 coastline and region checks, nearest major city
  config.py              thresholds and the city list, each citing the EDA section behind it
notebooks/01_eda.ipynb   exploratory data analysis and the cleaning decisions
tests/                   pytest suite (conftest.py loads the spreadsheets once for the real-data tests)
scripts/                 one-off maintenance scripts (see below)
data/geo/                UK boundaries (committed; the rest of data/ is gitignored)
```

## Maintenance

`data/geo/uk_boundaries.geojson` (UK land outline and regions, from [Natural Earth](https://www.naturalearthdata.com/),
public domain) is committed, so there's nothing to build. To refresh it from Natural Earth:

```bash
uv run python scripts/build_boundaries.py
```
