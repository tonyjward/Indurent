# Design: matching marketed warehouses to the Indurent portfolio

How the pieces fit together, and why. The data-quality evidence behind each rule is in
[`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb); section 7 of the notebook is the decisions table this design
implements.

## Principle

**Code computes, the model interprets.** Every number in an answer (similarity, distance, score, count) comes
from deterministic, tested Python. The LLM turns the user's request into tool calls, and turns the tools' results
into a caveated answer. It never estimates a value, and never fills in missing data.

## Components

```
src/indurent_matching/
  data.py        load + clean both files, attach quality flags          (step 1: preprocess)
  geo.py         coastline/region checks, nearest major city
  config.py      thresholds, city list, known-issues register
  features.py    per-block feature vectors and similarity               (step 2: vectorise)
  tools.py       filter_rows, find_similar, homogeneity_scores, ...     (step 3: retrieval)
  graph.py       LangGraph StateGraph, system prompt, output schema     (steps 4-5: prompt + answer)
app/app.py       Streamlit front end over the same modules
tests/           tools, graph (scripted model), LLM tool-call evals
deploy/          systemd unit for beast
```

The notebook explores; the package decides. The package never imports from the notebook.

## 1. Preprocessing (`data.py`, `geo.py`, `config.py`)

`load_portfolio(path)` and `load_marketed(path)` run the **same** cleaning steps (in EDA order), so new listings
get the same treatment as the portfolio. Thresholds, the city list and the known-issues register live in
`config.py`, each citing the EDA section behind it. Column names are the spreadsheet's, in snake_case
(`Min. Eaves m` -> `min_eaves_m`); the portfolio is keyed by `property_id`, the listings by `listing_id` (M1-M5).

| Kind | Columns |
|---|---|
| Identity | `property_id` / `listing_id`, `industrial_estate_name`, `unit_name` |
| Raw, parsed and cleaned | `region`, `latitude`, `longitude`, `car_parking_spaces`, `size_sqft` (renamed from `size_sqm`), `build_year`, `yard_depth_m`, `min_eaves_m`, `max_eaves_m`, `doors`, `epc_rating` |
| Estate structure (portfolio only) | `estate_units`, `is_single_unit` |
| Location | `km_offshore`, `km_to_region`, `location_confidence` (high / medium / unknown), `derived_region`, `nearest_city`, `miles_to_city` |
| Doors | `doors_describe_unit` (False where the count is a copied estate total), `doors_per_unit_estimate` (display only) |
| Quality | `<field>_doubtful` flags (size, build year, yard depth, min/max eaves, doors), `quality_notes` (plain-English reasons, passed to the LLM) |

A doubtful value is kept but never scored. A note without a flag is context only (e.g. "very large").

Rules come in two kinds, kept separate in the code:

**General rules** (run on any row, including new listings):

- placeholders (`X`, `-`, `TBC`) and zeros in eaves/doors -> missing; ranges (`20-28`) -> midpoint
- `Size sqm` is read as square feet (the total matches Indurent's published 28-40m sq ft); under 500 sq ft is
  doubtful, over 600k sq ft is noted as very large
- location confidence from the coastline and labelled-region checks (1 km tolerance); no city distance for
  `unknown`
- a door count that is the same on every unit of a multi-unit estate is an estate total, not the unit's own
- a unit's own door count is doubtful if it has more than 3 doors and fewer than 2,000 sq ft per door
- EPC keeps A-G only; it is context only, with one column-wide caveat (ratings don't fit build year)

**Known-issues register** (`KNOWN_ISSUES`: findings tied to particular Property Ids that no general rule can
express). Fields are flagged only where a value is recorded, with one note per finding:

| Property Ids | Fields | Reason |
|---|---|---|
| 1173-1187 | doors, min/max eaves, yard depth | specs look attached to the wrong units: eaves-size correlation reverses inside the block (EDA 2.8) |
| 259-263 (Granite Mews 59) | min/max eaves, build year | 2.1-2.4 m eaves, 1939-1953, EPC A (EDA 2.9) |

Opal Forge 42 and Granite Summit 84 (implausible doors, EDA 2.6) need no entry: the door-density rule catches
them. The register must be reviewed whenever the portfolio data is refreshed; in production it would go back to
the owners of the source data so the records get fixed.

## 2. Vectorisation and similarity (`features.py`)

Structured data gets **numeric** features, not text embeddings: text embeddings handle magnitudes badly
("12,000" and "120,000" look alike) and can't apply the quality rules. The evidence for each choice below is in
the similarity notebooks: [`02_similarity_build_year`](notebooks/02_similarity_build_year.ipynb),
[`03_similarity_location`](notebooks/03_similarity_location.ipynb) and
[`04_similarity_physical`](notebooks/04_similarity_physical.ipynb).

Three blocks, matching the brief's prompt 2, each scored on its own feature:

| Block | Feature | Usable values |
|---|---|---|
| Age | build year | recorded and not doubtful |
| Location | latitude/longitude | high and medium confidence only |
| Physical | size (log scale) | recorded and not doubtful |

Every score is 0-1. A block a listing can't be scored on is **"not available"**, never 0 (M2 has no coordinates
or build year; M5 is offshore).

**Not scored, shown as context:** eaves height (only M4 and M5 have it; scoring it would put them on a different
basis from the other listings), doors (39 usable portfolio counts), volume (99% correlated with size), EPC
(unreliable, EDA 2.9), parking (estate-level), yard depth (34 units, no listings), region labels (only 26% agree
with coordinates). All can be filtered on.

### Homogeneity scores (prompt 2): how typical is the listing of the portfolio?

"Correlation score" is ambiguous; the interpretation used, and stated in every answer, is *how typical the
listing is of the portfolio on each block*. Each score is scaled so the most typical value in the portfolio
scores 1:

| Block | Score | Context reported alongside |
|---|---|---|
| Age | Units in the listing's 5-year build-year band (1955-59, ..., 2020-22) / units in the busiest band | |
| Location | Best portfolio estate's mean miles to its 5 nearest other estates / the listing's mean miles to its 5 nearest estates, leaving out the estate it sits on | The estate it sits on ("sits on Velvet Park 60, which you own"); the nearest other estate; the percentile among portfolio estates |
| Physical | Portfolio units within a doubling-wide window centred on the listing's size (size / 1.41 to size x 1.41) / the busiest such window | Eaves height against similar-sized units ("tall for its size") |

Why these forms: build years are flat from 1970 to 2021, so fixed bands work (the jump at 1970 is in the data);
sizes slope steadily, so a moving window avoids jumps at band edges. Every listing with coordinates sits on an
existing Indurent estate, so location uses estates, not units, as the neighbours: nearest units would all be
from the listing's own estate at about 0 miles.

### Pairwise similarity (prompt 1): how similar is the listing to one portfolio unit?

Each block uses the same form, similarity = 0.5^(gap / h), so it is 1 for an identical value and halves with
every step of h:

| Block | Gap | Halves every |
|---|---|---|
| Age | years between build years | 10 years |
| Location | miles between the two (great-circle) | 10 miles |
| Physical | doublings between the sizes, \|log2(size ratio)\| | 1 doubling (twice or half the size) |

- **Combined score**: the mean of the blocks, equally weighted by default; weights are parameters.
- **Missing blocks**: a block the *listing* lacks is dropped for that listing (M2 is matched on size only, and
  the answer says so). A block a *portfolio unit* lacks scores 0, so a unit with less known about it can't
  outrank one that matches on everything; the answer shows it as "unknown".
- **Estate cap**: at most 2 units per estate in a result list. Without it, a listing's top 10 would be mostly
  units on the estate it sits on, which all score about 1 on location.

## 3. Tools (`tools.py`)

The working candidate set (a list of Property Ids) lives in the graph state. Tools update it and return short
summaries, so the model never receives the full table.

| Tool | Purpose | Arguments |
|---|---|---|
| `list_listings` | What can be compared | none |
| `get_listing` | Resolve "the marketed property" to one listing, with its own quality notes | `name` |
| `filter_rows` | Narrow the candidate set | `conditions: [{field, op, value}]`, `include_unknown: bool` (required) |
| `find_similar` | Rank candidates against a listing | `listing`, `k=10`, `weights`, `max_per_estate=2`, `use_filtered_set` |
| `homogeneity_scores` | Prompt 2 | `listing` |
| `reset_filters` | Start again within a conversation | none |

- `field` and `op` are enums; schemas are strict (`additionalProperties: false`).
- `include_unknown` is required so every filter decides explicitly what happens to units with missing values
  (the within / beyond / unknown rule). Summaries always report how many were excluded as unknown.
- EPC is allowed as a *filter* (the user's explicit choice), with the reliability caveat carried in the summary.
- Tool descriptions state conventions, e.g. "excluding properties more than 10 miles from a major city" means
  `miles_to_city <= 10`.

## 4. Orchestration and prompting (`graph.py`)

A hand-built LangGraph `StateGraph`:

```
START -> agent <-> tools -> validate -> respond -> END
                              |
                              +-> agent (once, with the error) if validation fails
```

- **State**: messages, candidate set, last results, listing in focus.
- **validate** (code): every Property Id cited in the answer was returned by a tool this turn; the answer matches
  the output schema. One retry, then answer with the failure stated.
- **System prompt**: acquisitions-analyst role; use only tool results; cite Property Ids; lead with limitations of
  the listing ("Tech Hub has no coordinates, so similarity is physical only"); report coverage and flags; never
  estimate missing values; ask when the listing is ambiguous rather than picking one.
- **Output**: structured (matches, block scores, caveats, summary), rendered by the app.
- **Model**: Claude (`claude-opus-5-5` by default, `CHAT_MODEL` / `CHAT_EFFORT` env overrides), key from `.env`.

## 5. Testing

| Layer | What | Cost |
|---|---|---|
| Tools | filter semantics (unknowns dropped/kept), estate cap, missing blocks ("not available" vs 0), known-issues flags, a listing with no coordinates | free, deterministic |
| Graph | scripted chat model replays chosen tool calls: state flows filter -> similarity, validation rejects an invented Property Id, retry stops after one | free, deterministic |
| Tool-call evals | real model, `@pytest.mark.llm` (skipped by default): each case = prompt + expected calls and arguments | paid, run on request |

Eval cases: the three brief prompts and rewordings; direction traps ("more than 10 miles" -> `<= 10`); ambiguity
("the marketed property" with five listings -> ask or run all five, never pick silently); defaults (k = 10).
Arguments are compared field by field; each case runs several times and reports a pass rate, since model output
varies.

## 6. Streamlit app and deployment

Pages: **Ask** (chat, with the tool calls and arguments shown), **Example prompts** (stored outputs for the three
brief prompts, no API call), **Explore** (the same tools with sliders, no LLM), **Data quality** (key EDA
findings and the decisions table).

Deployment on beast, the same mechanism as the fpl API:

1. `deploy/indurent-app.service` runs Streamlit bound to `127.0.0.1:8503` (8501 and 8502 are taken by fpl).
2. A public hostname on the existing shed Cloudflare Tunnel points at `http://localhost:8503`.
3. `data/` is copied across by hand (gitignored).

Safeguards before sharing:

- **Cloudflare Access** (email one-time PIN, allowlist of the interviewers and Tony): the assessment data is not
  for publication.
- A separate Anthropic workspace/key with its own spend limit, and a per-session message cap.
- Stored example outputs, so the app is useful even if the key is off or the limit is hit.

## 7. Production notes (for the write-up)

- Geocode from postcodes (ONS lookup) instead of trusting supplied coordinates; validate on ingest.
- Feed the known-issues register back to data owners; track fixes.
- Move vectors to pgvector/FAISS once the portfolio or listing volume justifies it; precompute portfolio blocks.
- Retrieval over the methodology itself (decisions, EDA findings) so the model can explain *why* a rule applies.
- Estate-level summaries as a second retrieval unit ("which estates does this listing resemble?").
- Tracing (LangSmith/OpenTelemetry), eval runs in CI on a fixed budget, prompt caching for the system prompt.

## Order of work

1. `data.py` + tests
2. `features.py` and `tools.py` + tests
3. `graph.py` + scripted-model tests
4. LLM evals and example outputs (needs an API key; spend agreed first)
5. Streamlit app, then deploy behind Access
