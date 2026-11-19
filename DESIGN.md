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
  features.py    per-block feature vectors and similarity               (step 2: vectorise)
  tools.py       filter_rows, find_similar, homogeneity_scores, ...     (step 3: retrieval)
  graph.py       LangGraph StateGraph, system prompt, output schema     (steps 4-5: prompt + answer)
app/app.py       Streamlit front end over the same modules
tests/           tools, graph (scripted model), LLM tool-call evals
deploy/          systemd unit for beast
```

The notebook explores; the package decides. The package never imports from the notebook.

## 1. Preprocessing (`data.py`)

`load_clean(path) -> DataFrame` is used for **both** files, so new listings get the same treatment as the
portfolio. It returns:

| Kind | Columns |
|---|---|
| Clean values | `property_id`, `estate`, `unit`, `size_sqft`, `eaves_min_m`, `eaves_max_m`, `doors`, `build_year`, `lat`, `lon`, `epc`, `parking`, `yard_depth_m` |
| Structure | `estate_units`, `is_single_unit` |
| Derived | `x_km`, `y_km` (British National Grid), `nearest_city`, `miles_to_city`, `derived_region` |
| Quality | `location_confidence` (high / medium / unknown), `doubtful_fields` (set of field names), `quality_notes` (human-readable reasons) |

Rules come in two kinds, kept separate in the code:

**General rules** (run on any row, including new listings):

- placeholders (`X`, `-`, `TBC`) and zeros in eaves/doors → missing; ranges (`20-28`) → midpoint
- `Size sqm` is read as square feet (the total matches Indurent's published 28-40m sq ft)
- location confidence from the coastline and labelled-region checks; no distances for `unknown`
- values constant across a multi-unit estate (doors, eaves, parking, yard depth) are estate-level, not unit facts
- implausible doors for the building's size; implausible eaves (under 5 m, or tall eaves on a tiny unit)

**Known-issues register** (findings specific to this dataset, each with its reason and EDA section):

| Property Ids | Fields | Reason |
|---|---|---|
| 1173-1187 | doors, eaves, yard depth | specs contradict size; order-blind scan (EDA 2.8) |
| 126 (Opal Forge 42), 548 (Granite Summit 84) | doors | ~7x denser than a typical shed (EDA 2.6) |
| 259-263 (Granite Mews 59) | all physical + age | 2.1-2.4 m eaves, 1939-1953, EPC A (EDA 2.9) |

In production this register would go back to the owners of the source data so the records get fixed.

## 2. Vectorisation and similarity (`features.py`)

Structured data gets **numeric** vectors, not text embeddings: text embeddings handle magnitudes badly ("12,000"
and "120,000" look alike) and can't apply the quality rules.

Three blocks, matching the brief's prompt 2:

| Block | Features | Similarity |
|---|---|---|
| Physical | log size, max eaves, doors (single-unit estates only) | robust-scaled distance over shared features |
| Location | BNG x/y | `exp(-miles / 25)`-style decay on great-circle miles |
| Age | build year | decay on the year gap (half-similar at ~15 years) |

- **Missing or doubtful values are masked, not imputed**: each block compares only the features both units have
  (Gower-style similarity for mixed, incomplete data). A block with nothing to compare is "not available", not 0.
- **Coverage** is recorded per match (which features it rested on) and penalises thin matches in the ranking.
- **Combined score**: weighted mean of the available blocks; weights are parameters (default equal).
- **Estate cap**: at most 2 units per estate in a result list (one estate has 64 units).
- **Not used for similarity**: EPC (unreliable, EDA 2.9), parking and yard depth (estate-level / sparse),
  region labels (only 26% agree with coordinates). EPC and parking are shown as context.

**Homogeneity score (prompt 2).** "Correlation score" is ambiguous; the interpretation used, and stated in every
answer: per block, the listing's mean similarity to its k nearest portfolio units, expressed as a percentile of the
same quantity for portfolio units against their own neighbours. "Physical 82" = more typical of the portfolio than
82% of portfolio units are of each other.

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
| Tools | filter semantics (unknowns dropped/kept), estate cap, masking, known-issues flags, a listing with no coordinates | free, deterministic |
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
