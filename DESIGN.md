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
  similarity.py  homogeneity scores and pairwise similarity per block   (step 2: vectorise)
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
| 259-263 (Granite Mews 59) | min/max eaves | 2.1-2.4 m eaves, house height (EDA 2.9). Its 1939-1953 build years are plausible and not flagged |

Opal Forge 42 and Granite Summit 84 (implausible doors, EDA 2.6) need no entry: the door-density rule catches
them. The register must be reviewed whenever the portfolio data is refreshed; in production it would go back to
the owners of the source data so the records get fixed.

## 2. Vectorisation and similarity (`similarity.py`)

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
| Age | Units in the listing's 5-year build-year band (1955-59, ..., 2020-22) / units in the most frequent band | |
| Location | Best portfolio estate's mean miles to its 5 nearest other estates / the listing's mean miles to its 5 nearest estates, leaving out the estate it sits on | The estate it sits on ("sits on Velvet Park 60, which you own"); the nearest other estate; the percentile among portfolio estates |
| Physical | Portfolio units of a similar size to the listing (size / 1.41 to size x 1.41: the largest twice the smallest) / the most any portfolio unit has of a similar size | Eaves height against similar-sized units ("tall for its size") |

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
| Physical | the ratio between the sizes (on a log scale, so half and double count the same) | time one size is twice the other |

Size uses a ratio where age and location use a fixed gap because what matters for size is the *relative*
difference: 1,000 vs 2,000 sq ft is a different class of unit, while 100,000 vs 101,000 sq ft is effectively the
same. Ten miles, or ten years, mean much the same wherever they fall.

- **Combined score**: the mean of the blocks, equally weighted by default; weights are parameters.
- **Missing blocks**: a block the *listing* lacks is dropped for that listing (M2 is matched on size only, and
  the answer says so). A block a *portfolio unit* lacks scores 0, so a unit with less known about it can't
  outrank one that matches on everything; the answer shows it as "unknown".
- **No estate cap by default**: the answer is the literal top 10. Units on the estate a listing sits on genuinely
  are among the most similar, so a top 10 can be concentrated (M3's are all on Velvet Park 60, its own estate);
  the answer says so. An optional `max_per_estate` limits units per estate when the user asks for a spread.

## 3. Tools (`tools.py`)

The working candidate set (a list of Property Ids) lives in the graph state. Tools update it and return short
summaries, so the model never receives the full table.

| Tool | Purpose | Arguments |
|---|---|---|
| `list_listings` | What can be compared | none |
| `get_listing` | Resolve "the marketed property" to one listing, with its own quality notes | `name` |
| `filter_rows` | Narrow the candidate set | `conditions: [{field, op, value}]`, `include_unknown: bool` (required) |
| `find_nearest` | Prompt 3: the nearest units on the map ("closest" means nearest, not most similar) | `listing`, `k=10`, `candidates` |
| `find_similar` | Rank candidates against a listing | `listing`, `k=10`, `weights`, `max_per_estate=None` (only when the user asks for a spread), `use_filtered_set` |
| `homogeneity_scores` | Prompt 2 | `listing` |
| `reset_filters` | Start again within a conversation | none |

- `field` and `op` are enums; schemas are strict (`additionalProperties: false`).
- `include_unknown` is required so every filter decides explicitly what happens to units with missing values
  (the within / beyond / unknown rule). Summaries always report how many were excluded as unknown.
- **Policy: exclude unknowns by default**, unless the user asks to include them or only asks to remove units known
  to fail the condition. The answer always states how many were left out and offers to include them. The system
  prompt repeats this (a production version would also check it, see section 7). For prompt
  3 the choice doesn't change the ranking: `find_nearest` can't rank units without a usable location anyway.
- EPC is allowed as a *filter* (the user's explicit choice), with the reliability caveat carried in the summary.
- Tool descriptions state conventions, e.g. "excluding properties more than 10 miles from a major city" means
  `miles_to_city <= 10`.

## 4. Orchestration and prompting (`graph.py`)

A hand-built LangGraph `StateGraph`, kept deliberately simple (the same pattern as the fpl chat app):

```
START -> agent <-> tools -> END
```

- **State**: the messages, and the candidate set from the last `filter_rows` (None for the whole portfolio).
- **Tools**: the functions in `tools.py` with the data filled in, plus `reset_filters`. `filter_rows` stores its
  result in the state (the ids aren't sent to the model); `find_similar` and `find_nearest` take a required
  `use_filter` argument, so a filter is only used when the model says so.
- **System prompt**: acquisitions-analyst role; use only tool results; cite Property Ids; lead with limitations of
  the listing ("Tech Hub has no coordinates, so similarity is physical only"); pass on caveats; never estimate
  missing values; ask when the listing is ambiguous rather than picking one; "similar" -> `find_similar`,
  "closest" -> `find_nearest`, "correlation score" -> how typical of the portfolio; exclude unknowns by default and
  state the count; mention the estate a listing sits on.
- **Model**: Claude (`claude-opus-5-5` by default, effort `medium`; `CHAT_MODEL` / `CHAT_EFFORT` env overrides),
  key from `.env`.
- **LangGraph Studio**: `langgraph.json` points `langgraph dev` at `studio.py`, which builds the graph on the real
  data without a checkpointer (Studio supplies its own memory).

## 5. Testing

| Layer | What | Cost |
|---|---|---|
| Tools | filter semantics (unknowns dropped/kept), optional estate cap, missing blocks ("not available" vs 0), known-issues flags, a listing with no coordinates | free, deterministic |
| Graph | scripted chat model replays chosen tool calls: the filter flows from `filter_rows` into the search tools only with `use_filter`, `refine` and `reset_filters` work, tool errors come back as messages | free, deterministic |
| App | Streamlit AppTest of the real page with the scripted model: prompt buttons, streaming, the tool-call panel, New chat, errors shown in the chat | free, deterministic |
| Evals | the real agent (Claude) on 12 cases, checked by code and by an LLM judge (`evals/`) | paid, run on request |

### Evals

`uv run python evals/run_evals.py` runs each case in a fresh conversation and applies three checks, cheapest first:

1. **Trajectory (code):** the tool calls each case expects, in order, with the arguments that matter (other calls
   in between are fine), and any tools it mustn't use.
2. **Numbers (code):** every number in the answer must appear in a tool result or the question. This catches
   invented scores, distances, counts and Property Ids, and is the check a production validation step would run.
3. **Judge (Claude, low effort, structured output):** reads the tool results and the answer, and lists claims the
   results don't support, caveats the answer leaves out, and whether it behaved as the case expects.

The 12 cases: the brief's three prompts word for word; variations (a named listing, a listing with size only,
nearest by estate name, location weighted double, a two-turn refine, including unknowns when asked, an EPC filter
that must carry the reliability warning); and two out-of-scope questions (a rent figure, a guess at a missing
build year), which it must decline. Each run's results and report are saved in a dated folder under
`data/evals/` (gitignored: the answers contain portfolio data). `--repeats 3` gives pass rates, since answers vary.

### What the first run found (2026-11-23)

6 of 12 cases passed every check; the run cost $1.01. The trajectory was right on every case: the right tools,
with the right arguments, including refine and include-unknown. The code checks found almost nothing. The judge
found the real problems.

**Claude filling gaps between tool facts.** It rarely invented anything outright. Instead it added one small,
plausible step of its own on top of what the tools said, and the step was often true, but nothing in the tool
results supported it:

| Case | The tools said | Claude said | Fix: make the tool say it |
|---|---|---|---|
| Guess M2's build year | M2 has no coordinates; the estate it sits on is blank | "M2 isn't on any estate Indurent owns" (unknown, not no) | "Unknown (no coordinates)" instead of blank |
| Include unknowns | 358 units have no city distance; 358 have no location | "Those are the same 358 units" (true, but unstated) | Say the distance is unknown *because* the location is |
| Prompt 2 | M4's size score is 0.17 (atypical) | "M4 is unusually small" (the score gives no direction) | Say whether the listing is smaller/larger, older/newer than typical |
| Nearest to M3 | The 10 nearest units, all 0.0 miles away | "There may be more units at 0.0 miles" (unstated) | Report ties, as `find_similar` already does for size |

The lesson: a rule in the prompt ("only say what the tools say") isn't enough on its own. The fix is to close
the gaps in the tools, so there's nothing left to infer. These gaps are also why the number check alone isn't
enough: every number in these answers was real.

**Problems with the eval itself**, fixed before the rerun:

- The EPC case expected ratings A, B and C exactly; Claude also included A+ (a real rating, absent from this
  data). The check now accepts it.
- The judge didn't see the system prompt, so it flagged the EPC warning the prompt tells Claude to give. It now
  gets the system prompt as a source.
- The judge listed claims it had checked and then accepted ("…so not flagged"). It's now told to list only the
  claims it flags.
- "51 miles or more" for a unit 51.5 miles away was flagged, because 51.5 rounds to 52. A number cut down to the
  precision written (rather than rounded) now counts too.

## 6. Streamlit app and deployment

Pages: **Ask** (chat, with the tool calls and arguments shown), **Example prompts** (stored outputs for the three
brief prompts, no API call), **Explore** (the same tools with sliders, no LLM), **Data quality** (key EDA
findings and the decisions table).

Deployment on beast, the same mechanism as the fpl API:

1. `deploy/indurent-app.service` runs Streamlit bound to `127.0.0.1:8503` (8501 and 8502 are taken by fpl).
2. A public hostname on the existing shed Cloudflare Tunnel points at `http://localhost:8503`.
3. `data/` is copied across by hand (gitignored).

Safeguards before sharing:

- **Cloudflare Access** (email one-time PIN, allowlist of the interviewers and Tony): the assessment data is kept
  private as a precaution (the brief doesn't say it's confidential), and access limits who can spend on the API key.
- A separate Anthropic workspace/key with its own spend limit, and a per-session message cap.
- Stored example outputs, so the app is useful even if the key is off or the limit is hit.

## 7. Production notes (for the write-up)

- Geocode from postcodes (ONS lookup) instead of trusting supplied coordinates; validate on ingest.
- Feed the known-issues register back to data owners; track fixes.
- Move vectors to pgvector/FAISS once the portfolio or listing volume justifies it; precompute portfolio blocks.
- Retrieval over the methodology itself (decisions, EDA findings) so the model can explain *why* a rule applies.
- Estate-level summaries as a second retrieval unit ("which estates does this listing resemble?").
- Tracing (LangSmith/OpenTelemetry), eval runs in CI on a fixed budget, prompt caching for the system prompt.
- **A validation step in the graph** (`agent <-> tools -> validate -> END`, with `validate` able to send an answer
  back once): plain code that checks every Property Id the answer cites was returned by a tool in this turn, and
  that a filter's unknown count is reported. A failing answer goes back to the model with the problem; a second
  failure goes out with the problem stated. Left out here to keep the demo graph simple: the system prompt carries
  the rules, and tests cover the tools and the state flow.
- Structured output for answers (matches, scores, caveats as fields), so the app renders tables rather than prose.

## Order of work

1. `data.py` + tests
2. `similarity.py` and `tools.py` + tests
3. `graph.py` + scripted-model tests
4. LLM evals and example outputs (needs an API key; spend agreed first)
5. Streamlit app, then deploy behind Access
