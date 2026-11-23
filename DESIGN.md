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
  similarity.py  homogeneity and pairwise similarity per dimension      (step 2: vectorise)
  tools.py       filter_rows, find_similar, homogeneity_scores, ...     (step 3: retrieval)
  graph.py       LangGraph StateGraph, system prompt                    (steps 4-5: prompt + answer)
  chat.py        stream an answer and read back its tool calls (for the app)
  evals.py       the eval checks: trajectory, numbers, LLM judge, right answer
  studio.py      the graph on the real data, for LangGraph Studio
app/app.py       Streamlit chat page over the same modules
evals/           eval cases, the runner (real Claude calls) and a free rescorer
tests/           data, similarity, tools, graph and app (scripted model), eval checks
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
[`notebooks/02_similarity`](notebooks/02_similarity.ipynb), one section per dimension.

Three dimensions, matching the brief's prompt 2, each scored on its own feature:

| Dimension | Feature | Usable values |
|---|---|---|
| Age | build year | recorded and not doubtful |
| Location | latitude/longitude | high and medium confidence only |
| Physical | size (log scale) | recorded and not doubtful |

Every score is 0-1. A dimension a listing can't be scored on is **"not available"**, never 0 (M2 has no coordinates
or build year; M5 is offshore).

**Not scored, shown as context:** eaves height (only M4 and M5 have it; scoring it would put them on a different
basis from the other listings), doors (39 usable portfolio counts), volume (99% correlated with size), EPC
(unreliable, EDA 2.9), parking (estate-level), yard depth (34 units, no listings), region labels (only 26% agree
with coordinates). All can be filtered on.

### Homogeneity scores (prompt 2): how typical is the listing of the portfolio?

"Correlation score" is ambiguous; the interpretation used, and stated in every answer, is *how typical the
listing is of the portfolio on each dimension*. Each score is scaled so the most typical value in the portfolio
scores 1:

| Dimension | Score | Context reported alongside |
|---|---|---|
| Age | Units in the listing's 5-year build-year band (1955-59, ..., 2020-22) / units in the most frequent band | |
| Location | Best portfolio estate's mean miles to its 5 nearest other estates / the listing's mean miles to its 5 nearest estates, leaving out the estate it sits on | The estate it sits on ("sits on Velvet Park 60, which you own"); the nearest other estate; the percentile among portfolio estates |
| Physical | Portfolio units of a similar size to the listing (size / 1.41 to size x 1.41: the largest twice the smallest) / the most any portfolio unit has of a similar size | Eaves height against similar-sized units ("tall for its size") |

Why these forms: build years are flat from 1970 to 2021, so fixed bands work (the jump at 1970 is in the data);
sizes slope steadily, so a moving window avoids jumps at band edges. Every listing with coordinates sits on an
existing Indurent estate, so location uses estates, not units, as the neighbours: nearest units would all be
from the listing's own estate at about 0 miles.

### Pairwise similarity (prompt 1): how similar is the listing to one portfolio unit?

Each dimension uses the same form, similarity = 0.5^(gap / h), so it is 1 for an identical value and halves with
every step of h:

| Dimension | Gap | Halves every |
|---|---|---|
| Age | years between build years | 10 years |
| Location | miles between the two (great-circle) | 10 miles |
| Physical | the ratio between the sizes (on a log scale, so half and double count the same) | time one size is twice the other |

Size uses a ratio where age and location use a fixed gap because what matters for size is the *relative*
difference: 1,000 vs 2,000 sq ft is a different class of unit, while 100,000 vs 101,000 sq ft is effectively the
same. Ten miles, or ten years, mean much the same wherever they fall.

- **Combined score**: the mean of the dimensions, equally weighted by default; weights are parameters.
- **Missing dimensions**: a dimension the *listing* lacks is dropped for that listing (M2 is matched on size only, and
  the answer says so). A dimension a *portfolio unit* lacks scores 0, so a unit with less known about it can't
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
| Tools | filter semantics (unknowns dropped/kept), optional estate cap, missing dimensions ("not available" vs 0), known-issues flags, a listing with no coordinates | free, deterministic |
| Graph | scripted chat model replays chosen tool calls: the filter flows from `filter_rows` into the search tools only with `use_filter`, `refine` and `reset_filters` work, tool errors come back as messages | free, deterministic |
| App | Streamlit AppTest of the real page with the scripted model: prompt buttons, streaming, the tool-call panel, New chat, errors shown in the chat | free, deterministic |
| Eval checks | each check against good and deliberately wrong answers (a dropped unit, a swapped value, an invented figure) | free, deterministic |
| Evals | the real agent (Claude) on 12 cases, checked by code and by an LLM judge (`evals/`) | paid, run on request |

### Evals

`uv run python evals/run_evals.py` runs each case in a fresh conversation and applies four checks, cheapest first:

1. **Trajectory (code):** the tool calls each case expects, in order, with the arguments that matter (other calls
   in between are fine), and any tools it mustn't use.
2. **Numbers (code):** every number in the answer must appear in a tool result or the question. This catches
   invented scores, distances, counts and Property Ids, and is the check a production validation step would run.
3. **Judge (Claude, low effort, structured output):** reads the system prompt, the tool results and the answer,
   and lists claims the results don't support, caveats the answer leaves out, and whether it behaved as the case
   expects.
4. **Right answer (code):** each case has a reference answer worked out by calling the tools directly with the
   right arguments (the tools' own tests check those results). The units the answer lists must be the reference
   ranking, in order, each with its own value on its row; required counts must appear; an out-of-scope answer
   must contain no invented figure. The first three checks can't see a dropped unit, a wrong order, or one unit's
   distance shown against another. Added after the runs below, and applied to the saved answers with
   `evals/rescore.py` (free).

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

### The second run (2026-11-23)

5 of 12 passed ($1.05). The four gaps above were gone: Claude now said "unknown" for M2's estate, explained the
358, and took direction and ties from the tools. The failures were new gaps of the same kind:

| Cases | The tools said | Claude said | Fix |
|---|---|---|---|
| 4 | M1 is labelled North West, nearest city Cardiff (its quality note was only in `get_listing`) | "M1's location doesn't look right"; the same for M3 (labelled Midlands, 14 miles from Manchester), which is wrong: M3's location is high confidence | Quality notes go with every listing's facts; system prompt: raise only the data-quality issues the tools report |
| 3 | After a filter, 0 of the 10 nearest are on the owned estate | "because the 10-mile filter removed them" (probably true, unstated) | The caveat says whether the filter removed the estate's units or they ranked lower |
| 1 | 358 units have no location, so couldn't be ranked | "Keeping the unknowns didn't change the result" (true, unstated) | The caveat says keeping or dropping them can't change the list |

Also one misquote ("15 km inside the East of England" for a note saying 15 km *outside* the South East, in the
East of England), and two judge calls that are arguably harmless ("the 10th place is shared"; "the main size band").

### The third run (2026-11-23)

7 of 12 passed ($1.16). Claude no longer raised data-quality issues of its own or guessed why the owned estate
was missing. Trajectory and numbers passed on every case, as in the second run. What was left:

- **A wrong caveat of ours.** `find_nearest` said tied units "are listed in Property Id order", but units that
  show the same rounded distance (0.0 miles) are in order of their exact distance. Claude repeated it, and the
  judge saw the table wasn't in Id order. The caveat now just says the distance is to 1 decimal place.
- **A definition the judge didn't have.** Claude said an owned estate's units are "probably M3's closest
  neighbours", which follows from what "sits on" means, but no result said so. The caveat now says it: at least
  one of the estate's units is within 0.25 miles of the listing.
- **Misreading its own numbers (2 cases).** "M3 is the most typical on all three dimensions" (M4 scores higher
  on location); "1157 ranks lower because it's much smaller" (1157 ranks 2nd). These aren't gaps in the tools,
  so a system prompt rule now says to compare only by the scores, and not to call one best overall or explain a
  rank by one factor unless the scores show it. A production validation step is the stronger guard here.
- One judge call that's arguably harmless (an opening sentence qualified two lines later).

| Run | Passed | Trajectory | Numbers | Judge |
|---|---|---|---|---|
| 1 (no fixes) | 6 of 12 | 11 of 12 | 11 of 12 | 6 of 12 |
| 2 (first fixes) | 5 of 12 | 12 of 12 | 12 of 12 | 5 of 12 |
| 3 (second fixes) | 7 of 12 | 12 of 12 | 12 of 12 | 7 of 12 |

With one run per case, totals move by a case or two from run to run (the agent and the judge both vary), so the
pass rate is measured with `--repeats 3`.

### Measurement after the third round (2026-11-23, 16:33)

All 12 cases, 3 runs each (36 runs, $3.52): trajectory 36 of 36, numbers 35, judge 25, right answer **36 of 36**
(scored afterwards on the saved answers), all four 24 of 36. Two cases failed every run:

- **Prompt 3:** "the owned estate's units were removed because they're over 10 miles from a city". The tool only
  said the filter removed them; they could also have had no location. One more gap of the tool kind.
- **Prompt 1:** "weighted equally" said of all five listings, when M2 is scored on size alone and M5 without
  location. Each tool result said this correctly: Claude over-summarised.

The fourth round closed the prompt 3 gap (the caveat now gives the estate's distance from a city and how many of
its units have no location) and said that tied units are in order of exact distance.

### The final measurement (2026-11-23, 18:56)

The same 36 runs ($3.31), after the fourth round and the rename of "block" to "dimension" throughout:

| Check | Passed |
|---|---|
| Trajectory (right tools and arguments) | **36 of 36** |
| Numbers (none invented) | 34 of 36 |
| Judge (no unsupported claims, right behaviour) | 23 of 36 |
| Right answer | **36 of 36** |
| All four | 23 of 36 (64%), against 24 of 36 in the previous measurement and 6 of 12 (50%) on the first run |

**Every one of the 36 answers reported the right result**, for the second measurement running: the right units, in
the right order, with the right values, and the counts the user needs. Tool use was right every time.

| Case | Runs | What went wrong |
|---|---|---|
| similar to M3, size only (M2), weighted location, guess M2's build year | PPP | |
| prompt 1, prompt 2, EPC filter | PFP / PPF / PFP | one slip each: "weighted equally" for all five again; "no other listing comes close to M3", which M4 beats on location; an invented "0.983" (caught by the number check) |
| prompt 3 | FFP | the estate-removal guess is gone. One failure was the judge listing a claim it then called "not flagged"; in the other, Claude said units with no location can't be ranked by distance instead of offering to include them, which is right (the case now accepts it) |
| nearest to Spitfire Park | PFF | once called the tie order "close to arbitrary", against the tool's caveat; once flagged for "up to 50 in one search", which is in the tool's own description (the checks now see the descriptions) |
| refine, include unknowns, rent for M3 | FFP / FPF / FFP | small overreaches: what a rerun "would bring back", "within about 9 miles" for 9.1-9.2, assuming Indurent has rent data for an estate |

Two things came out of this run's failures for the evals themselves: the right-answer check first marked two
correct answers wrong (Claude wrote tied units together, "| 170, 169, 167 |" and "689 and 221 (...)"; it now reads
both), and the two eval fixes noted in the table.

**Where this leaves it.** Across the last two measurements, 23-24 of 36 runs pass all four checks, and all 72 report
the right result. The gaps that were in the tools were closed round by round, and what's left is the model
overreaching in its own words, a different sentence each run. Tuning further against these 12 cases would make the
score say less about new questions. The production validation step (section 7) is the guard for the rest: its
number and right-answer checks would have caught the invented "0.983" before it reached the user.

## 6. Streamlit app and deployment

`app/app.py` is one chat page, like Claude or ChatGPT: the sidebar has New chat and the marketed listings (their
facts and data-quality notes on hover); a new conversation offers the brief's three prompts as buttons; answers
stream in as they're written; and **How this was worked out** under each answer lists the tool calls, with their
arguments and a one-line outcome, so every number can be traced to a tool. Conversation memory is per browser
session (`InMemorySaver`). The data is loaded once for all visitors.

It's live, served from beast the same way as the fpl app. The address isn't written in the repo, because the app
is open (below) and every question costs money:

1. `~/repos/Indurent` on beast is a git remote (`beast`) of the laptop's repo. `data/*.xlsx` and `.env` are
   copied across by hand (both gitignored).
2. `deploy/indurent-app.service` (systemd) runs Streamlit bound to `127.0.0.1:8503` (fpl has 8501 and 8502), and
   restarts it if it stops.
3. A published application route on the existing Cloudflare Tunnel maps the app's hostname to
   `http://localhost:8503`. Nothing on beast is open to the internet; Cloudflare provides HTTPS.

To deploy a change: `git push beast main` from the laptop, then `sudo systemctl restart indurent-app` on beast.

**Open for now.** Anyone with the link can use it, and each question costs about $0.10-0.15 on the Anthropic key.
To restrict it, add a Cloudflare Access application for the hostname (email one-time PIN, an allowlist of
addresses); a spend limit on the Anthropic workspace is a backstop either way.

## 7. Production notes: extending this behind a web app

- Geocode from postcodes (ONS lookup) instead of trusting supplied coordinates; validate on ingest.
- Feed the known-issues register back to data owners; track fixes.
- Move vectors to pgvector/FAISS once the portfolio or listing volume justifies it; precompute the portfolio's scores.
- Retrieval over the methodology itself (decisions, EDA findings) so the model can explain *why* a rule applies.
- Estate-level summaries as a second retrieval unit ("which estates does this listing resemble?").
- Tracing is on (LangSmith, with evals in their own project); next would be eval runs in CI on a fixed budget, and
  prompt caching for the system prompt.
- **A validation step in the graph** (`agent <-> tools -> validate -> END`, with `validate` able to send an answer
  back once): plain code that checks every Property Id the answer cites was returned by a tool in this turn, and
  that a filter's unknown count is reported. A failing answer goes back to the model with the problem; a second
  failure goes out with the problem stated. Left out here to keep the demo graph simple: the system prompt carries
  the rules, and tests cover the tools and the state flow. The evals' number and right-answer checks are this
  step's code, and the judge's findings (section 5) are what it would catch.
- Structured output for answers (matches, scores, caveats as fields), so the app renders tables rather than prose.
- Spend and access controls for a shared app: a spend-limited Anthropic key per environment, Cloudflare Access (or
  the company's single sign-on) in front of it, and per-user rate limits.
- A data refresh pipeline: run the cleaning rules on each new portfolio extract, and alert when a known-issues
  register entry no longer matches (its Property Ids may have changed).
- Conversation history in a database rather than memory (`InMemorySaver` loses it on a restart), and an API (e.g.
  FastAPI) between the agent and any front end, so other tools can use it.

## Order of work

1. `data.py` + tests
2. `similarity.py` and `tools.py` + tests
3. `graph.py` + scripted-model tests
4. LLM evals and example outputs (needs an API key; spend agreed first)
5. Streamlit app, then deploy (open for now; Access when needed)
