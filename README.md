# Indurent: matching marketed warehouses to the portfolio

An assistant for Indurent's Acquisitions team. Ask it about a newly marketed warehouse, and it compares the listing
with Indurent's portfolio of 1,250 industrial units: the most similar units, how typical the listing is of the
portfolio, and the nearest units on the map, with filters such as distance to a major city. Python does every
calculation; Claude turns the question into tool calls and explains the results, with their caveats.

```
two spreadsheets ──> clean + quality flags ──> similarity functions ──> tools ──> Claude (LangGraph) ──> answer
                       (data.py, geo.py)         (similarity.py)      (tools.py)     (graph.py)         (app/)
```

- [**The approach**](#the-approach): why not text embeddings, and what instead
- [**How it maps to the brief**](#how-it-maps-to-the-brief)
- [**Answers to the three prompts**](#answers-to-the-three-prompts), and
  [**example conversations**](outputs/README.md) with the agent
- [**What we found in the data**](#what-we-found-in-the-data)
- [**How we know it works**](#how-we-know-it-works): tests, and evals of the agent
- [**Extending to a production web app**](#extending-to-a-production-web-app): what's built, and what it would add
- [**Running it**](#running-it)

More detail: [`DESIGN.md`](DESIGN.md) (how it works and why), the [notebooks](#notebooks) (the evidence),
[`DEVELOPMENT.md`](DEVELOPMENT.md) (keys, LangSmith Studio, deploying).

## The approach

**What we considered first.** The usual RAG pattern over documents: turn each portfolio row into a text chunk
("Unit 5, Quartz Wharf 16, 17,340 sq ft, built 1976, ..."), embed the chunks, embed the listing the same way, and
retrieve the nearest by cosine similarity.

**Why we didn't.** The data is a table of numbers, and text embeddings don't treat numerical distance well:
"12,000 sq ft" and "120,000 sq ft" look almost the same as text, and an embedding has no sense that 1976 is closer
to 1977 than to 1997, or that two points 4 miles apart are neighbours. Nor can it apply rules: "exclude units more
than 10 miles from a major city" is a filter, not a similarity, and units with unknown or doubtful values need to be
left out and counted, not quietly matched.

**What we did instead.** Explicit, tested functions for each part of the question, given to the model as tools:

1. **Similarity functions** (`similarity.py`), one for each of the three dimensions the brief names, each scored
   from 0 to 1:
   - **Age:** compares build years.
   - **Location:** compares straight-line miles.
   - **Physical:** compares size, as a ratio (twice the size counts the same at 2,000 sq ft as at 200,000).

   Each dimension is scored in two ways:
   - **Homogeneity:** how typical the listing is of the whole portfolio (prompt 2).
   - **Pairwise similarity:** how similar the listing is to one portfolio unit (prompt 1).

   This is the **vectorisation of the structured data**: each unit becomes a vector of numbers (build year, the
   log of its size, latitude and longitude), and each similarity metric is computed with numpy for the whole
   portfolio at once, one array operation rather than a loop over units. Every choice is backed by evidence in the
   notebooks.
2. **Tools** (`tools.py`): plain Python that the model calls, such as `filter_rows`, `find_similar`, `find_nearest`
   and `get_homogeneity_scores`. Each returns a short summary, the rows that matter and the caveats to pass on;
   never the whole table. This is the *retrieval* in our RAG: exact retrieval over the portfolio, by filter and
   score, rather than by embedding.
3. **An agent** (`graph.py`): a LangGraph graph that gives the tools to Claude. A filtered set is held in the graph
   state, so a follow-up ("now only those over 10,000 sq ft") narrows it further. The **prompt engineering** is in
   the system prompt's rules (every number from a tool; ask which listing if it's unclear; lead with what limits
   the answer; exclude unknowns by default and say how many) and in the tools' descriptions, which tell the model
   when to use each one.
4. **Tests and evals** to give guarantees that the system behaves as intended: unit tests for every function and
   tool, and evals of the agent itself for **trajectory** (the right tools, with the right arguments),
   **faithfulness** (nothing the tools didn't say), **scope** (declining what the data can't answer) and the
   **right answer**.

### How it maps to the brief

| The brief's task | Where it's done |
|---|---|
| 1. Load and preprocess the datasets | `data.py` and `geo.py`: cleaning, quality flags and location checks, each rule backed by the EDA |
| 2. Vectorise the relevant features | `similarity.py`: numeric feature vectors per unit, and similarity metrics computed across the portfolio with numpy |
| 3. Apply RAG to enrich or contextualise the data | `tools.py`: retrieval over the portfolio (filter, rank, score), whose results ground every answer |
| 4. Use prompt engineering to query an AI model | `graph.py`: Claude through the Anthropic API, in a LangGraph agent, with the system prompt's rules and the tools' descriptions |
| 5. Return results based on user-defined prompts | `app/app.py` (a chat app) and [`outputs/`](outputs/README.md) (example conversations) |

And the deliverables: the code is the package and [notebooks](#notebooks), with docstrings throughout; the output
from testing the example prompts is in [`outputs/`](outputs/README.md); and for a production system behind a web
app, there's a working web app, with what production would add beyond it, in
[Extending to a production web app](#extending-to-a-production-web-app).

## Answers to the three prompts

There are five marketed listings (M1-M5). Two can't be compared on everything: **M2** has no build year or
coordinates (size only), and **M5**'s coordinates are 27 km offshore (no location). Full results for every listing
are in [`notebooks/03_answers`](notebooks/03_answers.ipynb), straight from the code. The agent's own answers, asked
each prompt word for word, are in [`outputs/`](outputs/README.md): [prompt 1](outputs/prompt_1_most_similar.md),
[prompt 2](outputs/prompt_2_homogeneity.md) and [prompt 3](outputs/prompt_3_closest_near_city.md), with the tool
calls behind each answer. The evals check that its answers match the code's.

### Prompt 1: "Find the 10 most similar properties in the estate to the newly marketed property."

**How we read it:** "the estate" is Indurent's portfolio. The prompt doesn't say which listing, so the agent asks,
or answers for all five. Each unit is scored on age, location and size, each 0-1 and halving every 10 years, every
10 miles, and each doubling or halving of size; the three are averaged with equal weights.

| Listing | Compared on | Top match (overall score) | Where the top 10 are |
|---|---|---|---|
| M1 Cherry Lane, 16,860 sq ft, 1977 | age, location, size | Property Id 586, Quartz Wharf 16 (0.88) | 5 on Quartz Wharf 16, 4.4 miles away; 2 on Scarlet Pavilion 78, the estate it sits on |
| M2 Tech Hub, 906 sq ft | size only | Property Id 602, Quartz Wharf 16 (1.00) | across estates: 11 units score 0.98 or more on size alone |
| M3 Spitfire Park, 2,294 sq ft, 1976 | age, location, size | Property Id 869, Velvet Park 60 (0.84) | all 10 on Velvet Park 60, the estate it sits on |
| M4 Stable Lane, 617 sq ft, 2000 | age, location, size | Property Id 26, Sable Mill 91 (0.70) | 7 on Sable Mill 91, the estate it sits on |
| M5 Chancery Depot, 16,614 sq ft, 1966 | age, size | Property Id 482, Sapphire Grove 50 (0.93) | across estates (no location to cluster on) |

**Worth knowing:** M1, M3 and M4 each sit on an estate Indurent already owns, so their best matches are often next
door. There's no cap on units per estate by default; the agent applies one if the user asks for a spread.

### Prompt 2: "Provide a correlation score for the homogeneity of the marketed property(ies) with the rest of the estate, scoring each by physical characteristics, location, and age separately."

**How we read it:** "correlation score for homogeneity" as **how typical the listing is of the portfolio**, scored
0-1 on each dimension, where 1 is the most typical value in the portfolio. The answer states this reading.

| Listing | Age | Location | Physical (size) | Which way it's atypical |
|---|---|---|---|---|
| M1 | 0.86 | 0.23 | 0.41 | an isolated location (it falls in Wales, far from most estates); larger than 78% of units |
| M2 | n/a | n/a | 0.38 | smaller than 95% of units |
| M3 | 0.86 | 0.62 | 0.87 | fairly typical on all three |
| M4 | 0.86 | 0.73 | 0.17 | smaller than 99% of units |
| M5 | 0.20 | n/a | 0.41 | older than 97% of units (built 1966); larger than 78% |

- **Age:** the share of portfolio units in the listing's 5-year build band, against the most frequent band.
- **Location:** how close the listing's 5 nearest estates are, against the best-placed estate in the portfolio.
- **Physical:** how many units are of a similar size (within a factor of 1.41 either way), against the most
  crowded size in the portfolio. Eaves height is shown as context; doors were too unreliable to score.

### Prompt 3: "Find the closest properties to the marketed property, after excluding any property in the portfolio that is more than 10 miles from a major city."

**How we read it:** "closest" means **nearest on the map**, not most similar. Keep units with `miles_to_city <= 10`
(16 major cities, listed in `config.py`), then rank by straight-line miles. **50 units** are within 10 miles of a
city; **358 have no usable location** and are left out, and the answer says so and offers to include them. M2 and
M5 can't be placed on the map, so they have no nearest units.

| Listing | Nearest units within 10 miles of a major city |
|---|---|
| M1 | Property Id 348, Marble Works 29 (74.6 miles away; 7.9 miles from Birmingham), then Sapphire Junction 78 (88.1) |
| M3 | Property Ids 170, 169, 167 and 168, Jade Bridge 6 (9.1-9.2 miles away; 7.1 miles from Manchester) |
| M4 | Property Id 350, Onyx Pointe 12 (33.2 miles away; 8.6 miles from London), then Sable Crescent 13 (39.9) |

**Worth knowing:** none of the estates M1, M3 and M4 sit on survives the filter (M3's Velvet Park 60 is 14 miles
from Manchester), which is why the nearest units are some way off.

## What we found in the data

The EDA ([`notebooks/01_eda`](notebooks/01_eda.ipynb)) shaped every cleaning rule. The main findings:

- **"Size sqm" holds square feet.** The values only make sense as sq ft, and their total matches Indurent's
  published floor area, so the column is renamed rather than converted.
- **358 of 1,250 units have no usable location:** no coordinates, or coordinates 1 km or more offshore. They're
  left out of location scores and distance filters, and always counted in answers. Another 652 fall outside their
  labelled region; they're kept, at medium confidence.
- **A block of records looks shuffled.** In Property Ids 1173-1187, small units carry a big shed's eaves, doors and
  yard depth, and big units a small one's. Those fields are flagged as doubtful for the block.
- **Door counts are often an estate total** copied onto every unit, so doors aren't scored.
- **EPC ratings look unreliable:** newer buildings are mostly rated E. The agent warns whenever a filter uses them.
- Smaller items: Granite Mews 59's eaves are house height (2.1-2.4 m); seven EPC cells hold Excel's FALSE.

Each value is kept, with a `_doubtful` flag and a plain-English note saying why it isn't used, so nothing is
silently dropped. Rules tied to particular Property Ids are in a register in `config.py`, to review whenever the
data is refreshed.

## How we know it works

**Unit tests** (132, no API calls): the cleaning rules, every similarity function and tool, the graph's state flow
with a scripted model (a filter only reaches a search when asked; refine and reset work), the web app, and the
eval checks themselves.

**Evals of the agent** with real Claude: 12 cases (the three prompts word for word, variations such as a two-turn
refine or an EPC filter, and two questions it should decline: a rent figure, and a guess at a missing build year).
Each answer gets four checks:

| Check | How | Final result (36 runs) |
|---|---|---|
| **Trajectory**: the right tools, with the right arguments | code | 36 of 36 |
| **Numbers**: every number in the answer comes from a tool | code | 34 of 36 |
| **Faithfulness and scope**: no unsupported claims; declines what it should | Claude as judge | 23 of 36 |
| **Right answer**: the right units, in order, with the right values | code, against a reference from the tools | **36 of 36** |

Every one of the 36 answers reported the right result (and all 36 in the measurement before). The judge's failures
are about wording: a sentence that goes a little beyond the tools, a different one each run.

**The evals changed the system.** The first run passed 6 of 12. The judge kept finding the same pattern: Claude
rarely invented anything outright, but it filled small gaps between tool facts with plausible claims (reading a
blank "estate" field as "none", or calling a listing "small" when the score said only "atypical"). Rules in the
prompt weren't enough; the fix was to make the tools state those facts, so there was nothing left to infer. Four
rounds of that took it to 23-24 of 36 runs passing all four checks (it varies from run to run). The details, run by
run, are in DESIGN.md section 5.

## Extending to a production web app

**What's built.** `app/app.py` is a Streamlit chat over the agent, deployed on a server behind a Cloudflare Tunnel
(DESIGN.md section 6):

- The brief's prompts as one-click buttons, and answers that stream in as they're written.
- **How this was worked out** under each answer: the tool calls behind it, so every number can be traced.
- Conversation memory, so follow-ups refine a filter; tracing of every run in LangSmith.
- The data loaded once and shared by every user; deploying a change is `git push` and a restart.

**What a production version would add:**

- **Access and spend:** the company's single sign-on (or Cloudflare Access) in front of the app, a spend-limited
  API key per environment, per-user rate limits, and prompt caching for the system prompt.
- **A check before answers are sent:** a validation step in the graph that checks every Property Id and number in
  an answer against the tool results, and sends a failing answer back once. The evals' code checks are that step.
- **Data:** a refresh pipeline that runs the cleaning rules on each new portfolio extract, geocodes from postcodes
  rather than trusting supplied coordinates, and sends the known-issues register back to the data owners.
- **Operations:** the evals in CI on a fixed budget, tracking pass rates over time; conversation history in a
  database rather than memory; and an API (e.g. FastAPI) between the agent and any front end, so other tools can
  use it.
- **Answers as data:** structured output (matches, scores and caveats as fields), so the app renders tables and
  charts rather than prose.
- **Scale:** for a much larger portfolio, precompute the portfolio's scores and move the numeric features into a
  vector index (pgvector, FAISS).

More in DESIGN.md section 7.

## Running it

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 if needed).

```bash
uv sync                                   # install into .venv
# copy the two spreadsheets into data/: "Current Portfolio.xlsx" and "Marketed Warehouses.xlsx"
uv run pytest                             # tests (those needing the spreadsheets skip without them)
uv run jupyter lab                        # the notebooks
```

The agent needs `ANTHROPIC_API_KEY` in `.env` (DEVELOPMENT.md lists the other keys). Each question costs about
$0.10-0.15.

```bash
uv run streamlit run app/app.py                      # the chat app, at http://localhost:8501
uv run python evals/run_evals.py                     # the evals once (about $1.20); --repeats 3 for pass rates
uv run python evals/rescore.py data/evals/<run>      # re-check a saved run's answers (free)
uv run python scripts/run_examples.py                # remake the example conversations in outputs/ (about $1.50)
```

The app is also deployed, on a home server behind a Cloudflare Tunnel (the address isn't in the repo while it's
open to anyone).

### Notebooks

| Notebook | What it shows |
|---|---|
| [`01_eda`](notebooks/01_eda.ipynb) | Every column examined; the evidence and decision behind each cleaning rule |
| [`02_similarity`](notebooks/02_similarity.ipynb) | How each dimension (age, location, physical) is scored, and why |
| [`03_answers`](notebooks/03_answers.ipynb) | The three prompts answered straight from the code, for every listing |

### Project layout

```
src/indurent_matching/   the package
  config.py              thresholds, city list, known-issues register and similarity settings, each citing its evidence
  geo.py                 coastline and region checks, distances, nearest major city
  data.py                load and clean both spreadsheets
  similarity.py          the three similarity dimensions (age, location, size) and how they combine
  tools.py               the functions the AI calls: plain Python, no LangGraph or model
  graph.py               the LangGraph agent: Claude with the tools
  chat.py                streams an answer and reads back its tool calls (for the app)
  evals.py               the eval checks (trajectory, numbers, judge, right answer)
  studio.py              builds the graph on the real data for LangGraph Studio
app/                     the Streamlit web app (app.py) and its logo
evals/                   eval cases, the runner and the rescorer
notebooks/               01 EDA, 02 similarity, 03 answers
outputs/                 example conversations with the agent, one per file
tests/                   pytest suite, no API calls (scripted.py is a fake chat model)
scripts/                 the example conversations run; the UK boundaries build
deploy/                  the systemd service that runs the app
data/geo/                UK boundaries (committed; the rest of data/ is gitignored)
```

Everything below `graph.py` is plain Python: `tools.py` and `similarity.py` work without LangGraph or a model, from
a notebook, the tests or the app.
