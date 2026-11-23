"""The graph for LangGraph Studio: `langgraph dev` (see langgraph.json at the repo root) loads `graph` from here.

It loads the two spreadsheets from data/ and builds the graph with Claude and no checkpointer: Studio supplies its own
conversation memory. Running the graph in Studio needs ANTHROPIC_API_KEY (in .env); viewing it doesn't.
"""

from pathlib import Path

from indurent_matching.data import load_marketed, load_portfolio
from indurent_matching.graph import build_graph
from indurent_matching.tools import Data

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

graph = build_graph(
    Data(
        load_portfolio(DATA_DIR / "Current Portfolio.xlsx"),
        load_marketed(DATA_DIR / "Marketed Warehouses.xlsx"),
    )
)
