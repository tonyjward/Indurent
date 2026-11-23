"""Run the eval cases against Claude and write a report (real API calls: this costs money).

    uv run python evals/run_evals.py                       # every case once
    uv run python evals/run_evals.py --repeats 3           # every case three times, for pass rates
    uv run python evals/run_evals.py --cases brief_3_closest_near_city scope_rent

Each run is a fresh conversation; the agent and the judge are both Claude Opus 5.5 (the judge at low effort).
It stops before starting a run once the spend reaches --budget, and saves after every run, so a stopped or
failed run keeps what it has. Each run of this script saves results.json and report.md in its own dated folder,
e.g. data/evals/2026-11-23-1505/ (gitignored: answers contain portfolio data), so earlier runs are kept. Needs ANTHROPIC_API_KEY in .env.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from cases import CASES

from indurent_matching.data import load_marketed, load_portfolio
from indurent_matching.evals import claude_judge, report_markdown, run_case
from indurent_matching.graph import MODEL, default_model
from indurent_matching.tools import Data

ROOT = Path(__file__).resolve().parents[1]
EVALS_DIR = ROOT / "data" / "evals"
# Roughly the most one run costs (agent plus judge), so the budget isn't overshot by a whole run.
MAX_RUN_USD = 0.30


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=1, help="runs per case (default 1)")
    parser.add_argument("--budget", type=float, default=50.00, help="stop before a run that could pass this, in $")
    parser.add_argument("--cases", nargs="*", help="case names to run (default all)")
    args = parser.parse_args()

    chosen = [case for case in CASES if not args.cases or case.name in args.cases]
    unknown = set(args.cases or []) - {case.name for case in CASES}
    if unknown:
        parser.error(f"unknown cases: {sorted(unknown)}")

    data = Data(
        load_portfolio(ROOT / "data" / "Current Portfolio.xlsx"),
        load_marketed(ROOT / "data" / "Marketed Warehouses.xlsx"),
    )
    judge = claude_judge()
    out = EVALS_DIR / datetime.now().strftime("%Y-%m-%d-%H%M")
    out.mkdir(parents=True, exist_ok=True)
    results, spent = [], 0.0

    # Every case once, then every case again for each repeat.
    runs = []
    for repeat in range(1, args.repeats + 1):
        for case in chosen:
            runs.append((repeat, case))

    for repeat, case in runs:
        if spent + MAX_RUN_USD > args.budget:
            print(f"Stopping: ${spent:.2f} spent, and another run could pass the ${args.budget:.2f} budget")
            break
        try:
            result = run_case(case, data, default_model(), judge)
        except Exception as error:  # noqa: BLE001 -- an API error fails this run, not the whole eval
            print(f"{case.name} (run {repeat}): error: {error}")
            continue
        result["run"] = repeat
        results.append(result)
        spent += result["cost_usd"]
        checks = []
        for check in ["trajectory", "numbers", "judge", "correct"]:
            checks.append(f"{check} {'ok' if result[check]['passed'] else 'FAIL'}")
        print(f"{case.name} (run {repeat}): {', '.join(checks)}; ${result['cost_usd']:.3f} (total ${spent:.2f})")
        # Save after every run, so a stopped or failed eval keeps what it has.
        (out / "results.json").write_text(json.dumps({"model": MODEL, "results": results}, indent=2))
        (out / "report.md").write_text(report_markdown(results, MODEL, spent))

    print(f"{sum(r['passed'] for r in results)} of {len(results)} runs passed every check. ${spent:.2f} spent.")
    print(f"Report: {(out / 'report.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
