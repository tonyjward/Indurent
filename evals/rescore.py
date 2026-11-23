"""Apply the right-answer check to a saved eval run, without calling Claude (free).

    uv run python evals/rescore.py data/evals/2026-11-23-1633

Runs saved before the check existed have only the first three checks. This adds the fourth to each saved answer,
recomputes `passed`, and saves rescored.json and rescored.md in the same folder (the originals are kept).
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from cases import CASES

from indurent_matching.data import load_marketed, load_portfolio
from indurent_matching.evals import check_answer, cost_usd, report_markdown
from indurent_matching.tools import Data

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    run_dir = Path(sys.argv[1])
    saved = json.loads((run_dir / "results.json").read_text())
    data = Data(
        load_portfolio(ROOT / "data" / "Current Portfolio.xlsx"),
        load_marketed(ROOT / "data" / "Marketed Warehouses.xlsx"),
    )
    cases = {case.name: case for case in CASES}

    for result in saved["results"]:
        problems = check_answer(cases[result["case"]], result["answer"], data)
        result["correct"] = {"passed": not problems, "problems": problems}
        three_checks = result["trajectory"]["passed"] and result["numbers"]["passed"] and result["judge"]["passed"]
        result["passed"] = three_checks and not problems

    results = saved["results"]
    spent = sum(r.get("cost_usd", cost_usd(r["tokens"]["agent"]) + cost_usd(r["tokens"]["judge"])) for r in results)
    (run_dir / "rescored.json").write_text(json.dumps(saved, indent=2))
    (run_dir / "rescored.md").write_text(report_markdown(results, saved["model"], spent))

    # A summary: each check's pass count, then each case's runs (P pass, F fail) on the right-answer check.
    for check in ["trajectory", "numbers", "judge", "correct"]:
        print(f"{check:10s} {sum(r[check]['passed'] for r in results)} of {len(results)}")
    print(f"{'overall':10s} {sum(r['passed'] for r in results)} of {len(results)}")
    by_case = defaultdict(list)
    for r in results:
        by_case[r["case"]].append(r)
    print()
    for name, runs in by_case.items():
        marks = "".join("P" if r["correct"]["passed"] else "F" for r in runs)
        print(f"{name:30s} right answer {marks}")
        for r in runs:
            for problem in r["correct"]["problems"]:
                print(f"    {problem}")


if __name__ == "__main__":
    main()
