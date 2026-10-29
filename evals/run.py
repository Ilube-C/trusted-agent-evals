"""Run the agent over the eval dataset and save raw outputs for grading. One file per repeat.

    python -m evals.run --split dev
    python -m evals.run --split heldout --version v2 --repeats 3
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from agent.graph import MODEL, PROMPTS, VERSION, check_claim

ROOT = Path(__file__).resolve().parent
PRICES = {"claude-sonnet-5-5": (2.00, 10.00)}  # $ per million input, output tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "heldout", "all"], default="dev")
    ap.add_argument("--version", choices=list(PROMPTS), default=VERSION)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    cases = [json.loads(line) for line in (ROOT / "dataset.jsonl").open()]
    cases = [c for c in cases if args.split == "all" or c["split"] == args.split]

    for rep in range(args.repeats):
        tags = [f"eval-{args.split}", f"repeat-{rep}"]

        def run(case):
            try:
                return check_claim(case["claim_id"], tags=tags, version=args.version)
            except Exception as e:
                return {"claim_id": case["claim_id"], "version": args.version, "error": repr(e)}

        with ThreadPoolExecutor(args.workers) as pool:
            results = list(pool.map(run, cases))

        tin = sum(r.get("usage", {}).get("input_tokens", 0) for r in results)
        tout = sum(r.get("usage", {}).get("output_tokens", 0) for r in results)
        pin, pout = PRICES[MODEL]
        cost = tin / 1e6 * pin + tout / 1e6 * pout

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = ROOT / "results" / f"{args.version}-{args.split}-{stamp}.json"
        out.write_text(json.dumps({"version": args.version, "model": MODEL, "split": args.split,
                                   "cost_usd": round(cost, 4), "results": results}, indent=2, ensure_ascii=False))
        errors = sum("error" in r for r in results)
        print(f"repeat {rep + 1}/{args.repeats}: {len(results)} claims, {errors} errors, ~${cost:.2f}. "
              f"Saved {out.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
