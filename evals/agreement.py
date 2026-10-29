"""Check the judge against hand labels with Cohen's kappa.

    python -m evals.agreement --sheet evals/results/graded/v1-dev.json   # write a labelling sheet
    python -m evals.agreement evals/results/graded/v1-dev.json           # score the judge against it

Fill in `supported` and `explained` (true/false) in evals/hand_labels.jsonl without looking at the
judge's answer; the sheet leaves it out on purpose.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LABELS = ROOT / "hand_labels.jsonl"
CRITERIA = ["supported", "explained"]


def kappa(a, b):
    """Agreement between two raters on yes/no labels, corrected for the agreement chance alone would give."""
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    chance = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if chance == 1 else (observed - chance) / (1 - chance)


def judged(graded):
    for cid, c in graded["claims"].items():
        for i, v in enumerate(c["verdicts"]):
            if "judge" in v:
                yield cid, i, v


def write_sheet(graded):
    runs = {}
    for src in graded["sources"]:
        for r in json.loads((ROOT / "results" / src).read_text())["results"]:
            runs.setdefault(r["claim_id"], []).append(r)
    existing = {(x["claim_id"], x["version"], x["repeat"]) for x in map(json.loads, LABELS.open())} \
        if LABELS.exists() else set()
    with LABELS.open("a") as f:
        for cid, i, v in judged(graded):
            if (cid, graded["version"], i) in existing:
                continue
            d = runs[cid][i]["decision"]
            f.write(json.dumps({"claim_id": cid, "version": graded["version"], "repeat": i,
                                "decision": d["decision"], "approved_amount": d["approved_amount"],
                                "rationale": d["rationale"], "supported": None, "explained": None},
                               ensure_ascii=False) + "\n")
    print(f"Added rows to {LABELS.relative_to(ROOT.parent)}. Fill in supported and explained.")


def score(graded):
    human = {(x["claim_id"], x["repeat"]): x for x in map(json.loads, LABELS.open())
             if x["version"] == graded["version"]}
    pairs = [(human[(cid, i)], v["judge"]) for cid, i, v in judged(graded)
             if (cid, i) in human and None not in (human[(cid, i)]["supported"], human[(cid, i)]["explained"])]
    if not pairs:
        raise SystemExit("No labelled rows yet.")
    print(f"{len(pairs)} labelled rationales\n")
    for c in CRITERIA:
        h, j = [p[0][c] for p in pairs], [p[1][c] for p in pairs]
        agree = sum(x == y for x, y in zip(h, j)) / len(h)
        print(f"  {c:10} agreement {agree:.2f}  kappa {kappa(h, j):+.2f}  "
              f"(you said yes {sum(h)}, judge said yes {sum(j)})")
        for (hp, jp) in pairs:
            if hp[c] != jp[c]:
                print(f"      {hp['claim_id']}: you {hp[c]}, judge {jp[c]}. {jp['reason'][:100]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graded")
    ap.add_argument("--sheet", action="store_true")
    args = ap.parse_args()
    graded = json.loads(Path(args.graded).read_text())
    write_sheet(graded) if args.sheet else score(graded)


if __name__ == "__main__":
    main()
