"""Paired before/after comparison of two graded versions on the same claims.

    python -m evals.compare evals/results/graded/v1-heldout.json evals/results/graded/v2-heldout.json

Each claim's score is the share of its repeats that reached the top rung. The difference is paired
by claim and the 95% interval comes from resampling claims, so it reflects which claims we happened
to write as well as run-to-run noise.
"""

import argparse
import json
import random
from pathlib import Path


def claim_scores(g, top):
    """Share of a claim's certain repeats at the top rung. Claims with no certain repeat drop out."""
    out = {}
    for cid, c in g["claims"].items():
        certain = [v for v in c["verdicts"] if not v.get("uncertain")]
        if certain:
            out[cid] = sum(v["rung"] >= top for v in certain) / len(certain)
    return out


def bootstrap(diffs, n=10_000, seed=0):
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    args = ap.parse_args()
    a, b = (json.loads(Path(p).read_text()) for p in (args.before, args.after))
    assert a["split"] == b["split"], "compare the same split"
    top = min(a["summary"]["top_rung"], b["summary"]["top_rung"])  # judged vs unjudged: compare on rung 3
    sa, sb = claim_scores(a, top), claim_scores(b, top)
    ids = sorted(set(sa) & set(sb))
    diffs = [sb[i] - sa[i] for i in ids]
    lo, hi = bootstrap(diffs)
    mean = sum(diffs) / len(diffs)

    print(f"{a['version']} -> {b['version']} on {a['split']}, {len(ids)} claims, "
          f"{a['summary']['repeats']} vs {b['summary']['repeats']} repeats, top rung {top}\n")
    print(f"  trusted   {sum(sa[i] for i in ids) / len(ids):.2f} -> {sum(sb[i] for i in ids) / len(ids):.2f}")
    print(f"  change    {mean:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]"
          f"  {'(includes 0: not shown to be real)' if lo <= 0 <= hi else ''}")
    print(f"  better on {sum(d > 0 for d in diffs)}, worse on {sum(d < 0 for d in diffs)}, "
          f"same on {sum(d == 0 for d in diffs)}\n")
    for i, d in zip(ids, diffs):
        if d:
            print(f"  {i}  {sa[i]:.2f} -> {sb[i]:.2f}")
    out = Path(args.after).parent / f"compare-{a['version']}-{b['version']}-{a['split']}.json"
    out.write_text(json.dumps({"before": a["version"], "after": b["version"], "split": a["split"], "top_rung": top,
                               "before_trusted": sum(sa[i] for i in ids) / len(ids),
                               "after_trusted": sum(sb[i] for i in ids) / len(ids),
                               "change": mean, "ci": [lo, hi], "claims": {i: [sa[i], sb[i]] for i in ids}}, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
