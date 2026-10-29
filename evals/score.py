"""Grade saved runs on the trust ladder. Pass several files to treat them as repeats of one version.

    python -m evals.score evals/results/v1-dev-*.json
    python -m evals.score evals/results/v1-dev-*.json --judge --langsmith
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from evals.graders import RUNGS, TAGS, climb

ROOT = Path(__file__).resolve().parent
GRADED = ROOT / "results" / "graded"


def grade_run(results, cases, use_judge):
    out, cost = [], 0.0
    for r in results:
        case = cases[r["claim_id"]]
        if "error" in r:
            out.append({"claim_id": r["claim_id"], "rung": -1, "grade": "unsafe", "failed": "safe",
                        "tag": "broken_action", "detail": r["error"]})
            continue
        v = climb(r, case)
        # The judge only runs where the rules leave a question: rungs 3-4.
        if use_judge and v["rung"] >= 3:
            from evals.judge import judge
            j = judge(r, case)
            cost += j["cost"]
            v = climb(r, case, judged=j)
            v["judge"] = j
        out.append({"claim_id": r["claim_id"], "run_id": r.get("run_id"),
                    "decision": (r.get("decision") or {}).get("decision"), **v})
    return out, cost


def summarise(claims, top):
    verdicts = [v for c in claims.values() for v in c["verdicts"]]
    certain = [v for v in verdicts if not v.get("uncertain")]
    by_claim = [[v for v in c["verdicts"] if not v.get("uncertain")] for c in claims.values()]
    k = min(len(c["verdicts"]) for c in claims.values())
    rungs = Counter(v["rung"] for v in certain)
    tags = {}
    for c in claims.values():
        for v in c["verdicts"]:
            if v["tag"]:
                tags.setdefault(v["tag"], Counter())[c["case_tag"]] += 1
    return {
        "runs": len(verdicts), "repeats": k, "top_rung": top,
        "rungs": {RUNGS[i] if i >= 0 else "unsafe": rungs.get(i, 0) for i in range(-1, 5)},
        # Uncertain runs are left out of the rates rather than counted as passes or failures.
        "uncertain": sum(v.get("uncertain", False) for v in verdicts),
        "trusted": sum(v["rung"] >= top for v in certain) / max(len(certain), 1),
        "pass_k": sum(all(v["rung"] >= top for v in sure) for sure in by_claim if sure)
                  / max(sum(bool(sure) for sure in by_claim), 1),
        "tags": {t: dict(tags[t]) for t in TAGS if t in tags},
        "flag_rate": sum(v.get("decision") == "flag" for v in verdicts) / len(verdicts),
        "key_match": sum(v.get("key_match", False) for v in verdicts) / len(verdicts),
    }


def post_feedback(claims, split):
    """Attach each verdict to its LangSmith trace, so a reviewer sees the grade next to the steps."""
    import os
    import uuid

    from dotenv import load_dotenv
    from langsmith import Client
    load_dotenv(ROOT.parent / ".env")
    client = Client()
    project = client.read_project(project_name=os.environ.get("LANGSMITH_PROJECT", "expense-checker"))
    app = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com").replace("api.", "")
    for cid, c in claims.items():
        for v in c["verdicts"]:
            run_id = v.get("run_id")
            if not run_id:  # runs from before run IDs were recorded: look up by name and tag
                found = list(client.list_runs(project_id=project.id, is_root=True, limit=1,
                                              filter=f'and(eq(name, "check {cid}"), has(tags, "eval-{split}"))'))
                if not found:
                    continue
                run_id = v["run_id"] = str(found[0].id)
            # Fixed feedback IDs, so re-scoring a run updates its feedback instead of adding more.
            for key, kw in (("trust_rung", {"score": v["rung"], "value": v["grade"]}),
                            ("failure_tag", {"value": v["tag"] or "none"})):
                fid = uuid.uuid5(uuid.NAMESPACE_URL, f"{run_id}/{key}")
                try:
                    client.create_feedback(run_id, key, trace_id=run_id, session_id=project.id, feedback_id=fid,
                                           comment=v["detail"] or None, **kw)
                except Exception:
                    client.update_feedback(fid, comment=v["detail"] or None, **kw)
            v["url"] = f"{app}/o/{project.tenant_id}/projects/p/{project.id}/r/{run_id}"


def report(g):
    s = g["summary"]
    print(f"{g['version']} {g['split']}: {len(g['claims'])} claims x {s['repeats']} repeats"
          f"{'' if g['judged'] else ' (not judged: top rung is faithful)'}\n")
    print("  rung reached     " + "  ".join(f"{k} {v}" for k, v in s["rungs"].items()))
    print(f"  trusted runs     {s['trusted']:.2f}  (of {s['runs'] - s['uncertain']} certain; "
          f"{s['uncertain']} uncertain, left for a human)")
    print(f"  pass^{s['repeats']}           {s['pass_k']:.2f}  (claims trusted on every repeat)")
    print(f"  flag rate        {s['flag_rate']:.2f}  (tracked, not failed: some claims should go to a human)")
    print(f"  agrees with key  {s['key_match']:.2f}  (offline check, not part of the ladder)\n")
    case_tags = ["normal", "edge", "failure"]
    print("  " + f"{'failure tag':22}" + "".join(f"{t:>9}" for t in case_tags))
    for tag, counts in s["tags"].items():
        print("  " + f"{tag:22}" + "".join(f"{counts.get(t, 0):>9}" for t in case_tags))
    print()
    for cid, c in g["claims"].items():
        for v in c["verdicts"]:
            if v["tag"]:
                print(f"  {cid}  failed at {v['failed']:10} {v['tag']:20} {v['detail'][:110]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--langsmith", action="store_true", help="post verdicts to LangSmith as feedback")
    args = ap.parse_args()

    files = [json.loads(Path(p).read_text()) for p in args.runs]
    version, split = files[0]["version"], files[0]["split"]
    assert all(f["version"] == version and f["split"] == split for f in files), "mix of versions or splits"
    cases = {c["claim_id"]: c for c in map(json.loads, (ROOT / "dataset.jsonl").open())}

    claims, cost = {}, 0.0
    for f in files:
        verdicts, c = grade_run(f["results"], cases, args.judge)
        cost += c
        for v in verdicts:
            claims.setdefault(v["claim_id"], {"case_tag": cases[v["claim_id"]]["tag"], "verdicts": []})
            claims[v["claim_id"]]["verdicts"].append(v)
    claims = dict(sorted(claims.items()))
    if args.langsmith:
        post_feedback(claims, split)

    top = 4 if args.judge else 3
    g = {"version": version, "split": split, "judged": args.judge, "sources": [Path(p).name for p in args.runs],
         "summary": summarise(claims, top), "claims": claims}
    report(g)
    if args.judge:
        judged = [v["judge"] for c in claims.values() for v in c["verdicts"] if "judge" in v]
        split_votes = [v for v in judged if not v["unanimous"]]
        print(f"\n  judge: {len(judged)} runs x {len(judged[0]['votes'])} votes, "
              f"{len(split_votes)} split, cost ~${cost:.3f}")
    GRADED.mkdir(exist_ok=True)
    out = GRADED / f"{version}-{split}.json"
    out.write_text(json.dumps(g, indent=2, default=str))
    print(f"\nSaved {out.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
