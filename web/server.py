"""Serves the synthetic expense claims to the inbox page, and runs the checker on request."""

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse

from agent.graph import check_claim
from agent.retrieval import load_clauses

ROOT = Path(__file__).resolve().parent.parent
DECISIONS = ROOT / "data/decisions.json"
app = FastAPI()


def load_decisions():
    return json.loads(DECISIONS.read_text()) if DECISIONS.exists() else {}


@app.get("/api/claims")
def claims():
    people = {e["id"]: e for e in json.loads((ROOT / "data/employees.json").read_text())}
    decisions = load_decisions()
    rows = json.loads((ROOT / "data/claims.json").read_text())
    for c in rows:
        c["from"] = people[c["employee"]]["name"]
        c["decision"] = decisions.get(c["id"])
    return sorted(rows, key=lambda c: c["sent_at"], reverse=True)


@app.post("/api/claims/{claim_id}/check")
def check(claim_id: str):
    result = check_claim(claim_id, tags=["ui"])
    if not result["decision"]:
        raise HTTPException(502, "The checker finished without submitting a decision.")
    decisions = load_decisions()
    decisions[claim_id] = result["decision"]
    DECISIONS.write_text(json.dumps(decisions, indent=2, ensure_ascii=False))
    return result["decision"]


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static/index.html")


@app.get("/api/evals")
def evals():
    graded = ROOT / "evals/results/graded"
    load = lambda pattern: [json.loads(p.read_text()) for p in sorted(graded.glob(pattern))]
    return {"graded": load("v*.json"), "comparisons": load("compare-*.json")}


@app.get("/api/rules")
def rules():
    return load_clauses()


@app.get("/api/trace/{run_id}")
def trace(run_id: str):
    """A saved eval run, with each tool call's output replayed (the tools are deterministic)."""
    from evals.graders import observations
    for path in sorted((ROOT / "evals/results").glob("*.json")):
        for r in json.loads(path.read_text())["results"]:
            if r.get("run_id") == run_id:
                steps = [{"name": n, "args": a, "output": o} for n, a, o in observations(r)]
                if r.get("decision"):
                    steps.append({"name": "submit_decision", "args": r["decision"], "output": "Decision recorded."})
                return {"file": path.name, "claim_id": r["claim_id"], "version": r["version"],
                        "decision": r.get("decision"), "steps": steps}
    raise HTTPException(404, "No saved run with that ID.")


@app.get("/evals")
def evals_page():
    return FileResponse(Path(__file__).parent / "static/evals.html")


@app.get("/graders")
def graders():
    return RedirectResponse("/evals#reference")
