# Trusted agent, with evals

An expense claim checker for a small company: it reads the expense policies,
looks up the claim and receipt, and approves, partially approves, rejects, or flags
the claim for a human. Built to answer the question the exercise actually asks: not
"does this agent work?" but "how would you know?"

Status: scaffolding only. See `HANDOVER.md` for the brief, the decisions already
made, and the constraints.

## Running it

The expenses inbox runs; the checker agent is not wired up yet.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in a key
uvicorn web.server:app --port 8765 --reload
```

Then open http://127.0.0.1:8765.

## Layout

| Path | What lives here |
|---|---|
| `agent/` | the graph, tools, retrieval, prompts |
| `data/` | synthetic expense policies, employees, claims and receipts |
| `evals/` | the dataset, the graders, the judge |
| `traces/` | captured runs |
| `web/` | the expenses inbox: FastAPI server and static UI |
| `docs/` | the technical write-up |
