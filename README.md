# Trusted agent, with evals

An HR policy assistant over a staff handbook, built to answer the question the
exercise actually asks: not "does this agent work?" but "how would you know?"

Status: scaffolding only. See `HANDOVER.md` for the brief, the decisions already
made, and the constraints.

## Running it

Not yet runnable.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in a key
```

## Layout

| Path | What lives here |
|---|---|
| `agent/` | the graph, tools, retrieval, prompts |
| `data/` | synthetic handbook and employee records |
| `evals/` | the dataset, the graders, the judge |
| `traces/` | captured runs |
| `docs/` | the technical write-up |
