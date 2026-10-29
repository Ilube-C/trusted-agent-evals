# Trusted agent, with evals

An expense claim checker for a small fictional consultancy. It reads the claim and receipt,
compares it with past claims, searches the expense policies, and approves, partially approves,
rejects, or flags the claim for a human, citing the rules it used. Built to answer the question
the exercise asks: not "does this agent work?" but "how would you know?"

**Start with [docs/WRITEUP.md](docs/WRITEUP.md)**: what it does, how it is graded, results
before and after, where it struggles, and what I'm unsure about.

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add an Anthropic key, and a LangSmith key for tracing
uvicorn web.server:app --port 8765 --reload
```

- http://127.0.0.1:8765 is the expenses inbox. "Check claim" runs the agent on one claim.
- http://127.0.0.1:8765/evals is the evals page: the policy rules, the checks and their
  equations, results by version, and every failing run with its trace. It reads the saved
  results in `evals/results/`, so it works without an API key.

## Evals

```bash
python -m evals.test_graders                                   # grader tests, no API key needed
python -m evals.run --split heldout --version v4 --repeats 3   # run the agent (costs money)
python -m evals.score evals/results/v4-heldout-*.json --judge  # grade, with the Haiku judge
python -m evals.compare evals/results/graded/v1-heldout.json evals/results/graded/v4-heldout.json
python -m agent.signals                                        # signals from past claims, per claim
```

## Layout

| Path | What lives here |
|---|---|
| `agent/` | the agent, its tools, policy search, prompts (v1 to v4), signals from past claims |
| `data/` | synthetic policies, employees, 40 claims, and 400 past paid claims (`make_history.py`) |
| `evals/` | the dataset, the checks, the judge, scoring, comparison, grader tests |
| `evals/results/` | every run, and the graded results the evals page reads |
| `web/` | the inbox and evals page: FastAPI server and static pages |
| `docs/` | the write-up |
