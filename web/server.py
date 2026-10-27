"""Serves the synthetic expense claims to the inbox page."""

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parent.parent
app = FastAPI()


@app.get("/api/claims")
def claims():
    people = {e["id"]: e for e in json.loads((ROOT / "data/employees.json").read_text())}
    rows = json.loads((ROOT / "data/claims.json").read_text())
    for c in rows:
        c["from"] = people[c["employee"]]["name"]
    return sorted(rows, key=lambda c: c["sent_at"], reverse=True)


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static/index.html")
