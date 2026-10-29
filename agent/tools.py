"""The agent's tools. Built per run so each run records its own decision and retrievals."""

import json
from pathlib import Path
from typing import Literal

from langchain_core.tools import tool

from agent import retrieval

DATA = Path(__file__).resolve().parent.parent / "data"
REVIEW_THRESHOLD = 500  # EXP-4.1: Finance reviews any claim over £500 before payment


def load(name):
    return json.loads((DATA / name).read_text())


class Recorder:
    def __init__(self):
        self.decisions = []
        self.retrieved = []
        self.blocked = []  # decisions the guard refused


def make_tools(rec: Recorder, claim_id: str, guard: bool = False):
    """guard: refuse to pay any claim over the review threshold, in code rather than in the prompt."""
    claims = {c["id"]: c for c in load("claims.json")}
    # Only claims that existed when this one arrived are visible to history searches.
    as_of = claims[claim_id]["sent_at"]
    amount = claims[claim_id]["amount"]  # submit_decision's own claim_id argument shadows this run's
    people = {e["id"]: e for e in load("employees.json")}

    @tool
    def search_policy(query: str) -> str:
        """Search the expense policies. Returns the most relevant clauses with their IDs."""
        hits = retrieval.search(query)
        rec.retrieved.append({"query": query, "clause_ids": [h["clause_id"] for h in hits]})
        if not hits:
            return "No matching clauses."
        return "\n\n".join(f"[{h['clause_id']}] {h['title']} ({h['document']})\n{h['text']}" for h in hits)

    @tool
    def get_claim(claim_id: str) -> str:
        """Get a claim: the email, its attachments, the employee who submitted it and their manager."""
        c = claims.get(claim_id)
        if not c:
            return f"No claim {claim_id}."
        c = {k: v for k, v in c.items() if k != "status"}
        e = people[c["employee"]]
        c["employee"] = e
        c["manager"] = people.get(e["manager"])
        return json.dumps(c, ensure_ascii=False)

    @tool
    def search_claims(employee_id: str | None = None, receipt_ref: str | None = None) -> str:
        """Find claims submitted up to now, by employee ID and/or receipt reference. Use this to check history and duplicates."""
        rows = []
        for c in claims.values():
            if c["sent_at"] > as_of:
                continue
            refs = [a.get("ref") for a in c["attachments"]]
            if employee_id and c["employee"] != employee_id:
                continue
            if receipt_ref and receipt_ref not in refs:
                continue
            rows.append({"id": c["id"], "employee": c["employee"], "category": c["category"], "incurred": c["incurred"],
                         "amount": c["amount"], "currency": c["currency"], "receipt_refs": refs, "subject": c["subject"]})
        return json.dumps(rows, ensure_ascii=False) if rows else "No matching claims."

    @tool
    def compare_to_history(claim_id: str) -> str:
        """Compare a claim with past claims: how unusual the amount is for its category and for this employee
        (robust z-scores, above 3 is unusual), merchants never seen before, possible repeats, organisations
        named in the email, and whether the content matches the declared category."""
        from agent.signals import signals  # signals imports load from here
        c = claims.get(claim_id)
        return json.dumps(signals(c), ensure_ascii=False) if c else f"No claim {claim_id}."

    @tool
    def submit_decision(
        claim_id: str,
        decision: Literal["approve", "partial", "reject", "flag"],
        approved_amount: float | None,
        clause_ids: list[str],
        rationale: str,
    ) -> str:
        """Record the decision on a claim. Call exactly once, at the end.
        approved_amount is in the claim's currency; use null for flag."""
        if guard and decision in ("approve", "partial") and amount > REVIEW_THRESHOLD:
            rec.blocked.append(decision)
            return (f"Refused: this claim is over {REVIEW_THRESHOLD} and EXP-4.1 requires Finance to review it "
                    "before payment. Call submit_decision again with flag, and say in the rationale what Finance "
                    "needs to check.")
        rec.decisions.append({"claim_id": claim_id, "decision": decision, "approved_amount": approved_amount,
                              "clause_ids": clause_ids, "rationale": rationale})
        return "Decision recorded."

    return [search_policy, get_claim, search_claims, compare_to_history, submit_decision]
