"""BM25 keyword search over policy clauses. One chunk per clause (### heading)."""

import re
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi

POLICIES = Path(__file__).resolve().parent.parent / "data" / "policies"
HEADING = re.compile(r"^### ([A-Z]+-[\w.-]+) (.+)$", re.M)
# Words people search with that the policy text never uses. BM25 only matches exact words, so a
# query for "duplicate" would never reach EXP-1.3 ("claimed once", "already been reimbursed").
SYNONYMS = {
    "duplicate": ["once", "already", "reimbursed", "again"],
    "duplicated": ["once", "already", "reimbursed", "again"],
    "resubmitted": ["once", "already", "reimbursed", "again"],
    "receipt": ["receipts"],
}


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def load_clauses():
    clauses = []
    for path in sorted(POLICIES.glob("*.md")):
        text = path.read_text().split("---", 2)[2]
        doc_title = re.search(r"^# (.+)$", text, re.M).group(1)
        matches = list(HEADING.finditer(text))
        for m, nxt in zip(matches, matches[1:] + [None]):
            body = text[m.end():nxt.start() if nxt else len(text)].strip()
            body = re.split(r"^## ", body, flags=re.M)[0].strip()
            clauses.append({"clause_id": m.group(1), "title": m.group(2), "document": doc_title, "text": body})
    return clauses


@lru_cache
def _index():
    clauses = load_clauses()
    corpus = [tokenize(f"{c['clause_id']} {c['title']} {c['text']}") for c in clauses]
    return clauses, BM25Okapi(corpus)


def search(query, k=4):
    clauses, bm25 = _index()
    words = tokenize(query)
    scores = bm25.get_scores(words + [s for w in words for s in SYNONYMS.get(w, [])])
    ranked = sorted(range(len(clauses)), key=lambda i: scores[i], reverse=True)
    return [clauses[i] for i in ranked[:k] if scores[i] > 0]
