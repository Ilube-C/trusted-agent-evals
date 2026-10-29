"""Signals from past claims, computed without an answer key: is this claim unusual, and is it what it says it is?

    python -m agent.signals            # print the signals for every pending claim
"""

import math
import re
from collections import Counter, defaultdict
from datetime import date
from functools import lru_cache
from statistics import median

from agent.tools import load

MIN_HISTORY = 20     # fewer past claims than this and a baseline is not worth trusting
ANOMALY_Z = 3.0      # robust z above this counts as unusual
GROUPS = {"Meals": "food", "Client entertainment": "food", "Rail": "travel", "Taxi": "travel",
          "Hotel": "travel", "Mileage": "travel", "Per diem": "travel", "Equipment": "equipment", "Other": "other"}


def robust_z(x, xs):
    """How far log(x) is from the median of log(xs), in MADs scaled to match a standard deviation."""
    logs = [math.log(v) for v in xs if v > 0]
    m = median(logs)
    mad = median(abs(v - m) for v in logs) * 1.4826
    return (math.log(x) - m) / mad if mad else 0.0


# ---- topic: multinomial Naive Bayes on words, trained on past claims ----

def words(text):
    return re.findall(r"[a-z]+", text.lower())


class NaiveBayes:
    def __init__(self, docs):
        """docs: (text, label) pairs."""
        self.prior = Counter(label for _, label in docs)
        self.counts = defaultdict(Counter)
        for text, label in docs:
            self.counts[label].update(words(text))
        self.vocab = {w for c in self.counts.values() for w in c}
        self.total = {label: sum(c.values()) for label, c in self.counts.items()}

    def predict(self, text):
        """Returns (label, probability), with add-one smoothing on word counts."""
        n, v = sum(self.prior.values()), len(self.vocab)
        scores = {label: math.log(self.prior[label] / n)
                  + sum(math.log((self.counts[label][w] + 1) / (self.total[label] + v)) for w in words(text))
                  for label in self.prior}
        top = max(scores, key=scores.get)
        p = 1 / sum(math.exp(s - scores[top]) for s in scores.values())
        return top, round(p, 2)


@lru_cache
def history():
    return load("history.json")


@lru_cache
def topic_model():
    return NaiveBayes([(f"{h['subject']} {h['line']} {h['merchant'] or ''}", h["category"]) for h in history()])


def claim_text(c):
    items = [f"{a.get('merchant', '')} " + " ".join(i["desc"] for i in a["items"])
             for a in c["attachments"] if a["kind"] == "receipt"]
    return " ".join([c["subject"], *items])


# ---- organisations: spaCy NER on the email, merchants on the receipts ----

@lru_cache
def nlp():
    import spacy
    return spacy.load("en_core_web_sm")


def organisations(c):
    return sorted({e.text for e in nlp()(c["body"]).ents if e.label_ == "ORG"})


def receipt_merchants(c):
    return [a["merchant"] for a in c["attachments"] if a["kind"] == "receipt" and a.get("merchant")]


def signals(c):
    """Everything the agent and graders need to judge whether a claim looks normal."""
    past = history()
    same_cat = [h["amount"] for h in past if h["category"] == c["category"]]
    # This person's spending relative to the firm median in each category, so categories can be pooled.
    cat_median = {cat: median(h["amount"] for h in past if h["category"] == cat) for cat in GROUPS}
    own = [h["amount"] / cat_median[h["category"]] for h in past if h["employee"] == c["employee"]]
    merchants = receipt_merchants(c)
    # Merchants and repeats also count real claims that arrived before this one, not just the paid history.
    earlier = [{"id": e["id"], "merchant": m, "amount": e["amount"], "incurred": e["incurred"]}
               for e in load("claims.json") if e["sent_at"] < c["sent_at"] for m in receipt_merchants(e)]
    known = {h["merchant"] for h in past + earlier}
    incurred = date.fromisoformat(c["incurred"])
    repeats = [h["id"] for h in past + earlier if h["merchant"] in merchants and abs(h["amount"] - c["amount"]) < 0.01
               and abs((date.fromisoformat(h["incurred"]) - incurred).days) <= 30]
    topic, confidence = topic_model().predict(claim_text(c))
    receipts = [a["total"] for a in c["attachments"] if a["kind"] == "receipt"]

    out = {"claim_id": c["id"], "category": c["category"], "amount": c["amount"]}
    if len(same_cat) >= MIN_HISTORY:
        out["category_z"] = round(robust_z(c["amount"], same_cat), 1)
    if len(own) >= MIN_HISTORY:
        out["employee_z"] = round(robust_z(c["amount"] / cat_median[c["category"]], own), 1)
    out["receipt_total"] = round(sum(receipts), 2) if receipts else None
    out["new_merchants"] = [m for m in merchants if m not in known]
    out["repeats"] = repeats
    out["organisations_mentioned"] = organisations(c)
    out["topic"] = {"predicted": topic, "group": GROUPS[topic], "confidence": confidence,
                    "matches_category": topic == c["category"]}
    out["unusual"] = [reason for reason, hit in [
        ("amount high for category", out.get("category_z", 0) > ANOMALY_Z),
        ("amount high for this employee", out.get("employee_z", 0) > ANOMALY_Z),
        ("claimed amount differs from receipts", bool(receipts) and abs(sum(receipts) - c["amount"]) >= 0.01),
        ("merchant never seen before", bool(out["new_merchants"])),
        ("same merchant and amount within 30 days", bool(repeats)),
        ("content does not match category", topic != c["category"] and confidence >= 0.8),
    ] if hit]
    return out


def main():
    claims = load("claims.json")
    right = sum(signals(c)["topic"]["matches_category"] for c in claims)
    right_group = sum(signals(c)["topic"]["group"] == GROUPS[c["category"]] for c in claims)
    print(f"{'claim':9} {'category':21} {'amount':>8} {'cat z':>6} {'emp z':>6}  topic (p)                    unusual")
    for c in claims:
        s = signals(c)
        t = s["topic"]
        print(f"{c['id']:9} {c['category']:21} {c['amount']:8.2f} {s.get('category_z', '-'):>6} "
              f"{s.get('employee_z', '-'):>6}  {t['predicted'] + ' (' + str(t['confidence']) + ')':28} "
              f"{'; '.join(s['unusual'])}")
    print(f"\ntopic matches declared category on {right}/{len(claims)}, group on {right_group}/{len(claims)}")


if __name__ == "__main__":
    main()
