"""Generate a year of past, already-paid claims to give the anomaly checks a baseline.

    python data/make_history.py

Amounts are log-normal per category (lots of small claims, a long tail), scaled a little per
employee so personal baselines differ. Seeded, so the file is the same on every run.
"""

import json
import math
import random
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
N = 400
START, END = date(2025, 9, 1), date(2026, 8, 31)  # ends before the pending claims begin

# category: (median £, spread of log amount, merchants, attendees range or None)
CATEGORIES = {
    "Meals": (19, 0.35, ["Pret Liverpool Street", "Leon Liverpool Street", "Dishoom Shoreditch",
                         "Pho Spitalfields", "Flat Iron Spitalfields", "Wasabi Moorgate", "Cask Ancoats",
                         "Itsu Bank"], None),
    "Client entertainment": (62, 0.25, ["Brasserie Annick", "The Gun, Spitalfields", "Hawksmoor Spitalfields",
                                        "Rudy's Ancoats", "St John Bread and Wine"], (2, 6)),  # median is per head
    "Rail": (80, 0.5, ["Avanti West Coast", "LNER", "Southeastern", "GWR", "Northern"], None),
    "Hotel": (190, 0.3, ["The Fenwick Hotel, Farringdon", "Quebecs Hotel, Leeds", "Harbourside Inn, Bristol",
                         "Premier Inn Manchester Central"], None),
    "Taxi": (24, 0.4, ["Addison Lee", "FreeNow", "Amber Cars Leeds", "Uber"], None),
    "Equipment": (140, 0.6, ["Currys", "John Lewis", "Posturite", "Argos"], None),
    "Mileage": (35, 0.6, [None], None),
    "Per diem": (110, 0.3, [None], None),
    "Other": (30, 0.5, ["Workhub Old Street", "WHSmith", "Royal Mail"], None),
}
WEIGHTS = {"Meals": 30, "Rail": 18, "Taxi": 14, "Client entertainment": 10, "Hotel": 10,
           "Equipment": 4, "Mileage": 6, "Per diem": 5, "Other": 3}
# A short subject and receipt line per claim, so the topic classifier has text to learn from.
TEXT = {
    "Meals": [("Late dinner in the office", "Chicken curry and rice"), ("Working late, dinner", "Noodle soup"),
              ("Evening meal, deadline", "Burrito and drink"), ("Dinner after late finish", "Salad and soup"),
              ("Late supper", "Sandwich, crisps"), ("Meal, worked late", "Steak frites")],
    "Client entertainment": [("Client dinner", "Set menu, wine"), ("Lunch with the client team", "Mains, desserts"),
                             ("Dinner with client sponsor", "Tasting menu, bottle of red"),
                             ("Client drinks after workshop", "Drinks and sharing plates")],
    "Rail": [("Train to client site", "Return, standard class"), ("Rail fare for workshop", "Off-peak return"),
             ("Train ticket", "Single, standard"), ("Return train for kickoff", "Advance return, standard class")],
    "Hotel": [("Hotel for client visit", "Room, one night"), ("Hotel stay, workshop", "Room, two nights, breakfast"),
              ("Overnight stay", "Standard double room")],
    "Taxi": [("Taxi home after late finish", "Trip, office to home"), ("Cab to client office", "Trip to client"),
             ("Taxi from station", "Trip, station to hotel")],
    "Equipment": [("Laptop stand", "Adjustable laptop stand"), ("Headset for calls", "USB headset"),
                  ("Replacement mouse", "Wireless mouse"), ("Home office desk lamp", "LED desk lamp")],
    "Mileage": [("Mileage to client site", "Business miles, own car"), ("Drove to workshop", "Return journey, miles")],
    "Per diem": [("Per diem, overseas trip", "Daily allowance"), ("Daily allowance, client visit abroad",
                                                                   "Per diem, days away")],
    "Other": [("Printing for workshop", "Colour printing"), ("Postage for contracts", "Recorded delivery"),
              ("Stationery", "Notebooks and pens")],
}
# How much each person tends to spend relative to the firm median (directors eat out pricier).
EMPLOYEE_SCALE = {"E01": 1.0, "E02": 0.9, "E03": 0.85, "E04": 1.1, "E05": 0.95, "E06": 1.3}


def main():
    rng = random.Random(7)
    days = (END - START).days
    rows = []
    for i in range(N):
        cat = rng.choices(list(WEIGHTS), weights=list(WEIGHTS.values()))[0]
        median, sigma, merchants, heads = CATEGORIES[cat]
        emp = rng.choice(list(EMPLOYEE_SCALE))
        subject, line = rng.choice(TEXT[cat])
        attendees = rng.randint(*heads) if heads else None
        each = math.exp(rng.gauss(math.log(median * EMPLOYEE_SCALE[emp]), sigma))
        amount = round(each * (attendees or 1), 2)
        rows.append({"id": f"HST-{i + 1:04d}", "employee": emp, "category": cat,
                     "incurred": (START + timedelta(days=rng.randrange(days))).isoformat(),
                     "amount": amount, "currency": "GBP", "merchant": rng.choice(merchants),
                     "attendees": attendees, "subject": subject, "line": line,
                     "status": "paid"})
    rows.sort(key=lambda r: r["incurred"])
    (ROOT / "history.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    print(f"Wrote {len(rows)} past claims to data/history.json")


if __name__ == "__main__":
    main()
