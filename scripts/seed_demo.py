"""Seed a realistic 60-day decline arc for the demo. Usage:
    WC_DB=data/demo.db python scripts/seed_demo.py
"""
from __future__ import annotations
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whatchanged.db import init_db, upsert_entry
from whatchanged.models import Entry


def seed(conn, days: int = 60) -> None:
    today = datetime.date.today()
    for i in range(days):
        day = today - datetime.timedelta(days=days - 1 - i)
        frac = i / max(1, days - 1)               # 0 -> 1 over the window
        mobility = max(1, round(5 - 3 * frac))    # gait/balance decline 5 -> 2
        tremor = max(1, round(5 - 3 * frac))      # tremor worsens 5 -> 2
        stiffness = max(1, round(5 - 2 * frac))
        sleep = 4 if i % 3 else 2
        mood = max(1, round(4 - 1.5 * frac))
        alertness = max(1, round(5 - 2 * frac))
        off_episodes = round(4 * frac) + (1 if i % 2 else 0)   # wearing-off creeps up
        freezing_episodes = 1 if (frac > 0.4 and i % 2 == 0) else 0
        dyskinesia_spells = 1 if (frac > 0.6 and i % 3 == 0) else 0
        falls = 1 if (frac > 0.5 and i % 7 == 0) else 0
        missed_or_late_meds = 1 if i % 5 == 0 else 0
        hallucinations = 1 if (frac > 0.8 and i % 6 == 0) else 0
        note = ("froze in the doorway, pill seemed to wear off before lunch"
                if (frac > 0.5 and i % 7 == 0) else "")
        upsert_entry(conn, Entry(
            date=str(day), mobility=mobility, tremor=tremor, stiffness=stiffness,
            mood=mood, sleep=sleep, alertness=alertness, off_episodes=off_episodes,
            dyskinesia_spells=dyskinesia_spells, freezing_episodes=freezing_episodes,
            falls=falls, missed_or_late_meds=missed_or_late_meds,
            hallucinations=hallucinations, note=note))


if __name__ == "__main__":
    conn = init_db(os.environ.get("WC_DB", "data/demo.db"))
    seed(conn)
    print("Seeded demo data.")
