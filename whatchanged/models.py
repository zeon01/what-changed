from __future__ import annotations
from dataclasses import dataclass, asdict, fields

DOMAINS = ["mobility", "tremor", "stiffness", "mood", "sleep", "alertness"]
EVENTS = ["off_episodes", "dyskinesia_spells", "freezing_episodes", "falls", "missed_or_late_meds", "hallucinations"]


@dataclass
class Entry:
    date: str  # ISO "YYYY-MM-DD"
    mobility: int | None = None
    tremor: int | None = None
    stiffness: int | None = None
    mood: int | None = None
    sleep: int | None = None
    alertness: int | None = None
    off_episodes: int = 0
    dyskinesia_spells: int = 0
    freezing_episodes: int = 0
    falls: int = 0
    missed_or_late_meds: int = 0
    hallucinations: int = 0
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Entry":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class Finding:
    kind: str          # "direction" | "streak" | "event"
    domain: str        # domain or event name
    summary: str       # human-readable, descriptive (never diagnostic)
    direction: str = ""  # "up" | "down" | "stable" | ""
    value: float = 0.0
    detail: str = ""
