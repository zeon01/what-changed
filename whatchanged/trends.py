from __future__ import annotations
from datetime import date as _date, timedelta
from whatchanged.models import Entry, Finding, DOMAINS, EVENTS


def _values(entries: list[Entry], domain: str) -> list[int]:
    """Chronological non-None ratings for a domain (entries assumed date-sorted)."""
    return [v for v in (getattr(e, domain) for e in entries) if v is not None]


def rolling_average(entries: list[Entry], domain: str) -> float | None:
    vals = _values(entries, domain)
    return sum(vals) / len(vals) if vals else None


def direction(entries: list[Entry], domain: str, window: int = 30,
              threshold: float = 0.5) -> str:
    """Compare the mean of the earlier half vs the later half of the last `window`
    entries with a rating. 'down' = concerning decline; 'up' = improvement."""
    vals = _values(entries, domain)[-window:]
    if len(vals) < 4:
        return "stable"
    mid = len(vals) // 2
    earlier = vals[:mid]
    later = vals[-mid:]
    delta = (sum(later) / len(later)) - (sum(earlier) / len(earlier))
    if delta <= -threshold:
        return "down"
    if delta >= threshold:
        return "up"
    return "stable"


DOMAIN_LABELS = {
    "mobility": "mobility (walking/balance)",
    "tremor": "tremor",
    "stiffness": "stiffness/rigidity",
    "mood": "mood",
    "sleep": "sleep",
    "alertness": "alertness/cognition",
}
EVENT_LABELS = {
    "off_episodes": "OFF (wearing-off) episodes",
    "dyskinesia_spells": "dyskinesia spells",
    "freezing_episodes": "freezing-of-gait episodes",
    "falls": "falls",
    "missed_or_late_meds": "missed/late medication doses",
    "hallucinations": "hallucination episodes",
}
EVENT_THRESHOLD = {
    "off_episodes": 10, "dyskinesia_spells": 8, "freezing_episodes": 6,
    "falls": 2, "missed_or_late_meds": 4, "hallucinations": 2,
}


def decline_streak(entries: list[Entry], domain: str) -> int:
    """Count consecutive strictly-decreasing steps at the tail of the series. Returns
    0 if the most recent change is flat or an increase, or if data is insufficient.
    Example: [3,3,3,2,1] -> 2 (the 3->2 and 2->1 steps); [5,4,5] -> 0."""
    vals = _values(entries, domain)
    steps = 0
    for i in range(len(vals) - 1, 0, -1):
        if vals[i] < vals[i - 1]:
            steps += 1
        else:
            break
    return steps


def event_total(entries: list[Entry], event: str, days: int, as_of: str) -> int:
    cutoff = _date.fromisoformat(as_of) - timedelta(days=days - 1)
    as_of_d = _date.fromisoformat(as_of)
    return sum(
        getattr(e, event) for e in entries
        if cutoff <= _date.fromisoformat(e.date) <= as_of_d
    )


def detect_findings(entries: list[Entry], as_of: str, window: int = 30) -> list[Finding]:
    findings: list[Finding] = []
    for d in DOMAINS:
        dirn = direction(entries, d, window=window)
        if dirn == "down":
            findings.append(Finding(
                kind="direction", domain=d, direction="down",
                summary=f"{DOMAIN_LABELS[d]} has been trending down over the last "
                        f"{window} days",
            ))
        streak = decline_streak(entries, d)
        if streak >= 3:
            findings.append(Finding(
                kind="streak", domain=d, direction="down", value=streak,
                summary=f"{DOMAIN_LABELS[d]} declined across {streak} consecutive "
                        f"logged days",
            ))
    for ev in EVENTS:
        total = event_total(entries, ev, days=window, as_of=as_of)
        if total >= EVENT_THRESHOLD.get(ev, 1):
            findings.append(Finding(
                kind="event", domain=ev, value=total,
                summary=f"{total} {EVENT_LABELS[ev]} logged in the last {window} days",
            ))
    return findings
