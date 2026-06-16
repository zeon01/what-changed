from whatchanged.models import Entry, Finding, DOMAINS, EVENTS


def test_domains_and_events_constants():
    assert DOMAINS == ["mobility", "tremor", "stiffness", "mood", "sleep", "alertness"]
    assert EVENTS == ["off_episodes", "dyskinesia_spells", "freezing_episodes", "falls", "missed_or_late_meds", "hallucinations"]


def test_entry_defaults():
    e = Entry(date="2026-06-01")
    assert e.date == "2026-06-01"
    for d in DOMAINS:
        assert getattr(e, d) is None
    for ev in EVENTS:
        assert getattr(e, ev) == 0
    assert e.note == ""


def test_entry_roundtrips_to_dict():
    e = Entry(date="2026-06-01", mobility=4, falls=1, note="ok")
    d = e.to_dict()
    assert d["mobility"] == 4 and d["falls"] == 1 and d["note"] == "ok"
    assert Entry.from_dict(d) == e


def test_finding_fields():
    f = Finding(kind="direction", domain="tremor", summary="tremor declining",
                direction="down", value=-1.2)
    assert f.kind == "direction" and f.direction == "down"
