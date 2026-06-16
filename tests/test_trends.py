from whatchanged.models import Entry
from whatchanged.trends import rolling_average, direction


def _series(values, start_day=1):
    # values: list of domain ratings (or None); dates are sequential June days
    return [Entry(date=f"2026-06-{start_day + i:02d}", mobility=v)
            for i, v in enumerate(values)]


def test_rolling_average_ignores_none():
    entries = _series([4, None, 2])
    assert rolling_average(entries, "mobility") == 3.0


def test_rolling_average_empty_returns_none():
    assert rolling_average([], "mobility") is None
    assert rolling_average(_series([None, None]), "mobility") is None


def test_direction_down_when_later_half_lower():
    entries = _series([5, 5, 5, 2, 2, 2])
    assert direction(entries, "mobility", window=6, threshold=0.5) == "down"


def test_direction_up_when_later_half_higher():
    entries = _series([2, 2, 2, 5, 5, 5])
    assert direction(entries, "mobility", window=6, threshold=0.5) == "up"


def test_direction_stable_when_flat():
    entries = _series([3, 3, 3, 3])
    assert direction(entries, "mobility", window=6, threshold=0.5) == "stable"


def test_direction_stable_when_insufficient_data():
    assert direction(_series([3]), "mobility") == "stable"


from whatchanged.trends import decline_streak, event_total, detect_findings


def test_decline_streak_counts_trailing_nonincreasing_drop():
    # 3,3 then steadily down 3->2->1 : trailing strictly-declining run of 2 steps
    entries = _series([3, 3, 3, 2, 1])
    assert decline_streak(entries, "mobility") == 2


def test_decline_streak_zero_when_last_increases():
    entries = _series([5, 4, 5])
    assert decline_streak(entries, "mobility") == 0


def test_event_total_sums_window():
    entries = [Entry(date=f"2026-06-{i:02d}", falls=1) for i in range(1, 6)]
    assert event_total(entries, "falls", days=30, as_of="2026-06-05") == 5
    assert event_total(entries, "falls", days=2, as_of="2026-06-05") == 2


def test_detect_findings_flags_decline_and_falls():
    entries = _series([5, 5, 5, 2, 2, 2])  # mobility declining
    for i, e in enumerate(entries):
        e.falls = 1 if i >= 4 else 0       # 2 recent falls
    findings = detect_findings(entries, as_of="2026-06-06")
    kinds = {(f.kind, f.domain) for f in findings}
    assert ("direction", "mobility") in kinds
    assert ("event", "falls") in kinds
    # every summary is descriptive, never diagnostic
    assert all("dementia" not in f.summary.lower() for f in findings)


def test_direction_filters_none_before_windowing():
    import datetime
    base = datetime.date(2026, 1, 1)
    ratings = [5] * 30 + [None] * 10 + [2] * 20  # 60 days; None gap before a decline
    entries = [Entry(date=str(base + datetime.timedelta(days=i)), mobility=v)
               for i, v in enumerate(ratings)]
    # last 30 RATED values = [5]*10 + [2]*20 -> later half clearly lower -> "down"
    assert direction(entries, "mobility", window=30, threshold=0.5) == "down"
