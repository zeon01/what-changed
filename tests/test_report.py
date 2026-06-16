from whatchanged.models import Entry
from whatchanged.inference import FakeBackend
from whatchanged.report import build_report, render_report_html, DoctorReport


def _declining():
    entries = []
    for i in range(30):
        cog = 5 if i < 15 else 2
        entries.append(Entry(date=f"2026-06-{i+1:02d}", mobility=cog))
    return entries


def test_build_report_populates_period_and_findings():
    entries = _declining()
    rep = build_report(entries, as_of="2026-06-30",
                       backend=FakeBackend("Cognition has been trending down."))
    assert isinstance(rep, DoctorReport)
    assert rep.period_start == "2026-06-01" and rep.period_end == "2026-06-30"
    assert any(f.domain == "mobility" for f in rep.findings)
    assert rep.narrative == "Cognition has been trending down."
    assert "not medical advice" in rep.disclaimer.lower()


def test_build_report_handles_empty():
    rep = build_report([], as_of="2026-06-30", backend=FakeBackend("unused"))
    assert rep.findings == []
    assert "no notable changes" in rep.narrative.lower()


def test_render_report_html_contains_key_sections():
    rep = build_report(_declining(), as_of="2026-06-30",
                       backend=FakeBackend("Cognition down."))
    html = render_report_html(rep)
    assert "Doctor Report" in html
    assert "2026-06-01" in html and "2026-06-30" in html
    assert "not medical advice" in html.lower()


def test_build_report_uses_full_selected_range():
    # 40 days, one fall each day -> a 40-day report must count all 40, not cap at 30
    entries = [Entry(date=f"2026-05-{i:02d}", falls=1) for i in range(1, 32)]  # May 1-31
    entries += [Entry(date=f"2026-06-{i:02d}", falls=1) for i in range(1, 10)]  # Jun 1-9
    rep = build_report(entries, as_of="2026-06-09", backend=FakeBackend("x"))
    falls = [f for f in rep.findings if f.domain == "falls"]
    assert falls and falls[0].value == 40


def test_render_report_html_escapes_narrative():
    from whatchanged.report import DoctorReport, render_report_html
    rep = DoctorReport(period_start="2026-06-01", period_end="2026-06-30",
                       directions={}, findings=[], narrative="<script>alert(1)</script>")
    html = render_report_html(rep)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
