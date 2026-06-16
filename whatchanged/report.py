from __future__ import annotations
import html
from dataclasses import dataclass, field
from datetime import date as _date
from whatchanged.models import Entry, Finding, DOMAINS
from whatchanged.trends import direction, detect_findings, DOMAIN_LABELS
from whatchanged.inference import LLMBackend, narrate_findings

DISCLAIMER = (
    "This report summarizes a caregiver's own day-to-day observations. It is not "
    "medical advice and is not a diagnosis. Please review it with a clinician."
)


@dataclass
class DoctorReport:
    period_start: str
    period_end: str
    directions: dict[str, str]
    findings: list[Finding]
    narrative: str
    disclaimer: str = DISCLAIMER


def build_report(entries: list[Entry], as_of: str, backend: LLMBackend) -> DoctorReport:
    entries = sorted(entries, key=lambda e: e.date)
    start = entries[0].date if entries else as_of
    if entries:
        window = max(1, (_date.fromisoformat(as_of) - _date.fromisoformat(start)).days + 1)
    else:
        window = 30
    directions = {d: direction(entries, d, window=window) for d in DOMAINS}
    findings = detect_findings(entries, as_of=as_of, window=window)
    narrative = narrate_findings(findings, backend)
    return DoctorReport(
        period_start=start, period_end=as_of, directions=directions,
        findings=findings, narrative=narrative,
    )


def render_report_html(rep: DoctorReport) -> str:
    arrows = {"down": "↓", "up": "↑", "stable": "→"}
    dir_rows = "".join(
        f"<tr><td>{DOMAIN_LABELS[d]}</td><td>{arrows.get(rep.directions.get(d, 'stable'), '→')} "
        f"{rep.directions.get(d, 'stable')}</td></tr>"
        for d in DOMAINS
    )
    finding_items = "".join(f"<li>{html.escape(f.summary)}</li>" for f in rep.findings) or \
        "<li>No notable changes detected.</li>"
    return f"""<div style="font-family: system-ui; max-width: 720px;">
<h1>Doctor Report — What Changed</h1>
<p><b>Period:</b> {html.escape(rep.period_start)} to {html.escape(rep.period_end)}</p>
<h2>Summary</h2>
<p>{html.escape(rep.narrative)}</p>
<h2>Domain directions</h2>
<table border="1" cellpadding="6" cellspacing="0">{dir_rows}</table>
<h2>Notable findings</h2>
<ul>{finding_items}</ul>
<hr>
<p style="font-size: 12px; color: #555;">{rep.disclaimer}</p>
</div>"""
