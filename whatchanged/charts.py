from __future__ import annotations
import datetime
from matplotlib.figure import Figure
from whatchanged.models import DOMAINS, EVENTS
from whatchanged.trends import DOMAIN_LABELS, EVENT_LABELS

# Trend colors (ratings only): red = deteriorating, green = improving, grey = no clear trend.
RED, GREEN, GREY = "#c0392b", "#2e8b57", "#7f8c8d"
NEUTRAL = "#5b7a9d"  # event bars — counts are discrete tallies, the shape shows the trend


def _d(s: str) -> datetime.date:
    return datetime.date.fromisoformat(s)


def _rolling(ys, w=7):
    """Trailing simple moving average (min_periods=1), ignoring None."""
    out = []
    for i in range(len(ys)):
        seg = [y for y in ys[max(0, i - w + 1): i + 1] if y is not None]
        out.append(sum(seg) / len(seg) if seg else None)
    return out


def _trend_color(series, higher_is_better, thresh, relative=False):
    """First-third vs last-third of the series -> directional color. With relative=True the
    move must clear max(thresh, 50% of the starting level), so count noise stays grey."""
    vals = [v for v in series if v is not None]
    if len(vals) < 2:
        return GREY
    half = max(1, len(vals) // 3)
    first, last = sum(vals[:half]) / half, sum(vals[-half:]) / half
    delta = last - first
    floor = max(thresh, 0.5 * first) if relative else thresh
    up, down = delta >= floor, delta <= -floor
    better = up if higher_is_better else down
    worse = down if higher_is_better else up
    return GREEN if better else (RED if worse else GREY)


def _sparse_dates(ax, dates):
    if not dates:
        return
    step = max(1, len(dates) // 3)
    ax.set_xticks(dates[::step])
    ax.tick_params(axis="x", rotation=45, labelsize=7)


def _empty(fig, msg):
    ax = fig.subplots()
    ax.text(0.5, 0.5, msg, ha="center", va="center")
    ax.axis("off")
    return fig


def domain_figure(entries):
    """Small multiples, one panel per symptom rating. Bold line = 7-day smoothed trend
    (red = declining, green = improving, grey = flat); faint line behind = raw daily."""
    entries = sorted(entries, key=lambda e: e.date)
    fig = Figure(figsize=(9.5, 5.2))
    if not entries:
        return _empty(fig, "No data yet — log a few days")
    dates = [_d(e.date) for e in entries]
    fig.suptitle("Symptom ratings — trend per domain (5 = best, smoothed 7-day)",
                 fontsize=11, fontweight="bold")
    axes = fig.subplots(2, 3).ravel()
    for ax, dom in zip(axes, DOMAINS):
        ys = [getattr(e, dom) for e in entries]
        roll = _rolling(ys)
        ax.plot(dates, ys, color=GREY, alpha=0.25, lw=1.0)            # raw (noise)
        ax.plot(dates, roll, color=_trend_color(roll, True, 0.4), lw=2.4)  # smoothed trend
        ax.set_title(DOMAIN_LABELS[dom], fontsize=9)
        ax.set_ylim(0.5, 5.5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(alpha=0.18)
        _sparse_dates(ax, dates)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _binned(entries, ev, bin_days):
    start = _d(entries[0].date)
    buckets: dict[int, int] = {}
    for e in entries:
        b = (_d(e.date) - start).days // bin_days
        buckets[b] = buckets.get(b, 0) + (getattr(e, ev) or 0)
    keys = sorted(buckets)
    xs = [start + datetime.timedelta(days=b * bin_days) for b in keys]
    ys = [buckets[b] for b in keys]
    return xs, ys


def events_figure(entries):
    """Small multiples, one panel per event type. Counts binned per day (<=14d window) or
    per week, drawn as neutral bars — the bar shape, not a color, conveys the trend."""
    entries = sorted(entries, key=lambda e: e.date)
    fig = Figure(figsize=(9.5, 5.2))
    if not entries:
        return _empty(fig, "No data yet — log a few days")
    span = (_d(entries[-1].date) - _d(entries[0].date)).days if len(entries) > 1 else 1
    bin_days = 1 if span <= 14 else 7
    unit = "day" if bin_days == 1 else "week"
    fig.suptitle(f"Event counts — per {unit} (total in each title)",
                 fontsize=11, fontweight="bold")
    axes = fig.subplots(2, 3).ravel()
    for ax, ev in zip(axes, EVENTS):
        xs, ys = _binned(entries, ev, bin_days)
        ax.bar(xs, ys, width=bin_days * 0.85, color=NEUTRAL, alpha=0.9)
        ax.set_title(f"{EVENT_LABELS[ev]}  (total {sum(ys)})", fontsize=9)
        ax.set_ylim(0, max(3, (max(ys) if ys else 0) + 1))
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(alpha=0.18, axis="y")
        _sparse_dates(ax, xs)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig
