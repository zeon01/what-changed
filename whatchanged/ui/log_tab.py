from __future__ import annotations
import calendar
import datetime
import gradio as gr
from whatchanged.models import Entry, DOMAINS, EVENTS
from whatchanged.db import upsert_entry
from whatchanged.inference import extract_note
from whatchanged.trends import DOMAIN_LABELS, EVENT_LABELS

# Plain-language anchors so a caregiver knows what each 1-5 rating means (5 = best). These
# rate the *quality* of the day, not a number of hours/etc.
ANCHORS = {
    "mobility":  "1 = could barely move  ·  5 = moved freely & steadily",
    "tremor":    "1 = severe tremor  ·  5 = almost no tremor",
    "stiffness": "1 = very stiff & rigid  ·  5 = loose, not stiff",
    "mood":      "1 = very low  ·  5 = bright & cheerful",
    "sleep":     "1 = barely slept  ·  5 = slept very well",
    "alertness": "1 = very confused & foggy  ·  5 = sharp & clear",
}


def _mk_date(y, m, d) -> str:
    """Assemble YYYY-MM-DD from the Y/M/D dropdowns, clamping the day to the month's length
    (so e.g. Feb 31 -> Feb 28/29 instead of crashing the save)."""
    try:
        y, m, d = int(y), int(m), int(d)
        last = calendar.monthrange(y, m)[1]
        return datetime.date(y, m, min(max(1, d), last)).isoformat()
    except Exception:  # noqa: BLE001
        return datetime.date.today().isoformat()


def build_log_tab(conn, backend):
    with gr.Tab("📝 Log Today"):
        gr.Markdown(
            "### Just describe the day — the on-device AI fills the form, you review.\n"
            "Type a sentence or two about how your parent did today, then tap **Auto-fill**.")
        _t = datetime.date.today()
        with gr.Row():
            year = gr.Dropdown(list(range(_t.year, _t.year - 4, -1)), value=_t.year, label="Year")
            month = gr.Dropdown(
                [(datetime.date(2000, mo, 1).strftime("%B"), mo) for mo in range(1, 13)],
                value=_t.month, label="Month")
            day = gr.Dropdown(list(range(1, 32)), value=_t.day, label="Day")
        note = gr.Textbox(
            label="Today's note", lines=3, autofocus=True,
            placeholder='e.g. "Rough day — Dad froze in the doorway twice, his meds wore off '
                        'before lunch, and he barely slept."')
        audio = gr.Audio(sources=["microphone", "upload"], type="filepath",
                         label="🎤 Or speak the note (transcribed on-device)")
        transcribe_btn = gr.Button("📝 Transcribe to note")
        transcribe_status = gr.Markdown()
        extract_btn = gr.Button("✨  Auto-fill from note (local AI)",
                                 variant="primary", size="lg")
        extract_status = gr.Markdown()

        gr.Markdown("#### Review & adjust")
        sliders = {}
        for d in DOMAINS:
            sliders[d] = gr.Slider(1, 5, step=1, value=3, label=DOMAIN_LABELS[d],
                                   info=ANCHORS[d])

        gr.Markdown("**Events today** — how many times each happened")
        counters = {}
        with gr.Row():
            for ev in EVENTS[:3]:
                counters[ev] = gr.Number(value=0, precision=0, label=EVENT_LABELS[ev], minimum=0)
        with gr.Row():
            for ev in EVENTS[3:]:
                counters[ev] = gr.Number(value=0, precision=0, label=EVENT_LABELS[ev], minimum=0)

        save_btn = gr.Button("Save today", variant="primary")
        save_status = gr.Markdown()

        def do_transcribe(audio_path):
            if not audio_path:
                return gr.update(), "Record or upload audio first."
            try:
                from whatchanged.stt import transcribe
                text = transcribe(audio_path)
            except Exception as e:  # noqa: BLE001 — voice is optional; typing always works
                return gr.update(), f"⚠️ Voice unavailable ({type(e).__name__}). Type the note instead."
            if not text:
                return gr.update(), "Couldn't make out any speech — try again or type it."
            return text, "✓ Transcribed — review/edit the note above, then tap Auto-fill."

        transcribe_btn.click(do_transcribe, inputs=[audio], outputs=[note, transcribe_status])

        def do_extract(note_text, *slider_vals):
            data = extract_note(note_text, backend)
            new = list(slider_vals)
            for i, d in enumerate(DOMAINS):
                if d in data:
                    new[i] = data[d]
            counter_updates = [data.get(ev, gr.update()) for ev in EVENTS]
            if backend is None:
                msg = "⚠️ The local model isn't loaded right now — enter values manually below."
            elif not note_text.strip():
                msg = "Write a note above first, then tap Auto-fill."
            else:
                msg = f"✓ Filled **{len(data)}** field(s) from your note — please review below."
            return (*new, *counter_updates, msg)

        extract_btn.click(
            do_extract,
            inputs=[note] + [sliders[d] for d in DOMAINS],
            outputs=[sliders[d] for d in DOMAINS] + [counters[ev] for ev in EVENTS]
                    + [extract_status],
        )

        def do_save(y, m, d, note_text, *vals):
            date_val = _mk_date(y, m, d)
            n = len(DOMAINS)
            domain_vals, event_vals = vals[:n], vals[n:]

            def _to_int(v):
                return int(v) if v not in (None, "") else 0

            entry = Entry(date=date_val, note=note_text,
                          **{d: _to_int(v) for d, v in zip(DOMAINS, domain_vals)},
                          **{ev: _to_int(v) for ev, v in zip(EVENTS, event_vals)})
            upsert_entry(conn, entry)
            return f"✅ Saved {date_val}."

        save_btn.click(
            do_save,
            inputs=[year, month, day, note] + [sliders[d] for d in DOMAINS]
                   + [counters[ev] for ev in EVENTS],
            outputs=[save_status],
        )
