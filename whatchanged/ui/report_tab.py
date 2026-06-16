from __future__ import annotations
import datetime
import gradio as gr
from whatchanged.db import get_range
from whatchanged.report import build_report, render_report_html


def build_report_tab(conn, backend):
    with gr.Tab("🩺 Doctor Report"):
        gr.Markdown("### Generate a one-page summary to share with the doctor.")
        with gr.Row():
            start = gr.Textbox(
                label="From",
                value=str(datetime.date.today() - datetime.timedelta(days=30)))
            end = gr.Textbox(label="To", value=str(datetime.date.today()))
        ai_summary = gr.Checkbox(
            value=False,
            label="Write the summary with the on-device AI (slower — ~20s on the free CPU)")
        gen = gr.Button("Generate report", variant="primary")
        out = gr.HTML()

        def do_generate(start_val, end_val, use_ai):
            entries = get_range(conn, start_val, end_val)
            # Default = instant deterministic summary; opt-in = AI-written narrative.
            rep = build_report(entries, as_of=end_val, backend=(backend if use_ai else None))
            return render_report_html(rep)

        gen.click(do_generate, inputs=[start, end, ai_summary], outputs=[out])
