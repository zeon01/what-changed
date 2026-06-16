from __future__ import annotations
import datetime
import gradio as gr
from whatchanged.db import get_range
from whatchanged.charts import domain_figure, events_figure
from whatchanged.trends import detect_findings


def build_trends_tab(conn):
    with gr.Tab("📈 Trends"):
        def render(win):
            today = datetime.date.today()
            start = today - datetime.timedelta(days=int(win) - 1)
            entries = get_range(conn, str(start), str(today))
            findings = detect_findings(entries, as_of=str(today), window=int(win))
            if findings:
                md = "### What changed\n" + "\n".join(f"- {f.summary}" for f in findings)
            else:
                md = "### What changed\nNo notable changes in this window."
            return domain_figure(entries), events_figure(entries), md

        # Populate on first open (conn is already seeded by the time the tab is built).
        init_dom, init_ev, init_md = render(30)

        window = gr.Radio([7, 30, 90], value=30, label="Window (days)")
        refresh = gr.Button("Refresh", variant="primary")
        plot = gr.Plot(value=init_dom, label="Symptom ratings")
        events_plot = gr.Plot(value=init_ev, label="Event counts")
        callouts = gr.Markdown(value=init_md)

        refresh.click(render, inputs=[window], outputs=[plot, events_plot, callouts])
        window.change(render, inputs=[window], outputs=[plot, events_plot, callouts])
