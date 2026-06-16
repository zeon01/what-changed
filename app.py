from __future__ import annotations
import os
import gradio as gr
from whatchanged.db import init_db
from whatchanged.inference import LlamaCppBackend
from whatchanged.ui.log_tab import build_log_tab
from whatchanged.ui.trends_tab import build_trends_tab
from whatchanged.ui.report_tab import build_report_tab

DB_PATH = os.environ.get("WC_DB", "data/whatchanged.db")
MODEL_PATH = os.environ.get("WC_MODEL", "")  # local GGUF path (wins if it exists)
HF_REPO = os.environ.get("WC_HF_REPO", "zeon01/what-changed-1b")  # else pull from the Hub
HF_FILE = os.environ.get("WC_HF_FILE", "MiniCPM5-1B.Q8_0.gguf")


def resolve_model_path() -> str:
    """Local WC_MODEL wins; otherwise download the published GGUF from the Hub (public,
    no token needed) so the Space is self-contained on first boot."""
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        return MODEL_PATH
    if HF_REPO:
        try:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(repo_id=HF_REPO, filename=HF_FILE)
        except Exception as e:  # noqa: BLE001
            print(f"[model] could not fetch {HF_REPO}/{HF_FILE}: {e}")
    return ""


def make_backend():
    path = resolve_model_path()
    if not path:
        return None
    # Pin threads (WC_THREADS) so llama.cpp doesn't oversubscribe the host's core count on a
    # 2-vCPU Space (the usual cause of absurdly slow CPU inference).
    backend = LlamaCppBackend(path, n_threads=(int(os.environ.get("WC_THREADS", "0")) or None))
    try:
        backend._ensure()  # warm-load now so a bad wheel/model degrades, never crashes mid-click
    except Exception as e:  # noqa: BLE001 — a model load failure must not 500 the UI
        print(f"[model] load failed ({type(e).__name__}: {e}); using deterministic fallback")
        return None
    return backend
    # When None: extraction is a no-op and the report narrative uses a deterministic
    # fallback (bullet summaries joined into a plain sentence).


def maybe_seed(conn) -> None:
    """On the demo Space (WC_SEED_DEMO=1) only, populate an empty DB with a realistic
    60-day decline arc so Trends/Report are non-empty on first open."""
    if os.environ.get("WC_SEED_DEMO") != "1":
        return
    if conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0] > 0:
        return
    try:
        from scripts.seed_demo import seed
        seed(conn)
        print("[seed] demo data loaded")
    except Exception as e:  # noqa: BLE001
        print(f"[seed] skipped: {e}")


# --- Off-Brand custom look: calm health palette, Inter, branded header, no Gradio footer ---
THEME = gr.themes.Soft(
    primary_hue="teal", secondary_hue="cyan", neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)
CSS = """
.gradio-container {max-width: 880px !important; margin: 0 auto !important;}
#wc-header {background: linear-gradient(135deg,#0d9488,#0e7490); color:#fff;
  padding:22px 26px; border-radius:16px; margin-bottom:6px;}
#wc-header h1 {margin:0; font-size:26px; font-weight:700; color:#fff;}
#wc-header p {margin:6px 0 0; opacity:.92; font-size:14px; color:#fff;}
#wc-header .wc-pill {display:inline-block; background:rgba(255,255,255,.18);
  padding:3px 11px; border-radius:999px; font-size:12px; margin-top:11px;}
footer {display:none !important;}
"""
HEADER = (
    '<div id="wc-header"><h1>📋 What Changed</h1>'
    "<p>A private, on-device Parkinson's diary — describe the day, and a local fine-tuned 1B "
    "structures it into a doctor-ready summary.</p>"
    '<span class="wc-pill">Runs 100% locally · fine-tuned 1B · llama.cpp · not medical advice</span>'
    "</div>"
)


def build_app():
    conn = init_db(DB_PATH)
    maybe_seed(conn)
    backend = make_backend()
    with gr.Blocks(title="What Changed", theme=THEME, css=CSS) as demo:
        gr.HTML(HEADER)
        build_log_tab(conn, backend)
        build_trends_tab(conn)
        build_report_tab(conn, backend)
    return demo


if __name__ == "__main__":
    build_app().launch()
