from __future__ import annotations
import functools
import os

# faster-whisper model name; "base" is a good CPU speed/accuracy balance. Override via env.
WHISPER_MODEL = os.environ.get("WC_WHISPER_MODEL", "base")


@functools.lru_cache(maxsize=1)
def _model():
    from faster_whisper import WhisperModel  # lazy: importing this module never needs the lib
    return WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")


def transcribe(audio_path: str | None) -> str:
    """On-device speech-to-text for a spoken caregiver note. Returns '' for no audio.
    Raises if faster-whisper isn't installed / the model can't load — the UI catches that
    and degrades gracefully (the user can always type instead)."""
    if not audio_path:
        return ""
    segments, _info = _model().transcribe(audio_path, language="en", vad_filter=True)
    return " ".join(seg.text.strip() for seg in segments).strip()
