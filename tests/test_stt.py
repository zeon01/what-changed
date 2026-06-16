from whatchanged.stt import transcribe


def test_no_audio_returns_empty():
    # The early-return guard means this path needs neither faster-whisper nor a model.
    assert transcribe(None) == ""
    assert transcribe("") == ""
