from whatchanged.inference import (
    LLMBackend, FakeBackend, parse_extraction, extract_note,
    build_extract_prompt, narrate_findings,
)
from whatchanged.models import Finding


def test_parse_extraction_handles_code_fence_and_extra_text():
    raw = 'Sure!\n```json\n{"tremor": 2, "falls": 1}\n```\nHope that helps'
    out = parse_extraction(raw)
    assert out == {"tremor": 2, "falls": 1}


def test_parse_extraction_clamps_and_drops_unknown_keys():
    raw = '{"tremor": 9, "mobility": 0, "falls": -1, "bogus": 5, "note": "x"}'
    out = parse_extraction(raw)
    assert out["tremor"] == 5        # clamped to 1..5
    assert out["mobility"] == 1      # clamped to 1..5
    assert out["falls"] == 0         # events clamped to >=0
    assert "bogus" not in out and "note" not in out


def test_parse_extraction_bad_json_returns_empty():
    assert parse_extraction("no json here") == {}


def test_extract_note_uses_backend_output():
    fake = FakeBackend('{"tremor": 2, "falls": 1}')
    out = extract_note("trembling, had a stumble", fake)
    assert out == {"tremor": 2, "falls": 1}
    assert "trembling" in fake.last_prompt  # the note is embedded in the prompt


def test_build_extract_prompt_lists_schema_keys():
    p = build_extract_prompt("slept poorly")
    assert "mobility" in p and "falls" in p and "slept poorly" in p


def test_narrate_findings_passes_findings_and_returns_text():
    fake = FakeBackend("Tremor and sleep have been trending down.")
    findings = [Finding(kind="direction", domain="tremor", summary="tremor down")]
    text = narrate_findings(findings, fake)
    assert text == "Tremor and sleep have been trending down."
    assert "tremor down" in fake.last_prompt


def test_narrate_findings_empty_short_circuits_without_calling_model():
    fake = FakeBackend("SHOULD NOT BE USED")
    text = narrate_findings([], fake)
    assert "no notable changes" in text.lower()
    assert fake.last_prompt is None  # model not called


def test_llamacpp_backend_importable():
    # The class must exist and be constructable lazily (no model load at import time).
    from whatchanged.inference import LlamaCppBackend
    assert callable(LlamaCppBackend)


def test_narrate_findings_none_backend_returns_deterministic_fallback():
    findings = [Finding(kind="direction", domain="tremor", summary="tremor trending down")]
    text = narrate_findings(findings, None)
    assert "tremor trending down" in text
    assert "{}" not in text


def test_extract_note_none_backend_returns_empty():
    assert extract_note("barely ate today", None) == {}
