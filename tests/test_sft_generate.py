import json
from sft.generate import build_teacher_prompt, generate


def test_build_teacher_prompt_includes_spec():
    p = build_teacher_prompt("appetite/eating rated 2 out of 5")
    assert "appetite/eating rated 2 out of 5" in p
    assert "caregiver" in p.lower()


def test_generate_writes_jsonl_with_stub_teacher(tmp_path):
    out = tmp_path / "d.jsonl"

    def stub(prompt, max_tokens=80):
        return '  "barely ate today, a bit confused"  '

    generate(5, str(out), seed=0, teacher=stub)
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 5
    for line in lines:
        row = json.loads(line)
        assert set(row) == {"note", "label"}
        assert isinstance(row["label"], dict)
        # note is stripped of whitespace and wrapping quotes
        assert row["note"] == "barely ate today, a bit confused"
