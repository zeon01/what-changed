import random
from sft.schema import sample_label, label_to_spec
from whatchanged.models import DOMAINS, EVENTS


def test_sample_label_keys_in_schema_and_ranges():
    rng = random.Random(0)
    for _ in range(200):
        lab = sample_label(rng)
        for k, v in lab.items():
            assert k in DOMAINS or k in EVENTS
            if k in DOMAINS:
                assert 1 <= v <= 5
            else:
                assert v >= 0


def test_sample_label_is_never_empty_and_can_be_multi():
    rng = random.Random(1)
    labels = [sample_label(rng) for _ in range(100)]
    assert all(lab for lab in labels)  # empty class dropped — always >= 1 field
    assert any(len(lab) >= 2 for lab in labels)


def test_label_to_spec_mentions_each_field():
    spec = label_to_spec({"tremor": 2, "falls": 1})
    assert "tremor" in spec                  # rendered as a natural phrase ("a bad tremor")
    assert "falls" in spec and "1" in spec   # events keep their count
