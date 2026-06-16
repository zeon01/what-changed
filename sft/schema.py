from __future__ import annotations
import random
from whatchanged.models import DOMAINS, EVENTS
from whatchanged.trends import DOMAIN_LABELS, EVENT_LABELS


def sample_label(rng: random.Random) -> dict:
    """Sample a ground-truth label: a random subset of domains (1-5) and events (>=0).
    ~15% of samples are empty to teach the 'nothing notable' case."""
    label: dict = {}
    for d in DOMAINS:
        if rng.random() < 0.5:
            label[d] = rng.randint(1, 5)
    for ev in EVENTS:
        if rng.random() < 0.12:
            label[ev] = rng.randint(1, 2)
    if not label:  # never empty — guarantee at least one field
        label[rng.choice(DOMAINS)] = rng.randint(1, 5)
    return label


# Per-domain, unambiguous level phrasing. A generic "very poor"->"very good" scale is
# ambiguous for SYMPTOM-named domains (a "poor tremor" reads as 'little tremor' = good),
# which made the teacher invert tremor/stiffness. These phrases remove that ambiguity.
_PHRASE = {
    "mobility":  {1: "could barely move or walk", 2: "moved with difficulty",
                  3: "moved okay", 4: "moved fairly well", 5: "moved freely and steadily"},
    "tremor":    {1: "a severe tremor", 2: "a bad tremor", 3: "a moderate tremor",
                  4: "only a slight tremor", 5: "almost no tremor"},
    "stiffness": {1: "very stiff and rigid", 2: "quite stiff", 3: "moderately stiff",
                  4: "only a little stiff", 5: "loose, not stiff at all"},
    "mood":      {1: "a very low mood", 2: "a low mood", 3: "an okay mood",
                  4: "a good mood", 5: "a bright, cheerful mood"},
    "sleep":     {1: "barely slept", 2: "slept poorly", 3: "slept okay",
                  4: "slept fairly well", 5: "slept very well"},
    "alertness": {1: "very confused and foggy", 2: "often confused",
                  3: "reasonably clear-headed", 4: "mostly sharp", 5: "sharp and clear"},
}


def label_to_spec(label: dict) -> str:
    """Human-readable spec of the label, used to instruct the teacher what to write.
    Ratings are rendered with valence words (5 = best) so the teacher writes the correct
    direction (e.g. appetite 1 -> 'barely ate', not 'ate well')."""
    if not label:
        return "nothing was recorded today"
    parts = []
    for k, v in label.items():
        if k in DOMAINS:
            parts.append(_PHRASE[k][v])
        else:
            parts.append(f"{v} {EVENT_LABELS[k]}")
    return "; ".join(parts)
