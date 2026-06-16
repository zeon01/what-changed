from __future__ import annotations


def field_f1(pred: dict, gold: dict) -> tuple[float, float, float]:
    """A (key, value) pair is a true positive only if both key and value match.
    Returns (precision, recall, f1). Two empty dicts score a perfect 1.0."""
    pred_pairs = set(pred.items())
    gold_pairs = set(gold.items())
    if not pred_pairs and not gold_pairs:
        return (1.0, 1.0, 1.0)
    tp = len(pred_pairs & gold_pairs)
    precision = tp / len(pred_pairs) if pred_pairs else 0.0
    recall = tp / len(gold_pairs) if gold_pairs else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return (precision, recall, f1)


def dataset_f1(preds: list[dict], golds: list[dict]) -> float:
    """Macro-average F1 over the dataset."""
    if not preds:
        return 0.0
    return sum(field_f1(p, g)[2] for p, g in zip(preds, golds)) / len(preds)


def field_f1_tol(pred: dict, gold: dict, tol: int = 1) -> tuple[float, float, float]:
    """Like field_f1 but a field is a true positive when the key matches AND the values
    are within `tol`. A 4-vs-5 rating counts as correct — exact 1-5 calibration is
    subjective and irrelevant to trend detection (the app's actual job)."""
    pred_keys, gold_keys = set(pred), set(gold)
    if not pred_keys and not gold_keys:
        return (1.0, 1.0, 1.0)
    tp = sum(1 for k in (pred_keys & gold_keys) if abs(pred[k] - gold[k]) <= tol)
    precision = tp / len(pred_keys) if pred_keys else 0.0
    recall = tp / len(gold_keys) if gold_keys else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return (precision, recall, f1)


def dataset_f1_tol(preds: list[dict], golds: list[dict], tol: int = 1) -> float:
    if not preds:
        return 0.0
    return sum(field_f1_tol(p, g, tol)[2] for p, g in zip(preds, golds)) / len(preds)
