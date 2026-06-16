from sft.metric import field_f1, dataset_f1, field_f1_tol, dataset_f1_tol


def test_field_f1_tol_allows_off_by_one():
    assert field_f1_tol({"mobility": 4}, {"mobility": 5}) == (1.0, 1.0, 1.0)


def test_field_f1_tol_rejects_beyond_tolerance():
    assert field_f1_tol({"mobility": 2}, {"mobility": 5})[2] == 0.0


def test_field_f1_tol_still_penalizes_extra_and_missing_keys():
    # extra key (not in gold) hurts precision; off-by-one value on the shared key is fine
    p, r, f = field_f1_tol({"mobility": 5, "tremor": 3}, {"mobility": 4})
    assert round(p, 2) == 0.5 and r == 1.0


def test_dataset_f1_tol_averages():
    assert dataset_f1_tol([{"mobility": 5}], [{"mobility": 4}]) == 1.0


def test_field_f1_perfect():
    assert field_f1({"appetite": 2, "falls": 1}, {"appetite": 2, "falls": 1}) == (1.0, 1.0, 1.0)


def test_field_f1_wrong_value_counts_as_miss():
    _, _, f = field_f1({"appetite": 5}, {"appetite": 2})
    assert f == 0.0


def test_field_f1_partial():
    p, r, f = field_f1({"appetite": 2, "sleep": 3}, {"appetite": 2, "falls": 1})
    assert round(p, 2) == 0.5 and round(r, 2) == 0.5 and round(f, 2) == 0.5


def test_both_empty_is_perfect():
    assert field_f1({}, {}) == (1.0, 1.0, 1.0)


def test_dataset_f1_averages():
    assert dataset_f1([{"appetite": 2}, {}], [{"appetite": 2}, {}]) == 1.0
