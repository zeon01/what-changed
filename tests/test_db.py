from whatchanged.db import init_db, upsert_entry, get_entry, get_range
from whatchanged.models import Entry


def test_init_and_insert_and_get():
    conn = init_db(":memory:")
    upsert_entry(conn, Entry(date="2026-06-01", mobility=4, falls=1, note="hi"))
    got = get_entry(conn, "2026-06-01")
    assert got == Entry(date="2026-06-01", mobility=4, falls=1, note="hi")


def test_upsert_overwrites_same_date():
    conn = init_db(":memory:")
    upsert_entry(conn, Entry(date="2026-06-01", mobility=4))
    upsert_entry(conn, Entry(date="2026-06-01", mobility=2, note="worse"))
    got = get_entry(conn, "2026-06-01")
    assert got.mobility == 2 and got.note == "worse"


def test_get_missing_returns_none():
    conn = init_db(":memory:")
    assert get_entry(conn, "2026-01-01") is None


def test_get_range_is_sorted_and_inclusive():
    conn = init_db(":memory:")
    for d in ["2026-06-03", "2026-06-01", "2026-06-02", "2026-05-30"]:
        upsert_entry(conn, Entry(date=d))
    rng = get_range(conn, "2026-06-01", "2026-06-03")
    assert [e.date for e in rng] == ["2026-06-01", "2026-06-02", "2026-06-03"]


def test_connection_usable_from_worker_thread():
    import threading
    conn = init_db(":memory:")
    errors = []

    def worker():
        try:
            upsert_entry(conn, Entry(date="2026-06-01", mobility=3))
            assert get_entry(conn, "2026-06-01").mobility == 3
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert not errors, errors
