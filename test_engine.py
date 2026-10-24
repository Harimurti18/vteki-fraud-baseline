"""Tests: duplicates, missing identifiers, TIME LEAKAGE, determinism, rule semantics, CLI contract."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import generate_data as gen
import scoring_engine as se

BASE = pd.Timestamp("2026-09-25 12:00")          # after ANOMALY["train_end"] -> IsolationForest inactive in small cases


def iso(minutes=0):
    return (BASE + pd.Timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_dir(tmp_path, logins=(), txns=(), customers=None):
    """Write a tiny dataset. logins: (id, cust, dev, min, ok, ingested_min)  txns: (id, cust, dev, amount, min, ingested_min)."""
    cust_ids = sorted({1, *[r[1] for r in logins], *[r[1] for r in txns]}) if customers is None else customers
    pd.DataFrame({"customer_id": [f"C{c:04d}" for c in cust_ids], "registered_at": "2026-01-01T00:00:00Z"}).to_csv(tmp_path / "customers.csv", index=False)
    pd.DataFrame([(i, f"CUST-{c:04d}", d, iso(m), ok, iso(g)) for i, c, d, m, ok, g in logins],
                 columns=["event_id", "customer_id", "device_id", "event_time", "success", "ingested_at"]).to_csv(tmp_path / "login_events.csv", index=False)
    pd.DataFrame([(i, str(c), d, a, iso(m), iso(g)) for i, c, d, a, m, g in txns],
                 columns=["transaction_id", "customer_id", "device_id", "amount", "event_time", "ingested_at"]).to_csv(tmp_path / "transactions.csv", index=False)
    return tmp_path


def feats(tmp_path, tid="T0", **kw):
    return se.Engine(make_dir(tmp_path, **kw)).score(tid)


# ----------------------------------------------------------------------------- validation
def test_exact_duplicates_and_replays_are_removed_earliest_copy_wins(tmp_path):
    d = make_dir(tmp_path, txns=[("T1", 1, "d1", 1000, 0, 0), ("T1", 1, "d1", 1000, 0, 0),
                                 ("T2", 1, "d1", 1000, 5, 90), ("T2", 1, "d1", 1000, 5, 6)])
    clean, rep = se.load_and_validate_data(d)
    tx = clean["transactions"].set_index("transaction_id")
    assert tx.index.is_unique and len(tx) == 2
    assert rep["transactions"]["duplicates_removed"] == 2
    assert tx.loc["T2", "ingested_at"] == pd.Timestamp(iso(6)).tz_localize(None)


def test_mixed_timestamp_formats_parse_to_the_same_instant():
    s = pd.Series(["2026-09-25T10:00:00Z", "25/09/2026 17:00", str(int(pd.Timestamp("2026-09-25 10:00").timestamp()))])
    assert (se.parse_ts(s) == pd.Timestamp("2026-09-25 10:00")).all()


def test_customer_id_formats_are_canonicalised():
    s = pd.Series(["C0012", "CUST-0012", "cust_12", "12", "x", ""])
    assert list(se.canonical_id(s).iloc[:4]) == [12] * 4 and se.canonical_id(s).iloc[4:].isna().all()


def test_bad_timestamp_and_orphan_customer_are_quarantined_not_crashing(tmp_path):
    d = make_dir(tmp_path, txns=[("T0", 1, "d1", 1000, 0, 0), ("T1", 1, "d1", 1000, 1, 1), ("T2", 999, "d1", 1000, 2, 2)], customers=[1])
    t = pd.read_csv(d / "transactions.csv", dtype=str)
    t.loc[t.transaction_id == "T1", "event_time"] = "not-a-date"
    t.to_csv(d / "transactions.csv", index=False)
    clean, rep = se.load_and_validate_data(d)
    assert rep["transactions"]["quarantined_invalid"] == 1 and rep["transactions"]["quarantined_orphan"] == 1
    assert list(clean["transactions"]["transaction_id"]) == ["T0"]
    assert "error" in se.Engine(d).score("T1")


def test_missing_device_id_stays_missing_and_is_not_a_shared_unknown_device(tmp_path):
    e = se.Engine(make_dir(tmp_path, txns=[("T0", 1, None, 2_000_000, 0, 0), ("T1", 2, None, 2_000_000, 1, 1)]))
    r = e.score("T0")
    assert r["evidence"]["device_id"] is None and r["evidence"]["device_n_customers"] is None
    assert r["evidence"]["device_age_hours"] is None and r["reasons"] == []      # absence of a device is not evidence


# ----------------------------------------------------------------------------- time leakage
def test_future_rows_and_late_arriving_rows_are_not_used(tmp_path):
    r = feats(tmp_path, logins=[("L1", 1, "d1", -10, False, -9),    # known failed login -> counts
                                ("L2", 1, "d1", -5, False, 3),      # happened before t but ingested AFTER t -> unknown
                                ("L3", 1, "d1", 5, False, 6),       # future
                                ("L4", 1, "d1", -120, False, -119)],  # outside 1h window
              txns=[("T0", 1, "d1", 50_000, 0, 0), ("T9", 1, "d1", 50_000, -3, 4), ("T8", 1, "d1", 50_000, 10, 10)])
    assert r["evidence"]["failed_logins_1h"] == 1 and r["evidence"]["txn_count_1h"] == 0


def test_transaction_is_excluded_from_its_own_history(tmp_path):
    e = se.Engine(make_dir(tmp_path, txns=[("T0", 1, "d1", 1000, 0, 0), ("T1", 1, "d1", 1000, 10, 10)]))
    assert e.score("T0")["evidence"]["txn_count_1h"] == 0 and e.score("T1")["evidence"]["txn_count_1h"] == 1


def test_device_cold_start_guard_and_new_device_rule(tmp_path):
    young = feats(tmp_path / "a" if (tmp_path / "a").mkdir() is None else tmp_path, txns=[("T0", 1, "dNEW", 2_000_000, 0, 0)])
    assert "NEW_DEVICE_HIGH_VALUE" not in young["reasons"]                       # no history -> cannot call it "new"
    old_dir = tmp_path / "b"; old_dir.mkdir()
    old = feats(old_dir, logins=[("L1", 1, "d1", -60 * 24 * 10, True, -60 * 24 * 10)], txns=[("T0", 1, "dNEW", 2_000_000, 0, 0)])
    assert "NEW_DEVICE_HIGH_VALUE" in old["reasons"] and old["evidence"]["device_age_hours"] == 0.0


def test_properties_future_data_never_changes_past_features():
    """THE leakage test: remove everything unknown at cut-off T, features of earlier transactions must be identical."""
    full = se.Engine(DATA)
    T = pd.Timestamp("2026-09-11").value                       # cuts through the first ring's activity (days 8-13)
    cut = {k: v[v["_g"] <= T].reset_index(drop=True) if "_g" in v else v for k, v in full.clean.items()}
    part = se.Engine(clean=cut)
    ids = [t for t in part.tx.index if part.tx.loc[t, "_e"] <= T]    # every transaction, no sampling
    assert len(ids) > 500
    for t in ids:
        assert full.features(full.tx.loc[t]) == part.features(part.tx.loc[t]), t


def test_isolation_forest_is_not_fit_on_the_period_it_scores():
    """Regression test for the earlier draft: model fit on ALL rows made today's flag depend on tomorrow's data."""
    full = se.Engine(DATA)
    T = pd.Timestamp("2026-09-26").value
    cut = {k: v[v["_g"] <= T].reset_index(drop=True) if "_g" in v else v for k, v in full.clean.items()}
    part = se.Engine(clean=cut)
    ids = [t for t in part.tx.index if pd.Timestamp("2026-09-20").value < part.tx.loc[t, "_e"] <= T][::5]
    assert len(ids) > 50
    for t in ids:
        a = full.anomaly_percentile(full.tx.loc[t], full.features(full.tx.loc[t]))
        b = part.anomaly_percentile(part.tx.loc[t], part.features(part.tx.loc[t]))
        assert a is not None and a == b, t


def test_anomaly_signal_is_unavailable_inside_training_window():
    e = se.Engine(DATA)
    row = e.tx[e.tx["_e"] < pd.Timestamp(se.ANOMALY["train_end"]).value].iloc[0]
    assert e.score(row["transaction_id"])["evidence"]["anomaly_percentile"] is None


# ----------------------------------------------------------------------------- rule semantics
def test_household_sharing_one_device_is_not_flagged(tmp_path):
    r = feats(tmp_path, logins=[(f"L{c}", c, "dHome", -60, True, -60) for c in (1, 2, 3)],
              txns=[("T0", 1, "dHome", 20_000, 0, 0)])
    assert r["evidence"]["device_n_customers"] == 3 and r["risk_band"] == "LOW" and r["reasons"] == []


def test_device_shared_by_many_accounts_is_flagged(tmp_path):
    r = feats(tmp_path, logins=[(f"L{c}", c, "dRing", -60, True, -60) for c in range(1, 7)],
              txns=[("T0", 1, "dRing", 20_000, 0, 0)])
    assert r["evidence"]["device_n_customers"] == 6 and r["reasons"] == ["DEVICE_MANY_ACCOUNTS"]


def test_device_sharing_count_ignores_future_and_late_users(tmp_path):
    r = feats(tmp_path, logins=[("L2", 2, "dS", -60, True, -59),       # known before t -> counts
                                ("L3", 3, "dS", 30, True, 31),         # future user
                                ("L4", 4, "dS", -30, True, 20)],       # used it before t but ingested after t
              txns=[("T0", 1, "dS", 1000, 0, 0)])
    assert r["evidence"]["device_n_customers"] == 2                    # customer 1 itself + customer 2


def test_reasons_are_ordered_by_points_and_score_is_numeric(tmp_path):
    r = feats(tmp_path, logins=[(f"L{i}", 1, "dA", -5 - i, False, -5 - i) for i in range(4)] + [("L9", 1, "dOld", -60 * 24 * 10, True, -60 * 24 * 10)],
              txns=[("T0", 1, "dA", 2_000_000, 0, 0)])
    assert r["reasons"][:2] == ["NEW_DEVICE_HIGH_VALUE", "LOGIN_FAILURE_BURST"] and r["risk_score"] == 55 and r["risk_band"] == "HIGH"
    assert isinstance(r["risk_score"], int)


def test_decision_version_changes_when_a_rule_changes(monkeypatch):
    v1 = se.decision_version()
    assert v1 == se.decision_version()
    changed = json.loads(json.dumps(se.RULES))
    changed[1]["all"][0][2] = 2                                # LOGIN_FAILURE_BURST threshold 3 -> 2
    monkeypatch.setattr(se, "RULES", changed)
    assert se.decision_version() != v1


# ----------------------------------------------------------------------------- determinism and CLI
def test_scores_do_not_depend_on_row_order(tmp_path):
    dst = tmp_path / "shuf"; shutil.copytree(DATA, dst)
    for f in ("customers", "login_events", "transactions"):
        pd.read_csv(dst / f"{f}.csv", dtype=str).sample(frac=1.0, random_state=3).to_csv(dst / f"{f}.csv", index=False)
    a, b = se.Engine(DATA), se.Engine(dst)
    for t in list(a.tx.index)[::25]:
        assert a.score(t) == b.score(t), t


def test_output_is_deterministic_and_schema_is_stable():
    t = se.Engine(DATA).tx.index[100]
    r1, r2 = se.Engine(DATA).score(t), se.Engine(DATA).score(t)
    assert r1 == r2 and {"transaction_id", "as_of", "risk_score", "risk_band", "reasons", "decision_version", "evidence"} <= set(r1)
    json.dumps(r1)                                              # native Python types only


def test_cli_works_from_any_directory_and_matches_in_process_result(tmp_path):
    t = se.Engine(DATA).tx.index[100]
    out = subprocess.run([sys.executable, str(Path(se.__file__)), t, "--data-dir", str(DATA)], cwd=tmp_path, capture_output=True, text=True)
    assert out.returncode == 0 and json.loads(out.stdout) == se.Engine(DATA).score(t)
    assert "error" in json.loads(subprocess.run([sys.executable, str(Path(se.__file__)), "NOPE", "--data-dir", str(DATA)],
                                                cwd=tmp_path, capture_output=True, text=True).stdout)


def test_ground_truth_is_never_needed(tmp_path):
    dst = tmp_path / "nt"; shutil.copytree(DATA, dst); (dst / "_truth.csv").unlink()
    e = se.Engine(dst)
    assert not {"is_fraud", "scenario"} & set(e.tx.columns)
    assert e.score(e.tx.index[0])["decision_version"] == se.decision_version()


# ----------------------------------------------------------------------------- shared synthetic dataset
@pytest.fixture(scope="session", autouse=True)
def _dataset(tmp_path_factory):
    global DATA
    DATA = tmp_path_factory.mktemp("synthetic")
    old = gen.OUT
    gen.OUT = DATA
    try:
        gen.main()
    finally:
        gen.OUT = old
    yield
