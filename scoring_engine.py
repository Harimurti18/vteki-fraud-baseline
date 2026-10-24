"""Fraud risk baseline: validation -> point-in-time features -> versioned rules + Isolation Forest signal.

Usage:  python scoring_engine.py T00855            (one transaction, JSON with evidence)
        python scoring_engine.py --batch out.csv   (all transactions)

Design rules (each has a test in test_engine.py):
  * A transaction is scored at as_of = its event_time. A history row may be used only if
    event_time <= as_of AND ingested_at <= as_of (late-arriving rows are NOT known yet).
  * The transaction itself is excluded from its own history.
  * Isolation Forest is fit ONLY on transactions before ANOMALY["train_end"], using their own
    point-in-time features. It never sees data from the period it scores.
  * Missing device_id stays missing (never a shared "UNKNOWN" id); missing features never fire rules.
  * The ground truth file (_truth.csv) is never read.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

DATA_DIR = Path(__file__).parent / "data"
LOCAL_UTC_OFFSET_H = 7                       # assumption: naive "dd/mm/yyyy HH:MM" strings are WIB
H, D = 3_600_000_000_000, 86_400_000_000_000  # ns per hour / day

# Expert-set starting points, NOT fitted to labels. Changing anything here changes decision_version().
RULES = [
    {"code": "NEW_DEVICE_HIGH_VALUE", "points": 30,
     "all": [["device_age_hours", "<", 24], ["amount", ">=", 1_000_000], ["history_days", ">=", 3]]},
    {"code": "LOGIN_FAILURE_BURST", "points": 25, "all": [["failed_logins_1h", ">=", 3]]},
    {"code": "TXN_VELOCITY_1H", "points": 25, "all": [["txn_count_1h", ">=", 5]]},
    {"code": "DEVICE_MANY_ACCOUNTS", "points": 25, "all": [["device_n_customers", ">=", 5]]},
    {"code": "NEW_ACCOUNT_HIGH_VALUE", "points": 15, "all": [["account_age_days", "<", 14], ["amount", ">=", 1_000_000]]},
    {"code": "AMOUNT_SPIKE_VS_HISTORY", "points": 15, "all": [["amount_ratio_30d", ">=", 5]]},
]
BANDS = {"MEDIUM": 25, "HIGH": 50}
ANOMALY = {"train_end": "2026-09-20", "n_estimators": 100, "random_state": 42,
           "flag_percentile": 0.99, "points": 15, "min_train_rows": 100}
FEATURES = ["device_age_hours", "device_n_customers", "failed_logins_1h", "txn_count_1h",
            "amount_ratio_30d", "account_age_days", "history_days"]
OPS = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal, "==": np.equal}


def decision_version():
    blob = json.dumps({"rules": RULES, "bands": BANDS, "anomaly": ANOMALY, "tz": LOCAL_UTC_OFFSET_H}, sort_keys=True)
    return "rules+iforest@" + hashlib.sha256(blob.encode()).hexdigest()[:8]


# ---------------------------------------------------------------- validation / ingestion
def canonical_id(s):
    """C0012 / CUST-0012 / cust_12 / 12 -> 12 ; anything else -> <NA>."""
    d = s.astype("string").str.strip().str.extract(r"(?i)^(?:cust|c)?[-_]?(\d+)$")[0]
    return pd.to_numeric(d, errors="coerce").astype("Int64")


def parse_ts(s):
    """ISO-UTC, 'dd/mm/yyyy HH:MM' (WIB) or epoch seconds -> naive UTC; invalid -> NaT."""
    s = s.astype("string").str.strip()
    iso = s.str.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z").fillna(False).astype(bool)
    loc = s.str.fullmatch(r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}").fillna(False).astype(bool)
    ep = s.str.fullmatch(r"\d{9,11}").fillna(False).astype(bool)
    parts = []
    if iso.any():
        parts.append(pd.to_datetime(s[iso], format="%Y-%m-%dT%H:%M:%SZ", errors="coerce"))
    if loc.any():
        parts.append(pd.to_datetime(s[loc], format="%d/%m/%Y %H:%M", errors="coerce") - pd.Timedelta(hours=LOCAL_UTC_OFFSET_H))
    if ep.any():
        parts.append(pd.to_datetime(pd.to_numeric(s[ep]).astype("int64"), unit="s"))
    out = (pd.concat(parts).reindex(s.index) if parts else pd.Series(pd.NaT, index=s.index)).astype("datetime64[ns]")
    return out.where((out >= pd.Timestamp("2015-01-01")) & (out < pd.Timestamp("2100-01-01")))


def _ns(s):
    return s.to_numpy("datetime64[ns]").astype("int64")


def _clean_events(raw, key, valid_ids, rep, name):
    raw = raw.reset_index(drop=True)
    cid, t, g = canonical_id(raw["customer_id"]), parse_ts(raw["event_time"]), parse_ts(raw["ingested_at"])
    bad = (cid.isna() | t.isna() | g.isna() | raw[key].isna()).to_numpy(bool)
    v = raw[~bad].copy()
    v["customer_id"] = cid[~bad].astype("int64")
    v["event_time"], v["ingested_at"] = t[~bad], g[~bad]
    n_valid = len(v)
    v = v.sort_values([key, "ingested_at"], kind="mergesort").drop_duplicates(key, keep="first")   # earliest copy wins
    orphan = ~v["customer_id"].isin(valid_ids)
    v = v[~orphan]
    dev = v["device_id"].astype("string").str.strip()
    v["device_id"] = dev.mask(dev == "").astype(object).where(dev.notna(), None)                    # missing stays missing
    rep[name] = {"rows_in": len(raw), "quarantined_invalid": int(bad.sum()), "duplicates_removed": n_valid - (len(v) + int(orphan.sum())),
                 "quarantined_orphan": int(orphan.sum()), "rows_out": len(v)}
    v["_e"], v["_g"] = _ns(v["event_time"]), _ns(v["ingested_at"])
    return v.sort_values(["_e", key], kind="mergesort").reset_index(drop=True)


def load_and_validate_data(data_dir=DATA_DIR):
    d = Path(data_dir)
    rep = {}
    cu = pd.read_csv(d / "customers.csv", dtype=str)
    cid, reg = canonical_id(cu["customer_id"]), parse_ts(cu["registered_at"])
    ok = (cid.notna() & reg.notna()).to_numpy(bool)
    cu = pd.DataFrame({"customer_id": cid[ok].astype("int64"), "registered_at": reg[ok]})
    cu = cu.sort_values(["customer_id", "registered_at"], kind="mergesort").drop_duplicates("customer_id")
    ids = set(cu["customer_id"])
    rep["customers"] = {"rows_out": len(cu)}
    lg = _clean_events(pd.read_csv(d / "login_events.csv", dtype=str), "event_id", ids, rep, "login_events")
    lg["success"] = lg["success"].astype("string").str.lower().eq("true")
    tx = _clean_events(pd.read_csv(d / "transactions.csv", dtype=str), "transaction_id", ids, rep, "transactions")
    tx["amount"] = pd.to_numeric(tx["amount"], errors="coerce")
    return {"customers": cu, "login_events": lg, "transactions": tx}, rep


# ---------------------------------------------------------------- point-in-time features
class Index:
    """Per-customer / per-device numpy views of the clean tables (built once)."""

    def __init__(self, clean):
        lg, tx, cu = clean["login_events"], clean["transactions"], clean["customers"]
        self.reg = {int(c): int(r) for c, r in zip(cu["customer_id"], _ns(cu["registered_at"]))}
        self.login = {int(c): dict(e=_ns(s["event_time"]), g=s["_g"].to_numpy(), ok=s["success"].to_numpy(bool))
                      for c, s in lg.groupby("customer_id")}
        self.txn = {int(c): dict(e=s["_e"].to_numpy(), g=s["_g"].to_numpy(), a=s["amount"].to_numpy(float),
                                 id=s["transaction_id"].to_numpy(object)) for c, s in tx.groupby("customer_id")}
        sight = pd.concat([lg[["customer_id", "device_id", "_e", "_g"]], tx[["customer_id", "device_id", "_e", "_g"]]])
        self.sight = {int(c): dict(d=s["device_id"].to_numpy(object), e=s["_e"].to_numpy(), g=s["_g"].to_numpy())
                      for c, s in sight.groupby("customer_id")}
        dv = sight.dropna(subset=["device_id"])
        self.dev = {d: dict(c=s["customer_id"].to_numpy(), e=s["_e"].to_numpy(), g=s["_g"].to_numpy())
                    for d, s in dv.groupby("device_id")}


def features_at(ix, cid, dev, t, amount, tid):
    """Features known at time t (ns) for one transaction. Only rows with event<=t AND ingested<=t are used."""
    f = dict.fromkeys(FEATURES)
    L, T, S = ix.login.get(cid), ix.txn.get(cid), ix.sight.get(cid)
    f["failed_logins_1h"] = int(((L["e"] <= t) & (L["e"] > t - H) & (L["g"] <= t) & ~L["ok"]).sum()) if L else 0
    if T:
        m = (T["e"] <= t) & (T["g"] <= t) & (T["id"] != tid)
        f["txn_count_1h"] = int((m & (T["e"] > t - H)).sum())
        prior = T["a"][m & (T["e"] > t - 30 * D)]
        if len(prior) >= 3 and np.median(prior) > 0:
            f["amount_ratio_30d"] = float(amount / np.median(prior))
    else:
        f["txn_count_1h"] = 0
    if S is not None:
        known = (S["e"] <= t) & (S["g"] <= t)
        f["history_days"] = float((t - min(S["e"][known].min(), t)) / D) if known.any() else 0.0
        if dev is not None:
            m = known & (S["d"] == dev)
            f["device_age_hours"] = float((t - min(S["e"][m].min(), t)) / H) if m.any() else 0.0
    if dev is not None and dev in ix.dev:
        v = ix.dev[dev]
        f["device_n_customers"] = len(set(v["c"][(v["e"] <= t) & (v["g"] <= t)].tolist()) | {cid})
    f["account_age_days"] = float((t - ix.reg[cid]) / D) if cid in ix.reg else None
    return f


# ---------------------------------------------------------------- scoring
def _py(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return x.item() if isinstance(x, np.generic) else x


def _vec(f, amount):
    return [np.log1p(amount), f["txn_count_1h"] or 0, f["failed_logins_1h"] or 0,
            f["amount_ratio_30d"] if f["amount_ratio_30d"] is not None else 1.0,
            f["device_n_customers"] if f["device_n_customers"] is not None else 1]


class Engine:
    def __init__(self, data_dir=DATA_DIR, clean=None):
        self.clean = clean if clean is not None else load_and_validate_data(data_dir)[0]
        self.ix = Index(self.clean)
        self.tx = self.clean["transactions"].set_index("transaction_id", drop=False)
        self._model = "untrained"

    def features(self, row):
        return features_at(self.ix, int(row["customer_id"]), row["device_id"], int(row["_e"]), float(row["amount"]),
                           row["transaction_id"])

    def _train(self):
        end = pd.Timestamp(ANOMALY["train_end"]).value
        rows = self.tx[self.tx["_e"] < end]
        if len(rows) < ANOMALY["min_train_rows"]:
            self._model = None
            return
        X = np.array([_vec(self.features(r), r["amount"]) for _, r in rows.iterrows()])
        clf = IsolationForest(n_estimators=ANOMALY["n_estimators"], random_state=ANOMALY["random_state"]).fit(X)
        self._model = (clf, np.sort(clf.score_samples(X)))

    def anomaly_percentile(self, row, f):
        """Share of training rows that look at least as normal as this one (1.0 = most extreme). None if unavailable."""
        if self._model == "untrained":
            self._train()
        if self._model is None or int(row["_e"]) < pd.Timestamp(ANOMALY["train_end"]).value:
            return None                                   # no model, or row lies inside the training window
        clf, train_scores = self._model
        s = clf.score_samples(np.array([_vec(f, row["amount"])]))[0]
        return float(1.0 - np.searchsorted(train_scores, s, side="left") / len(train_scores))

    def score(self, transaction_id):
        if transaction_id not in self.tx.index:
            return {"error": "Transaction not found (unknown id or quarantined during validation)"}
        row = self.tx.loc[transaction_id]
        f = self.features(row)
        feats = {**f, "amount": float(row["amount"])}
        fired = []
        for r in RULES:
            if all(feats[n] is not None and bool(OPS[op](feats[n], v)) for n, op, v in r["all"]):
                fired.append((r["points"], r["code"]))
        pct = self.anomaly_percentile(row, f)
        if pct is not None and pct >= ANOMALY["flag_percentile"]:
            fired.append((ANOMALY["points"], "ANOMALY_TOP_1PCT"))
        fired.sort(key=lambda x: (-x[0], x[1]))
        score = min(100, sum(p for p, _ in fired))
        band = "HIGH" if score >= BANDS["HIGH"] else "MEDIUM" if score >= BANDS["MEDIUM"] else "LOW"
        return {"transaction_id": transaction_id, "as_of": str(row["event_time"]), "risk_score": int(score),
                "risk_band": band, "reasons": [c for _, c in fired], "decision_version": decision_version(),
                "evidence": {**{k: _py(v) for k, v in feats.items()}, "customer_id": int(row["customer_id"]),
                             "device_id": row["device_id"], "anomaly_percentile": _py(pct)}}


def score_transaction(transaction_id, data_dir=DATA_DIR):
    return json.dumps(Engine(data_dir).score(transaction_id), indent=2)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("transaction_id", nargs="?")
    ap.add_argument("--batch", metavar="OUT_CSV")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    a = ap.parse_args(argv)
    if a.batch:
        eng = Engine(a.data_dir)
        res = [eng.score(t) for t in eng.tx.index]
        pd.DataFrame([{"transaction_id": r["transaction_id"], "as_of": r["as_of"], "risk_score": r["risk_score"],
                       "risk_band": r["risk_band"], "reasons": ";".join(r["reasons"]),
                       "decision_version": r["decision_version"]} for r in res]).to_csv(a.batch, index=False)
        print(f"wrote {len(res)} scores to {a.batch}")
    elif a.transaction_id:
        print(score_transaction(a.transaction_id, a.data_dir))
    else:
        print("Usage: python scoring_engine.py <transaction_id> | --batch out.csv")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
