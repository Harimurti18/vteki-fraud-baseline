"""Synthetic data generator (deterministic, seed=42).

Writes three linked tables to data/ and the ground truth to data/_truth.csv.
`_truth.csv` is for sanity checks only: scoring_engine.py never reads it.

Documented imperfections (measured values are written to data/data_notes.json):
  * customer_id has a different format in every table (C0001 / CUST-0001 / 1)
  * timestamps are strings in three formats: ISO-UTC, "dd/mm/yyyy HH:MM" (naive, assumed WIB) and epoch
  * exact duplicates and replays (same id, later ingested_at)
  * ~6% of device_id values are missing
  * ~0.5% of events reference customers that are not in customers.csv (orphans)
  * ~3% of events arrive late (ingested_at 1-3 days after event_time)
Synthetic scenarios (ground truth, NOT available operationally):
  * account takeover (8): new device, burst of failed logins, large purchases right after
  * ring (2 x 6 accounts): new accounts sharing one device, repeated small purchases
  * household devices (~15% of customers share a device with 1-2 relatives): legitimate
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "data"
START, DAYS, N = pd.Timestamp("2026-09-01"), 30, 300
HOUR_W = np.array([1, .5, .3, .2, .2, .5, 1.5, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4.5, 5, 5.5, 6, 6, 5, 3, 2], float)
HOUR_P = HOUR_W / HOUR_W.sum()


def rand_times(rng, k, lo=0, hi=DAYS):
    secs = rng.integers(lo, hi, k) * 86400 + (rng.choice(24, k, p=HOUR_P) - 7) * 3600 + rng.integers(0, 3600, k)
    return START + pd.to_timedelta(np.maximum(secs, 0), unit="s")


def render(ts, rng, mix):
    """Raw timestamp strings in mixed formats."""
    t = pd.DatetimeIndex(ts)
    iso = np.asarray(t.strftime("%Y-%m-%dT%H:%M:%SZ"), dtype=object)
    loc = np.asarray((t + pd.Timedelta(hours=7)).strftime("%d/%m/%Y %H:%M"), dtype=object)
    ep = np.asarray(((t - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).astype(str), dtype=object)
    kind = rng.choice(list(mix), len(t), p=list(mix.values()))
    return np.where(kind == "iso", iso, np.where(kind == "local", loc, ep))


def make_noisy(df, rng, idcol, prefix, width, cust_fmt, mix):
    """Turn clean event rows into an imperfect source extract."""
    df = df.sort_values("t", kind="mergesort").reset_index(drop=True)
    df[idcol] = [f"{prefix}{i + 1:0{width}d}" for i in range(len(df))]
    clean_ids = df[[idcol]].copy()
    orphan = rng.random(len(df)) < 0.005
    df.loc[orphan, "customer_id"] = rng.integers(N + 1, N + 11, orphan.sum())
    df.loc[rng.random(len(df)) < 0.06, "device_id"] = None
    late = rng.random(len(df)) < 0.03
    delay = np.where(late, rng.integers(86400, 3 * 86400, len(df)), rng.exponential(20, len(df)).astype(int))
    df["ingested_at"] = np.asarray((df["t"] + pd.to_timedelta(delay, unit="s")).dt.strftime("%Y-%m-%dT%H:%M:%SZ"), dtype=object)
    df["event_time"] = render(df["t"], rng, mix)
    df["customer_id"] = df["customer_id"].map(cust_fmt)
    df = df.drop(columns=["t"])
    exact = df.sample(frac=0.015, random_state=int(rng.integers(1 << 31)))
    replay = df.sample(frac=0.01, random_state=int(rng.integers(1 << 31))).copy()
    replay["ingested_at"] = np.asarray((pd.to_datetime(replay["ingested_at"], format="%Y-%m-%dT%H:%M:%SZ")
                                        + pd.to_timedelta(rng.integers(3600, 86400, len(replay)), unit="s"))
                                       .dt.strftime("%Y-%m-%dT%H:%M:%SZ"), dtype=object)
    out = pd.concat([df, exact, replay], ignore_index=True)
    return out.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).reset_index(drop=True), clean_ids


def main(seed=42):
    rng = np.random.default_rng(seed)
    ids = list(range(1, N + 1))
    ring_ids, pool = ids[-12:], ids[:-12]
    rings = [ring_ids[:6], ring_ids[6:]]
    ato = [int(x) for x in rng.choice(pool, 8, replace=False)]
    rest = [i for i in pool if i not in set(ato)]
    n_dev = [0]

    def new_dev():
        n_dev[0] += 1
        return f"dev_{n_dev[0]:04d}"

    home = {}
    hh = [int(x) for x in rng.choice(rest, int(0.15 * len(rest)), replace=False)]
    i = 0
    while i < len(hh):                       # households: 2-3 relatives share one device (legitimate)
        grp = hh[i:i + int(rng.integers(2, 4))]
        i += len(grp)
        d = new_dev()
        for c in grp:
            home[c] = d
    ring_dev = [new_dev(), new_dev()]
    for r, members in enumerate(rings):
        for c in members:
            home[c] = ring_dev[r]
    for c in ids:
        home.setdefault(c, new_dev())

    logins, txns = [], []                    # rows: (customer_id, device_id, t, ...) ; scenario kept for truth
    base = {c: float(np.exp(rng.normal(np.log(150_000), 0.6))) for c in ids}
    for c in pool:                           # normal behaviour
        k = rng.poisson(1.0 * DAYS)
        for t, ok in zip(rand_times(rng, k), rng.random(k) < 0.96):
            logins.append((c, home[c], t, bool(ok)))
        k = rng.poisson(0.3 * DAYS)
        amt = base[c] * np.exp(rng.normal(0, 0.5, k)) * np.where(rng.random(k) < 0.01, 10, 1)
        for t, a in zip(rand_times(rng, k), amt):
            txns.append((c, home[c], t, max(round(float(a), -3), 1000.0), "normal"))
    for n, c in enumerate(ato):              # account takeover
        t0 = START + pd.Timedelta(days=int(rng.integers(6, DAYS - 1)), hours=int(rng.integers(0, 24)))
        d = f"dev_atk{n}"
        kf = int(rng.integers(3, 9))
        for s in np.sort(rng.integers(0, 600, kf)):
            logins.append((c, d, t0 + pd.Timedelta(seconds=int(s)), False))
        logins.append((c, d, t0 + pd.Timedelta(seconds=620), True))
        for s in np.sort(rng.integers(700, 3000, int(rng.integers(1, 4)))):
            txns.append((c, d, t0 + pd.Timedelta(seconds=int(s)), max(round(base[c] * float(rng.uniform(5, 20)), -3), 1_500_000.0), "ato"))
    reg = {c: START - pd.Timedelta(days=int(rng.integers(20, 400))) for c in ids}
    for r, members in enumerate(rings):      # new accounts sharing one device
        r0 = 8 if r == 0 else 22
        for c in members:
            reg[c] = START + pd.Timedelta(days=r0 - int(rng.integers(3, 10)))
            for t in rand_times(rng, int(rng.integers(3, 7)), r0, r0 + 5):
                logins.append((c, home[c], t - pd.Timedelta(seconds=60), True))
                txns.append((c, home[c], t, float(round(rng.uniform(50_000, 120_000), -3)), "ring"))

    lg = pd.DataFrame(logins, columns=["customer_id", "device_id", "t", "success"])
    tx = pd.DataFrame(txns, columns=["customer_id", "device_id", "t", "amount", "scenario"])
    truth_src = tx.sort_values("t", kind="mergesort").reset_index(drop=True)
    tx_n, tx_ids = make_noisy(tx.drop(columns=["scenario"]), rng, "transaction_id", "T", 5, lambda x: str(int(x)),
                              {"iso": 0.8, "local": 0.2})
    lg_n, _ = make_noisy(lg, rng, "event_id", "L", 6, lambda x: f"CUST-{int(x):04d}", {"iso": 0.6, "local": 0.3, "epoch": 0.1})
    cu = pd.DataFrame({"customer_id": [f"C{c:04d}" for c in ids], "t": [reg[c] for c in ids]})
    cu["registered_at"] = render(cu["t"], rng, {"iso": 0.7, "local": 0.3})
    cu = pd.concat([cu.drop(columns="t"), cu.drop(columns="t").sample(2, random_state=1)], ignore_index=True)

    OUT.mkdir(exist_ok=True)
    cu.to_csv(OUT / "customers.csv", index=False)
    lg_n[["event_id", "customer_id", "device_id", "event_time", "success", "ingested_at"]].to_csv(OUT / "login_events.csv", index=False)
    tx_n[["transaction_id", "customer_id", "device_id", "amount", "event_time", "ingested_at"]].to_csv(OUT / "transactions.csv", index=False)
    truth = pd.DataFrame({"transaction_id": tx_ids["transaction_id"], "scenario": truth_src["scenario"]})
    truth["is_fraud"] = truth["scenario"] != "normal"
    truth.to_csv(OUT / "_truth.csv", index=False)

    notes = {"seed": seed, "rows": {"customers": len(cu), "login_events": len(lg_n), "transactions": len(tx_n)},
             "transactions": {"exact_duplicate_rows": int(tx_n.duplicated().sum()),
                              "repeated_ids_total": int(tx_n.duplicated("transaction_id").sum()),
                              "missing_device_rate": round(float(tx_n["device_id"].isna().mean()), 3),
                              "raw_timestamp_formats": {"iso": int(tx_n["event_time"].str.contains("T").sum()),
                                                        "local": int(tx_n["event_time"].str.contains("/").sum())}},
             "ground_truth": {"fraud_txn_rate": round(float(truth["is_fraud"].mean()), 4),
                              "by_scenario": truth["scenario"].value_counts().to_dict(),
                              "note": "scenario rates are my construction, not evidence about real fraud"}}
    (OUT / "data_notes.json").write_text(json.dumps(notes, indent=2))
    print(json.dumps(notes, indent=2))


if __name__ == "__main__":
    main()
