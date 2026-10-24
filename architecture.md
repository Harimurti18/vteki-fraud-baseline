# Architecture Overview: Fraud Risk Baseline

This document summarises the high-level architecture decisions for the fraud detection and customer operations system, scoped to a two-week Minimum Viable Product (MVP).

---

## 1. System Architecture — Data Flow & Trust Boundaries

The diagram below shows internal components, secure network trust boundaries, and entity interactions.

![System Architecture Data Flow](Data-Flow%20Diagram.png)

### Context Diagram

![Context Diagram](Context%20Diagram.png)

---

## 2. Component Overview

```
data/
├── customers.csv        # 300 synthetic customers; canonical ID normalised on load
├── login_events.csv     # ~8 900 login events (multi-format timestamps, duplicates, orphans)
├── transactions.csv     # ~2 700 transactions (amounts, device IDs, 30-day history)
└── _truth.csv           # Ground-truth labels — read ONLY by tests, never by the engine

generate_data.py         # Reproducible synthetic dataset generator (seed 42)
scoring_engine.py        # Core pipeline: validation → features → rules + Isolation Forest
test_engine.py           # 20 pytest tests (including mutation checks)
```

### Pipeline Stages (inside `scoring_engine.py`)

| Stage | Function / Class | Responsibility |
|---|---|---|
| **Ingestion & Validation** | `load_and_validate_data()` | Reads CSVs, normalises IDs, parses 3 timestamp formats, deduplicates, quarantines bad rows |
| **Indexing** | `Index` class | Builds in-memory per-customer / per-device numpy views for O(1) feature lookups |
| **Feature Engineering** | `features_at()` | Computes 7 point-in-time features; enforces strict `event_time ≤ as_of AND ingested_at ≤ as_of` |
| **Rule Scoring** | `Engine.score()` | Evaluates 6 expert-set rules; sums points (capped at 100); assigns risk band |
| **Anomaly Signal** | `Engine.anomaly_percentile()` | Isolation Forest trained on pre-`2026-09-20` data; adds 15 pts if percentile ≥ 99th |
| **Decision Version** | `decision_version()` | SHA-256 hash of all rules, bands, and model parameters — changes on any config edit |
| **CLI / Batch** | `main()` | Single-transaction JSON output or full-batch CSV output |

---

## 3. Architecture Decision Records (ADRs)

### ADR-1 — Architecture Pattern: Modular Monolith

- **Decision:** Use a modular monolith rather than microservices.
- **Rationale:** Significantly lower operational cost for a small team; no internal network latency; minimal deployment complexity. Optimal for a two-week MVP target. A service-oriented split can be introduced post-MVP once traffic patterns are known.

### ADR-2 — Data Processing: Batch + Synchronous API

- **Decision:** Run historical feature extraction in a daily batch job combined with a lightweight synchronous API for near-real-time risk scoring at transaction time.
- **Rationale:** Avoids the high cost and operational complexity of a pure streaming system (e.g. Kafka), while still meeting the latency Service Level Objective of < 200 ms. The `Engine` class is designed to be instantiated once per process and reused across requests.

### ADR-3 — Primary Storage: Relational Database (PostgreSQL)

- **Decision:** Use a relational database as the primary store.
- **Rationale:** Guarantees data integrity through a strict schema and is efficient for standard aggregation queries. A graph database was considered and rejected — too costly and complex for the MVP scope. Graph relationships (e.g. shared devices across accounts) are handled at the feature-engineering layer instead.

### ADR-4 — AI Infrastructure: Self-Hosted Small LLM

- **Decision:** Deploy a small-parameter LLM locally (self-hosted within the client's VPC), combined with Retrieval-Augmented Generation (RAG) for the internal SOP assistant.
- **Rationale:** Absolute mitigation of PII data-leakage risk. The client has not approved data transfer outside their perimeter; use of external LLM APIs (e.g. OpenAI, Anthropic) is therefore architecturally blocked.

---

## 4. Point-in-Time Feature Design

All 7 features are computed strictly at `as_of = transaction.event_time`. A history row is usable only when **both** conditions hold:

```
history_row.event_time  ≤ as_of
history_row.ingested_at ≤ as_of   # late-arriving rows are not yet "known"
```

The scored transaction itself is always excluded from its own history.

| Feature | Description |
|---|---|
| `device_age_hours` | Hours since the device was first seen by this customer |
| `device_n_customers` | Number of distinct customers observed on the same device |
| `failed_logins_1h` | Failed login attempts in the last hour |
| `txn_count_1h` | Number of transactions in the last hour |
| `amount_ratio_30d` | Transaction amount ÷ median amount over the last 30 days (requires ≥ 3 prior txns) |
| `account_age_days` | Days since customer registration |
| `history_days` | Days of observable history (guards against cold-start false positives) |

---

## 5. Scoring & Decision Version

- **Risk score:** integer 0–100 = sum of fired rule points, capped at 100.
- **Risk bands:** `LOW` (< 25) · `MEDIUM` (25–49) · `HIGH` (≥ 50).
- **Reason codes:** ordered descending by points contributed.
- **Anomaly signal:** `ANOMALY_TOP_1PCT` (+15 pts) when the Isolation Forest percentile ≥ 99th.
  - The model is trained **only** on transactions before `2026-09-20` using their own point-in-time features.
  - Transactions inside the training window receive `anomaly_percentile: null`.
- **`decision_version`:** `rules+iforest@<hash>` — a SHA-256 hash of all rules, bands, and model parameters. Any configuration change produces a new version, enabling audit traceability.

### Rules Summary

| Code | Points | Conditions |
|---|---|---|
| `NEW_DEVICE_HIGH_VALUE` | 30 | device_age_hours < 24 AND amount ≥ 1 000 000 AND history_days ≥ 3 |
| `LOGIN_FAILURE_BURST` | 25 | failed_logins_1h ≥ 3 |
| `TXN_VELOCITY_1H` | 25 | txn_count_1h ≥ 5 |
| `DEVICE_MANY_ACCOUNTS` | 25 | device_n_customers ≥ 5 |
| `NEW_ACCOUNT_HIGH_VALUE` | 15 | account_age_days < 14 AND amount ≥ 1 000 000 |
| `AMOUNT_SPIKE_VS_HISTORY` | 15 | amount_ratio_30d ≥ 5 |
| `ANOMALY_TOP_1PCT` | 15 | Isolation Forest percentile ≥ 99th (post-training-window only) |

> **Note:** Rule thresholds are expert-set starting values, not fitted to labels. They are documented heuristics and carry no claims about real-world fraud detection accuracy.

---

## 6. Failure Mode & Mitigation

| Failure | Behaviour | Mitigation |
|---|---|---|
| **API Degradation (Fail-Open)** | If the Scoring API exceeds 500 ms under traffic spikes, the system switches to graceful degradation: transactions are accepted by default (*fail-open*) to avoid business disruption, then flagged for retroactive analysis in the next batch run. | Circuit-breaker pattern at the API gateway layer |
| **Missing `device_id`** | Device stays `None`; device-dependent features return `None`; no device-based rule fires on `None`. Shared-device features are not polluted by an artificial `"UNKNOWN"` sentinel. | Handled in `_clean_events()` |
| **Invalid / unparseable timestamps** | Row is quarantined (counted in the validation report). | `parse_ts()` + `_clean_events()` |
| **Orphan customer references** | Rows referencing unknown customer IDs are quarantined. | `_clean_events()` orphan check |
| **Late-arriving rows** | Rows with `ingested_at > as_of` are invisible to the feature engine at scoring time. | Point-in-time filter in `features_at()` |
| **Isolation Forest unavailable** | If fewer than 100 training rows exist, the model is skipped; `anomaly_percentile` returns `null`. | `min_train_rows` guard in `Engine._train()` |

---

## 7. Known Limitations

- Synthetic scenarios and scoring rules were authored by the same person; results on this dataset make no claims about real fraud-detection accuracy. A real evaluation requires an independent adjudicated sample.
- Thresholds are not tuned. The 5-account device rule is a documented heuristic; rings with fewer accounts, or early-stage rings, may be missed.
- The Isolation Forest uses 5 inputs and is a ranking signal, not a classifier. It is fit on a fixed historical window and is not retrained or loaded from a model registry.
- CLI / batch only (no HTTP API). Pandas on a single process is adequate for thousands of rows but not for production-scale volumes.
- Naive local timestamps are assumed to be WIB (UTC+7). The validation report counts quarantined rows but does not yet verify the hour-of-day distribution per format.
