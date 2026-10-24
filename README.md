# V-TEKI AI Solution Engineer - Assignment E, Option 1: Fraud Risk Baseline

A small, tested vertical slice: **synthetic data -> validation -> point-in-time features -> versioned rule score + anomaly signal -> JSON with reason codes and evidence.**
Architecture and trust boundaries: `architecture.md`, `Context Diagram.png`, `Data-Flow Diagram.png`.

## Run it
Tested with Python 3.12 (dependency versions pinned in `requirements.txt`; on another Python run `pip freeze` and update the pins).
```
pip install -r requirements.txt
python generate_data.py                      # writes data/*.csv, data/_truth.csv, data/data_notes.json (seed 42)
python scoring_engine.py T00630              # one transaction -> JSON (works from any directory)
python scoring_engine.py --batch scores.csv  # all transactions
python -m pytest test_engine.py -v           # 20 tests, ~12 s
```
Sample inputs: `data/`. Sample outputs and the test log: `samples/` (`ato_*.json`, `ring_*.json`, `household_*.json`, `missing_device_*.json`, `not_found.json`, `batch_scores.csv`, `pytest_output.txt`).

## Data (synthetic) and its documented imperfections
Three linked tables: `customers`, `login_events`, `transactions` (300 customers, 30 days, ~2.7k transactions, ~8.9k logins).

| Imperfection | How it is injected | How the engine handles it |
|---|---|---|
| Inconsistent customer IDs | `C0012` / `CUST-0012` / `12` per table | canonical numeric id |
| String timestamps, 3 formats | ISO-UTC, naive `dd/mm/yyyy HH:MM`, epoch | parsed to UTC; naive strings assumed WIB (UTC+7) - an assumption; invalid -> quarantine |
| Duplicates and replays | ~1.5% exact, ~1% same id with later `ingested_at` | one row per id, earliest `ingested_at` wins; counted in the validation report |
| Missing `device_id` | ~6% | stays missing (no shared "UNKNOWN" id); device features are `None`, rules never fire on `None` |
| Orphan customers | ~0.5% of events | quarantined |
| Late-arriving events | ~3% arrive 1-3 days late | a row is usable only if `ingested_at <= as_of` |
| Shared devices | ~15% of customers share a device with relatives (legitimate) | not flagged by themselves (rule needs >= 5 accounts) |

**Ground truth vs operational labels.** `data/_truth.csv` (injected scenarios: account takeover, ring) exists only for sanity checks. `scoring_engine.py` never reads it (tested). There are **no operational fraud labels**, so this repo makes **no claim about fraud-detection accuracy**; scenario rates and thresholds are my construction, not evidence about real fraud.

## Point-in-time features (7)
`device_age_hours`, `device_n_customers`, `failed_logins_1h`, `txn_count_1h`, `amount_ratio_30d`, `account_age_days`, `history_days`.
A transaction is scored at its own `event_time`; history may be used only if `event_time <= as_of` **and** `ingested_at <= as_of`; the transaction itself is excluded. `history_days` guards against cold start (a device cannot look "new" for a customer we barely observed).

## Scoring and decision version
Numeric `risk_score` 0-100 = sum of rule points (cap 100); bands LOW < 25 <= MEDIUM < 50 <= HIGH. Reason codes are ordered by points. Rule points are expert-set starting values, not fitted.
`ANOMALY_TOP_1PCT` adds 15 points. The Isolation Forest is fit **only on transactions before 2026-09-20** (using their own point-in-time features); transactions inside that window have no anomaly signal (`anomaly_percentile: null`).
`decision_version` is `rules+iforest@<hash>` of all rules, bands and model parameters, so any change produces a new version.

## Tests (20)
Duplicates and replays, mixed timestamps, canonical IDs, quarantine of bad timestamp/orphan, missing device id, **time leakage** (future rows, late arrivals, self-exclusion, a property test that truncating everything unknown at time T leaves earlier features unchanged, Isolation Forest not fit on the scored period), household vs ring device sharing, reason-code order, decision-version change, row-order independence, determinism, CLI from another directory, ground truth never needed.
I also mutation-tested the suite by deliberately breaking the code (fit on all rows, ignoring `ingested_at`, counting future device users, keeping the wrong replay copy, counting a transaction in its own history, replacing missing device by "UNKNOWN"); each change made a test fail. The first run of the device-sharing mutation survived, which exposed a test gap that was then closed.

## Known limitations
* Synthetic scenarios were written by the same author as the rules: results on this data say nothing about real fraud. A real evaluation needs an independent adjudicated sample.
* Thresholds are not tuned; the 5-account device rule is a documented heuristic and a ring using fewer accounts would be missed (early ring transactions are also missed until enough accounts have been seen).
* The Isolation Forest uses 5 simple inputs and is a ranking signal, not a classifier; it is fit on a fixed window, not retrained or loaded from a registry.
* CLI/batch only (no HTTP API); pandas on one process, fine for thousands of rows, not for production volumes.
* Naive local timestamps are assumed to be WIB; the validation report counts quarantined rows but does not yet check the hour-of-day distribution per format.

## AI-tool-use disclosure  (author must edit - see brackets)
* Tools used: Gemini and Claude.
* Gemini was used to scan all the tasks that needed to be completed, and then to evaluate potential solutions. Furthermore, Gemini was employed for the initial creation of `generate_data.py`, `scoring_engine.py`, and `test_engine.py`.
* Claude reviewed my earlier draft (3 tables with 4 transactions) and wrote this version of `generate_data.py`, `scoring_engine.py` and `test_engine.py`, and ran the tests and mutation checks described above.
* Problems found in the earlier draft by running it: (1) the Isolation Forest was fit on all transactions including later ones, so a flag could disappear when later data was added; (2) `tail(1)` on unsorted rows made results depend on file order; (3) any shared device produced HIGH, which would flag families; (4) mixed or invalid timestamps crashed; (5) the decision version was a fixed string. Each is now covered by a test.
* Mistakes made while building this version and caught by testing: a test helper silently registered the orphan customer; one mutation survived because the leakage test sampled too little, so I added a direct test and widened the property test.
* AI-use disclosure: I used AI assistance throughout the preparation of this assessment. Gemini helped develop the initial draft and frame the early solution. During implementation, Claude reviewed and revised the initial Gemini-generated work to better align it with the take-home assessment criteria and assisted with coding and tests. I did not accept every AI recommendation. In Assessment E, I chose Option 1 and rejected Option 2. I also rejected the AI's suggestion to start by writing code first. Instead, I worked through the assessments in sequence because I viewed the earlier assessments as the foundation for the decisions and implementation in the later assessments. Examples of decisions made independently of AI recommendations: (1) In Assessment E, I selected Option 1 and rejected Option 2. (2) Although AI recommended writing the code first, I chose to complete the assessments in sequence because I considered the earlier assessments to be the foundation for the subsequent decisions and implementation. These examples describe the decisions I made; the technical rationale for Option 1 should also be explained in Assessment E itself.
* Time spent: 8 hours.
