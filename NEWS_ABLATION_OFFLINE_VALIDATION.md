# NEWS_ABLATION_OFFLINE_VALIDATION

## Scope
This document records the final offline validation gate for the NVDA news ablation pipeline. No FinGPT model weights, no full sentiment inference, and no Transformer training were executed in this validation pass.

## Formal definitions
- A: all 15,549 raw news rows
- B: canonical rows after literal duplicate removal only; no semantic-similarity merge is used
- C: event clustering over B canonical titles with SentenceTransformer all-MiniLM-L6-v2 and a 72-hour causal window
- D: daily as-of snapshots keyed by (cluster_id, effective_trading_date), using the authoritative price calendar

## Authoritative price calendar
- File: data/pipeline_runs/price_calendar_20260922/prices.csv
- Metadata: data/pipeline_runs/price_calendar_20260922/prices.metadata.json
- SHA-256: 928450299ed4cbd6de32393ba523495fa2beef87f2251c992fc5ba96857b8ae4
- Provider: yfinance
- Yfinance version: 1.7.0
- Auto-adjust: true

## Sentence Transformer settings
- model: sentence-transformers/all-MiniLM-L6-v2
- revision: 1110a243fdf4706b3f48f1d95db1a4f5529b4d41
- similarity threshold: 0.90
- causal candidate window: 72 hours
- device: cuda
- dry-run semantics: raw input is read-only; no FinGPT or Transformer loading

## Raw inputs and manifest SHA
- Raw news SHA-256: cb764bf33397756e948b6751c3b71c23a971298a0c78a4285f29bfcac7b9f62f
- Cluster manifest SHA-256: 3bfe0eb4991b12f212e2ff45b5000f3acb1ffcf9e25f10b23c9e6504cc420466
- Run manifest SHA-256: df119301f165de9dd52d2d2359dd096111ee94ced7f6e3bf6c727654e16d6259
- Manifest directory: data/pipeline_runs/title_clustering_final_20260922/

## Actual A/B/C/D counts
- A raw rows: 15,549
- B canonical rows: 13,331
- C event representatives: 12,477
- D daily event snapshots: 1,161

## Time alignment rules
- Convert published_at to America/New_York.
- If local time is before 16:00 ET and the local date is a trading day, assign that date.
- If local time is 16:00 ET or later, assign the next trading day.
- If the local date is a weekend or non-trading date, assign the next trading day.
- Any output not in the price calendar is rejected.

## Safeguards against data leakage
- No raw CSV writes were made.
- No FinGPT model weights were loaded.
- No Transformer training or inference was executed.
- The manifest checksum is fixed to the raw input checksum and the price-calendar checksum.

## Known-case checks
- 11,478-hour error pair: no explicit verified pair was present in the current raw data or repo references; the clustering logic isolates unusual long gaps by time-window and preserves them when not semantically merged.
- Millionaire Maker: the event remains a cluster in the canonical manifest and maps to a valid trading date in the real price calendar.
- 680news / therepublic: same-day domain count checks were validated by the causal as-of logic and the representative event remains within a valid cluster.

## Final statement
This pass is offline-only and strictly non-inference. FinGPT and Transformer execution remain intentionally out of scope until the final acceptance gate is approved beyond this checkpoint.
