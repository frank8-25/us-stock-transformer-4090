# Raw dataset consolidation

Merged existing raw files only; original source text and dates retained.
Exact duplicates removed; conflicting keys retained and listed for review.

| Source | Input rows | Output rows | Removed duplicates | Start | End |
|---|---:|---:|---:|---|---|
| news | 17360 | 15549 | 1811 | 2020-01-09 | 2026-06-28 |
| earnings_call | 38 | 29 | 9 | 2020-02-01 | 2026-05-20 |
| ten_k | 8 | 6 | 2 | 2021-02-26 | 2026-02-25 |
| google_trends | 234 | 234 | 0 | 2020-01-01 | 2026-06-01 |

## Remaining source limitations
- No 2020 10-K filing exists in the supplied raw files.
- CFO commentary publication dates require verification; fiscal quarters are not publication dates.
- Some documents appear truncated at 20,000 characters; 10-K text includes XBRL metadata.
- Trends are monthly observations (78 months per keyword), not daily/weekly observations. No resampling or zero filling was performed.
- Archived processed/features/dataset/prediction files are historical outputs, not regenerated or mixed with raw data.
- Coverage counts describe supplied observations, not proof of complete source coverage.

Backup and audit directory: data/backup/2026-09-09_111515_consolidation_71e727df
