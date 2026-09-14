# Real Data Annotation Audit

Audit date: 2026-09-14, Asia/Taipei

## Scope and safeguards

This is a read-only audit of News, Earnings Call, and 10-K CSVs under data/raw. No source CSV was changed. No model, CUDA, QLoRA, consolidation apply operation, formal pipeline, or Transformer operation was used. This report contains aggregate statistics only and reproduces no source text or URL value. The optional manual Earnings Call CSV is absent.

The Git gate passed before inspection: the worktree was clean, commit 9411520 exists, HEAD and the local origin/main tracking ref both resolved to 9411520, and ahead/behind was 0/0.

## Readiness conclusion

The collection has enough temporal and event coverage to plan a 400-unit pilot, but the raw rows are not yet 400 ready-to-label units. All 15,549 News rows contain titles only. The 29 Earnings Call rows are long documents: 24 are titled CFO Commentary and only 5 are titled transcript. The six 10-K rows point to six unique SEC filings, but every body is exactly 20,000 characters and contains XBRL markers. News enrichment, EC segmentation, and complete 10-K extraction are prerequisites.

A 50-unit double-annotated calibration round should precede expansion to 400. Every heuristic in this report selects candidates only and is not a label or ground truth.

## Files and columns

| Source | File | Size | Rows | Columns |
|---|---|---:|---:|---|
| News | data/raw/news/NVDA_news_raw.csv | 5,039,075 bytes | 15,549 | date, title, url, source, language, query, search_date, domain, seendate, language_raw |
| Earnings Call | data/raw/earnings_call/NVDA_earnings_call_raw.csv | 549,189 bytes | 29 | date, quarter, title, content, url, source |
| Earnings Call manual | data/raw/earnings_call/NVDA_earnings_call_manual.csv | missing | — | — |
| 10-K | data/raw/ten_k/NVDA_10k_raw.csv | 120,907 bytes | 6 | date, filing_type, title, content, url, source |

## Field mapping

| Annotation field | News | Earnings Call | 10-K |
|---|---|---|---|
| source_type | Constant news | Constant earnings_call | Constant ten_k |
| published_at | Prefer seendate after source verification; all values retain UTC timestamp format YYYYMMDDTHHMMSSZ | Verify actual call or document publication timestamp; raw date is day-only and CFO Commentary dates are unverified | Verify SEC acceptance or filing publication timestamp; raw date is only a candidate filing day |
| as_of_date | UTC day of verified published_at | Same | Same |
| title | title | title | title |
| text | Summary or body must be acquired; headline-only samples must be explicitly identified if retained | Cleaned semantic segment from content, with derived speaker and section metadata | Cleaned chunk from a verified extracted filing section |
| stable document ID | Hash of normalized resolved URL, seendate, and title | Hash of verified event identity plus source document URL | SEC accession parsed from URL |
| group ID | Causal news cluster ID | Call event ID shared by related documents and all chunks | SEC accession shared by all sections and chunks |

Canonical input remains title, a newline, and text when both exist; otherwise it is the nonblank field. The raw files have no explicit document_id, accession, section, speaker, or chunk_id columns. A separate versioned preprocessing stage must create these fields.

## Dates and availability

| Source | Earliest raw date | Latest raw date | Unparseable | Raw date values with time |
|---|---|---|---:|---:|
| News | 2020-01-09 | 2026-06-28 | 0 | 0 |
| Earnings Call | 2020-02-01 | 2026-05-20 | 0 | 0 |
| 10-K | 2021-02-26 | 2026-02-25 | 0 | 0 |

News seendate is populated and parseable in all 15,549 rows, retains UTC time, and its UTC day equals raw date for every row. It is the best existing published_at candidate. search_date is collection metadata and must not define availability.

EC date is structurally valid but has only day precision. Quarter labels extend through First Quarter 2027 while calendar dates end in 2026, showing that the quarter is a fiscal label rather than publication time. Verify call time for transcripts and publication time for CFO Commentary. The current pipeline already flags the latter as unverified.

All six 10-K URLs are SEC accession-like URLs. Use SEC acceptance or filing-publication time. Never substitute fiscal period end for information availability.

| Year | News rows | EC documents | 10-K documents |
|---:|---:|---:|---:|
| 2020 | 1,811 | 4 | 0 |
| 2021 | 2,192 | 4 | 1 |
| 2022 | 2,103 | 4 | 1 |
| 2023 | 2,606 | 4 | 1 |
| 2024 | 3,056 | 4 | 1 |
| 2025 | 2,600 | 7 | 1 |
| 2026 | 1,181 | 2 | 1 |

The 2020–2026 split can be retained, but 2026 News is partial through June and no 2020 10-K is supplied.

## Empty fields and duplicates

| Source | Blank title | Blank content or text | Both blank | Exact duplicate extras | Dedup key | Key duplicate extras | Rows after key dedup |
|---|---:|---:|---:|---:|---|---:|---:|
| News | 0 | 15,549 because content is absent | 0 | 0 | date, title, url | 0 | 15,549 |
| Earnings Call | 0 | 0 | 0 | 0 | date, title, url | 0 | 29 |
| 10-K | 0 | 0 | 0 | 0 | date, title, url | 0 | 6 |

News has 2,177 duplicate extras by exact date and title, 2,240 by normalized date and title, and 2,601 global normalized-title duplicate extras. A same-day normalized sequence comparison at threshold 0.90 found 456 near-duplicate pairs involving 473 rows. These counts are audit indicators, not semantic clusters. Five URL values repeat globally, so URL alone is not a unique News ID.

Every EC date and URL is unique. Every 10-K date and URL is unique. This supports document identification but does not make the rows annotation chunks.

## Canonical input length

| Source and measure | Min | P25 | Median | P75 | P90 | P95 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| News characters | 10 | 59 | 71 | 88 | 110 | 130 | 253 |
| News token estimate | 3 | 15 | 18 | 22 | 28 | 33 | 64 |
| EC characters | 16,121 | 17,305 | 19,057 | 20,050 | 20,054 | 20,054 | 20,054 |
| EC token estimate | 4,031 | 4,327 | 4,765 | 5,013 | 5,014 | 5,014 | 5,014 |
| 10-K characters | 20,027 | 20,028 | 20,028 | 20,028 | 20,028 | 20,028 | 20,028 |
| 10-K token estimate | 5,007 | 5,007 | 5,007 | 5,007 | 5,007 | 5,007 | 5,007 |

Token values are planning estimates using ceiling of characters divided by four, not a model tokenizer. Recalculate with the exact training tokenizer and prompt after preparation.

For screening, over-short means under 50 characters, long means over 4,000, and very long means over 8,000. News has 1,663 over-short titles and no long row. All 29 EC rows and all six 10-K rows are very long.

## News findings

- All 15,549 rows are title-only. Neither summary nor content exists.
- Observed daily count distribution is min 1, P25 6, median 10, P75 17, P90 24, P95 30, max 56.
- All URLs are nonblank; 15,544 are unique.
- Pattern screens find 4,195 possible price-only narratives, 564 rows without the literal name NVIDIA or NVDA, 1,480 positive-word candidates, 820 negative or risk-word candidates, and 82 mixed-word candidates. Human review is required. Name absence does not establish irrelevance.
- Two titles match a mojibake indicator. No HTML or XBRL indicator was found.
- Future Sentence Transformer and BERTopic preprocessing is suitable only after summary or body enrichment. It must operate causally at each cutoff so future reports cannot change historical membership, centroid, representative, size, similarity, or span.

## Earnings Call findings

- The 29 rows have 29 unique dates, URLs, and quarter labels, so they are 29 distinct source documents in this file. They are not 29 confirmed complete call transcripts.
- Titles classify 5 documents as transcript and 24 as CFO Commentary. Five documents contain an Operator marker, four contain a Q-and-A marker, and none contains the literal Prepared Remarks marker.
- Eleven bodies are exactly 20,000 characters. Every row exceeds 8,000 characters. Completeness must be checked.
- Forward-looking or safe-harbor template language occurs in 28 of 29 documents. Operator introductions, legal text, and other low-information sections must not dominate annotation samples.
- No HTML, XBRL, or mojibake pattern was found. speaker, section, and chunk_id are absent.
- EC must be segmented. Identify document type, speaker turns, management discussion, and Q-and-A; then form coherent tokenizer-checked chunks. All chunks and related documents for the same earnings event share one call event group.
- A stable event ID should combine ticker, verified public timestamp, and fiscal event identity. URL identifies a document, while transcript and commentary for one event should share the event group.

## 10-K findings

- Six rows have six unique SEC accession-like URLs and filing_type is 10-K for all rows. They plausibly identify six filings from 2021 through 2026, but the bodies are not complete filings.
- Every body is exactly 20,000 characters and all six contain XBRL namespace markers.
- Item 1, Item 1A, Item 7, Item 7A, Risk Factors, and MD-and-A markers are absent from these supplied prefixes.
- The simple screen finds no literal HTML tags or mojibake, but XBRL contamination remains present.
- Retrieve or extract complete filing text, remove inline-XBRL metadata, verify Item boundaries, and then create coherent tokenizer-limited chunks. Use the SEC accession as filing group ID.
- Six independent filing groups can contribute to a pilot, but cannot support a strong source-specific generalization claim.

## Group-aware temporal split

Create stable group IDs before selecting samples:

1. News: normalize exact duplicates, create time-causal semantic clusters, and keep all cluster members together.
2. EC: associate source documents with the same earnings event, and keep all speaker or section chunks together.
3. 10-K: keep every section and chunk from one SEC accession together.
4. Sort groups by verified publication time. A practical pilot split is train for 2020–2023, validation for 2024, and test for 2025–2026. If an event crosses a boundary, move the whole event to one side.
5. Check exact and normalized near duplicates across proposed splits and resolve collisions at group level.
6. Only after groups receive a split, sample source, year, and candidate strata within each split.

Never use future five-day returns, later news, later filings, or later calls to create multidimensional labels or choose candidates.

## Proposed 400-unit pilot

The target is 400 prepared annotation units, conditional on News enrichment, EC segmentation, and complete clean 10-K extraction.

| Source | Units | Sampling unit |
|---|---:|---|
| News | 260 | One representative per causal cluster; headline-only records form an explicit limited stratum if retained |
| Earnings Call | 100 | Coherent speaker or section chunks distributed across the 29 document and event records |
| 10-K | 40 | Verified section chunks across all six accession groups |
| Total | 400 | 4,000 dimension decisions per annotator plus evidence review |

The 10-K quota is limited because only six independent filing groups exist. EC can contribute more chunks, but all chunks remain nested in event groups.

| Year | News | EC chunks | 10-K chunks | Total | Split |
|---:|---:|---:|---:|---:|---|
| 2020 | 38 | 12 | 0 | 50 | train |
| 2021 | 34 | 14 | 7 | 55 | train |
| 2022 | 34 | 14 | 7 | 55 | train |
| 2023 | 38 | 15 | 7 | 60 | train |
| 2024 | 45 | 17 | 8 | 70 | validation |
| 2025 | 48 | 20 | 7 | 75 | test |
| 2026 | 23 | 8 | 4 | 35 | test, partial News year |
| Total | 260 | 100 | 40 | 400 | — |

This matrix is applied after group assignment. Preserve whole groups and rebalance within the same source and split if an exact quota would split an event or force poor-quality text.

Candidate strata overlap and do not sum to source totals:

- News: cluster size and similarity bins; unique and near-duplicate reports; length bins; price-narrative candidates; possible irrelevant candidates; positive-word, negative or risk-word, mixed-word, and no-direction candidates; domain and year.
- EC: transcript and CFO Commentary; management narrative and Q-and-A; guidance, growth, financial strength, concrete risk, uncertainty, mixed information, and template controls; length, year, and event.
- 10-K: after verified extraction, target 8 Business chunks, 12 Risk Factors, 12 MD-and-A, 4 Item 7A, and 4 other or low-information controls, spread across accessions and years.

Keywords, source type, section, date, and length may locate candidates. They must never assign sentiment, relevance, price narrative, risk, uncertainty, or other labels.

## Fifty-unit inter-annotator pilot

First double-annotate 50 prepared units: 25 News representatives, 15 EC chunks from distinct events where possible, and 10 10-K section chunks covering all six filings. Cover years and candidate strata rather than drawing random rows.

Each sample needs ten label decisions: six directional and four binary dimensions. Fifty samples produce 500 label decisions per annotator, 1,000 across two annotators, plus evidence-span selection and programmatic offset checks. Adjudicate disagreements and freeze the revised annotation guide before expansion.

Expand in batches of about 75 to 100 while monitoring per-dimension distribution, evidence validity, source and event coverage, and label collapse. Stop at 400 if coverage is adequate; extend toward 500 only to fill rare strata or independent event groups. Do not map insufficient_information to neutral.

## Required preparation before formal annotation

1. Enrich News with summary or body, or explicitly accept a narrow headline-only stratum with likely insufficient-information labels.
2. Verify every EC availability timestamp, especially all CFO Commentary documents, and determine which documents share an event.
3. Check the eleven exactly 20,000-character EC bodies, then segment every EC document with speaker and section metadata.
4. Replace the six truncated and XBRL-contaminated 10-K prefixes with complete filing text; record SEC acceptance time and accession; extract and verify sections before chunking.
5. Create group IDs and chronological splits before candidate sampling, and check duplicate leakage across splits.
6. Complete the 50-unit human calibration before producing the remaining 300 to 500 formal labels.

## Method limitations

Near-duplicate detection used normalized same-day titles and character sequence similarity; it is not semantic clustering. Token counts are character estimates. HTML, XBRL, mojibake, document-type, and keyword counts are pattern screens. The audit assesses readiness and sampling design; it makes no model labels, prediction claims, or source-completeness claims.
