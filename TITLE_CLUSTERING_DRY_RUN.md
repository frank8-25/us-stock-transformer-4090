# News Title Clustering Dry Run

Run date: 2026-09-14, Asia/Taipei

## Purpose and scope

This dry-run reduces the set of News URLs that may later be considered for body retrieval. It is a title-only planning baseline. It did not fetch any URL, summary, or article body; did not run BERTopic, FinGPT, QLoRA, or the Transformer; and did not modify the raw News CSV.

The 500-row smoke run established only that field mapping, embedding, output schema, causal ordering, and SHA safeguards worked. All statistics below are from the separate complete 15,549-row run unless explicitly labeled smoke.

## Input and mapping

Input: data/raw/news/NVDA_news_raw.csv

Original rows: 15,549

Columns: date, title, url, source, language, query, search_date, domain, seendate, language_raw

Mapping:

| Output concept | Input field or method |
|---|---|
| title | title |
| URL | url |
| publication timestamp | seendate, with date fallback |
| domain | domain |
| source | source |
| original row ID | Not present; stable SHA-256 of source row fields plus identical-row occurrence number |

All seendate values were parsed as UTC. Canonical rows were sorted by published_at and stable row ID before causal comparison. The output manifest retains original CSV row order so every source row remains traceable.

Input SHA-256 before and after:

cb764bf33397756e948b6751c3b71c23a971298a0c78a4285f29bfcac7b9f62f

The hashes match.

## Normalization and exact duplicate handling

Normalization uses Unicode NFKC, conservative quote and dash normalization, lowercase, trim, whitespace collapse, punctuation spacing normalization, and removal of a trailing media suffix only when it is verified against domain or source metadata. Numbers and directional terms such as not, up, and down are retained. No stemming is used.

Rows are never deleted. A canonical row is selected for exact URL matches or same-day normalized-title matches. Other rows retain their stable ID and receive duplicate_of_row_id.

| Measure | Count |
|---|---:|
| Exact URL extra rows | 5 |
| Global normalized-title extra rows | 2,578 |
| Same-date normalized-title extra rows | 2,217 |
| Normalized titles repeated across different dates | 241 groups |
| Rows marked exact duplicate | 2,222 |
| Canonical title rows embedded | 13,327 |

Global normalized-title repetition on different dates is reported but is not automatically collapsed because a repeated title may describe a later event.

## Embedding

| Setting | Actual value |
|---|---|
| Model | sentence-transformers/all-MiniLM-L6-v2 |
| Model revision | 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 |
| sentence-transformers | 6.0.1 |
| torch | 2.14.0+cu130 |
| Device | CUDA, NVIDIA GeForce RTX 4090 |
| Embedded rows | 13,327 canonical titles |
| Dimension | 384 |
| Dtype | float32 |
| Normalized embeddings | yes |
| Batch size | 64 |
| Seed | 42 |
| First embedding time | 9.621 seconds |
| Verified cache reload time | 0.007 seconds |

The model was loaded once for the full embedding pass. All four thresholds share the same normalized matrix. The ignored cache is stored at data/pipeline_runs/title_clustering_20260914_full/canonical_title_embeddings.npy with validation metadata in embedding_cache_metadata.json. Cache reuse requires matching model ID, input SHA-256, canonical-ID SHA-256, row count, normalization flag, dtype, and embedding dimension.

No large embedding file is tracked by Git.

## Causal clustering method

Canonical titles are processed chronologically. A title is compared only with already-published canonical titles inside the previous 72 hours. The implementation holds only the active time window and computes vector-to-window similarities; it never allocates a complete all-pairs matrix.

The complete run performed 314,805 candidate comparisons, with at most 95 active prior titles. A direction-conflict guard prevents direct joining when matched titles contain explicit opposing pairs such as raises and cuts, beats and misses, up and down, approval and rejection, or growth and decline.

cluster_size_as_of_publication counts only rows visible by that publication timestamp. cluster_size_final is an offline retrieval-planning value and must never be used as a historical Transformer feature.

## Threshold sensitivity

Cluster size columns below use canonical titles. The largest all-row group, after attaching exact duplicates, is 53 rows at every threshold.

| Threshold | Clusters | Singletons | Singleton ratio | Groups size 2–3 | Groups size 4+ | Max canonical group | Mean group | Near-duplicate canonical titles | URL candidates | Reduction from 15,549 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.85 | 11,988 | 11,120 | 92.76% | 773 | 95 | 16 | 1.112 | 1,339 | 13,911 | 1,638, or 10.53% |
| 0.90 | 12,473 | 11,843 | 94.95% | 576 | 54 | 12 | 1.068 | 854 | 14,143 | 1,406, or 9.04% |
| 0.93 | 12,720 | 12,247 | 96.28% | 444 | 29 | 9 | 1.048 | 607 | 14,233 | 1,316, or 8.46% |
| 0.95 | 12,846 | 12,447 | 96.89% | 385 | 14 | 9 | 1.037 | 481 | 14,291 | 1,258, or 8.09% |

Each group supplies at most three candidate URLs: the representative, then up to two additional URLs from different domains. Candidate ordering considers representative status, URL presence, similarity to the centroid, domain diversity, earlier publication, and stable row ID.

## Pilot threshold decision and human review

The human review covered the random multi-news groups, largest groups, and near-threshold boundary groups in cluster_review_sample.csv. The reviewer found the inspected titles to describe the same or highly similar news events and found no obvious incorrect merge in that sample.

Threshold 0.90 is therefore adopted for the current fetch-candidate pilot. It retains a conservative 94.95 percent singleton rate while identifying 854 additional semantically similar canonical titles and reducing the retrieval candidate set from 15,549 rows to 14,143 URLs. Threshold 0.85 saves only 232 more URL candidates than 0.90 but creates more and larger multi-title groups, increasing the risk of merging related but distinct events.

This is a sampled review conclusion and a pilot threshold. It does not prove that all 12,473 groups are correct, and it is not a validated production feature setting.

## Review sample

The ignored file cluster_review_sample.csv contains 236 rows from 48 unique clusters at threshold 0.90:

- 20 randomly selected multi-news clusters.
- The 10 largest clusters.
- 20 clusters with a joining similarity nearest the threshold.
- Direction-conflict clusters detected by the explicit antonym screen.

Categories overlap, which is why the union contains 48 rather than 50 clusters. No merged direction-conflict cluster was detected by the limited antonym screen at threshold 0.90. The completed sample review found no obvious incorrect merge. This does not prove semantic consistency across all groups; future reviews should continue checking entity, event, time, and direction errors.

The review file contains titles, publication times, domains, and URLs only. It is located inside the ignored pipeline run directory and will not be committed.

## Outputs

Complete run directory: data/pipeline_runs/title_clustering_20260914_full

- run_manifest.json
- title_clustering_manifest.csv, containing all 15,549 original rows
- threshold_sensitivity.json
- threshold_sensitivity.csv
- cluster_review_sample.csv
- canonical_title_embeddings.npy
- embedding_cache_metadata.json

The title manifest includes original_row_id, published_at, seendate, title, normalized_title, url, domain, exact_duplicate, duplicate_of_row_id, preliminary_cluster_id, cluster_representative_row_id, similarity_to_representative, cluster_size_final, cluster_size_as_of_publication, candidate_fetch_rank, and clustering_status.

## Limitations and next review

This is short-title similarity within a fixed time window. It is not final event clustering. It has no summary or article body, and it does not use BERTopic. Similar headlines can still refer to different events, while differently worded reports of one event can remain separate.

Manual review should focus on the largest groups, near-threshold groups, recurring generic titles, cross-domain duplicates, entity mismatches, and positive-versus-negative direction. After review, adjust normalization, direction guards, time window, or threshold and repeat the dry-run. Only then should candidate URLs be considered for a separate retrieval stage. No offline final group size may be backfilled as a historical feature.
