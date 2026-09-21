"""Select deterministic causal representatives and a stratified News annotation pilot."""
import argparse
from collections import defaultdict
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import random
import re

import pandas as pd

YEAR_QUOTAS = {2020: 37, 2021: 34, 2022: 34, 2023: 38, 2024: 46, 2025: 48, 2026: 23}
SPLIT_QUOTAS = {"train": 143, "validation": 46, "test": 71}
PILOT_COLUMNS = [
    "sample_id", "split", "preliminary_cluster_id", "cluster_size_final",
    "representative_original_row_id", "published_at", "title",
    "representative_domain", "representative_url", "candidate_url_1",
    "candidate_url_2", "candidate_url_3", "cluster_first_published_at",
    "cluster_last_published_at", "dissemination_span_hours",
    "unique_domain_count", "sampling_seed", "selection_reason",
]
PRIMARY_DOMAINS = ("nvidia.com", "sec.gov")
WIRE_DOMAINS = ("reuters.com", "apnews.com", "businesswire.com", "globenewswire.com")
ESTABLISHED_DOMAINS = (
    "bloomberg.com", "wsj.com", "ft.com", "cnbc.com", "finance.yahoo.com",
    "marketwatch.com", "barrons.com", "investors.com", "nasdaq.com",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_for_year(year):
    if 2020 <= year <= 2023:
        return "train"
    if year == 2024:
        return "validation"
    if 2025 <= year <= 2026:
        return "test"
    raise ValueError("Publication year is outside the configured 2020 through 2026 pilot range.")


def domain_priority(domain):
    value = str(domain).strip().casefold()
    if not value:
        return 4
    if any(value == item or value.endswith("." + item) for item in PRIMARY_DOMAINS):
        return 0
    if any(value == item or value.endswith("." + item) for item in WIRE_DOMAINS):
        return 1
    if any(value == item or value.endswith("." + item) for item in ESTABLISHED_DOMAINS):
        return 2
    return 3


def title_quality(title):
    value = str(title).strip()
    return len(re.findall(r"[A-Za-z0-9]", value))


def parse_manifest(frame):
    required = {
        "original_row_id", "published_at", "title", "normalized_title", "url", "domain",
        "preliminary_cluster_id", "cluster_size_final",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("Cluster manifest is missing columns: " + ", ".join(sorted(missing)))
    result = frame.copy().fillna("").astype(str)
    parsed = pd.to_datetime(result["published_at"], errors="coerce", utc=True)
    if parsed.isna().any():
        raise ValueError("Cluster manifest contains missing or unparseable published_at values.")
    if result["preliminary_cluster_id"].str.strip().eq("").any():
        raise ValueError("Cluster manifest contains blank preliminary_cluster_id values.")
    result["_published"] = parsed
    result["_domain_priority"] = result["domain"].map(domain_priority)
    result["_title_quality"] = result["title"].map(title_quality)
    actual_sizes = result.groupby("preliminary_cluster_id")["original_row_id"].transform("size").astype(int)
    reported = pd.to_numeric(result["cluster_size_final"], errors="coerce")
    if reported.isna().any() or not reported.astype(int).equals(actual_sizes):
        raise ValueError("cluster_size_final does not match rows in the cluster manifest.")
    result["cluster_size_final"] = actual_sizes
    return result


def choose_representative(group):
    ordered = group.sort_values(
        ["_published", "_domain_priority", "_title_quality", "original_row_id"],
        ascending=[True, True, False, True], kind="stable",
    )
    return ordered.iloc[0]


def choose_url_candidates(group, representative):
    records = []
    representative_url = representative["url"].strip()
    representative_domain = representative["domain"].strip().casefold()
    if representative_url:
        records.append({"rank": 1, "url": representative_url, "domain": representative["domain"],
                        "original_row_id": representative["original_row_id"], "is_representative": True})
    remaining = group.loc[group["original_row_id"].ne(representative["original_row_id"])].copy()
    remaining = remaining.loc[remaining["url"].str.strip().ne("")]
    remaining = remaining.sort_values(
        ["_domain_priority", "_published", "_title_quality", "original_row_id"],
        ascending=[True, True, False, True], kind="stable",
    )
    used_urls = {representative_url} if representative_url else set()
    used_domains = {representative_domain} if representative_domain else set()
    selected_indices = set()
    maximum_records = 3 if representative_url else 2
    for prefer_new_domain in (True, False):
        for index, row in remaining.iterrows():
            if len(records) >= maximum_records:
                break
            url = row["url"].strip()
            domain = row["domain"].strip().casefold()
            if index in selected_indices or url in used_urls:
                continue
            if prefer_new_domain and domain in used_domains:
                continue
            records.append({"rank": len(records) + 1, "url": url, "domain": row["domain"],
                            "original_row_id": row["original_row_id"], "is_representative": False})
            selected_indices.add(index)
            used_urls.add(url)
            if domain:
                used_domains.add(domain)
        if len(records) >= maximum_records:
            break
    slots = ["", "", ""]
    if representative_url:
        slots[0] = representative_url
        for record in records[1:]:
            slots[record["rank"] - 1] = record["url"]
    else:
        for offset, record in enumerate(records[:2], start=1):
            slots[offset] = record["url"]
            record["rank"] = offset + 1
    return slots, records


def make_representatives(rows):
    records = []
    candidate_records = {}
    boundary_groups = 0
    for cluster_id, group in rows.groupby("preliminary_cluster_id", sort=True):
        representative = choose_representative(group)
        first = group["_published"].min()
        last = group["_published"].max()
        start_split = split_for_year(first.year)
        if split_for_year(last.year) != start_split:
            boundary_groups += 1
        slots, candidates = choose_url_candidates(group, representative)
        candidate_records[cluster_id] = candidates
        records.append({
            "preliminary_cluster_id": cluster_id,
            "cluster_size_final": len(group),
            "representative_original_row_id": representative["original_row_id"],
            "published_at": representative["published_at"], "title": representative["title"],
            "representative_normalized_title": representative["normalized_title"],
            "representative_domain": representative["domain"],
            "representative_url": representative["url"],
            "candidate_url_1": slots[0], "candidate_url_2": slots[1], "candidate_url_3": slots[2],
            "cluster_first_published_at": first.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "cluster_last_published_at": last.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "dissemination_span_hours": (last - first).total_seconds() / 3600,
            "unique_domain_count": group.loc[group["domain"].str.strip().ne(""), "domain"].str.casefold().nunique(),
            "year": first.year, "split": start_split,
        })
    return pd.DataFrame(records), candidate_records, boundary_groups


def sampling_stratum(row):
    size = int(row["cluster_size_final"])
    size_bin = "singleton" if size == 1 else "size_2_3" if size <= 3 else "size_4_plus"
    title = row["representative_normalized_title"].casefold()
    positive = bool(re.search(r"\b(raise[sd]?|beat[s]?|growth|gain[s]?|surge[sd]?|up|approval|approved)\b", title))
    negative = bool(re.search(r"\b(cut[s]?|miss(?:es)?|decline[s]?|drop[s]?|down|rejection|rejected|risk)\b", title))
    price = bool(re.search(r"\b(stock|share|price|target|rally|momentum|support|resistance)\b", title))
    theme = "mixed" if positive and negative else "positive" if positive else "negative_risk" if negative else "price" if price else "other"
    length = title_quality(row["title"])
    length_bin = "short" if length < 50 else "long" if length >= 120 else "medium"
    return size_bin + ":" + theme + ":" + length_bin


def cross_split_near_duplicate(candidate, selected, threshold=0.92):
    title = candidate["representative_normalized_title"]
    for other in selected:
        if candidate["split"] == other["split"]:
            continue
        other_title = other["representative_normalized_title"]
        if title == other_title or SequenceMatcher(None, title, other_title).ratio() >= threshold:
            return True
    return False


def select_pilot(representatives, year_quotas=None, seed=42):
    quotas = YEAR_QUOTAS if year_quotas is None else year_quotas
    selected = []
    shortages = {}
    rejected_cross_split = 0
    for year in sorted(quotas):
        target = quotas[year]
        year_frame = representatives.loc[representatives["year"].eq(year)].copy()
        if year_frame.empty:
            shortages[str(year)] = {"requested": target, "actual": 0,
                                    "reason": "no eligible groups for this year"}
            continue
        year_frame["_stratum"] = year_frame.apply(sampling_stratum, axis=1)
        buckets = {}
        for stratum, group in year_frame.groupby("_stratum", sort=True):
            indices = group.index.tolist()
            random.Random(seed + year + int(hashlib.sha256(stratum.encode()).hexdigest()[:8], 16)).shuffle(indices)
            buckets[stratum] = indices
        year_selected = []
        while len(year_selected) < target and buckets:
            progressed = False
            for stratum in sorted(list(buckets)):
                while buckets[stratum]:
                    index = buckets[stratum].pop()
                    candidate = representatives.loc[index].to_dict()
                    if cross_split_near_duplicate(candidate, selected + year_selected):
                        rejected_cross_split += 1
                        continue
                    candidate["selection_reason"] = "year_quota_stratified:" + stratum
                    year_selected.append(candidate)
                    progressed = True
                    break
                if not buckets[stratum]:
                    del buckets[stratum]
                if len(year_selected) >= target:
                    break
            if not progressed:
                break
        selected.extend(year_selected)
        if len(year_selected) < target:
            shortages[str(year)] = {"requested": target, "actual": len(year_selected),
                                    "reason": "insufficient eligible groups after cross-split near-duplicate guard"}
    result = pd.DataFrame(selected)
    if not result.empty:
        result["sample_id"] = [
            "news_pilot_" + hashlib.sha256((str(seed) + ":" + cluster).encode()).hexdigest()[:16]
            for cluster in result["preliminary_cluster_id"]
        ]
        result["sampling_seed"] = seed
        split_order = pd.Categorical(result["split"], ["train", "validation", "test"], ordered=True)
        result = result.assign(_split_order=split_order).sort_values(
            ["_split_order", "published_at", "sample_id"], kind="stable"
        ).drop(columns="_split_order").reset_index(drop=True)
    return result, shortages, rejected_cross_split


def cluster_size_distribution(selected):
    values = selected["cluster_size_final"].astype(int)
    return {
        "singleton": int(values.eq(1).sum()), "size_2_to_3": int(values.between(2, 3).sum()),
        "size_4_plus": int(values.ge(4).sum()), "minimum": int(values.min()),
        "median": float(values.median()), "maximum": int(values.max()),
    }


def run(args):
    cluster_manifest = Path(args.cluster_manifest).resolve()
    clustering_report = Path(args.clustering_report).resolve()
    raw_news = Path(args.raw_news).resolve()
    output_dir = Path(args.output_dir).resolve()
    for path in (cluster_manifest, clustering_report, raw_news):
        if not path.is_file():
            raise FileNotFoundError(path)
    raw_root = (Path(__file__).resolve().parent / "data/raw").resolve()
    if output_dir == raw_root or raw_root in output_dir.parents:
        raise ValueError("Output directory cannot be inside data/raw.")
    clustering = json.loads(clustering_report.read_text(encoding="utf-8"))
    if float(clustering.get("selected_similarity_threshold", -1)) != args.similarity_threshold:
        raise ValueError("Clustering run threshold does not match the requested pilot threshold.")
    raw_sha_before = sha256_file(raw_news)
    if raw_sha_before != clustering.get("input_sha256_after"):
        raise ValueError("Raw News SHA-256 does not match the clustering run.")
    frame = pd.read_csv(cluster_manifest, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    rows = parse_manifest(frame)
    representatives, candidates, boundary_groups = make_representatives(rows)
    selected, shortages, rejected_cross_split = select_pilot(representatives, seed=args.seed)
    output_dir.mkdir(parents=True, exist_ok=False)
    if len(selected) != sum(YEAR_QUOTAS.values()) or shortages:
        status = "quota_shortfall"
    else:
        status = "complete"
    output = selected.copy()
    output["cluster_size_final"] = output["cluster_size_final"].astype(int)
    output["unique_domain_count"] = output["unique_domain_count"].astype(int)
    output["dissemination_span_hours"] = output["dissemination_span_hours"].astype(float)
    output[PILOT_COLUMNS].to_csv(output_dir / "news_pilot_260.csv", index=False, encoding="utf-8-sig")
    output[PILOT_COLUMNS].to_json(
        output_dir / "news_pilot_260.jsonl", orient="records", lines=True, force_ascii=False
    )
    representative_columns = [
        "preliminary_cluster_id", "cluster_size_final", "representative_original_row_id",
        "published_at", "title", "representative_domain", "representative_url",
        "candidate_url_1", "candidate_url_2", "candidate_url_3",
        "cluster_first_published_at", "cluster_last_published_at",
        "dissemination_span_hours", "unique_domain_count", "year", "split",
    ]
    representatives[representative_columns].to_csv(
        output_dir / "news_cluster_representatives.csv", index=False, encoding="utf-8-sig"
    )
    fetch_rows = []
    selected_ids = set(output["preliminary_cluster_id"])
    sample_by_cluster = output.set_index("preliminary_cluster_id")["sample_id"].to_dict()
    for cluster_id in sorted(selected_ids):
        for record in candidates[cluster_id]:
            item = dict(record)
            item.update(sample_id=sample_by_cluster[cluster_id], preliminary_cluster_id=cluster_id)
            fetch_rows.append(item)
    fetch = pd.DataFrame(fetch_rows, columns=[
        "sample_id", "preliminary_cluster_id", "rank", "url", "domain",
        "original_row_id", "is_representative",
    ])
    fetch.to_csv(output_dir / "news_fetch_candidate_manifest.csv", index=False, encoding="utf-8-sig")
    raw_sha_after = sha256_file(raw_news)
    split_actual = output["split"].value_counts().reindex(SPLIT_QUOTAS, fill_value=0).astype(int).to_dict()
    year_actual = output["published_at"].str[:4].value_counts().sort_index().astype(int).to_dict()
    backup_rows = fetch.loc[~fetch["is_representative"]].merge(
        output[["sample_id", "representative_domain"]], on="sample_id"
    )
    cross_domain_backups = 0 if backup_rows.empty else int(backup_rows.apply(
        lambda row: row["domain"].casefold() != row["representative_domain"].casefold(), axis=1
    ).sum())
    report = {
        "status": status, "similarity_threshold": args.similarity_threshold, "sampling_seed": args.seed,
        "input_cluster_rows": len(rows), "representative_cluster_rows": len(representatives),
        "requested_samples": sum(YEAR_QUOTAS.values()), "actual_samples": len(output),
        "year_quotas_requested": {str(k): v for k, v in YEAR_QUOTAS.items()},
        "year_counts_actual": year_actual, "split_quotas_requested": SPLIT_QUOTAS,
        "split_counts_actual": split_actual, "shortages": shortages,
        "clusters_crossing_split_year_boundary_assigned_by_first_publication": boundary_groups,
        "cross_split_near_duplicate_candidates_rejected": rejected_cross_split,
        "selected_cluster_size_distribution": cluster_size_distribution(output),
        "url_statistics": {
            "representative_url_valid": int(output["representative_url"].str.strip().ne("").sum()),
            "representative_url_missing": int(output["representative_url"].str.strip().eq("").sum()),
            "groups_with_any_backup_url": int(output[["candidate_url_2", "candidate_url_3"]].ne("").any(axis=1).sum()),
            "groups_with_two_backup_urls": int(output["candidate_url_3"].ne("").sum()),
            "fetch_candidate_rows": len(fetch),
            "cross_domain_backup_rows": cross_domain_backups,
        },
        "representative_rule": [
            "earliest published_at", "domain trust tier when timestamps tie",
            "longer title completeness when still tied", "original_row_id lexical order",
        ],
        "raw_news_sha256_before": raw_sha_before, "raw_news_sha256_after": raw_sha_after,
        "raw_news_unchanged": raw_sha_before == raw_sha_after,
        "causal_limitations": {
            "offline_only_fields": ["cluster_size_final", "cluster_last_published_at", "dissemination_span_hours", "unique_domain_count"],
            "historical_transformer_feature_allowed": False,
            "future_causal_replacements": ["cluster_size_asof_t", "unique_domain_count_asof_t", "dissemination_span_hours_asof_t"],
        },
        "safeguards": {"url_fetch_performed": False, "raw_csv_modified": False,
                       "fingpt_called": False, "qlora_run": False, "transformer_modified": False},
    }
    (output_dir / "news_pilot_selection_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if raw_sha_before != raw_sha_after:
        raise RuntimeError("Raw News CSV changed during selection.")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-manifest", required=True, type=Path)
    parser.add_argument("--clustering-report", required=True, type=Path)
    parser.add_argument("--raw-news", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--similarity-threshold", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except Exception:
        import traceback
        traceback.print_exc()
        return 1
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
