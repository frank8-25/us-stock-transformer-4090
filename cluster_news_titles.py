"""Rule deduplication and causal title clustering for URL-fetch planning only."""
import argparse
from collections import defaultdict, deque
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import re
import unicodedata

import numpy as np
import pandas as pd

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_THRESHOLDS = (0.85, 0.90, 0.93, 0.95)
REQUIRED_OUTPUT_COLUMNS = [
    "original_row_id", "published_at", "seendate", "title", "normalized_title",
    "url", "domain", "exact_duplicate", "duplicate_of_row_id",
    "preliminary_cluster_id", "cluster_representative_row_id",
    "similarity_to_representative", "cluster_size_final",
    "cluster_size_as_of_publication", "candidate_fetch_rank", "clustering_status",
]
DIRECTION_PAIRS = (
    ("raises", "cuts"), ("raised", "cut"), ("raise", "cut"),
    ("beats", "misses"), ("beat", "miss"), ("up", "down"),
    ("approval", "rejection"), ("approved", "rejected"),
    ("growth", "decline"), ("grows", "declines"),
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain_words(value):
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _literal_title_similarity(left, right):
    left_title = normalize_title(left)
    right_title = normalize_title(right)
    if not left_title or not right_title:
        return 0.0
    if left_title == right_title:
        return 1.0
    left_tokens = set(re.findall(r"[a-z0-9]+", left_title))
    right_tokens = set(re.findall(r"[a-z0-9]+", right_title))
    if not left_tokens or not right_tokens:
        return 0.0
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union)


def _verified_media_suffix(suffix, domain, source):
    suffix_words = _plain_words(suffix)
    if len(suffix_words) < 3:
        return False
    host_words = _plain_words(re.sub(r"\.(com|net|org|io|co|news)$", "", domain.casefold()))
    source_words = _plain_words(source)
    suffix_set = set(suffix_words.split())
    host_set = set(host_words.split())
    source_set = set(source_words.split())
    return bool(
        suffix_words == source_words
        or suffix_words == host_words
        or suffix_set == host_set
        or suffix_set == source_set
        or suffix_words in host_words
        or suffix_words in source_words
    )


def normalize_title(title, domain="", source=""):
    value = unicodedata.normalize("NFKC", str(title))
    value = value.translate(str.maketrans({
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-",
    }))
    for separator in (" | ", " - "):
        if separator in value:
            head, suffix = value.rsplit(separator, 1)
            if head.strip() and _verified_media_suffix(suffix, domain, source):
                value = head
                break
    value = value.casefold().strip()
    value = re.sub(r"\s*([,;:!?|])\s*", r"\1 ", value)
    value = re.sub(r"\s*-\s*", " - ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" \t\r\n-|:")


def stable_row_ids(frame):
    occurrences = defaultdict(int)
    result = []
    for record in frame.astype(str).to_dict("records"):
        payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        occurrences[payload] += 1
        identity = payload + "#occurrence=" + str(occurrences[payload])
        result.append("news_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20])
    return result


def parse_published_at(frame):
    if "seendate" in frame:
        raw = frame["seendate"].astype(str).str.strip()
        parsed = pd.to_datetime(raw, format="%Y%m%dT%H%M%SZ", errors="coerce", utc=True)
        fallback = pd.to_datetime(raw, format="mixed", errors="coerce", utc=True)
        parsed = parsed.fillna(fallback)
    else:
        parsed = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    dates = pd.to_datetime(frame.get("date", ""), format="mixed", errors="coerce", utc=True)
    return parsed.fillna(dates)


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def prepare_rows(frame):
    required = {"title", "url", "date"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("Missing required News columns: " + ", ".join(sorted(missing)))
    result = frame.copy().fillna("").astype(str)
    if result["title"].str.strip().eq("").any():
        raise ValueError("News title contains blank values.")
    result["original_row_id"] = stable_row_ids(result)
    result["normalized_title"] = [
        normalize_title(title, domain, source)
        for title, domain, source in zip(
            result["title"], result.get("domain", ""), result.get("source", "")
        )
    ]
    if result["normalized_title"].eq("").any():
        raise ValueError("Title normalization produced a blank value.")
    published = parse_published_at(result)
    if published.isna().any():
        raise ValueError("News contains missing or unparseable publication dates.")
    result["_published"] = published
    result["published_at"] = published.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    if "seendate" not in result:
        result["seendate"] = ""
    if "domain" not in result:
        result["domain"] = ""

    union = UnionFind(len(result))
    first_url = defaultdict(list)
    first_date_title = {}
    for index, row in result.iterrows():
        url = row["url"].strip()
        date_title = (row["published_at"][:10], row["normalized_title"])
        if url:
            for prior_index in first_url[url]:
                prior_row = result.iloc[prior_index]
                if row["normalized_title"] == prior_row["normalized_title"]:
                    union.union(prior_index, index)
                    continue
                score = _literal_title_similarity(row["normalized_title"], prior_row["normalized_title"])
                if score >= 0.85 and abs((row["_published"] - prior_row["_published"]).total_seconds()) <= 72 * 3600:
                    union.union(prior_index, index)
            first_url[url].append(index)
        if date_title in first_date_title:
            union.union(first_date_title[date_title], index)
        else:
            first_date_title[date_title] = index
    groups = defaultdict(list)
    for index in result.index:
        groups[union.find(index)].append(index)
    canonical_for = {}
    for members in groups.values():
        canonical = min(members, key=lambda i: (result.at[i, "_published"], i))
        for member in members:
            canonical_for[member] = canonical
    result["_canonical_index"] = [canonical_for[i] for i in result.index]
    result["exact_duplicate"] = [canonical_for[i] != i for i in result.index]
    result["duplicate_of_row_id"] = [
        "" if canonical_for[i] == i else result.at[canonical_for[i], "original_row_id"]
        for i in result.index
    ]
    canonical = result.loc[[i for i in result.index if canonical_for[i] == i]].copy()
    canonical = canonical.sort_values(["_published", "original_row_id"], kind="stable").reset_index()
    return result, canonical


def exact_duplicate_statistics(rows):
    normalized = rows["normalized_title"]
    date = rows["published_at"].str[:10]
    global_groups = rows.groupby("normalized_title")["published_at"].agg(lambda x: len(set(v[:10] for v in x)))
    return {
        "url_same_extra_rows": int(rows.loc[rows["url"].str.strip().ne("")].duplicated("url").sum()),
        "normalized_title_same_extra_rows": int(normalized.duplicated().sum()),
        "same_date_normalized_title_extra_rows": int(pd.DataFrame({"date": date, "title": normalized}).duplicated().sum()),
        "normalized_title_across_different_dates": int(global_groups.gt(1).sum()),
        "marked_exact_duplicate_rows": int(rows["exact_duplicate"].sum()),
        "canonical_title_rows": int((~rows["exact_duplicate"]).sum()),
    }


def direction_conflict(left, right):
    a = set(re.findall(r"[a-z]+", left.casefold()))
    b = set(re.findall(r"[a-z]+", right.casefold()))
    return any((x in a and y in b) or (y in a and x in b) for x, y in DIRECTION_PAIRS)


def cluster_causal(embeddings, published, row_ids, titles, threshold, window_hours=72):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or len(embeddings) != len(row_ids):
        raise ValueError("Embedding row count does not match canonical titles.")
    norms = np.linalg.norm(embeddings, axis=1)
    if np.any(norms == 0):
        raise ValueError("Zero-length embedding encountered.")
    embeddings = embeddings / norms[:, None]
    times = pd.DatetimeIndex(published)
    if not times.is_monotonic_increasing:
        raise ValueError("Canonical rows must be sorted by publication time.")
    window = pd.Timedelta(hours=window_hours)
    active = deque()
    assignments = []
    link_similarity = []
    comparisons = 0
    max_active = 0
    for index in range(len(row_ids)):
        while active and times[index] - times[active[0]] > window:
            active.popleft()
        max_active = max(max_active, len(active))
        chosen = None
        chosen_similarity = -1.0
        if active:
            candidates = np.asarray(active, dtype=np.int64)
            similarities = embeddings[candidates] @ embeddings[index]
            comparisons += len(candidates)
            order = np.argsort(-similarities, kind="stable")
            for offset in order:
                prior = int(candidates[offset])
                score = float(similarities[offset])
                if score < threshold:
                    break
                if not direction_conflict(titles[prior], titles[index]):
                    chosen, chosen_similarity = prior, score
                    break
        if chosen is None:
            cluster_id = "cluster_" + row_ids[index]
            link_similarity.append(1.0)
        else:
            cluster_id = assignments[chosen]
            link_similarity.append(chosen_similarity)
        assignments.append(cluster_id)
        active.append(index)
    return {
        "assignments": assignments,
        "link_similarity": link_similarity,
        "embeddings": embeddings,
        "candidate_comparisons": comparisons,
        "max_active_window": max_active,
        "full_matrix_allocated": False,
    }


def materialize_manifest(rows, canonical, clustered, threshold):
    canonical = canonical.copy()
    canonical["preliminary_cluster_id"] = clustered["assignments"]
    canonical["_link_similarity"] = clustered["link_similarity"]
    cluster_members = defaultdict(list)
    for index, cluster_id in enumerate(clustered["assignments"]):
        cluster_members[cluster_id].append(index)
    representatives = {}
    for cluster_id, members in cluster_members.items():
        matrix = clustered["embeddings"][members]
        centroid = matrix.mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        scores = matrix @ centroid
        best = members[int(np.argmax(scores))]
        representatives[cluster_id] = best
    canonical_cluster = dict(zip(canonical["index"], canonical["preliminary_cluster_id"]))
    raw = rows.copy()
    raw["preliminary_cluster_id"] = raw["_canonical_index"].map(canonical_cluster)
    canonical_position = {source_index: pos for pos, source_index in enumerate(canonical["index"])}
    raw["_canonical_position"] = raw["_canonical_index"].map(canonical_position)
    rep_row_ids = {
        cluster_id: canonical.iloc[position]["original_row_id"]
        for cluster_id, position in representatives.items()
    }
    raw["cluster_representative_row_id"] = raw["preliminary_cluster_id"].map(rep_row_ids)
    raw["similarity_to_representative"] = [
        float(clustered["embeddings"][position] @ clustered["embeddings"][representatives[cluster_id]])
        for position, cluster_id in zip(raw["_canonical_position"], raw["preliminary_cluster_id"])
    ]
    raw["cluster_size_final"] = raw.groupby("preliminary_cluster_id")["original_row_id"].transform("size")
    as_of = pd.Series(index=raw.index, dtype="int64")
    for _, group in raw.groupby("preliminary_cluster_id"):
        sorted_times = np.sort(group["_published"].to_numpy(dtype="datetime64[ns]"))
        for index in group.index:
            value = raw.at[index, "_published"].to_datetime64()
            as_of.at[index] = int(np.searchsorted(sorted_times, value, side="right"))
    raw["cluster_size_as_of_publication"] = as_of.astype(int)
    raw["candidate_fetch_rank"] = pd.Series("", index=raw.index, dtype="object")
    for cluster_id, group in raw.groupby("preliminary_cluster_id"):
        representative_id = rep_row_ids[cluster_id]
        ordered = group.assign(
            _is_rep=group["original_row_id"].eq(representative_id),
            _has_url=group["url"].str.strip().ne(""),
        ).sort_values(
            ["_is_rep", "_has_url", "similarity_to_representative", "_published", "original_row_id"],
            ascending=[False, False, False, True, True], kind="stable"
        )
        used_domains, used_urls, rank = set(), set(), 1
        for index, row in ordered.iterrows():
            url = row["url"].strip()
            domain = row["domain"].strip().casefold()
            if not url or url in used_urls or (rank > 1 and domain in used_domains):
                continue
            raw.at[index, "candidate_fetch_rank"] = rank
            used_urls.add(url)
            used_domains.add(domain)
            rank += 1
            if rank > 3:
                break
    raw["clustering_status"] = np.where(
        raw["exact_duplicate"], "exact_duplicate",
        np.where(raw["cluster_size_final"].eq(1), "canonical_singleton", "canonical_clustered")
    )
    raw["_threshold"] = threshold
    return raw, canonical, cluster_members


def sensitivity_statistics(raw, canonical, clustered, cluster_members, threshold):
    canonical_sizes = pd.Series([len(v) for v in cluster_members.values()], dtype="int64")
    all_sizes = raw.groupby("preliminary_cluster_id").size()
    candidates = int(raw["candidate_fetch_rank"].ne("").sum())
    return {
        "similarity_threshold": threshold,
        "canonical_titles": len(canonical),
        "preliminary_clusters": len(canonical_sizes),
        "singletons": int(canonical_sizes.eq(1).sum()),
        "singleton_ratio": float(canonical_sizes.eq(1).mean()),
        "clusters_size_2_to_3": int(canonical_sizes.between(2, 3).sum()),
        "clusters_size_4_plus": int(canonical_sizes.ge(4).sum()),
        "maximum_cluster_size_canonical": int(canonical_sizes.max()),
        "average_cluster_size_canonical": float(canonical_sizes.mean()),
        "maximum_cluster_size_all_rows": int(all_sizes.max()),
        "near_duplicate_canonical_titles": int(len(canonical) - len(canonical_sizes)),
        "candidate_fetch_urls": candidates,
        "candidate_reduction_from_original_rows": int(len(raw) - candidates),
        "candidate_reduction_ratio": float(1 - candidates / len(raw)),
        "candidate_comparisons": int(clustered["candidate_comparisons"]),
        "max_active_window": int(clustered["max_active_window"]),
        "full_matrix_allocated": False,
    }


def build_review_sample(raw, canonical, clustered, threshold, seed):
    grouped = {cluster: group.copy() for cluster, group in raw.groupby("preliminary_cluster_id")}
    multi = sorted([cluster for cluster, group in grouped.items() if len(group) > 1])
    rng = random.Random(seed)
    random_clusters = rng.sample(multi, min(20, len(multi)))
    largest = sorted(multi, key=lambda c: (-len(grouped[c]), c))[:10]
    link_by_cluster = defaultdict(list)
    for cluster, score in zip(clustered["assignments"], clustered["link_similarity"]):
        if score < 1:
            link_by_cluster[cluster].append(score)
    boundary = sorted(
        link_by_cluster,
        key=lambda c: (min(abs(score - threshold) for score in link_by_cluster[c]), c)
    )[:20]
    conflicts = []
    for cluster in multi:
        values = grouped[cluster]["normalized_title"].tolist()
        if any(direction_conflict(a, b) for i, a in enumerate(values) for b in values[i + 1:]):
            conflicts.append(cluster)
    reasons = defaultdict(set)
    for cluster in random_clusters: reasons[cluster].add("random_multi_news")
    for cluster in largest: reasons[cluster].add("largest")
    for cluster in boundary: reasons[cluster].add("near_threshold")
    for cluster in conflicts: reasons[cluster].add("direction_conflict")
    rows = []
    for cluster in sorted(reasons):
        group = grouped[cluster].sort_values(["_published", "original_row_id"], kind="stable")
        for _, row in group.head(10).iterrows():
            rows.append({
                "review_reason": ";".join(sorted(reasons[cluster])),
                "preliminary_cluster_id": cluster,
                "cluster_size_final": len(group),
                "original_row_id": row["original_row_id"],
                "published_at": row["published_at"],
                "title": row["title"], "domain": row["domain"], "url": row["url"],
                "similarity_to_representative": row["similarity_to_representative"],
            })
    return pd.DataFrame(rows), len(conflicts)


def load_embeddings(titles, model_name, device, batch_size, seed, output_dir, input_sha, canonical_ids):
    cache_file = output_dir / "canonical_title_embeddings.npy"
    metadata_file = output_dir / "embedding_cache_metadata.json"
    canonical_hash = hashlib.sha256(chr(10).join(canonical_ids).encode("utf-8")).hexdigest()
    if cache_file.is_file() and metadata_file.is_file():
        cached = json.loads(metadata_file.read_text(encoding="utf-8"))
        expected = {
            "model": model_name,
            "input_sha256": input_sha,
            "canonical_ids_sha256": canonical_hash,
            "embedding_rows": len(canonical_ids),
            "normalized": True,
            "dtype": "float32",
        }
        mismatches = [key for key, value in expected.items() if cached.get(key) != value]
        if not mismatches:
            started = datetime.now(timezone.utc)
            embeddings = np.load(cache_file, allow_pickle=False)
            elapsed = (datetime.now(timezone.utc) - started).total_seconds()
            if embeddings.ndim != 2 or embeddings.shape[0] != len(canonical_ids):
                raise ValueError("Cached embedding shape does not match metadata.")
            if embeddings.shape[1] != cached.get("embedding_dimension"):
                raise ValueError("Cached embedding dimension does not match metadata.")
            runtime_metadata = dict(cached)
            runtime_metadata["cache_hit"] = True
            runtime_metadata["cache_load_seconds"] = elapsed
            return np.asarray(embeddings, dtype=np.float32), runtime_metadata

    import torch
    from sentence_transformers import SentenceTransformer
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    selected_device = device
    if device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    started = datetime.now(timezone.utc)
    model = SentenceTransformer(model_name, device=selected_device)
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(
            list(titles), batch_size=batch_size, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        )
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    embeddings = np.asarray(embeddings, dtype=np.float32)
    np.save(cache_file, embeddings)
    from huggingface_hub.constants import HF_HUB_CACHE
    ref_path = Path(HF_HUB_CACHE) / ("models--" + model_name.replace("/", "--")) / "refs/main"
    model_revision = ref_path.read_text(encoding="utf-8").strip() if ref_path.is_file() else "unknown"
    device_name = (
        torch.cuda.get_device_name(torch.cuda.current_device())
        if selected_device.startswith("cuda") else "CPU"
    )
    metadata = {
        "model": model_name, "model_revision": model_revision,
        "sentence_transformers_version": importlib.metadata.version("sentence-transformers"),
        "torch_version": torch.__version__, "device": selected_device, "device_name": device_name,
        "embedding_rows": len(embeddings), "embedding_dimension": int(embeddings.shape[1]),
        "embedding_seconds": elapsed, "normalized": True, "dtype": str(embeddings.dtype),
        "batch_size": batch_size, "seed": seed, "input_sha256": input_sha,
        "canonical_ids_sha256": canonical_hash, "cache_hit": False,
    }
    metadata_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return embeddings, metadata


def run(args, embedding_loader=load_embeddings):
    if not args.dry_run:
        raise ValueError("This planning tool requires --dry-run.")
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    raw_root = (Path(__file__).resolve().parent / "data/raw").resolve()
    if output_dir == input_path or raw_root in output_dir.parents or output_dir == raw_root:
        raise ValueError("Output directory cannot overlap data/raw or the input CSV.")
    output_dir.mkdir(parents=True, exist_ok=True)
    before_sha = sha256_file(input_path)
    frame = pd.read_csv(input_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if args.limit is not None:
        frame = frame.head(args.limit).copy()
    rows, canonical = prepare_rows(frame.reset_index(drop=True))
    duplicate_stats = exact_duplicate_statistics(rows)
    embeddings, embedding_meta = embedding_loader(
        canonical["normalized_title"].tolist(), args.model, args.device, args.batch_size,
        args.seed, output_dir, before_sha, canonical["original_row_id"].tolist()
    )
    sensitivity = []
    selected = None
    thresholds = sorted(set(args.thresholds + [args.similarity_threshold]))
    for threshold in thresholds:
        clustered = cluster_causal(
            embeddings, canonical["_published"], canonical["original_row_id"].tolist(),
            canonical["normalized_title"].tolist(), threshold, args.time_window_hours
        )
        manifest, canonical_run, cluster_members = materialize_manifest(rows, canonical, clustered, threshold)
        sensitivity.append(sensitivity_statistics(manifest, canonical_run, clustered, cluster_members, threshold))
        if math.isclose(threshold, args.similarity_threshold):
            selected = (manifest, canonical_run, clustered, cluster_members)
    manifest, canonical_run, clustered, cluster_members = selected
    review, conflict_groups = build_review_sample(
        manifest, canonical_run, clustered, args.similarity_threshold, args.seed
    )
    manifest[REQUIRED_OUTPUT_COLUMNS].to_csv(output_dir / "title_clustering_manifest.csv", index=False, encoding="utf-8-sig")
    review.to_csv(output_dir / "cluster_review_sample.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sensitivity).to_csv(output_dir / "threshold_sensitivity.csv", index=False)
    (output_dir / "threshold_sensitivity.json").write_text(json.dumps(sensitivity, indent=2), encoding="utf-8")
    after_sha = sha256_file(input_path)
    run_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(), "dry_run": True,
        "input": str(input_path), "input_rows_processed": len(rows),
        "input_sha256_before": before_sha, "input_sha256_after": after_sha,
        "input_unchanged": before_sha == after_sha, "input_columns": list(frame.columns),
        "field_mapping": {name: name if name in frame else None for name in ("title", "url", "date", "seendate", "domain", "source")},
        "original_row_id_present": "original_row_id" in frame,
        "original_row_id_strategy": "stable SHA-256 of source row fields plus identical-row occurrence number",
        "duplicate_statistics": duplicate_stats, "embedding": embedding_meta,
        "time_window_hours": args.time_window_hours,
        "causal_processing_order": ["published_at", "original_row_id"],
        "output_row_order": "original CSV row order",
        "thresholds_share_one_embedding_matrix": True,
        "embedding_cache_path": str(output_dir / "canonical_title_embeddings.npy"),
        "selected_similarity_threshold": args.similarity_threshold,
        "sensitivity": sensitivity, "direction_conflict_groups_selected_threshold": conflict_groups,
        "output_columns": REQUIRED_OUTPUT_COLUMNS,
        "safeguards": {
            "urls_fetched": False, "fingpt_called": False, "raw_input_modified": False,
            "formal_bertopic_run": False,
            "cluster_size_final_historical_feature_allowed": False,
        },
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if before_sha != after_sha:
        raise RuntimeError("Input SHA-256 changed during dry-run.")
    print(json.dumps(run_manifest, indent=2, ensure_ascii=False))
    return run_manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--time-window-hours", type=float, default=72)
    parser.add_argument("--similarity-threshold", type=float, default=0.90)
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.time_window_hours <= 0 or not 0 <= args.similarity_threshold <= 1:
        raise SystemExit("time window must be positive and threshold must be between zero and one")
    try:
        run(args)
    except Exception:
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
