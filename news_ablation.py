"""Offline FinGPT News ablation helpers for A/B/C/D experiments.

This module deliberately avoids loading the 13B model and only provides
causal, resumable, and offline-safe logic used to define the experiment
pipeline and validate the design without touching the raw dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


LABEL_SCORES = {"negative": -1, "neutral": 0, "positive": 1}
US_MARKET_TZ = ZoneInfo("America/New_York")


class ExperimentSpec(Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_title(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.translate(str.maketrans({
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-",
    }))
    text = text.casefold().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([,;:!?|])\s*", r"\1 ", text)
    text = re.sub(r"\s*\-\s*", " - ", text)
    text = text.strip(" \t\r\n-|:")
    return text


def build_cache_key(
    normalized_input: str,
    *,
    base_model: str,
    adapter_name: str,
    model_revision: str,
    quantization: str,
    prompt_version: str,
    generation_settings: dict[str, Any],
) -> str:
    payload = {
        "input_sha256": sha256_text(normalized_input),
        "base_model": base_model,
        "adapter_name": adapter_name,
        "model_revision": model_revision,
        "quantization": quantization,
        "prompt_version": prompt_version,
        "generation_settings": generation_settings,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime().isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return value


class NewsInferenceCache:
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def load(self, key: str) -> dict[str, Any]:
        path = self._path(key)
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, key: str, payload: dict[str, Any]) -> None:
        path = self._path(key)
        safe_payload = _json_safe(payload)
        path.write_text(json.dumps(safe_payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    def save_error(self, key: str, payload: dict[str, Any]) -> None:
        path = self._path(key)
        existing = self.load(key)
        merged = {**existing, **payload}
        path.write_text(json.dumps(_json_safe(merged), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _is_direction_conflict(left: str, right: str) -> bool:
    left_tokens = set(re.findall(r"[a-z0-9]+", left.casefold()))
    right_tokens = set(re.findall(r"[a-z0-9]+", right.casefold()))
    pairs = (
        ("raises", "cuts"), ("raised", "cut"), ("raise", "cut"),
        ("beats", "misses"), ("beat", "miss"), ("up", "down"),
        ("approval", "rejection"), ("approved", "rejected"),
        ("growth", "decline"), ("grows", "declines"),
    )
    for a, b in pairs:
        if (a in left_tokens and b in right_tokens) or (b in left_tokens and a in right_tokens):
            return True
    return False


def _token_overlap_similarity(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[a-z0-9]+", normalize_title(left).casefold()))
    right_tokens = set(re.findall(r"[a-z0-9]+", normalize_title(right).casefold()))
    if not left_tokens and not right_tokens:
        return 1.0 if normalize_title(left) == normalize_title(right) else 0.0
    if not left_tokens or not right_tokens:
        return 0.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def _is_literal_title_duplicate(left: str, right: str, *, time_window_hours: float = 72.0) -> bool:
    left_title = normalize_title(left)
    right_title = normalize_title(right)
    if not left_title or not right_title:
        return False
    if left_title == right_title:
        return True
    similarity = _token_overlap_similarity(left_title, right_title)
    return similarity >= 0.85


def _is_same_event_candidate(left: pd.Series, right: pd.Series, *, time_window_hours: float = 72.0) -> bool:
    left_url = str(left.get("url", "")).strip()
    right_url = str(right.get("url", "")).strip()
    if not left_url or not right_url:
        return False
    if left_url != right_url:
        return False

    left_ts = pd.to_datetime(left.get("published_at"), errors="coerce", utc=True)
    right_ts = pd.to_datetime(right.get("published_at"), errors="coerce", utc=True)
    if pd.isna(left_ts) or pd.isna(right_ts):
        return False
    delta_hours = abs((right_ts - left_ts).total_seconds()) / 3600.0
    if delta_hours > time_window_hours:
        return False

    left_title = normalize_title(left.get("title", ""))
    right_title = normalize_title(right.get("title", ""))
    if left_title and left_title == right_title:
        return True
    return _token_overlap_similarity(left_title, right_title) >= 0.90


def deduplicate_news(raw_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df = raw_df.copy()
    if "original_row_id" in df.columns:
        df["original_row_id"] = df["original_row_id"].astype(str)
    else:
        df["original_row_id"] = [str(idx + 1) for idx in range(len(df))]
    df["normalized_title"] = df["title"].map(normalize_title)
    df["published_at"] = pd.to_datetime(df.get("published_at", df.get("seendate", df.get("date", ""))), errors="coerce", utc=True)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    if df["published_at"].isna().any():
        fallback = pd.to_datetime(df.get("date", ""), errors="coerce", utc=True)
        df["published_at"] = df["published_at"].fillna(fallback)
    df["exact_duplicate"] = False
    df["duplicate_of_row_id"] = ""

    seen: list[tuple[int, pd.Timestamp, str, str]] = []
    for idx, row in df.iterrows():
        current_url = str(row.get("url", "")).strip()
        current_title = normalize_title(row.get("title", ""))
        current_ts = pd.to_datetime(row.get("published_at"), errors="coerce", utc=True)
        match_idx = None
        for prev_idx, prev_ts, prev_url, prev_title in seen:
            if pd.isna(current_ts) or pd.isna(prev_ts):
                continue
            delta_hours = abs((current_ts - prev_ts).total_seconds()) / 3600.0
            if delta_hours > 72.0:
                continue
            url_match = bool(current_url and prev_url and current_url == prev_url)
            same_date_title = bool(
                current_title and prev_title and current_ts.date() == prev_ts.date() and current_title == prev_title
            )
            url_title_match = bool(
                url_match and current_title and prev_title and (
                    current_title == prev_title or _token_overlap_similarity(current_title, prev_title) >= 0.85
                )
            )
            if same_date_title or url_title_match:
                match_idx = prev_idx
                break
        if match_idx is not None:
            df.at[idx, "exact_duplicate"] = True
            df.at[idx, "duplicate_of_row_id"] = df.at[match_idx, "original_row_id"]
            continue
        seen.append((idx, current_ts, current_url, current_title))

    raw = df.copy()
    canonical = df.loc[~df["exact_duplicate"]].copy().reset_index(drop=True)
    canonical["cluster_representative_row_id"] = canonical["original_row_id"]
    canonical["cluster_size_final"] = 1
    canonical["cluster_members"] = 1
    canonical["event_cluster_id"] = [f"cluster_{idx:05d}" for idx in range(len(canonical))]
    return {"raw": raw, "canonical": canonical}


def choose_cluster_representative(rows: pd.DataFrame) -> pd.Series:
    if rows.empty:
        raise ValueError("No rows to choose a representative from.")
    candidate = rows.copy()
    candidate["_published"] = pd.to_datetime(candidate["published_at"], errors="coerce", utc=True)
    if "domain" in candidate.columns:
        domain_series = candidate["domain"]
    else:
        domain_series = pd.Series([""] * len(candidate), index=candidate.index)
    candidate["_domain_priority"] = domain_series.map(
        lambda value: 0 if "nvidia" in str(value).lower() or "sec" in str(value).lower() else 1
    )
    candidate["_title_len"] = candidate["title"].str.len()
    result = candidate.sort_values(
        ["_published", "_domain_priority", "_title_len", "original_row_id"],
        ascending=[True, True, False, True],
        kind="stable",
    ).iloc[0]
    return result


def make_event_cluster_manifest(
    canonical_df: pd.DataFrame,
    similarity_threshold: float = 0.90,
    time_window_hours: float = 72,
) -> pd.DataFrame:
    df = canonical_df.copy().sort_values("published_at", kind="stable").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=[
            "original_row_id",
            "published_at",
            "event_cluster_id",
            "cluster_size_final",
            "cluster_representative_row_id",
            "similarity_to_representative",
        ])
    df["_published"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    active: list[tuple[int, str]] = []
    assignments: dict[str, str] = {}
    cluster_rows: dict[str, list[str]] = {}
    representative: dict[str, str] = {}
    similarity: dict[str, float] = {}

    for idx, row in df.iterrows():
        current_id = row["original_row_id"]
        best_idx = None
        best_score = -1.0
        active_filtered = []
        for prior_idx, prior_cluster in active:
            if row["_published"] - df.iloc[prior_idx]["_published"] > pd.Timedelta(hours=time_window_hours):
                continue
            active_filtered.append((prior_idx, prior_cluster))
            if _is_direction_conflict(str(row.get("title", "")), str(df.iloc[prior_idx].get("title", ""))):
                continue
            score = float(len(set(normalize_title(row.get("title", "")).split()) & set(normalize_title(df.iloc[prior_idx].get("title", "")).split()))) / max(1, len(set(normalize_title(row.get("title", "")).split()) | set(normalize_title(df.iloc[prior_idx].get("title", "")).split())))
            if score >= similarity_threshold and score > best_score:
                best_score = score
                best_idx = prior_idx
        if best_idx is not None:
            cluster_id = assignments[df.iloc[best_idx]["original_row_id"]]
        else:
            cluster_id = f"cluster_{len(cluster_rows) + 1:05d}"
        assignments[current_id] = cluster_id
        cluster_rows.setdefault(cluster_id, []).append(current_id)
        active = active_filtered + [(idx, cluster_id)]
        similarity[current_id] = best_score if best_score >= 0 else 1.0

    rep_lookup = {}
    for cluster_id, member_ids in cluster_rows.items():
        subset = df[df["original_row_id"].isin(member_ids)].copy()
        rep = choose_cluster_representative(subset)
        rep_lookup[cluster_id] = rep["original_row_id"]

    result = df[["original_row_id", "published_at"]].copy()
    result["event_cluster_id"] = result["original_row_id"].map(assignments)
    result["cluster_size_final"] = result["event_cluster_id"].map({cid: len(ids) for cid, ids in cluster_rows.items()})
    result["cluster_representative_row_id"] = result["event_cluster_id"].map(rep_lookup)
    result["similarity_to_representative"] = result["original_row_id"].map(similarity)
    result["similarity_to_representative"] = result["similarity_to_representative"].replace(-1.0, 1.0)
    return result


def _coerce_price_calendar(price_calendar: pd.Index | list | tuple | set | str | Path | None) -> pd.DatetimeIndex:
    if price_calendar is None:
        price_calendar = pd.bdate_range(start="2020-01-01", end="2035-12-31")
    if isinstance(price_calendar, (str, Path)):
        prices = pd.read_csv(price_calendar)
        if "date" not in prices.columns:
            raise ValueError("Price CSV must contain a date column.")
        dates = pd.to_datetime(prices["date"], errors="raise").dt.tz_localize(None).dt.normalize()
        price_calendar = pd.DatetimeIndex(dates)
    else:
        price_calendar = pd.DatetimeIndex(pd.to_datetime(price_calendar, errors="raise")).tz_localize(None).normalize()
    if price_calendar.empty:
        raise ValueError("Price calendar is empty.")
    if price_calendar.has_duplicates:
        raise ValueError("Price calendar dates must be unique.")
    if not price_calendar.is_monotonic_increasing:
        raise ValueError("Price calendar dates must be sorted ascending.")
    if (price_calendar.weekday >= 5).any():
        raise ValueError("Price calendar must not include Saturday or Sunday dates.")
    return price_calendar


def _next_calendar_date(candidate_date: date, trading_dates: pd.DatetimeIndex) -> date | None:
    if candidate_date in set(trading_dates.date):
        return candidate_date
    later = trading_dates[trading_dates >= pd.Timestamp(candidate_date)]
    if later.empty:
        return None
    next_date = later[0].date()
    if next_date < candidate_date:
        return None
    return next_date


def align_news_to_trading_date(
    frame: pd.DataFrame,
    *,
    price_calendar: pd.Index | list | tuple | set | str | Path | None = None,
    prices_path: str | Path | None = None,
) -> pd.DataFrame:
    trading_dates = _coerce_price_calendar(price_calendar if price_calendar is not None else prices_path)
    out = frame.copy()
    published = pd.to_datetime(
        out.get("published_at", out.get("seendate", out.get("date", ""))),
        errors="coerce",
        utc=True,
    )
    out["_published_at"] = pd.to_datetime(published, errors="coerce", utc=True)
    out["effective_trading_date"] = pd.NA
    for idx, row in out.iterrows():
        ts = row["_published_at"]
        if pd.isna(ts):
            fallback = pd.to_datetime(row.get("date", ""), errors="coerce")
            ts = pd.Timestamp(fallback, tz="UTC") if pd.notna(fallback) else pd.NaT
        if pd.isna(ts):
            out.at[idx, "effective_trading_date"] = pd.NA
            continue
        local = ts.tz_convert(US_MARKET_TZ)
        candidate_date = local.date()
        if local.time() >= datetime.strptime("16:00:00", "%H:%M:%S").time():
            candidate_date += timedelta(days=1)
        if candidate_date.weekday() >= 5:
            next_date = _next_calendar_date(candidate_date, trading_dates)
            if next_date is None:
                raise ValueError(f"News publication date {ts.isoformat()} falls outside the price calendar range.")
            out.at[idx, "effective_trading_date"] = next_date.isoformat()
            continue
        if candidate_date not in set(trading_dates.date):
            next_date = _next_calendar_date(candidate_date, trading_dates)
            if next_date is None:
                raise ValueError(f"News publication date {ts.isoformat()} falls outside the price calendar range.")
            out.at[idx, "effective_trading_date"] = next_date.isoformat()
            continue
        out.at[idx, "effective_trading_date"] = candidate_date.isoformat()
    bad = out["effective_trading_date"].isna() | ~out["effective_trading_date"].isin(trading_dates.strftime("%Y-%m-%d"))
    if bad.any():
        raise ValueError("Effective trading dates must all belong to the authoritative price calendar.")
    return out.drop(columns=["_published_at"], errors="ignore")


def _normalize_effective_trading_date_series(series: pd.Series) -> pd.Series:
    values = pd.Series(series, copy=True)
    if values.empty:
        return pd.Series([], index=values.index, dtype="datetime64[ns]")
    try:
        normalized = pd.to_datetime(values, errors="raise", utc=True, format="mixed")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid effective_trading_date values encountered: {exc}") from exc
    if normalized.isna().any():
        raise ValueError("Invalid effective_trading_date: missing dates are not allowed.")
    return normalized.dt.tz_localize(None).dt.normalize().astype("datetime64[ns]")


def build_daily_dissemination_features(rows: pd.DataFrame) -> pd.DataFrame:
    df = rows.copy()
    if df.empty:
        return pd.DataFrame(columns=[
            "effective_trading_date",
            "cluster_size_asof_t",
            "unique_domain_count_asof_t",
            "dissemination_span_hours_asof_t",
        ])
    if "effective_trading_date" not in df.columns:
        df = align_news_to_trading_date(df)
    df["effective_trading_date"] = _normalize_effective_trading_date_series(df["effective_trading_date"])
    required = {"cluster_size_asof_t", "unique_domain_count_asof_t", "dissemination_span_hours_asof_t"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=[
            "effective_trading_date",
            "cluster_size_asof_t",
            "unique_domain_count_asof_t",
            "dissemination_span_hours_asof_t",
        ])
    daily = (
        df.groupby("effective_trading_date", dropna=False)
        .agg(
            cluster_size_asof_t=("cluster_size_asof_t", "sum"),
            unique_domain_count_asof_t=("unique_domain_count_asof_t", "max"),
            dissemination_span_hours_asof_t=("dissemination_span_hours_asof_t", "max"),
        )
        .reset_index()
    )
    return daily


def build_daily_news_features(rows: pd.DataFrame, *, include_dissemination: bool = False) -> pd.DataFrame:
    df = rows.copy()
    if df.empty:
        empty = pd.DataFrame(columns=[
            "effective_trading_date",
            "news_sentiment_mean",
            "news_sentiment_sum",
            "news_sentiment_std",
            "news_positive_count",
            "news_neutral_count",
            "news_negative_count",
            "news_positive_ratio",
            "news_neutral_ratio",
            "news_negative_ratio",
            "news_item_or_event_count",
            "news_has_data",
            "news_inference_failure_count",
        ])
        if include_dissemination:
            empty["cluster_size_asof_t"] = pd.Series(dtype="float64")
            empty["unique_domain_count_asof_t"] = pd.Series(dtype="float64")
            empty["dissemination_span_hours_asof_t"] = pd.Series(dtype="float64")
        empty["effective_trading_date"] = pd.Series(dtype="datetime64[ns]")
        return empty
    df["effective_trading_date"] = _normalize_effective_trading_date_series(df["effective_trading_date"])
    df["sentiment_score"] = pd.to_numeric(df.get("sentiment_score", pd.Series(index=df.index, dtype=float)), errors="coerce")
    df["sentiment_label"] = df.get("sentiment_label", pd.Series("", index=df.index)).fillna("").astype(str).str.strip().str.lower()
    df["fingpt_error"] = df.get("fingpt_error", pd.Series("", index=df.index)).fillna("").astype(str)
    valid = (df["sentiment_label"].isin(LABEL_SCORES)
             & df["fingpt_error"].eq("") & df["sentiment_score"].notna())
    daily = []
    for day, group in df.groupby("effective_trading_date", sort=True):
        successful = group.loc[valid.loc[group.index]]
        score = successful["sentiment_score"]
        n = len(successful)
        row = {
            "effective_trading_date": day,
            "news_sentiment_mean": float(score.mean()) if n else np.nan,
            "news_sentiment_sum": float(score.sum()) if n else np.nan,
            "news_sentiment_std": float(score.std(ddof=0)) if n else np.nan,
            "news_item_or_event_count": len(group),
            "news_has_data": int(n > 0),
            "news_inference_failure_count": len(group) - n,
        }
        for label in LABEL_SCORES:
            count = int(successful["sentiment_label"].eq(label).sum())
            row[f"news_{label}_count"] = count
            row[f"news_{label}_ratio"] = count / n if n else np.nan
        daily.append(row)
    result = pd.DataFrame(daily)
    result["effective_trading_date"] = _normalize_effective_trading_date_series(result["effective_trading_date"])
    if include_dissemination:
        dissemination = build_daily_dissemination_features(df)
        dissemination["effective_trading_date"] = _normalize_effective_trading_date_series(dissemination["effective_trading_date"])
        result = result.merge(dissemination, on="effective_trading_date", how="left", validate="one_to_one")
    return result


def build_dissemination_features(rows: pd.DataFrame, *, as_of_date: str | date | None = None) -> pd.DataFrame:
    df = rows.copy()
    if df.empty:
        return df.copy()
    if "published_at" not in df.columns:
        raise ValueError("Rows must carry a published_at column to compute dissemination features.")
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    cluster_col = "event_cluster_id" if "event_cluster_id" in df.columns else "preliminary_cluster_id"
    if cluster_col not in df.columns:
        raise ValueError("Rows must carry an event cluster identifier to compute dissemination features.")

    df["cluster_size_asof_t"] = 0
    df["unique_domain_count_asof_t"] = 0
    df["dissemination_span_hours_asof_t"] = 0.0

    cutoff_ts = None
    if as_of_date is not None:
        raw = pd.Timestamp(as_of_date)
        if getattr(raw, "tzinfo", None) is None:
            cutoff_ts = raw.tz_localize("UTC")
        else:
            cutoff_ts = raw.tz_convert("UTC")
        cutoff_ts = cutoff_ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    for cluster_id, group in df.groupby(cluster_col, dropna=False):
        ordered = group.sort_values("published_at", kind="stable").copy()
        for idx in ordered.index:
            row = ordered.loc[idx]
            ts = row["published_at"]
            if pd.isna(ts):
                continue
            if cutoff_ts is not None and ts > cutoff_ts:
                continue
            prior = ordered[ordered["published_at"] <= ts].copy()
            if cutoff_ts is not None:
                prior = prior[prior["published_at"] <= cutoff_ts].copy()
            size = int(len(prior))
            unique_count = int(prior.loc[prior["domain"].astype(str).str.strip().ne(""), "domain"].astype(str).str.strip().nunique())
            span_hours = 0.0
            if len(prior) >= 2:
                span_hours = float((prior["published_at"].max() - prior["published_at"].min()).total_seconds() / 3600.0)
            df.at[idx, "cluster_size_asof_t"] = size
            df.at[idx, "unique_domain_count_asof_t"] = unique_count
            df.at[idx, "dissemination_span_hours_asof_t"] = span_hours
    return df


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_cluster_manifest(cluster_manifest_path: str | Path, *, raw_input_path: str | Path | None = None) -> pd.DataFrame:
    path = Path(cluster_manifest_path)
    if not path.exists():
        raise ValueError("A valid cluster manifest is required for C and D. Provide --cluster-manifest PATH.")
    manifest = pd.read_csv(path)
    required = {"original_row_id", "published_at", "title", "domain", "preliminary_cluster_id"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Cluster manifest missing required columns: {', '.join(missing)}")

    run_manifest_path = path.parent / "run_manifest.json"
    if not run_manifest_path.exists():
        raise ValueError("Cluster manifest is missing its run_manifest.json metadata; cannot validate the manifest source.")
    metadata = json.loads(run_manifest_path.read_text(encoding="utf-8"))

    if raw_input_path is not None and str(raw_input_path) != "<in-memory>":
        raw_path = Path(raw_input_path)
        if raw_path.exists():
            raw_sha = _sha256_file(raw_path)
            manifest_sha = metadata.get("input_sha256_before")
            if manifest_sha and manifest_sha != raw_sha:
                raise ValueError(f"Cluster manifest belongs to raw CSV SHA-256 {manifest_sha}, not {raw_sha}.")

    threshold = metadata.get("selected_similarity_threshold")
    if threshold is not None and not pd.isna(threshold) and float(threshold) != 0.90:
        raise ValueError("Cluster manifest must use similarity threshold 0.90.")
    window = metadata.get("time_window_hours")
    if window is not None and not pd.isna(window) and float(window) != 72.0:
        raise ValueError("Cluster manifest must use a 72-hour causal time window.")
    model = metadata.get("embedding", {}).get("model")
    if model is None:
        raise ValueError("Cluster manifest metadata does not include the Sentence Transformer model name.")
    if metadata.get("embedding", {}).get("model_revision") is None:
        raise ValueError("Cluster manifest metadata does not include the model revision.")
    return manifest


def _stable_row_ids(frame: pd.DataFrame) -> pd.Series:
    occurrences: dict[str, int] = {}
    result: list[str] = []
    for record in frame.astype(str).to_dict("records"):
        payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        occurrences[payload] = occurrences.get(payload, 0) + 1
        identity = payload + "#occurrence=" + str(occurrences[payload])
        result.append("news_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20])
    return pd.Series(result, index=frame.index, dtype="string")


def _manifest_duplicate_flags(manifest: pd.DataFrame) -> pd.DataFrame:
    out = manifest.copy()
    exact_flags = out.get("exact_duplicate", pd.Series([False] * len(out), index=out.index))
    is_exact = exact_flags.fillna(False).astype(str).str.lower().eq("true")
    out["retained_in_B"] = (~is_exact).astype(bool)
    out["duplicate_reason"] = np.where(is_exact, "same-day-title-or-exact-repeat", "")
    out["canonical_row_id"] = out["original_row_id"].where(out["retained_in_B"], out["duplicate_of_row_id"].fillna(""))
    return out


def build_stage2_rows(raw_news: pd.DataFrame, cluster_manifest: pd.DataFrame, *, price_calendar: pd.Index | list | tuple | set | str | Path | None = None, prices_path: str | Path | None = None) -> dict[str, Any]:
    raw = raw_news.copy()
    if "original_row_id" not in raw.columns:
        raw["original_row_id"] = _stable_row_ids(raw)
    raw["published_at"] = pd.to_datetime(raw.get("published_at", raw.get("seendate", raw.get("date", ""))), errors="coerce", utc=True)
    if "normalized_title" not in raw.columns:
        raw["normalized_title"] = [normalize_title(str(title)) for title in raw["title"]]
    manifest = _manifest_duplicate_flags(cluster_manifest.copy())
    if "published_at" not in manifest.columns:
        manifest["published_at"] = pd.to_datetime(manifest.get("seendate", manifest.get("date", "")), errors="coerce", utc=True)
    if "normalized_title" not in manifest.columns:
        manifest["normalized_title"] = [normalize_title(str(title)) for title in manifest["title"]]
    manifest["_published_ts"] = pd.to_datetime(manifest["published_at"], errors="coerce", utc=True)
    manifest["_published_date"] = manifest["_published_ts"].dt.strftime("%Y-%m-%d")
    manifest["eligible_for_dissemination"] = manifest["retained_in_B"].astype(bool).copy()
    if len(manifest):
        for (title_key, date_key), group in manifest.groupby(["normalized_title", "_published_date"], sort=False):
            seen_urls = set()
            ordered = group.sort_values("_published_ts", kind="stable")
            for idx in ordered.index:
                row = ordered.loc[idx]
                url = str(row.get("url", "")).strip()
                if bool(row["retained_in_B"]):
                    if url:
                        seen_urls.add(url)
                    continue
                if not url:
                    manifest.at[idx, "eligible_for_dissemination"] = False
                    continue
                if url in seen_urls:
                    manifest.at[idx, "eligible_for_dissemination"] = False
                else:
                    manifest.at[idx, "eligible_for_dissemination"] = True
                    seen_urls.add(url)
    B = manifest.loc[manifest["retained_in_B"]].copy().reset_index(drop=True)
    B_count = int(len(B))
    if len(manifest["original_row_id"].dropna()) != len(manifest):
        raise ValueError("Stage 2 B manifest rows must be preserved.")
    if B_count != int((~manifest["exact_duplicate"].fillna(False).astype(str).str.lower().eq("true")).sum()):
        raise ValueError("Stage 2 B count must equal the Stage 1 manifest B count.")

    trading_dates = _coerce_price_calendar(price_calendar if price_calendar is not None else prices_path)
    eligible = manifest.copy()
    if eligible.empty:
        D = pd.DataFrame(columns=["original_row_id", "published_at", "effective_trading_date", "cluster_id", "cluster_size_asof_t", "unique_domain_count_asof_t", "dissemination_span_hours_asof_t", "domain", "title", "url"])
    else:
        aligned = align_news_to_trading_date(eligible[["published_at", "title", "url", "domain", "preliminary_cluster_id", "original_row_id"]], price_calendar=trading_dates)
        aligned["published_at"] = pd.to_datetime(aligned["published_at"], errors="coerce", utc=True)
        asof_rows: list[dict[str, Any]] = []
        for (cluster_id, effective_date), group in aligned.groupby(["preliminary_cluster_id", "effective_trading_date"], sort=False):
            ordered = group.sort_values("published_at", kind="stable").copy()
            size = int(len(ordered))
            unique_count = int(ordered["domain"].fillna("").astype(str).str.strip().nunique())
            span = float((ordered["published_at"].max() - ordered["published_at"].min()).total_seconds() / 3600.0) if len(ordered) >= 2 else 0.0
            asof_rows.append({
                "original_row_id": ordered["original_row_id"].iloc[0],
                "cluster_id": str(cluster_id),
                "effective_trading_date": str(effective_date),
                "published_at": ordered["published_at"].max(),
                "domain": ordered["domain"].iloc[0],
                "title": ordered["title"].iloc[0],
                "url": ordered["url"].iloc[0],
                "cluster_size_asof_t": size,
                "unique_domain_count_asof_t": unique_count,
                "dissemination_span_hours_asof_t": span,
            })
        D = pd.DataFrame(asof_rows)
        if D.empty:
            D = pd.DataFrame(columns=["original_row_id", "published_at", "effective_trading_date", "cluster_id", "cluster_size_asof_t", "unique_domain_count_asof_t", "dissemination_span_hours_asof_t", "domain", "title", "url"])
        else:
            D = D.sort_values(["cluster_id", "effective_trading_date", "published_at"], kind="stable").reset_index(drop=True)
            if D.duplicated(subset=["cluster_id", "effective_trading_date"]).any():
                raise ValueError("Duplicate D keys are not allowed.")
            if not set(["cluster_size_asof_t", "unique_domain_count_asof_t", "dissemination_span_hours_asof_t"]).issubset(D.columns):
                raise ValueError("D snapshot output is missing dissemination columns.")
            if not (D["cluster_size_asof_t"].ge(1).all() and D["unique_domain_count_asof_t"].ge(1).all() and D["dissemination_span_hours_asof_t"].ge(0).all()):
                raise ValueError("D snapshot as-of columns must be valid and non-negative.")
    C_count = int(manifest["preliminary_cluster_id"].dropna().nunique())
    D_count = int(len(D))
    if D_count < C_count:
        raise ValueError("D snapshot rows must be >= the number of C clusters.")
    if D_count:
        invalid_dates = int((~D["effective_trading_date"].isin(trading_dates.strftime("%Y-%m-%d"))).sum())
    else:
        invalid_dates = 0
    if invalid_dates != 0:
        raise ValueError("D output contains invalid effective_trading_date values outside the price calendar.")
    summary = {
        "A": {"rows": int(len(raw)), "experiment": ExperimentSpec.A.value},
        "B": {"rows": B_count, "experiment": ExperimentSpec.B.value},
        "C": {"rows": C_count, "experiment": ExperimentSpec.C.value},
        "D": {"rows": D_count, "experiment": ExperimentSpec.D.value},
        "duplicate_D_keys": int(D.duplicated(subset=["cluster_id", "effective_trading_date"]).sum()) if not D.empty else 0,
        "B_source": "Stage 1 manifest retained rows only",
        "invalid_effective_trading_dates": int(invalid_dates),
        "cluster_size_mean": float(D["cluster_size_asof_t"].mean()) if not D.empty else 0.0,
        "cluster_size_max": int(D["cluster_size_asof_t"].max()) if not D.empty else 0,
        "domain_count_mean": float(D["unique_domain_count_asof_t"].mean()) if not D.empty else 0.0,
        "domain_count_max": int(D["unique_domain_count_asof_t"].max()) if not D.empty else 0,
        "dissemination_span_mean": float(D["dissemination_span_hours_asof_t"].mean()) if not D.empty else 0.0,
        "dissemination_span_max": float(D["dissemination_span_hours_asof_t"].max()) if not D.empty else 0.0,
    }
    return {
        "B": B,
        "B_count": B_count,
        "C_count": C_count,
        "D": D,
        "D_count": D_count,
        "summary": summary,
        "invalid_effective_trading_dates": invalid_dates,
    }


def write_offline_validation_report(report_path: str | Path, *, raw_file: str | Path, price_file: str | Path, manifest_file: str | Path, stage2_dir: str | Path) -> Path:
    report_path = Path(report_path)
    raw_path = Path(raw_file)
    price_path = Path(price_file)
    manifest_path = Path(manifest_file)
    stage2_output = Path(stage2_dir)
    stage2_summary = json.loads((stage2_output / "experiment_summary.json").read_text(encoding="utf-8"))
    d_file = stage2_output / "d_snapshot_manifest.csv"
    if d_file.exists():
        d_df = pd.read_csv(d_file)
    else:
        d_df = pd.DataFrame()
    raw_sha = _sha256_file(raw_path)
    price_sha = _sha256_file(price_path)
    manifest_sha = _sha256_file(manifest_path)
    stage2_sha = _sha256_file(stage2_output / "experiment_summary.json")
    lines = [
        "# NEWS_ABLATION_OFFLINE_VALIDATION",
        "",
        "## Raw file SHA",
        f"- raw SHA: {raw_sha}",
        f"- price SHA: {price_sha}",
        f"- manifest SHA: {manifest_sha}",
        f"- stage2 output SHA: {stage2_sha}",
        "",
        "## A/B/C/D counts",
        f"- A raw rows: {stage2_summary['A']['rows']}",
        f"- B canonical rows: {stage2_summary['B']['rows']}",
        f"- C representative clusters: {stage2_summary['C']['rows']}",
        f"- D daily event snapshots: {stage2_summary['D']['rows']}",
        "",
        "## D snapshot validation",
        f"- duplicate D keys: {int(d_df.duplicated(subset=['cluster_id', 'effective_trading_date']).sum()) if not d_df.empty else 0}",
        f"- cluster_size_asof_t present: {'yes' if 'cluster_size_asof_t' in d_df.columns else 'no'}",
        f"- unique_domain_count_asof_t present: {'yes' if 'unique_domain_count_asof_t' in d_df.columns else 'no'}",
        f"- dissemination_span_hours_asof_t present: {'yes' if 'dissemination_span_hours_asof_t' in d_df.columns else 'no'}",
        f"- invalid effective dates: {int(d_df.loc[~d_df['effective_trading_date'].isin(pd.read_csv(price_file)['date'].astype(str))].shape[0]) if not d_df.empty else 0}",
        "",
        "## Final statement",
        "This report is generated from actual output files and is not hand-entered.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_experiment_matrix(raw_news: pd.DataFrame, *, cache_dir: str | Path = "data/pipeline_runs/test_cache", cluster_manifest: str | Path | None = None, prices_path: str | Path | None = None) -> dict[str, Any]:
    if cluster_manifest is None:
        raise ValueError("A valid cluster manifest is required for C and D. Provide --cluster-manifest PATH.")
    raw = raw_news.copy()
    if "published_at" not in raw.columns and "seendate" in raw.columns:
        raw["published_at"] = raw["seendate"]
    if "published_at" in raw.columns:
        raw["published_at"] = pd.to_datetime(raw["published_at"], errors="coerce", utc=True)
    manifest = validate_cluster_manifest(cluster_manifest, raw_input_path="<in-memory>")
    price_calendar = None
    if prices_path is not None:
        price_calendar = _coerce_price_calendar(prices_path)
    elif "published_at" in raw.columns:
        dates = pd.to_datetime(raw["published_at"], errors="coerce", utc=True).dt.tz_convert(US_MARKET_TZ).dt.date
        price_calendar = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates.dropna().drop_duplicates()).astype(str), errors="coerce").normalize())
    stage2 = build_stage2_rows(raw, manifest, price_calendar=price_calendar, prices_path=prices_path)
    aligned = align_news_to_trading_date(raw, price_calendar=price_calendar if price_calendar is not None else None)
    summary = {
        "A": {"rows": len(raw), "experiment": ExperimentSpec.A.value},
        "B": {"rows": stage2["B_count"], "experiment": ExperimentSpec.B.value},
        "C": {"rows": stage2["C_count"], "experiment": ExperimentSpec.C.value},
        "D": {"rows": stage2["D_count"], "experiment": ExperimentSpec.D.value},
        "cache_dir": str(Path(cache_dir)),
        "aligned_dates": [
            {"published_at": _json_safe(row["published_at"]), "effective_trading_date": _json_safe(row["effective_trading_date"])}
            for row in aligned[["published_at", "effective_trading_date"]].to_dict("records")[:5]
        ],
    }
    return summary


def _aggregate_existing_inference_run(run_dir: str | Path, *, prices_path: str | Path | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    if prices_path is None:
        raise ValueError("Postprocess-only requires a frozen --prices calendar.")
    calendar = _coerce_price_calendar(prices_path).astype("datetime64[ns]")
    inference_dir = run_dir / "inference_results"
    if not inference_dir.is_dir():
        raise FileNotFoundError(f"Inference directory not found: {inference_dir}")
    frames: dict[str, pd.DataFrame] = {}
    for experiment in ("A", "B", "C", "D"):
        csv_path = inference_dir / f"{experiment}_inference_results.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing inference CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        if df.empty:
            raise ValueError(f"Empty inference CSV: {csv_path}")
        df["effective_trading_date"] = _normalize_effective_trading_date_series(df["effective_trading_date"])
        if not df["effective_trading_date"].isin(calendar).all():
            raise ValueError(f"{experiment}: dates outside frozen price calendar")
        if experiment == "D":
            cluster_column = "cluster_id" if "cluster_id" in df else "document_id"
            if df[cluster_column].isna().any() or df.duplicated([cluster_column, "effective_trading_date"]).any():
                raise ValueError("Duplicate or missing D cluster/date keys")
            for column, minimum in (("cluster_size_asof_t", 1), ("unique_domain_count_asof_t", 1), ("dissemination_span_hours_asof_t", 0)):
                values = pd.to_numeric(df[column], errors="raise")
                if not (np.isfinite(values) & values.ge(minimum)).all():
                    raise ValueError(f"Invalid D dissemination column: {column}")
        frames[experiment] = df
    inference_hashes = {experiment: _sha256_file(inference_dir / f"{experiment}_inference_results.csv") for experiment in frames}
    daily_dir = run_dir / "daily_features"
    daily_dir.mkdir(parents=True, exist_ok=True)
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    daily_outputs: dict[str, pd.DataFrame] = {}
    for experiment, frame in frames.items():
        daily = build_daily_news_features(frame, include_dissemination=(experiment == "D"))
        daily["effective_trading_date"] = pd.to_datetime(daily["effective_trading_date"], errors="raise").dt.strftime("%Y-%m-%d")
        daily_outputs[experiment] = daily
        csv_path = daily_dir / f"{experiment}_daily_features.csv"
        daily.to_csv(csv_path, index=False)

    report_lines = [
        "# POSTPROCESS-ONLY DAILY AGGREGATION REPORT",
        "",
        f"run_dir: {run_dir}",
        f"prices_path: {str(prices_path) if prices_path is not None else 'not provided'}",
        "",
        "## inference counts",
    ]
    for experiment, frame in frames.items():
        report_lines.append(f"- {experiment}: {len(frame)} rows")
    report_lines.extend([
        "",
        "## daily feature files",
    ])
    for experiment in ("A", "B", "C", "D"):
        report_lines.append(f"- {experiment}: {daily_dir / f'{experiment}_daily_features.csv'}")
    (report_dir / "postprocess_summary.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    run_manifest = {
        "mode": "postprocess-only",
        "new_inference_count": 0,
        "inference_sha256": inference_hashes,
        "prices_path": str(Path(prices_path).resolve()),
        "prices_sha256": _sha256_file(Path(prices_path)),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "inference_csvs": {experiment: str(inference_dir / f"{experiment}_inference_results.csv") for experiment in ("A", "B", "C", "D")},
        "daily_feature_csvs": {experiment: str(daily_dir / f"{experiment}_daily_features.csv") for experiment in ("A", "B", "C", "D")},
        "safeguards": {
            "fingpt_loaded": False,
            "base_model_loaded": False,
            "lora_loaded": False,
            "cuda_inference_called": False,
            "transformer_training_run": False,
            "model_inference_restarted": False,
        },
        "inference_counts": {experiment: int(len(frame)) for experiment, frame in frames.items()},
        "daily_row_counts": {experiment: int(len(daily_outputs[experiment])) for experiment in ("A", "B", "C", "D")},
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return run_manifest


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None, help="Optional raw news CSV for an offline dry-run.")
    parser.add_argument("--prices", type=Path, default=None, help="Authoritative NVDA price calendar CSV with a date column; used to map published_at to effective_trading_date.")
    parser.add_argument("--cluster-manifest", type=Path, default=None, help="Required manifest from cluster_news_titles.py for C/D event identity.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory for offline manifests.")
    parser.add_argument("--cache-dir", type=Path, default="data/pipeline_runs/news_cache", help="Directory for resumable inference cache.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing cache and do not re-run successful rows.")
    parser.add_argument("--dry-run", action="store_true", help="Offline-only dry run; never writes to raw data.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for offline testing.")
    parser.add_argument("--aggregate-only", "--postprocess-only", dest="aggregate_only", action="store_true", help="Read existing A/B/C/D inference CSVs and generate only the daily feature and report artifacts without loading the FinGPT model or any model weights.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Existing pipeline run directory containing inference_results/, cache/, and reports/. Used with --aggregate-only.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)
    if args.aggregate_only:
        if args.run_dir is None or args.prices is None:
            parser.error("--postprocess-only requires --run-dir and --prices")
        run_dir = args.run_dir
        report = _aggregate_existing_inference_run(run_dir, prices_path=args.prices)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    if args.input is None and args.output_dir is None and not args.dry_run:
        parser.print_help()
        return 0

    if args.input is not None:
        df = pd.read_csv(args.input)
        if args.limit is not None:
            df = df.head(args.limit)
        if args.cluster_manifest is None:
            raise ValueError("A valid cluster manifest is required for C and D. Provide --cluster-manifest PATH.")
        report = run_experiment_matrix(df, cache_dir=args.cache_dir, cluster_manifest=args.cluster_manifest, prices_path=args.prices)
        if args.output_dir is not None:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            manifest = validate_cluster_manifest(args.cluster_manifest, raw_input_path=str(args.input))
            stage2 = build_stage2_rows(df, manifest, prices_path=args.prices)
            stage2["B"].to_csv(output_dir / "b_manifest.csv", index=False)
            stage2["D"].to_csv(output_dir / "d_snapshot_manifest.csv", index=False)
            (output_dir / "experiment_summary.json").write_text(json.dumps(stage2["summary"], indent=2, ensure_ascii=False), encoding="utf-8")
            (output_dir / "experiment_summary.csv").write_text("experiment,rows\n" + "\n".join(f"{k},{v['rows']}" for k, v in stage2["summary"].items() if k in {"A", "B", "C", "D"}), encoding="utf-8")
            write_offline_validation_report(output_dir / "NEWS_ABLATION_OFFLINE_VALIDATION.md", raw_file=args.input, price_file=args.prices, manifest_file=args.cluster_manifest, stage2_dir=output_dir)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(json.dumps({"dry_run": bool(args.dry_run), "resume": bool(args.resume), "cache_dir": str(args.cache_dir)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
