"""Offline validator for FinGPT multidimensional annotation JSONL."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import sys
from typing import Any

SOURCE_TYPES = {"news", "earnings_call", "ten_k"}
SPLITS = {"train", "validation", "test"}
DIRECTIONAL_DIMENSIONS = (
    "sentiment", "growth_outlook", "financial_strength", "fundamental_impact",
    "short_term_impact", "long_term_impact",
)
BINARY_DIMENSIONS = ("risk_presence", "uncertainty_presence", "relevance", "price_narrative")
DIMENSIONS = DIRECTIONAL_DIMENSIONS + BINARY_DIMENSIONS
DIRECTIONAL_LABELS = {"negative", "neutral", "positive", "insufficient_information"}
BINARY_LABELS = {"yes", "no"}
REQUIRED_FIELDS = {
    "sample_id", "ticker", "source_type", "published_at", "as_of_date", "title",
    "text", "labels", "evidence", "annotator", "annotation_version", "split",
}


def canonical_input(record: dict[str, Any]) -> str:
    """Return the exact string against which all evidence offsets are measured."""
    title, text = record.get("title", ""), record.get("text", "")
    return f"{title}\n{text}" if title else text


def _normalized_text(record: dict[str, Any]) -> str:
    value = canonical_input(record).casefold()
    return " ".join(re.findall(r"\w+", value))


def _parse_published_at(value: Any) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("must be a nonempty ISO-8601 string")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(candidate)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("must include Z or an explicit UTC offset")
    return parsed.astimezone(timezone.utc)


def validate_record(record: Any, line_number: int) -> list[str]:
    prefix = f"line {line_number}"
    errors: list[str] = []
    if not isinstance(record, dict):
        return [f"{prefix}: record must be a JSON object"]
    missing = sorted(REQUIRED_FIELDS - record.keys())
    if missing:
        errors.append(f"{prefix}: missing required fields: {', '.join(missing)}")
    extra = sorted(record.keys() - REQUIRED_FIELDS)
    if extra:
        errors.append(f"{prefix}: unknown top-level fields: {', '.join(extra)}")
    sample_id = record.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        errors.append(f"{prefix}: sample_id must be a nonempty string")
    ticker = record.get("ticker")
    if not isinstance(ticker, str) or not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", ticker):
        errors.append(f"{prefix}: ticker must be an uppercase ticker symbol")
    if record.get("source_type") not in SOURCE_TYPES:
        errors.append(f"{prefix}: invalid source_type: {record.get('source_type')!r}")
    if record.get("split") not in SPLITS:
        errors.append(f"{prefix}: invalid split: {record.get('split')!r}")
    for field in ("title", "text", "annotator", "annotation_version"):
        if not isinstance(record.get(field), str):
            errors.append(f"{prefix}: {field} must be a string")
    if isinstance(record.get("text"), str) and not record["text"].strip():
        errors.append(f"{prefix}: text must not be empty")
    if isinstance(record.get("annotator"), str) and not record["annotator"].strip():
        errors.append(f"{prefix}: annotator must not be empty")
    if isinstance(record.get("annotation_version"), str) and record["annotation_version"] != "v1":
        errors.append(f"{prefix}: annotation_version must be 'v1'")
    published = None
    try:
        published = _parse_published_at(record.get("published_at"))
    except (ValueError, TypeError) as exc:
        errors.append(f"{prefix}: invalid published_at: {exc}")
    try:
        as_of = datetime.strptime(record.get("as_of_date", ""), "%Y-%m-%d").date()
        if published is not None and as_of != published.date():
            errors.append(f"{prefix}: as_of_date must equal the UTC date of published_at")
    except (ValueError, TypeError):
        errors.append(f"{prefix}: as_of_date must use YYYY-MM-DD")

    labels, evidence = record.get("labels"), record.get("evidence")
    if not isinstance(labels, dict):
        errors.append(f"{prefix}: labels must be an object")
        labels = {}
    if not isinstance(evidence, dict):
        errors.append(f"{prefix}: evidence must be an object")
        evidence = {}
    missing_dimensions = sorted(set(DIMENSIONS) - labels.keys())
    extra_dimensions = sorted(labels.keys() - set(DIMENSIONS))
    if missing_dimensions:
        errors.append(f"{prefix}: missing label dimensions: {', '.join(missing_dimensions)}")
    if extra_dimensions:
        errors.append(f"{prefix}: unknown label dimensions: {', '.join(extra_dimensions)}")
    unknown_evidence = sorted(evidence.keys() - set(DIMENSIONS))
    if unknown_evidence:
        errors.append(f"{prefix}: unknown evidence dimensions: {', '.join(unknown_evidence)}")

    source = canonical_input(record) if isinstance(record.get("title"), str) and isinstance(record.get("text"), str) else ""
    for dimension in DIMENSIONS:
        label = labels.get(dimension)
        allowed = DIRECTIONAL_LABELS if dimension in DIRECTIONAL_DIMENSIONS else BINARY_LABELS
        if label not in allowed:
            errors.append(f"{prefix}: invalid label for {dimension}: {label!r}")
        items = evidence.get(dimension, [])
        if not isinstance(items, list):
            errors.append(f"{prefix}: evidence.{dimension} must be a list")
            continue
        required = dimension in DIRECTIONAL_DIMENSIONS and label in {"negative", "neutral", "positive"}
        required = required or dimension in BINARY_DIMENSIONS and label == "yes"
        if required and not items:
            errors.append(f"{prefix}: evidence required for {dimension}={label}")
        if label == "insufficient_information" and items:
            errors.append(f"{prefix}: insufficient_information must not have evidence for {dimension}")
        for index, item in enumerate(items):
            location = f"{prefix}: evidence.{dimension}[{index}]"
            if not isinstance(item, dict) or set(item) != {"quote", "start", "end"}:
                errors.append(f"{location} must contain exactly quote/start/end")
                continue
            quote, start, end = item.get("quote"), item.get("start"), item.get("end")
            if not isinstance(quote, str) or not quote:
                errors.append(f"{location}.quote must be nonempty")
                continue
            if type(start) is not int or type(end) is not int or start < 0 or end <= start:
                errors.append(f"{location} has invalid offsets")
                continue
            if end > len(source) or source[start:end] != quote:
                errors.append(f"{location} quote does not exactly match canonical_input[start:end]")
    return errors


def validate_jsonl(path: str | Path, similarity_threshold: float = 0.92) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    records: list[dict[str, Any]] = []
    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return {"valid": False, "errors": [f"cannot read {path}: {exc}"], "warnings": [], "record_count": 0}
    if not lines:
        errors.append("file contains no JSONL records")
    for number, line in enumerate(lines, 1):
        if not line.strip():
            errors.append(f"line {number}: blank JSONL lines are not allowed")
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {number}: invalid JSON: {exc.msg}")
            continue
        record_errors = validate_record(record, number)
        errors.extend(record_errors)
        if isinstance(record, dict):
            records.append(record)

    ids: dict[str, int] = {}
    for number, record in enumerate(records, 1):
        sample_id = record.get("sample_id")
        if isinstance(sample_id, str) and sample_id in ids:
            errors.append(f"duplicate sample_id {sample_id!r} at parsed records {ids[sample_id]} and {number}")
        elif isinstance(sample_id, str):
            ids[sample_id] = number

    comparable = [(r, _normalized_text(r)) for r in records if isinstance(r.get("text"), str)]
    for left in range(len(comparable)):
        first, first_text = comparable[left]
        if not first_text:
            continue
        for right in range(left + 1, len(comparable)):
            second, second_text = comparable[right]
            if first.get("split") == second.get("split"):
                continue
            ratio = 1.0 if first_text == second_text else SequenceMatcher(None, first_text, second_text).ratio()
            if ratio == 1.0:
                errors.append(f"cross-split duplicate text: {first.get('sample_id')} / {second.get('sample_id')}")
            elif ratio >= similarity_threshold:
                warnings.append(
                    f"cross-split highly similar text ({ratio:.3f}): "
                    f"{first.get('sample_id')} / {second.get('sample_id')}"
                )

    source_counts = Counter(r.get("source_type") for r in records)
    distributions: dict[str, Counter[str]] = {dimension: Counter() for dimension in DIMENSIONS}
    for record in records:
        labels = record.get("labels")
        if isinstance(labels, dict):
            for dimension in DIMENSIONS:
                if labels.get(dimension) is not None:
                    distributions[dimension][str(labels[dimension])] += 1
    for dimension, counts in distributions.items():
        if records and len(counts) == 1:
            warnings.append(f"label collapse for {dimension}: all records use {next(iter(counts))!r}")
        allowed = DIRECTIONAL_LABELS if dimension in DIRECTIONAL_DIMENSIONS else BINARY_LABELS
        if len(records) >= 10:
            for label in sorted(allowed):
                fraction = counts.get(label, 0) / len(records)
                if fraction < 0.10:
                    warnings.append(f"severe imbalance for {dimension}={label}: {counts.get(label, 0)}/{len(records)}")

    return {
        "valid": not errors,
        "record_count": len(records),
        "errors": errors,
        "warnings": warnings,
        "source_counts": dict(sorted(source_counts.items(), key=lambda item: str(item[0]))),
        "label_distributions": {key: dict(sorted(value.items())) for key, value in distributions.items()},
        "similarity_threshold": similarity_threshold,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate multidimensional financial-text annotation JSONL offline.")
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--similarity-threshold", type=float, default=0.92)
    parser.add_argument("--summary-json", type=Path, help="Optional path for the validation summary.")
    args = parser.parse_args(argv)
    if not 0.0 < args.similarity_threshold <= 1.0:
        parser.error("--similarity-threshold must be in (0, 1]")
    result = validate_jsonl(args.jsonl, args.similarity_threshold)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.summary_json:
        args.summary_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
