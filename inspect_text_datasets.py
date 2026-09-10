"""Read-only CSV inventory. No model, tokenizer, network, clustering or text rewriting."""
import argparse
from datetime import datetime
import hashlib
from html import escape
import json
from pathlib import Path
import re
import sys

import pandas as pd

import config

PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_STEM = "dataset_inspection_report"
SPECS = {
    "news": {"config": "RAW_NEWS_FILE", "required": ["date", "title"],
             "recommended": ["url", "source"], "required_file": True},
    "earnings_call": {"config": "RAW_EARNINGS_CALL_FILE", "required": ["date", "content"],
                      "recommended": ["title", "speaker", "section", "quarter", "url"], "required_file": True},
    "manual_earnings_call": {"config": "MANUAL_EARNINGS_CALL_FILE", "required": ["date", "content"],
                             "recommended": ["title", "speaker", "section", "quarter", "url"], "required_file": False},
    "ten_k": {"config": "RAW_TEN_K_FILE", "required": ["date", "content"],
              "recommended": ["title", "section", "fiscal_year", "filing_date", "url"], "required_file": True},
}
NOTES = [
    "Read-only inspection: original CSV content, row order and dates are never changed.",
    "Ratios are fractions from 0 to 1. Missing values mean empty CSV cells; whitespace-only text is counted separately.",
    "Literal strings such as NA/null are preserved, not automatically converted to missing values.",
    "Dates use mixed-format parsing, month-first for ambiguous numeric dates, and UTC normalization.",
    "Daily/weekly volume includes observed dates/weeks only; missing calendar periods are not filled with zeros.",
    "Weeks start on Monday. Timestamp parsing does not verify the actual publication date.",
    "Over 4,000 / 8,000 characters are provisional warnings only; formal chunking will use the real tokenizer.",
    "Item mentions do not prove that complete 10-K sections have been extracted.",
    "manual_earnings_call is an optional supplement; a missing supplement is warning-only even in strict mode.",
    "No raw text, raw field values or invalid date samples are included in any report.",
]


def file_sha256(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def stats(values, percentiles=False):
    values = pd.Series(values, dtype="float64")
    if values.empty:
        return dict.fromkeys(["minimum", "median", "p90", "p95", "p99", "maximum"] if percentiles
                             else ["mean", "median", "maximum"])
    if percentiles:
        return {"minimum": int(values.min()), "median": float(values.median()),
                "p90": float(values.quantile(.90)), "p95": float(values.quantile(.95)),
                "p99": float(values.quantile(.99)), "maximum": int(values.max())}
    return {"mean": float(values.mean()), "median": float(values.median()), "maximum": int(values.max())}


def blank_series(frame, name):
    return frame[name].str.strip().eq("") if name in frame else pd.Series(True, index=frame.index)


def inspect_dataset(source, path):
    spec = SPECS[source]
    path = Path(path).resolve()
    report = {"source": source, "path": str(path), "required_file": spec["required_file"],
              "checked_at": datetime.now().astimezone().isoformat(), "exists": path.is_file(),
              "size_bytes": None, "sha256": None, "rows": None, "columns": [],
              "status": "ok", "warnings": [], "errors": []}
    if not report["exists"]:
        report["status"] = "missing"
        report["severity"] = "error" if spec["required_file"] else "warning"
        report["errors" if spec["required_file"] else "warnings"].append("CSV file is missing or not a regular file.")
        return report
    try:
        before = path.stat()
        report["size_bytes"] = before.st_size
        report["sha256"] = file_sha256(path)
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        frame = pd.DataFrame()
        report["errors"].append("CSV contains no header or columns.")
    except Exception as exc:
        report["status"] = report["severity"] = "error"
        # Exception text can contain source data; retain the type and a safe description only.
        report["errors"].append(f"{type(exc).__name__}: could not read CSV as UTF-8; inspect file encoding, structure and permissions locally.")
        return report
    rows = len(frame)
    report["rows"], report["columns"] = rows, list(frame.columns)
    missing = [name for name in spec["required"] if name not in frame]
    recommended = [name for name in spec["recommended"] if name not in frame]
    if source == "news" and not {"content", "summary"}.intersection(frame.columns):
        recommended.append("content OR summary")
    report["fields"] = {"required": spec["required"], "missing_required": missing,
                        "missing_recommended": recommended}
    if missing:
        report["errors"].append("Missing required columns: " + ", ".join(missing))
    if recommended:
        report["warnings"].append("Missing recommended columns: " + ", ".join(recommended))
    if rows == 0:
        report["warnings"].append("CSV has no data rows.")
    report["missing_values"] = {
        column: {"empty_cells": int(frame[column].eq("").sum()),
                 "empty_ratio": float(frame[column].eq("").mean()) if rows else None,
                 "blank_including_whitespace": int(blank_series(frame, column).sum()),
                 "blank_ratio": float(blank_series(frame, column).mean()) if rows else None}
        for column in frame
    }
    date_values = frame["date"].str.strip() if "date" in frame else pd.Series("", index=frame.index)
    parsed = pd.to_datetime(date_values, errors="coerce", format="mixed", utc=True)
    good = parsed.dropna()
    days = good.dt.normalize()
    weeks = days - pd.to_timedelta(days.dt.weekday, unit="D")
    daily = days.value_counts().sort_index()
    weekly = weeks.value_counts().sort_index()
    report["dates"] = {
        "parsed_success": len(good), "parsed_failure": rows-len(good),
        "parse_success_ratio": len(good)/rows if rows else None,
        "earliest": good.min().isoformat() if len(good) else None,
        "latest": good.max().isoformat() if len(good) else None,
        "daily_volume": stats(daily.tolist()), "weekly_volume": stats(weekly.tolist()),
    }
    report["daily_counts"] = {day.strftime("%Y-%m-%d"): int(count) for day, count in daily.items()}
    report["weekly_counts"] = {day.strftime("%Y-%m-%d"): int(count) for day, count in weekly.items()}
    if rows-len(good):
        report["errors"].append(f"{rows-len(good)} rows have missing or unparseable date values.")
    duplicates = {"all_columns_extra_rows": int(frame.duplicated().sum()) if len(frame.columns) else 0}
    keys = [["date", "title", "url"], ["date", "title"]] if source == "news" else [["date", "title", "content"]]
    for key in keys:
        duplicates["+".join(key) + "_extra_rows"] = (
            int(frame.duplicated(key).sum()) if set(key).issubset(frame.columns) else None)
    report["duplicates"] = duplicates
    if any(value for value in duplicates.values() if value is not None):
        report["warnings"].append("Duplicate rows/keys detected; counts exclude the first occurrence.")
    title = frame.get("title", pd.Series("", index=frame.index)).str.strip()
    content = frame.get("content", pd.Series("", index=frame.index)).str.strip()
    summary = frame.get("summary", pd.Series("", index=frame.index)).str.strip()
    body = content.where(content.ne(""), summary) if source == "news" else content
    text = (title + "\n" + body).str.strip()
    lengths = text.str.len()
    report["text"] = {
        "selection": "title + content; otherwise title + summary; otherwise title" if source == "news"
                     else "title + content; otherwise content",
        "blank_analysis_rows": int(text.eq("").sum()),
        "blank_analysis_ratio": float(text.eq("").mean()) if rows else None,
        "analysis_characters": stats(lengths.tolist(), True),
        "over_4000_characters": int(lengths.gt(4000).sum()),
        "over_8000_characters": int(lengths.gt(8000).sum()),
        "field_presence": {field: field in frame for field in ["title", "content", "summary", "speaker", "section", "chunk_id"]},
        "blank_title_rows": int(blank_series(frame, "title").sum()) if "title" in frame else None,
        "blank_content_rows": int(blank_series(frame, "content").sum()) if "content" in frame else None,
        "blank_summary_rows": int(blank_series(frame, "summary").sum()) if "summary" in frame else None,
    }
    if source == "news":
        report["text"].update(
            title_only_rows=int((title.ne("") & body.eq("")).sum()),
            title_and_body_or_summary_rows=int((title.ne("") & body.ne("")).sum()))
        if report["text"]["title_only_rows"]:
            report["warnings"].append("Some news rows contain only a title; body/summary preparation is recommended.")
    else:
        original_lengths = frame["content"].str.len() if "content" in frame else pd.Series(dtype=int)
        report["text"]["content_characters"] = stats(original_lengths.tolist(), True)
        report["text"]["content_over_4000_characters"] = int(original_lengths.gt(4000).sum())
        report["text"]["content_over_8000_characters"] = int(original_lengths.gt(8000).sum())
        report["content_lengths_by_row"] = [
            {"data_row": index+1, "characters": int(length)} for index, length in enumerate(original_lengths)]
    if report["text"]["over_4000_characters"]:
        report["warnings"].append("Long analysis texts exceed provisional character thresholds; tokenizer-based chunking needs review.")
    for field in spec["required"]:
        if field != "date" and field in frame and blank_series(frame, field).any():
            report["warnings"].append(f"Required text column {field} contains blank values.")
    if source == "ten_k":
        sections = frame.get("section", pd.Series("", index=frame.index))
        report["ten_k_items"] = {}
        for item in ["1", "1A", "7", "7A"]:
            pattern = rf"\bItem\s+{item}(?![A-Za-z0-9])"
            in_content = content.str.contains(pattern, flags=re.I, regex=True)
            in_section = sections.str.contains(pattern, flags=re.I, regex=True) | sections.str.strip().str.upper().eq(item)
            report["ten_k_items"]["Item " + item] = {
                "content_mention_rows": int(in_content.sum()), "section_label_rows": int(in_section.sum()),
                "either_rows": int((in_content | in_section).sum())}
        if any(item["either_rows"] == 0 for item in report["ten_k_items"].values()):
            report["warnings"].append("One or more target 10-K Items are not found or marked; this does not prove missing source sections.")
    try:
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            report["errors"].append("File changed during inspection; rerun after writes finish.")
    except OSError:
        report["errors"].append("File became unavailable during inspection; rerun when stable.")
    report["status"] = report["severity"] = "error" if report["errors"] else "warning" if report["warnings"] else "ok"
    report["preprocessing_readiness"] = (
        "blocked: resolve schema/date/read errors" if report["errors"] else
        "review: resolve listed warnings; tokenizer not used" if report["warnings"] else
        "structural checks passed; semantic/date-publication quality not verified")
    return report


def visible_stats(report):
    # Never include per-row content, raw dates, credentials or environment variables.
    return {key: value for key, value in report.items()
            if key not in {"daily_counts", "weekly_counts", "content_lengths_by_row"}}


def write_reports(report, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    (output / f"{REPORT_STEM}.json").write_text(payload, encoding="utf-8")
    markdown = ["# Text dataset inspection", "", f"Checked at: {report['checked_at']}", ""]
    sections = []
    for source in report["datasets"]:
        stats_text = json.dumps(visible_stats(source), ensure_ascii=False, indent=2, allow_nan=False)
        markdown.extend([f"## {source['source']}: {source['status']}", "", "```json", stats_text, "```", ""])
        def flatten(value, prefix=""):
            if isinstance(value, dict):
                for key, child in value.items():
                    yield from flatten(child, f"{prefix}.{key}" if prefix else key)
            else:
                yield prefix, value

        table_rows = []
        for key, value in flatten(visible_stats(source)):
            rendered = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, bool)) else "N/A" if value is None else str(value)
            table_rows.append(f"<tr><th>{escape(key)}</th><td>{escape(rendered)}</td></tr>")
        sections.append(
            f'<section><h2>{escape(source["source"])} '
            f'<span class="{source["severity"]}">{escape(source["status"].upper())}</span></h2>'
            f'<table>{"".join(table_rows)}</table></section>')
    markdown.extend(["## Interpretation", "", *["- " + note for note in NOTES]])
    (output / f"{REPORT_STEM}.md").write_text("\n".join(markdown), encoding="utf-8")
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Text dataset inspection</title>
<style>
body {{margin:0;background:#f4f6f8;color:#182332;font:15px/1.5 system-ui,sans-serif}}
main {{max-width:1200px;margin:auto;padding:24px}}
section {{background:white;border:1px solid #dce2e8;border-radius:10px;padding:20px;margin:18px 0}}
h2 {{font-size:20px}} h2 span {{padding:4px 10px;border-radius:5px;font-size:14px}}
.ok {{background:#dcfce7;color:#166534}} .warning {{background:#fff3bf;color:#744800}}
.error {{background:#fee2e2;color:#991b1b}}
table {{width:100%;border-collapse:collapse;font-size:13px}}
th,td {{padding:8px;border-bottom:1px solid #edf0f3;text-align:left;vertical-align:top;overflow-wrap:anywhere}}
th {{width:40%;font-weight:500}}
</style></head><body><main><h1>Text dataset inspection</h1>
<p>Checked at: {escape(report["checked_at"])}</p>
<p>Statistics only. Original CSVs remain unchanged.</p>
{"".join(sections)}
<section><h2>Interpretation</h2><ul>{"".join("<li>"+escape(note)+"</li>" for note in NOTES)}</ul></section>
</main></body></html>"""
    (output / f"{REPORT_STEM}.html").write_text(document, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for source in SPECS:
        parser.add_argument("--" + source.replace("_", "-") + "-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--strict", action="store_true",
                        help="Nonzero for missing required files, invalid CSV/schema or unparseable dates.")
    args = parser.parse_args(argv)
    paths = {}
    for source, spec in SPECS.items():
        override = getattr(args, source + "_file")
        default = Path(getattr(config, spec["config"]))
        paths[source] = override.resolve() if override else (
            default if default.is_absolute() else PROJECT_ROOT / default).resolve()
    # Refuse collisions, including symlinks, rather than risk overwriting an input CSV.
    destinations = [(args.output_dir / f"{REPORT_STEM}.{ext}").resolve() for ext in ("json", "md", "html")]
    if any(destination == source or (destination.exists() and source.exists() and destination.samefile(source))
           for destination in destinations for source in paths.values()):
        parser.error("Report output path overlaps an input file.")
    report = {"checked_at": datetime.now().astimezone().isoformat(), "notes": NOTES,
              "datasets": [inspect_dataset(source, path) for source, path in paths.items()]}
    report["summary"] = {status: sum(item["status"] == status for item in report["datasets"])
                         for status in ("ok", "warning", "error", "missing")}
    try:
        write_reports(report, args.output_dir)
    except OSError as exc:
        print(f"{type(exc).__name__}: unable to write inspection reports; check output directory permissions.", file=sys.stderr)
        return 1
    for item in report["datasets"]:
        print(f"{item['source']}: {item['status']} (rows={item['rows']}, bytes={item['size_bytes']})")
    print(f"Reports: {args.output_dir.resolve()}")
    return int(args.strict and any(item["severity"] == "error" for item in report["datasets"]))


if __name__ == "__main__":
    raise SystemExit(main())
