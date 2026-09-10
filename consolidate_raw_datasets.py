"""Consolidate existing raw CSVs without fetching data or changing source text."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from uuid import uuid4
import pandas as pd

ROOT = Path(__file__).resolve().parent
FILES = {
    "news": "NVDA_news_raw.csv",
    "earnings_call": "NVDA_earnings_call_raw.csv",
    "ten_k": "NVDA_10k_raw.csv",
    "google_trends": "NVDA_google_trends_raw.csv",
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Consolidate existing local raw CSVs after creating a byte-for-byte backup."
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Required acknowledgement: back up and replace active data/raw CSVs with consolidated copies.",
    )
    args = parser.parse_args(argv)
    if not args.apply:
        parser.error("refusing to modify data without explicit --apply")

    run = ROOT / "data/backup" / (datetime.now().strftime("%Y-%m-%d_%H%M%S") + "_consolidation_" + uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    summaries, provenance, issues, plans, coverage = [], [], [], [], []
    for source, filename in FILES.items():
        active = ROOT / "data/raw" / source / filename
        paths = [active] + sorted((ROOT / "data/archive").glob(f"*/raw/{source}/{filename}"))
        frames = []
        schema = None
        for path in paths:
            if not path.exists():
                continue
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            if schema is None:
                schema = list(frame.columns)
            if set(frame.columns) != set(schema):
                raise ValueError(f"Schema mismatch: {path}")
            frame = frame[schema]
            dates = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
            if dates.isna().any():
                raise ValueError(f"Missing dates: {path}")
            provenance.append({"source": source, "path": str(path.relative_to(ROOT)), "rows": len(frame),
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            frames.append(frame)
        if not frames:
            raise ValueError(f"No files for {source}")
        combined = pd.concat(frames, ignore_index=True)
        # Remove identical rows only. Conflicting records remain available for review.
        merged = combined.drop_duplicates().sort_values(
            ["date", "trend_keyword"] if source == "google_trends" else ["date", "url", "title"],
            kind="stable").reset_index(drop=True)
        keys = ["date", "trend_keyword"] if source == "google_trends" else ["url"]
        conflicts = merged.duplicated(keys, keep=False)
        for idx, row in merged.iterrows():
            flags = []
            if conflicts.iloc[idx]: flags.append("duplicate_key_review")
            if source in {"earnings_call", "ten_k"}:
                if not row["content"].strip(): flags.append("empty_content")
                if len(row["content"]) == 20000: flags.append("possible_20000_character_truncation")
            if source == "earnings_call" and "CFO Commentary" in row["title"]:
                flags.append("cfo_commentary_not_transcript;publication_date_unverified")
            if source == "ten_k" and ("us-gaap:" in row["content"] or "xbrli:" in row["content"]):
                flags.append("xbrl_metadata_in_text")
            if flags:
                issues.append({"source": source, "csv_row": idx + 2, "date": row["date"],
                               "url": row.get("url", ""), "issues": ";".join(flags)})
        for month, count in merged.groupby(merged.date.str[:7]).size().items():
            coverage.append({"source": source, "month": month, "rows": int(count)})
        summaries.append({"source": source, "input_rows": len(combined), "output_rows": len(merged),
                          "exact_duplicates_removed": len(combined)-len(merged),
                          "start": merged.date.min(), "end": merged.date.max(),
                          "conflicting_key_rows": int(conflicts.sum())})
        plans.append((active, merged))
    # Prepare all outputs before replacing active files; preserve byte-for-byte originals.
    for active, merged in plans:
        relative = active.relative_to(ROOT / "data/raw")
        backup = run / "original_raw" / relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        if active.exists(): shutil.copy2(active, backup)
        staged = run / "merged_raw" / relative
        staged.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(staged, index=False, encoding="utf-8-sig")
        reread = pd.read_csv(staged, dtype=str, keep_default_na=False)
        pd.testing.assert_frame_equal(merged, reread)
    for active, merged in plans:
        staged = run / "merged_raw" / active.relative_to(ROOT / "data/raw")
        active.parent.mkdir(parents=True, exist_ok=True)
        temporary = active.with_suffix(".csv.consolidating")
        shutil.copyfile(staged, temporary)
        temporary.replace(active)
    pd.DataFrame(issues, columns=["source", "csv_row", "date", "url", "issues"]).to_csv(run / "quality_issues.csv", index=False)
    pd.DataFrame(coverage).to_csv(run / "monthly_coverage.csv", index=False)
    (run / "manifest.json").write_text(json.dumps({"summary": summaries, "inputs": provenance}, indent=2), encoding="utf-8")
    lines = ["# Raw dataset consolidation", "", "Merged existing raw files only; original source text and dates retained.",
             "Exact duplicates removed; conflicting keys retained and listed for review.", "",
             "| Source | Input rows | Output rows | Removed duplicates | Start | End |",
             "|---|---:|---:|---:|---|---|"]
    for item in summaries:
        lines.append("| {source} | {input_rows} | {output_rows} | {exact_duplicates_removed} | {start} | {end} |".format(**item))
    lines += ["", "## Remaining source limitations",
              "- No 2020 10-K filing exists in the supplied raw files.",
              "- CFO commentary publication dates require verification; fiscal quarters are not publication dates.",
              "- Some documents appear truncated at 20,000 characters; 10-K text includes XBRL metadata.",
              "- Trends are monthly observations (78 months per keyword), not daily/weekly observations. No resampling or zero filling was performed.",
              "- Archived processed/features/dataset/prediction files are historical outputs, not regenerated or mixed with raw data.",
              "- Coverage counts describe supplied observations, not proof of complete source coverage.", "",
              f"Backup and audit directory: {run.relative_to(ROOT)}"]
    report = "\n".join(lines) + "\n"
    (run / "README.md").write_text(report, encoding="utf-8")
    (ROOT / "data/DATASET_INVENTORY.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
