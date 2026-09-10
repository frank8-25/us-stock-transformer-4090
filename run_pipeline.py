"""End-to-end financial text pipeline. --help never imports or loads models."""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback
from uuid import uuid4

import pandas as pd

from config import TICKER, START_DATE, END_DATE, FINGPT_DEFAULT_MODEL_PROFILE, FINGPT_DEFAULT_QUANTIZATION
from pipeline_documents import SOURCES, RAW_NAMES, prepare_documents, read_csv, save_json, write_csv
from pipeline_sentiment import build_features, build_dataset, download_prices, infer_documents

ROOT = Path(__file__).resolve().parent
STAGES = ("audit", "prepare", "sentiment", "features", "prices", "dataset", "train")
DEFAULTS = dict(ticker=TICKER, start=START_DATE, end=END_DATE, frequency="weekly", horizon=5,
                model_profile=FINGPT_DEFAULT_MODEL_PROFILE, quantization=FINGPT_DEFAULT_QUANTIZATION,
                fetch_body=False, allow_title_only=False, allow_incomplete_text=False,
                allow_unverified_dates=False, limit=None, date_overrides=None)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_stage(manifest, run, name, allow_partial):
    record = manifest["stages"].get(name, {})
    accepted = ["completed", "completed_with_errors"] if allow_partial else ["completed"]
    if record.get("status") not in accepted:
        raise ValueError(f"Stage {name} must complete first; inspect manifest.json and error outputs.")
    for filename, checksum in record.get("artifacts", {}).items():
        if sha(run / filename) != checksum:
            raise ValueError(f"Artifact changed since {name}: {filename}. Rerun that stage.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=(*STAGES, "all"))
    parser.add_argument("--run-dir", type=Path, help="Resume an existing run; stored data/model settings are reused.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Project root containing data/raw.")
    parser.add_argument("--ticker")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--frequency", choices=["weekly", "daily"])
    parser.add_argument("--horizon", type=int, help="Future trading observations after entry close; default 5.")
    parser.add_argument("--model-profile", choices=["sentiment-llama2-13b", "mt-llama2-7b", "sentiment-chatglm2-6b"])
    parser.add_argument("--quantization", choices=["4bit", "8bit", "fp16"])
    for option in ["fetch-body", "allow-title-only", "allow-incomplete-text", "allow-unverified-dates"]:
        parser.add_argument("--" + option, action="store_true", default=None)
    parser.add_argument("--date-overrides", type=str, help="CSV: document_id,available_date,evidence.")
    parser.add_argument("--limit", type=int, help="Maximum documents per source; marks the run as a sample.")
    parser.add_argument("--user-agent", default="", help="Contact identity for optional document fetching.")
    parser.add_argument("--request-delay", type=float, default=1.0)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--allow-partial", action="store_true", help="Continue with excluded failed documents; exit remains nonzero.")
    parser.add_argument("--prices", type=Path, help="Local CSV with date,close; use split-adjusted daily closes.")
    parser.add_argument("--download-prices", action="store_true", help="Explicitly download adjusted prices using optional yfinance.")
    parser.add_argument("--lookback", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32, help="Transformer training batch size; FinGPT uses one chunk.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args(argv)
    if args.prices and args.download_prices:
        parser.error("Choose --prices or --download-prices.")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")
    if args.request_delay < 0 or args.epochs < 1 or args.lookback < 1 or args.batch_size < 1:
        parser.error("Invalid delay or training parameters.")
    root = args.root.resolve()
    run = args.run_dir.resolve() if args.run_dir else root / "data/pipeline_runs" / (
        datetime.now().strftime("%Y-%m-%d_%H%M%S") + "_" + uuid4().hex[:8])
    manifest_path = run / "manifest.json"
    try:
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            settings = manifest["settings"]
            if manifest["root"] != str(root):
                raise ValueError("Run belongs to a different --root.")
            for name in DEFAULTS:
                supplied = getattr(args, name)
                if supplied is not None and supplied != settings[name]:
                    raise ValueError(f"Cannot change {name} inside an existing run; create a new run.")
        else:
            settings = {key: getattr(args, key) if getattr(args, key) is not None else value
                        for key, value in DEFAULTS.items()}
            if pd.Timestamp(settings["start"]) > pd.Timestamp(settings["end"]) or settings["horizon"] < 1:
                raise ValueError("Invalid date range or horizon.")
            if settings["date_overrides"]:
                settings["date_overrides"] = str(Path(settings["date_overrides"]).resolve())
            run.mkdir(parents=True, exist_ok=False)
            manifest = {"run_id": run.name, "root": str(root), "settings": settings, "stages": {},
                        "created_at": datetime.now().astimezone().isoformat(),
                        "scope": "sample" if settings["limit"] else "full"}
            save_json(manifest_path, manifest)
        print(f"Run directory: {run}", flush=True)
        if args.download_prices and importlib.util.find_spec("yfinance") is None:
            raise ImportError("Optional yfinance is missing; provide --prices CSV or install the optional dependency separately.")
        if args.prices and not args.prices.is_file():
            raise FileNotFoundError(args.prices)
        actions = STAGES if args.stage == "all" else [args.stage]
        if args.stage == "all" and not args.prices and not args.download_prices:
            require_stage(manifest, run, "prices", args.allow_partial)
        any_errors = False
        for stage in actions:
            if stage == "prices" and not args.prices and not args.download_prices:
                require_stage(manifest, run, "prices", args.allow_partial)
                continue
            print(f"Stage: {stage}", flush=True)
            manifest["stages"][stage] = {"status": "running", "started_at": datetime.now().astimezone().isoformat()}
            # Invalidate downstream stage status before replacing upstream results.
            descendants = {"prepare": ["sentiment", "features", "dataset", "train"],
                           "sentiment": ["features", "dataset", "train"], "features": ["dataset", "train"],
                           "prices": ["dataset", "train"], "dataset": ["train"]}.get(stage, [])
            for child in descendants:
                manifest["stages"].pop(child, None)
            save_json(manifest_path, manifest)
            errors, files = 0, []
            try:
                if stage == "audit":
                    inventory = []
                    for source, suffix in RAW_NAMES.items():
                        path = root / "data/raw" / source / f"{settings['ticker']}_{suffix}_raw.csv"
                        data = read_csv(path)
                        inventory.append({"source": source, "rows": len(data), "columns": list(data.columns),
                                          "sha256": sha(path), "missing_body": int(
                                              data.get("content", pd.Series("", index=data.index)).fillna("").eq("").sum())})
                    save_json(run / "inventory.json", inventory)
                    files = ["inventory.json"]
                elif stage == "prepare":
                    docs = prepare_documents(root, settings["ticker"], run, settings["start"], settings["end"],
                                             fetch=settings["fetch_body"], user_agent=args.user_agent,
                                             allow_title_only=settings["allow_title_only"],
                                             allow_incomplete=settings["allow_incomplete_text"],
                                             allow_unverified=settings["allow_unverified_dates"],
                                             overrides=settings["date_overrides"], limit=settings["limit"],
                                             delay=args.request_delay, retry_errors=args.retry_errors)
                    errors = int(docs.preparation_error.ne("").sum()) or int(docs.empty)
                    files = ["documents.csv", "preparation_errors.csv", "preparation_summary.json"]
                elif stage == "sentiment":
                    require_stage(manifest, run, "prepare", args.allow_partial)
                    docs = read_csv(run / "documents.csv")
                    result = infer_documents(docs, run, settings["model_profile"], settings["quantization"], args.retry_errors)
                    errors = int(result.fingpt_error.ne("").sum()) or int(result.empty)
                    files = ["sentiment_documents.csv", "sentiment_chunks.csv", "sentiment_report.html", "inference_config.json"]
                elif stage == "features":
                    require_stage(manifest, run, "sentiment", args.allow_partial)
                    sentiments = read_csv(run / "sentiment_documents.csv")
                    trends_path = root / "data/raw/google_trends" / f"{settings['ticker']}_google_trends_raw.csv"
                    trends = read_csv(trends_path)
                    features, columns = build_features(sentiments, trends, settings["start"], settings["end"],
                                                       settings["frequency"])
                    write_csv(features, run / "features.csv")
                    save_json(run / "feature_schema.json",
                              {"columns": columns, "frequency": settings["frequency"],
                               "trends_sha256": sha(trends_path),
                               "trend_columns": {str(k): "trend_" + hashlib.sha256(str(k).encode()).hexdigest()[:8]
                                                 for k in trends.trend_keyword.unique()},
                               "limitations": ["Trends are historical normalized monthly values, available next month.",
                                               "Zero features are paired with missing indicators.",
                                               "Sentiment scores are not confidence or risk measures."]})
                    files = ["features.csv", "feature_schema.json"]
                elif stage == "prices":
                    if args.download_prices:
                        download_prices(settings["ticker"], settings["start"],
                                        (pd.Timestamp(settings["end"]) + pd.Timedelta(days=settings["horizon"]*3+14)).strftime("%Y-%m-%d"),
                                        run / "prices.csv")
                    elif args.prices:
                        data = read_csv(args.prices)
                        build_dataset(pd.DataFrame({"date": [settings["start"]]}), data, settings["horizon"])
                        write_csv(data, run / "prices.csv")
                        save_json(run / "prices.metadata.json", {"source": str(args.prices.resolve()),
                                                                 "sha256": sha(args.prices),
                                                                 "assumption": "User-supplied split-adjusted daily close."})
                    else:
                        raise ValueError("Supply --prices or --download-prices.")
                    files = ["prices.csv", "prices.metadata.json"]
                elif stage == "dataset":
                    require_stage(manifest, run, "features", args.allow_partial)
                    require_stage(manifest, run, "prices", args.allow_partial)
                    dataset = build_dataset(read_csv(run / "features.csv"), read_csv(run / "prices.csv"),
                                            settings["horizon"])
                    write_csv(dataset, run / "dataset.csv")
                    save_json(run / "dataset_summary.json", {"rows": len(dataset),
                              "labeled_rows": int(dataset.label.notna().sum()),
                              "unlabeled_rows": int(dataset.label.isna().sum()),
                              "label": "1 if future adjusted close > entry adjusted close; otherwise 0",
                              "entry": "First trading close strictly after feature cutoff",
                              "horizon": settings["horizon"]})
                    if dataset.label.notna().sum() == 0:
                        raise ValueError("No labels could be constructed. Check price date coverage.")
                    files = ["dataset.csv", "dataset_summary.json"]
                elif stage == "train":
                    require_stage(manifest, run, "dataset", args.allow_partial)
                    require_stage(manifest, run, "features", args.allow_partial)
                    from pipeline_transformer import train_transformer
                    schema = json.loads((run / "feature_schema.json").read_text())
                    train_transformer(read_csv(run / "dataset.csv"), schema["columns"], run,
                                      lookback=args.lookback, epochs=args.epochs, batch_size=args.batch_size,
                                      device=args.device)
                    files = ["transformer.pt", "transformer_predictions.csv", "training_history.csv",
                             "metrics.json", "training_config.json"]
                manifest["stages"][stage].update(
                    status="completed_with_errors" if errors else "completed",
                    errors=errors, artifacts={name: sha(run / name) for name in files},
                    finished_at=datetime.now().astimezone().isoformat())
                save_json(manifest_path, manifest)
                any_errors |= bool(errors)
                if errors and not args.allow_partial:
                    print(f"{errors} rows failed/excluded; inspect {run}.", file=sys.stderr)
                    return 1
            except Exception as exc:
                manifest["stages"][stage].update(status="failed", error=f"{type(exc).__name__}: {exc}",
                                                traceback=traceback.format_exc())
                save_json(manifest_path, manifest)
                raise
        return 1 if any_errors or any(s.get("status") == "completed_with_errors"
                                     for s in manifest["stages"].values()) else 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
