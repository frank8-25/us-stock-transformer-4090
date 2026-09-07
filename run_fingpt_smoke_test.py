import argparse
import importlib.util
import json
import platform
import shutil
from uuid import uuid4
import time
import traceback
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from config import (
    FINGPT_DEFAULT_MODEL_PROFILE,
    FINGPT_DEFAULT_QUANTIZATION,
    RAW_EARNINGS_CALL_FILE,
    RAW_NEWS_FILE,
    RAW_TEN_K_FILE,
    TICKER,
    ensure_directories,
)
from fingpt_sentiment import (
    FINGPT_ADAPTER_TYPE,
    FINGPT_BASE_MODEL,
    FINGPT_MODEL_PROFILES,
    FINGPT_SENTIMENT_MODEL,
    get_torch_hardware,
    load_fingpt_model,
    parse_sentiment_label,
    run_fingpt_batch,
)
from pipeline_utils import save_csv
from fingpt_report import HTML_REPORT_FILE, summarize_results, write_html_report


REPORT_FILE = "fingpt_smoke_test_report.md"
ARCHIVE_ROOT = Path("data/archive")


def create_run_directory(started_at: datetime) -> Path:
    # Exclusive creation protects existing runs even in the unlikely event of a collision.
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    while True:
        run_id = started_at.strftime("%Y-%m-%d_%H%M%S_%f") + "_" + uuid4().hex[:8]
        directory = ARCHIVE_ROOT / f"{run_id}_fingpt_experiment"
        try:
            directory.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return directory


def dependency_status() -> dict:
    packages = ["torch", "transformers", "peft", "accelerate", "sentencepiece", "bitsandbytes"]
    return {package: importlib.util.find_spec(package) is not None for package in packages}


def read_sample(path: str, limit: int, source: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame(columns=["source_type", "date", "text"])
    if df.empty:
        return pd.DataFrame(columns=["source_type", "date", "text"])

    df = df.copy()
    if source == "news":
        df["text"] = df.get("title", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
        dedupe_columns = [column for column in ["date", "title", "url"] if column in df.columns]
    else:
        df["text"] = (df.get("title", pd.Series("", index=df.index)).fillna("").astype(str) + "\n" + df.get("content", pd.Series("", index=df.index)).fillna("").astype(str)).str.strip()
        dedupe_columns = [column for column in ["date", "title", "url"] if column in df.columns]
    if dedupe_columns:
        df = df.drop_duplicates(subset=dedupe_columns)
    df = df[df["text"] != ""].head(limit)
    df["source_type"] = source
    if "date" not in df.columns:
        df["date"] = ""
    return df[["source_type", "date", "text"]]


def build_sample(news_limit: int, call_limit: int, tenk_limit: int) -> pd.DataFrame:
    frames = [
        read_sample(RAW_NEWS_FILE, news_limit, "news"),
        read_sample(RAW_EARNINGS_CALL_FILE, call_limit, "earnings_call"),
        read_sample(RAW_TEN_K_FILE, tenk_limit, "ten_k"),
    ]
    return pd.concat(frames, ignore_index=True)


def build_direct_text_sample(text: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_type": "direct_text",
                "date": date.today().isoformat(),
                "text": text.strip(),
            }
        ]
    )


def write_report(report: dict, sample_outputs: pd.DataFrame,
                 path: str | Path = REPORT_FILE) -> None:
    examples = sample_outputs.head(8).to_dict(orient="records") if not sample_outputs.empty else []
    lines = [
        "# FinGPT Smoke Test Report",
        "",
        f"- Date: {date.today().isoformat()}",
        f"- Run ID: {report.get('run_id')}",
        f"- Started at: {report.get('started_at')}",
        f"- Finished at: {report.get('finished_at')}",
        f"- Model profile: {report.get('model_profile')}",
        f"- Ticker: {TICKER}",
        f"- FinGPT model: `{report.get('model_name') or FINGPT_SENTIMENT_MODEL}`",
        f"- Base model: `{report.get('base_model') or FINGPT_BASE_MODEL}`",
        f"- LoRA / adapter: `{FINGPT_ADAPTER_TYPE}`",
        f"- Quantization: `{report.get('quantization')}`",
        f"- Probability / confidence output: Not directly produced by this official generate-style inference path.",
        "",
        "## Environment",
        "",
        f"- Python: `{platform.python_version()}`",
        f"- Platform: `{platform.platform()}`",
        f"- Device selected: `{report.get('device')}`",
        f"- CUDA available: `{report.get('cuda_available')}`",
        f"- GPU: `{report.get('gpu_name')}`",
        f"- Total VRAM GB: `{report.get('gpu_total_vram_gb')}`",
        f"- Peak allocated VRAM GB: `{report.get('peak_allocated_vram_gb')}`",
        f"- Peak reserved VRAM GB: `{report.get('peak_reserved_vram_gb')}`",
        f"- Torch version: `{report.get('torch_version')}`",
        f"- Dependencies: `{json.dumps(report.get('dependencies', {}), sort_keys=True)}`",
        "",
        "## Sample Size",
        "",
        f"- Input mode: `{report.get('input_mode')}`",
        f"- News rows requested: `{report.get('news_limit')}`",
        f"- Earnings call chunks requested: `{report.get('call_limit')}`",
        f"- 10-K chunks requested: `{report.get('tenk_limit')}`",
        f"- Rows available for smoke test: `{report.get('sample_rows')}`",
        f"- Rows completed: `{report.get('completed_rows')}`",
        f"- Successful rows: `{report.get('successful_rows')}`",
        f"- Failed / incomplete rows: `{report.get('failed_rows')}`",
        "Only sentiment is validated. Multidimensional outputs will be designed and validated in a later phase.",
        "",
        "## Timing",
        "",
        f"- Model load seconds: `{report.get('model_load_seconds')}`",
        f"- Total inference seconds: `{report.get('total_inference_seconds')}`",
        f"- Average inference seconds per row: `{report.get('avg_inference_seconds')}`",
        f"- Batch size tested: `{report.get('batch_size')}`",
        "",
        "## Output Format",
        "",
        "- `date`",
        "- `text`",
        "- `sentiment_label`",
        "- `sentiment_score`",
        "- `model_profile`",
        "- `model_name`",
        "- `fingpt_raw_output`",
        "- `fingpt_inference_seconds`",
        "- `fingpt_error`",
        "",
        "No `positive_probability`, `negative_probability`, or `neutral_probability` columns were emitted, because the official model-card inference example decodes generated text labels rather than calibrated class probabilities.",
        "",
        "## Output Examples",
        "",
        "```json",
        json.dumps(examples, ensure_ascii=False, indent=2)[:4000],
        "```",
        "",
        "## Result",
        "",
        f"- Status: `{report.get('status')}`",
        f"- fingpt_error: `{report.get('fingpt_error')}`",
        "",
        "```text",
        report.get("traceback", ""),
        "```",
        f"- Suitable for full 2020-2026 dataset now: `{report.get('suitable_for_full_dataset')}`",
        f"- Recommendation: {report.get('recommendation')}",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--news-limit", type=int, default=100)
    parser.add_argument("--call-limit", type=int, default=5)
    parser.add_argument("--tenk-limit", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--model-profile",
        choices=sorted(FINGPT_MODEL_PROFILES),
        default=FINGPT_DEFAULT_MODEL_PROFILE,
    )
    parser.add_argument("--quantization", choices=["4bit", "8bit", "fp16"], default=FINGPT_DEFAULT_QUANTIZATION)
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Run one direct-text smoke test without reading News, Earnings Call, or 10-K CSV files.",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or min(args.news_limit, args.call_limit, args.tenk_limit) < 0:
        parser.error("batch-size must be positive and sample limits must be nonnegative.")
    direct_text = args.text.strip() if args.text is not None else None
    if args.text is not None and not direct_text:
        parser.error("--text cannot be empty.")

    ensure_directories()
    started_at = datetime.now().astimezone()
    archive_dir = create_run_directory(started_at)
    sample_output = archive_dir / "processed_sample" / f"{TICKER}_fingpt_smoke_sample_outputs.csv"
    model_config_output = archive_dir / "fingpt_model_config.json"

    hardware = get_torch_hardware()
    deps = dependency_status()
    input_mode = "direct_text" if direct_text is not None else "csv"
    effective_batch_size = 1 if direct_text is not None else args.batch_size
    sample_df = (
        build_direct_text_sample(direct_text)
        if direct_text is not None
        else build_sample(args.news_limit, args.call_limit, args.tenk_limit)
    )
    report = {
        **hardware,
        "run_id": archive_dir.name.removesuffix("_fingpt_experiment"),
        "started_at": started_at.isoformat(timespec="seconds"),
        "dependencies": deps,
        "news_limit": args.news_limit,
        "call_limit": args.call_limit,
        "tenk_limit": args.tenk_limit,
        "batch_size": effective_batch_size,
        "input_mode": input_mode,
        "model_profile": args.model_profile,
        "model_name": FINGPT_MODEL_PROFILES[args.model_profile]["model_name"],
        "base_model": FINGPT_MODEL_PROFILES[args.model_profile]["base_model"],
        "quantization": args.quantization,
        "peak_allocated_vram_gb": None,
        "peak_reserved_vram_gb": None,
        "sample_rows": len(sample_df),
        "completed_rows": 0,
        "model_load_seconds": None,
        "total_inference_seconds": None,
        "avg_inference_seconds": None,
        "status": "not_started",
        "error": None,
        "fingpt_error": "",
        "suitable_for_full_dataset": False,
        "recommendation": "Smoke test did not complete.",
    }
    model_config_output.write_text(
        json.dumps(
            {
                "run_id": report["run_id"],
                "started_at": report["started_at"],
                "model_profile": args.model_profile,
                "model_name": FINGPT_MODEL_PROFILES[args.model_profile]["model_name"],
                "base_model": FINGPT_MODEL_PROFILES[args.model_profile]["base_model"],
                "base_model_description": FINGPT_MODEL_PROFILES[args.model_profile]["base_model_description"],
                "adapter": FINGPT_ADAPTER_TYPE,
                "quantization": args.quantization,
                "official_source": "https://github.com/AI4Finance-Foundation/FinGPT",
                "huggingface_model": f"https://huggingface.co/{FINGPT_MODEL_PROFILES[args.model_profile]['model_name']}",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    output_df = sample_df.copy()
    rows = []
    for column in ["sentiment_label", "sentiment_score", "fingpt_raw_output",
                   "fingpt_inference_seconds", "fingpt_error"]:
        output_df[column] = None
    output_df["model_profile"] = args.model_profile
    output_df["model_name"] = report["model_name"]

    try:
        if sample_df.empty:
            raise ValueError("No sample rows were available from the configured raw CSV files.")
        load_start = time.perf_counter()
        tokenizer, model, load_info = load_fingpt_model(
            allow_cpu=args.allow_cpu,
            model_profile=args.model_profile,
            quantization=args.quantization,
        )
        report["model_load_seconds"] = round(time.perf_counter() - load_start, 4)
        report["device"] = load_info.device
        report["model_name"] = load_info.model_name
        report["base_model"] = load_info.base_model
        report["quantization"] = load_info.quantization
        report["peak_allocated_vram_gb"] = load_info.peak_allocated_vram_gb
        report["peak_reserved_vram_gb"] = load_info.peak_reserved_vram_gb

        rows = []
        total_inference_seconds = 0.0
        texts = sample_df["text"].fillna("").astype(str).tolist()
        for start_idx in range(0, len(texts), effective_batch_size):
            batch = texts[start_idx : start_idx + effective_batch_size]
            batch_results, seconds = run_fingpt_batch(
                tokenizer,
                model,
                batch,
                model_name=load_info.model_name,
            )
            if len(batch_results) != len(batch):
                raise RuntimeError("FinGPT returned a different number of results than inputs.")
            total_inference_seconds += seconds
            for source_row, result in zip(sample_df.iloc[start_idx : start_idx + effective_batch_size].to_dict("records"), batch_results):
                raw_output = result.get("fingpt_raw_output", "")
                if result.get("fingpt_error") or not raw_output.strip():
                    raise RuntimeError(result.get("fingpt_error") or "FinGPT generated an empty response.")
                rows.append(
                    {
                        **source_row,
                        "sentiment_label": result.get("sentiment_label"),
                        "sentiment_score": result.get("sentiment_score"),
                        "model_profile": args.model_profile,
                        "model_name": result.get("model_name"),
                        "fingpt_raw_output": raw_output,
                        "fingpt_error": "",
                        "parsed_from_raw_output": parse_sentiment_label(raw_output),
                        "fingpt_inference_seconds": round(seconds / max(len(batch), 1), 4),
                    }
                )
        output_df = pd.DataFrame(rows)
        report["completed_rows"] = len(output_df)
        report["total_inference_seconds"] = round(total_inference_seconds, 4)
        report["avg_inference_seconds"] = round(total_inference_seconds / max(len(output_df), 1), 4)
        report["status"] = "completed"
        report["suitable_for_full_dataset"] = bool(hardware["cuda_available"] and report["avg_inference_seconds"] < 5)
        report["recommendation"] = (
            "Full dataset run is reasonable only after confirming GPU VRAM headroom and stable parsed labels."
            if report["suitable_for_full_dataset"]
            else "Do not run the full 2020-2026 dataset on this environment yet."
        )
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["fingpt_error"] = report["error"]
        report["traceback"] = traceback.format_exc()
        traceback.print_exc()
        report["completed_rows"] = len(rows)
        # Preserve completed rows; mark failed and unattempted rows explicitly.
        pending = output_df.iloc[len(rows):].copy()
        pending["fingpt_error"] = report["fingpt_error"]
        output_df = pd.concat([pd.DataFrame(rows), pending], ignore_index=True)
        if not hardware["cuda_available"]:
            report["recommendation"] = (
                "Current Torch environment has no CUDA GPU. FinGPT v3.3 is a 13B LoRA model, "
                "so use a CUDA build with sufficient VRAM before running full inference."
            )
        elif not all(deps.get(package, False) for package in ["peft", "accelerate", "sentencepiece"]):
            report["recommendation"] = "Install missing FinGPT inference dependencies, then rerun the smoke test."
        else:
            report["recommendation"] = "Resolve the reported model load/inference error before scaling beyond smoke test."

    report["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    summary = summarize_results(output_df)
    report["successful_rows"] = summary["success"]
    report["failed_rows"] = summary["failed"]
    report["avg_inference_seconds"] = summary["average_seconds"]
    save_csv(output_df, str(sample_output))
    archived_markdown = archive_dir / REPORT_FILE
    archived_html = archive_dir / HTML_REPORT_FILE
    write_report(report, output_df, archived_markdown)
    write_html_report(report, output_df, archived_html)
    # Archive first; only these documented latest-report aliases are replaced.
    shutil.copyfile(archived_markdown, REPORT_FILE)
    shutil.copyfile(archived_html, HTML_REPORT_FILE)
    if direct_text is not None and not output_df.empty:
        direct_fields = [
            "sentiment_label",
            "sentiment_score",
            "fingpt_raw_output",
            "model_profile",
            "model_name",
            "fingpt_inference_seconds",
            "fingpt_error",
        ]
        direct_output = {
            field: output_df.head(1).to_dict(orient="records")[0].get(field)
            for field in direct_fields
            if field in output_df.columns
        }
        print(json.dumps(direct_output, ensure_ascii=False, indent=2))
    print(f"Report: {REPORT_FILE}")
    print(f"HTML report (latest): {HTML_REPORT_FILE}")
    print(f"Archive: {archive_dir}")
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
