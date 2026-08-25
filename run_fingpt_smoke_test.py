import argparse
import importlib.util
import json
import platform
import time
import traceback
from datetime import date
from pathlib import Path

import pandas as pd

from config import (
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


REPORT_FILE = "fingpt_smoke_test_report.md"
ARCHIVE_DIR = Path(f"data/archive/{date.today().isoformat()}_fingpt_experiment")
SAMPLE_OUTPUT = ARCHIVE_DIR / "processed_sample" / f"{TICKER}_fingpt_smoke_sample_outputs.csv"
MODEL_CONFIG_OUTPUT = ARCHIVE_DIR / "fingpt_model_config.json"


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
        df["text"] = df.get("title", "").fillna("").astype(str).str.strip()
        dedupe_columns = [column for column in ["date", "title", "url"] if column in df.columns]
    else:
        df["text"] = (df.get("title", "").fillna("").astype(str) + "\n" + df.get("content", "").fillna("").astype(str)).str.strip()
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


def write_report(report: dict, sample_outputs: pd.DataFrame) -> None:
    examples = sample_outputs.head(8).to_dict(orient="records") if not sample_outputs.empty else []
    lines = [
        "# FinGPT Smoke Test Report",
        "",
        f"- Date: {date.today().isoformat()}",
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
        f"- Error: `{report.get('error')}`",
        f"- Suitable for full 2020-2026 dataset now: `{report.get('suitable_for_full_dataset')}`",
        f"- Recommendation: {report.get('recommendation')}",
    ]
    Path(REPORT_FILE).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--news-limit", type=int, default=100)
    parser.add_argument("--call-limit", type=int, default=5)
    parser.add_argument("--tenk-limit", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--model-profile",
        choices=sorted(FINGPT_MODEL_PROFILES),
        default="sentiment-llama2-13b",
    )
    parser.add_argument("--quantization", choices=["4bit", "8bit", "fp16"], default="4bit")
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Run one direct-text smoke test without reading News, Earnings Call, or 10-K CSV files.",
    )
    args = parser.parse_args()
    direct_text = args.text.strip() if args.text is not None else None
    if args.text is not None and not direct_text:
        parser.error("--text cannot be empty.")

    ensure_directories()
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_CONFIG_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

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
        "suitable_for_full_dataset": False,
        "recommendation": "Smoke test did not complete.",
    }
    MODEL_CONFIG_OUTPUT.write_text(
        json.dumps(
            {
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
    if sample_df.empty:
        report["status"] = "failed"
        report["error"] = "No sample rows were available from current NVDA raw files."
        write_report(report, output_df)
        save_csv(output_df, str(SAMPLE_OUTPUT))
        return

    try:
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
            total_inference_seconds += seconds
            for source_row, result in zip(sample_df.iloc[start_idx : start_idx + effective_batch_size].to_dict("records"), batch_results):
                raw_output = result.get("fingpt_raw_output", "")
                rows.append(
                    {
                        **source_row,
                        "sentiment_label": result.get("sentiment_label"),
                        "sentiment_score": result.get("sentiment_score"),
                        "model_profile": args.model_profile,
                        "model_name": result.get("model_name"),
                        "fingpt_raw_output": raw_output,
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
        report["traceback"] = traceback.format_exc(limit=6)
        output_df["sentiment_label"] = ""
        output_df["sentiment_score"] = ""
        output_df["model_profile"] = args.model_profile
        output_df["model_name"] = FINGPT_MODEL_PROFILES[args.model_profile]["model_name"]
        output_df["fingpt_raw_output"] = ""
        output_df["fingpt_inference_seconds"] = ""
        if not hardware["cuda_available"]:
            report["recommendation"] = (
                "Current Torch environment has no CUDA GPU. FinGPT v3.3 is a 13B LoRA model, "
                "so use a CUDA build with sufficient VRAM before running full inference."
            )
        elif not all(deps.get(package, False) for package in ["peft", "accelerate", "sentencepiece"]):
            report["recommendation"] = "Install missing FinGPT inference dependencies, then rerun the smoke test."
        else:
            report["recommendation"] = "Resolve the reported model load/inference error before scaling beyond smoke test."

    save_csv(output_df, str(SAMPLE_OUTPUT))
    write_report(report, output_df)
    if direct_text is not None and not output_df.empty:
        direct_fields = [
            "sentiment_label",
            "sentiment_score",
            "fingpt_raw_output",
            "model_profile",
            "model_name",
            "fingpt_inference_seconds",
        ]
        direct_output = {
            field: output_df.iloc[0].get(field)
            for field in direct_fields
            if field in output_df.columns
        }
        print(json.dumps(direct_output, ensure_ascii=False, indent=2))
    print(f"Report: {REPORT_FILE}")
    print(f"Archive: {ARCHIVE_DIR}")


if __name__ == "__main__":
    main()
