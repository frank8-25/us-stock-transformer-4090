"""Probe FinGPT MT Llama-2 7B with independent fixed-label classifications.

This diagnostic is isolated from the production pipeline and multidimensional
schema. It never maps, repairs, or guesses an out-of-vocabulary model answer.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Callable

from config import FINGPT_MODEL_PROFILES
from fingpt_sentiment import load_fingpt_model


MODEL_PROFILE = "mt-llama2-7b"
DEFAULT_JSON_OUTPUT = "probe_mt_classification_dimensions_results.json"
DEFAULT_CSV_OUTPUT = "probe_mt_classification_dimensions_results.csv"
MAX_NEW_TOKENS = 16

TEST_CASES = {
    "positive_fundamentals": (
        "NVIDIA reported strong AI chip demand, record data center revenue, "
        "and raised its revenue outlook."
    ),
    "negative_risk": (
        "NVIDIA warned that export restrictions, weaker demand, and rising "
        "costs may reduce future revenue."
    ),
    "mixed_message": (
        "NVIDIA raised its revenue outlook due to strong AI demand, but warned "
        "that export restrictions may reduce sales in China."
    ),
}

DIMENSIONS = [
    {
        "name": "sentiment",
        "allowed": ["negative", "neutral", "positive"],
        "instruction": (
            "What is the overall sentiment toward NVIDIA in this financial text? "
            "Please choose an answer from {negative/neutral/positive}."
        ),
    },
    {
        "name": "risk",
        "allowed": ["low", "medium", "high"],
        "instruction": (
            "How severe is the financial or business risk to NVIDIA described in "
            "this text? Please choose an answer from {low/medium/high}."
        ),
    },
    {
        "name": "uncertainty",
        "allowed": ["low", "medium", "high"],
        "instruction": (
            "How much uncertainty about NVIDIA's future business performance is "
            "expressed in this text? Please choose an answer from {low/medium/high}."
        ),
    },
    {
        "name": "growth_outlook",
        "allowed": ["negative", "neutral", "positive"],
        "instruction": (
            "What growth outlook for NVIDIA is described in this text? Please "
            "choose an answer from {negative/neutral/positive}."
        ),
    },
    {
        "name": "financial_strength",
        "allowed": ["weak", "neutral", "strong"],
        "instruction": (
            "What level of financial or operational strength does this text "
            "indicate for NVIDIA? Please choose an answer from {weak/neutral/strong}."
        ),
    },
    {
        "name": "fundamental_impact",
        "allowed": ["negative", "neutral", "positive"],
        "instruction": (
            "What is the likely impact of the described information on NVIDIA's "
            "business fundamentals? Please choose an answer from "
            "{negative/neutral/positive}."
        ),
    },
    {
        "name": "price_narrative",
        "allowed": ["no", "partial", "yes"],
        "instruction": (
            "Is this text mainly a stock-price movement narrative rather than "
            "fundamental business information? Please choose an answer from "
            "{no/partial/yes}."
        ),
    },
    {
        "name": "relevance",
        "allowed": ["low", "medium", "high"],
        "instruction": (
            "How relevant is this text to NVIDIA's future business performance? "
            "Please choose an answer from {low/medium/high}."
        ),
    },
    {
        "name": "short_term_impact",
        "allowed": ["negative", "neutral", "positive"],
        "instruction": (
            "What short-term business impact on NVIDIA is supported by this text? "
            "Please choose an answer from {negative/neutral/positive}."
        ),
    },
    {
        "name": "long_term_impact",
        "allowed": ["negative", "neutral", "positive"],
        "instruction": (
            "What long-term business impact on NVIDIA is supported by this text? "
            "Please choose an answer from {negative/neutral/positive}."
        ),
    },
]

CSV_FIELDS = [
    "test_case",
    "input_text",
    "dimension",
    "allowed_labels",
    "raw_output_run_1",
    "raw_output_run_2",
    "normalized_output_run_1",
    "normalized_output_run_2",
    "valid_run_1",
    "valid_run_2",
    "consistent",
    "inference_seconds_run_1",
    "inference_seconds_run_2",
    "full_prompt",
]


def build_prompt(instruction: str, input_text: str) -> str:
    """Use the exact FinGPT Instruction/Input/Answer shape, including final space."""
    return f"Instruction: {instruction}\nInput: {input_text}\nAnswer: "


def normalize_output(output: str) -> str:
    """Normalize only surrounding whitespace and case; never infer intent."""
    return output.strip().casefold()


def _model_input_device(model: Any):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return model.device


def generate_once(
    tokenizer: Any,
    model: Any,
    prompt: str,
    synchronize: Callable[[], None],
) -> dict[str, Any]:
    """Generate one continuation and decode only tokens after the prompt."""
    import torch

    tokens = tokenizer(prompt, return_tensors="pt", truncation=False)
    input_token_count = int(tokens["input_ids"].shape[1])
    input_device = _model_input_device(model)
    tokens = {name: value.to(input_device) for name, value in tokens.items()}
    generation_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "num_beams": 1,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**tokens, **generation_kwargs)
    synchronize()
    elapsed = time.perf_counter() - started

    new_token_ids = generated[:, input_token_count:]
    raw_output = tokenizer.batch_decode(
        new_token_ids, skip_special_tokens=True
    )[0]
    return {
        "raw_output": raw_output,
        "normalized_output": normalize_output(raw_output),
        "input_token_count": input_token_count,
        "generated_token_count": int(new_token_ids.shape[1]),
        "inference_seconds": round(elapsed, 6),
    }


def run_probes(tokenizer: Any, model: Any) -> list[dict[str, Any]]:
    """Run all 30 identical-prompt pairs with one already-loaded model."""
    import torch

    synchronize = torch.cuda.synchronize if torch.cuda.is_available() else lambda: None
    rows = []
    for test_case, input_text in TEST_CASES.items():
        for dimension in DIMENSIONS:
            prompt = build_prompt(dimension["instruction"], input_text)
            run_1 = generate_once(tokenizer, model, prompt, synchronize)
            run_2 = generate_once(tokenizer, model, prompt, synchronize)
            allowed = dimension["allowed"]
            normalized_1 = run_1["normalized_output"]
            normalized_2 = run_2["normalized_output"]
            rows.append(
                {
                    "test_case": test_case,
                    "input_text": input_text,
                    "dimension": dimension["name"],
                    "allowed_labels": allowed,
                    "raw_output_run_1": run_1["raw_output"],
                    "raw_output_run_2": run_2["raw_output"],
                    "normalized_output_run_1": normalized_1,
                    "normalized_output_run_2": normalized_2,
                    "valid_run_1": normalized_1 in allowed,
                    "valid_run_2": normalized_2 in allowed,
                    "consistent": normalized_1 == normalized_2,
                    "inference_seconds_run_1": run_1["inference_seconds"],
                    "inference_seconds_run_2": run_2["inference_seconds"],
                    "full_prompt": prompt,
                    "input_token_count": run_1["input_token_count"],
                    "generated_token_count_run_1": run_1["generated_token_count"],
                    "generated_token_count_run_2": run_2["generated_token_count"],
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_generations = len(rows) * 2
    valid_generations = sum(
        int(row["valid_run_1"]) + int(row["valid_run_2"]) for row in rows
    )
    consistent_pairs = sum(int(row["consistent"]) for row in rows)
    by_dimension = {}
    for dimension in DIMENSIONS:
        name = dimension["name"]
        selected = [row for row in rows if row["dimension"] == name]
        valid = sum(
            int(row["valid_run_1"]) + int(row["valid_run_2"])
            for row in selected
        )
        consistent = sum(int(row["consistent"]) for row in selected)
        by_dimension[name] = {
            "valid_generations": valid,
            "total_generations": len(selected) * 2,
            "valid_answer_rate": valid / (len(selected) * 2) if selected else None,
            "consistent_pairs": consistent,
            "total_pairs": len(selected),
            "consistency_rate": consistent / len(selected) if selected else None,
        }

    invalid_outputs = []
    inconsistent_cases = []
    for row in rows:
        for run_number in (1, 2):
            if not row[f"valid_run_{run_number}"]:
                invalid_outputs.append(
                    {
                        "test_case": row["test_case"],
                        "dimension": row["dimension"],
                        "run": run_number,
                        "raw_output": row[f"raw_output_run_{run_number}"],
                        "normalized_output": row[
                            f"normalized_output_run_{run_number}"
                        ],
                        "allowed_labels": row["allowed_labels"],
                    }
                )
        if not row["consistent"]:
            inconsistent_cases.append(
                {
                    "test_case": row["test_case"],
                    "dimension": row["dimension"],
                    "normalized_output_run_1": row[
                        "normalized_output_run_1"
                    ],
                    "normalized_output_run_2": row[
                        "normalized_output_run_2"
                    ],
                    "valid_run_1": row["valid_run_1"],
                    "valid_run_2": row["valid_run_2"],
                }
            )

    valid_rate = valid_generations / total_generations if rows else None
    consistency_rate = consistent_pairs / len(rows) if rows else None
    return {
        "total_prompt_pairs": len(rows),
        "total_generations": total_generations,
        "valid_generations": valid_generations,
        "valid_answer_rate": valid_rate,
        "valid_answer_threshold": 0.90,
        "meets_valid_answer_threshold": (
            valid_rate >= 0.90 if valid_rate is not None else False
        ),
        "consistent_pairs": consistent_pairs,
        "consistency_rate": consistency_rate,
        "consistency_threshold": 0.95,
        "meets_consistency_threshold": (
            consistency_rate >= 0.95 if consistency_rate is not None else False
        ),
        "by_dimension": by_dimension,
        "invalid_outputs": invalid_outputs,
        "inconsistent_cases": inconsistent_cases,
    }


def _csv_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        field: (
            "{" + "/".join(row[field]) + "}"
            if field == "allowed_labels"
            else row[field]
        )
        for field in CSV_FIELDS
    }


def write_outputs(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    json_path: str | Path,
    csv_path: str | Path,
) -> dict[str, Any]:
    summary = summarize(rows)
    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": metadata,
        "method": {
            "model_load_count": 1,
            "batch_size": 1,
            "repeat_count": 2,
            "do_sample": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "comparison": "strip surrounding whitespace, remove special tokens during decode, then casefold; no regex or label mapping",
        },
        "summary": summary,
        "results": rows,
    }
    Path(json_path).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with Path(csv_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_csv_row(row) for row in rows)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Probe MT Llama-2 7B using repeated independent fixed-label "
            "classifications; does not modify the production pipeline."
        )
    )
    parser.add_argument(
        "--quantization", choices=["4bit", "8bit", "fp16"], default="4bit"
    )
    parser.add_argument("--json-output", default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--csv-output", default=DEFAULT_CSV_OUTPUT)
    args = parser.parse_args()

    profile = FINGPT_MODEL_PROFILES[MODEL_PROFILE]
    try:
        tokenizer, model, load_info = load_fingpt_model(
            model_profile=MODEL_PROFILE,
            quantization=args.quantization,
        )
        rows = run_probes(tokenizer, model)
        metadata = {
            "model_profile": MODEL_PROFILE,
            "model_name": profile["model_name"],
            "base_model": profile["base_model"],
            "quantization": args.quantization,
            "device": load_info.device,
            "gpu_name": load_info.gpu_name,
            "model_load_seconds": round(load_info.load_seconds, 6),
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
        report = write_outputs(
            rows, metadata, args.json_output, args.csv_output
        )
    except Exception as exc:
        traceback.print_exc()
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Saved JSON: {Path(args.json_output).resolve()}")
    print(f"Saved CSV: {Path(args.csv_output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
