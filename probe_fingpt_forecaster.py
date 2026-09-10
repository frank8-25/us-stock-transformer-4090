"""Real-inference probe for the native FinGPT Forecaster output format.

This file is intentionally independent of the production data and Transformer
pipelines. All company, market, news, and financial inputs are synthetic
fixtures created only for format and directional-behavior diagnostics.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Any, Callable


BASE_MODEL = "meta-llama/Llama-2-7b-chat-hf"
ADAPTER_MODEL = "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora"
DEFAULT_JSON_OUTPUT = "probe_fingpt_forecaster_results.json"
DEFAULT_CSV_OUTPUT = "probe_fingpt_forecaster_results.csv"
MAX_NEW_TOKENS = 512

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

FULL_SYSTEM_PROMPT = (
    "You are a seasoned stock market analyst. Your task is to list the positive "
    "developments and potential concerns for companies based on relevant news and "
    "basic financials from the past weeks, then provide an analysis and prediction "
    "for the companies' stock price movement for the upcoming week. Your answer "
    "format should be as follows:\n\n"
    "[Positive Developments]:\n"
    "1. ...\n\n"
    "[Potential Concerns]:\n"
    "1. ...\n\n"
    "[Prediction & Analysis]\n"
    "Prediction: ...\n"
    "Analysis: ..."
)

ABLATION_SYSTEM_PROMPT = (
    "You are a seasoned stock market analyst. Your task is to list the positive "
    "developments and potential concerns for companies based on relevant news and "
    "basic financials from the past weeks, then provide a concise analysis. Do not "
    "make a stock-price prediction. Your answer format should be as follows:\n\n"
    "[Positive Developments]:\n"
    "1. ...\n\n"
    "[Potential Concerns]:\n"
    "1. ...\n\n"
    "[Analysis]:\n"
    "..."
)

COMPANY_INTRODUCTION = """[Company Introduction]:

NVIDIA Corporation is a leading semiconductor and accelerated-computing company.
It develops GPUs and computing platforms for data centers, artificial intelligence,
gaming, and professional visualization. NVIDIA trades under the ticker NVDA on
NASDAQ and primarily operates from the United States."""

TEST_CASES = {
    "positive": {
        "expected_direction": "positive",
        "weekly_context": """From 2024-06-14 to 2024-06-21, NVDA's stock price increased from 120.00 to 124.00. Company news during this period are listed below:

[Headline]: NVIDIA sees strong demand for new AI accelerators
[Summary]: Cloud customers expanded orders for NVIDIA AI chips, supporting strong data center growth.

[Headline]: Supply timing remains a minor constraint
[Summary]: Management said a small number of shipments could move between quarters while total demand remains strong.

From 2024-06-21 to 2024-06-28, NVDA's stock price increased from 124.00 to 130.00. Company news during this period are listed below:

[Headline]: NVIDIA raises revenue and profit outlook
[Summary]: Record data center revenue and improving operating leverage led management to raise its outlook.

[Headline]: New AI platform receives broad customer commitments
[Summary]: Major cloud providers announced plans to deploy NVIDIA's latest platform.""",
        "financials": """Revenue Growth YoY: 40%
Gross Margin: 74%
Operating Margin: 58%
Debt to Equity: 0.40
Free Cash Flow Growth YoY: 35%""",
    },
    "negative": {
        "expected_direction": "negative",
        "weekly_context": """From 2024-06-14 to 2024-06-21, NVDA's stock price decreased from 120.00 to 114.00. Company news during this period are listed below:

[Headline]: NVIDIA customers report weaker accelerator demand
[Summary]: Several customers delayed orders and channel inventory increased as demand weakened.

[Headline]: Component and logistics costs rise
[Summary]: Higher input and logistics costs are expected to pressure NVIDIA's margins.

From 2024-06-21 to 2024-06-28, NVDA's stock price decreased from 114.00 to 105.00. Company news during this period are listed below:

[Headline]: Export restrictions reduce NVIDIA sales opportunities
[Summary]: New restrictions limit shipments to important overseas customers and may reduce future revenue.

[Headline]: NVIDIA lowers its revenue outlook
[Summary]: Management reduced its outlook because of weaker demand, higher costs, and export limits.""",
        "financials": """Revenue Growth YoY: -8%
Gross Margin: 55%
Previous Gross Margin: 62%
Operating Expense Growth YoY: 18%
Debt to Equity: 0.80""",
    },
    "mixed": {
        "expected_direction": "mixed",
        "weekly_context": """From 2024-06-14 to 2024-06-21, NVDA's stock price increased from 120.00 to 126.00. Company news during this period are listed below:

[Headline]: AI infrastructure demand supports NVIDIA growth
[Summary]: Cloud providers increased orders for NVIDIA accelerators as generative AI investment expanded.

[Headline]: NVIDIA raises its revenue outlook
[Summary]: Management expects strong AI demand to lift data center revenue in the next quarter.

From 2024-06-21 to 2024-06-28, NVDA's stock price decreased from 126.00 to 124.00. Company news during this period are listed below:

[Headline]: China export restrictions may reduce NVIDIA sales
[Summary]: Tighter export controls could block some accelerator shipments and lower sales in China.

[Headline]: NVIDIA develops compliant products for restricted markets
[Summary]: The company is working on alternative products, but approval timing and customer demand remain uncertain.""",
        "financials": """Revenue Growth YoY: 25%
Gross Margin: 70%
Operating Margin: 52%
China Revenue Exposure: 20%
Free Cash Flow Growth YoY: 18%""",
    },
}

HEADER_PATTERNS = {
    "positive_developments": re.compile(
        r"(?im)^\s*\[Positive Developments\]\s*:?\s*$"
    ),
    "potential_concerns": re.compile(
        r"(?im)^\s*\[Potential Concerns\]\s*:?\s*$"
    ),
    "prediction_analysis": re.compile(
        r"(?im)^\s*\[Prediction\s*&\s*Analysis\]\s*:?\s*$"
    ),
    "analysis": re.compile(r"(?im)^\s*\[Analysis\]\s*:?\s*$"),
}
PREDICTION_LINE = re.compile(r"(?im)^\s*Prediction\s*:\s*(.+?)\s*$")
ANALYSIS_LINE = re.compile(r"(?im)^\s*Analysis\s*:\s*(.*)$")
NUMBERED_ITEM = re.compile(r"(?m)^\s*\d+[.)]\s+(.+?)\s*$")

CSV_FIELDS = [
    "test_case_id",
    "case_direction",
    "variant",
    "run_number",
    "synthetic_fixture",
    "full_prompt",
    "full_decoded_sequence",
    "new_tokens_decoded_output",
    "input_token_count",
    "generated_token_count",
    "generation_kwargs",
    "gpu_inference_seconds",
    "base_model",
    "adapter_model",
    "active_adapter",
    "found_positive_developments",
    "found_potential_concerns",
    "found_prediction",
    "found_analysis",
    "all_expected_blocks_found",
    "parsed_positive_developments",
    "parsed_potential_concerns",
    "parsed_prediction",
    "parsed_analysis",
    "parse_error",
    "empty_output",
    "consistent",
]


def build_case_information(case: dict[str, str]) -> str:
    return (
        f"{COMPANY_INTRODUCTION}\n\n"
        f"{case['weekly_context']}\n\n"
        "Some recent basic financials of NVDA, reported at 2024-06-28, are "
        "presented below. These figures are a synthetic fixture and are not "
        "historical observations:\n\n"
        f"[Basic Financials]:\n\n{case['financials']}"
    )


def build_prompt(case: dict[str, str], variant: str) -> str:
    information = build_case_information(case)
    if variant == "full":
        request = (
            "Based on all the information before 2024-06-28, let's first analyze "
            "the positive developments and potential concerns for NVDA. Come up "
            "with 2-4 most important factors respectively and keep them concise. "
            "Most factors should be inferred from company related news. Then make "
            "your prediction of the NVDA stock price movement for next week "
            "(2024-06-28 to 2024-07-05). Provide a summary analysis to support "
            "your prediction."
        )
        system_prompt = FULL_SYSTEM_PROMPT
    elif variant == "no_prediction":
        request = (
            "Based on all the information before 2024-06-28, analyze the positive "
            "developments and potential concerns for NVDA. Come up with 2-4 most "
            "important factors respectively and keep them concise. Most factors "
            "should be inferred from company related news. Then provide a summary "
            "analysis of the competing business factors. Do not make a prediction "
            "of stock-price movement for 2024-06-28 to 2024-07-05."
        )
        system_prompt = ABLATION_SYSTEM_PROMPT
    else:
        raise ValueError(f"Unknown prompt variant: {variant}")
    return (
        B_INST + B_SYS + system_prompt + E_SYS
        + information + "\n\n" + request + E_INST
    )


def _section_bounds(
    text: str, start_pattern: re.Pattern[str], following: list[re.Pattern[str]]
) -> tuple[str | None, str]:
    match = start_pattern.search(text)
    if not match:
        return None, ""
    end = len(text)
    for pattern in following:
        candidate = pattern.search(text, match.end())
        if candidate and candidate.start() < end:
            end = candidate.start()
    return text[match.end():end].strip(), ""


def _numbered_items(section: str | None) -> list[str]:
    if not section:
        return []
    return [match.group(1).strip() for match in NUMBERED_ITEM.finditer(section)]


def parse_native_output(output: str, variant: str) -> dict[str, Any]:
    """Parse only explicitly written official headings and labeled content."""
    positive_section, _ = _section_bounds(
        output,
        HEADER_PATTERNS["positive_developments"],
        [
            HEADER_PATTERNS["potential_concerns"],
            HEADER_PATTERNS["prediction_analysis"],
            HEADER_PATTERNS["analysis"],
        ],
    )
    concern_section, _ = _section_bounds(
        output,
        HEADER_PATTERNS["potential_concerns"],
        [HEADER_PATTERNS["prediction_analysis"], HEADER_PATTERNS["analysis"]],
    )
    positives = _numbered_items(positive_section)
    concerns = _numbered_items(concern_section)

    prediction = ""
    analysis = ""
    found_prediction = False
    found_analysis = False
    if variant == "full":
        combined_header = HEADER_PATTERNS["prediction_analysis"].search(output)
        tail = output[combined_header.end():] if combined_header else ""
        prediction_match = PREDICTION_LINE.search(tail)
        analysis_match = ANALYSIS_LINE.search(tail)
        if prediction_match:
            prediction = prediction_match.group(1).strip()
            found_prediction = bool(prediction)
        if analysis_match:
            analysis = tail[analysis_match.end():].strip()
            first_line = analysis_match.group(1).strip()
            analysis = (first_line + ("\n" + analysis if analysis else "")).strip()
            found_analysis = bool(analysis)
    else:
        analysis_header = HEADER_PATTERNS["analysis"].search(output)
        if analysis_header:
            analysis = output[analysis_header.end():].strip()
            found_analysis = bool(analysis)

    found_positive = bool(positive_section is not None and positives)
    found_concerns = bool(concern_section is not None and concerns)
    expected = {
        "positive_developments": found_positive,
        "potential_concerns": found_concerns,
        "analysis": found_analysis,
    }
    if variant == "full":
        expected["prediction"] = found_prediction
    errors = [name for name, found in expected.items() if not found]
    return {
        "found_positive_developments": found_positive,
        "found_potential_concerns": found_concerns,
        "found_prediction": found_prediction,
        "found_analysis": found_analysis,
        "all_expected_blocks_found": not errors,
        "parsed_positive_developments": positives,
        "parsed_potential_concerns": concerns,
        "parsed_prediction": prediction,
        "parsed_analysis": analysis,
        "parse_error": (
            "" if not errors else "Missing or empty expected blocks: " + ", ".join(errors)
        ),
    }


def _active_adapters(model: Any) -> list[str]:
    active = getattr(model, "active_adapters", None)
    if callable(active):
        active = active()
    if active is None:
        active = getattr(model, "active_adapter", None)
    if active is None:
        return []
    return list(active) if isinstance(active, (list, tuple, set)) else [str(active)]


def _checkpoint_name(value: str) -> str:
    return Path(value.rstrip("/")).name.casefold()


def load_forecaster(local_files_only: bool = False):
    import torch
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this RTX 4090 Forecaster probe.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The active CUDA device does not support bfloat16.")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        device_map="auto",
        dtype=torch.bfloat16,
        quantization_config=quantization_config,
        local_files_only=local_files_only,
    )
    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_MODEL,
        is_trainable=False,
        local_files_only=local_files_only,
    ).eval()
    load_seconds = time.perf_counter() - started

    active = _active_adapters(model)
    configs = getattr(model, "peft_config", {})
    declared_bases = [
        str(getattr(config, "base_model_name_or_path", ""))
        for config in configs.values()
    ]
    diagnostics = {
        "base_model": BASE_MODEL,
        "adapter_model": ADAPTER_MODEL,
        "device": str(model.get_input_embeddings().weight.device),
        "gpu_name": torch.cuda.get_device_name(0),
        "quantization": "4bit",
        "bnb_4bit_quant_type": "nf4",
        "compute_dtype": "torch.bfloat16",
        "device_map": "auto",
        "load_seconds": round(load_seconds, 6),
        "local_files_only": local_files_only,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "model_is_peft_model": bool(configs),
        "available_adapters": sorted(str(name) for name in configs),
        "active_adapters": active,
        "adapter_is_active": bool(active),
        "adapter_declared_base_models": declared_bases,
        "base_identifier_exact_match": BASE_MODEL in declared_bases,
        "base_checkpoint_name_match": any(
            _checkpoint_name(BASE_MODEL) == _checkpoint_name(item)
            for item in declared_bases
        ),
        "peak_allocated_vram_gb": round(
            torch.cuda.max_memory_allocated(0) / (1024 ** 3), 2
        ),
        "peak_reserved_vram_gb": round(
            torch.cuda.max_memory_reserved(0) / (1024 ** 3), 2
        ),
    }
    if not diagnostics["adapter_is_active"]:
        raise RuntimeError("Forecaster LoRA adapter loaded but is not active.")
    return tokenizer, model, diagnostics


def _input_device(model: Any):
    return model.get_input_embeddings().weight.device


def generate_once(
    tokenizer: Any,
    model: Any,
    prompt: str,
    synchronize: Callable[[], None],
) -> dict[str, Any]:
    import torch

    tokens = tokenizer(prompt, return_tensors="pt", padding=False, truncation=False)
    input_token_count = int(tokens["input_ids"].shape[1])
    context = getattr(model.config, "max_position_embeddings", 4096)
    if input_token_count + MAX_NEW_TOKENS > context:
        raise ValueError(
            f"Prompt plus generation budget exceeds context: "
            f"{input_token_count}+{MAX_NEW_TOKENS}>{context}"
        )
    tokens = {name: value.to(_input_device(model)) for name, value in tokens.items()}
    generation_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "num_beams": 1,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
    }
    synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**tokens, **generation_kwargs)
    synchronize()
    elapsed = time.perf_counter() - started

    new_ids = generated[:, input_token_count:]
    full_decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    new_decoded = tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0]
    return {
        "full_decoded_sequence": full_decoded,
        "new_tokens_decoded_output": new_decoded,
        "raw_output": new_decoded,
        "input_token_count": input_token_count,
        "generated_token_count": int(new_ids.shape[1]),
        "generation_kwargs": generation_kwargs,
        "gpu_inference_seconds": round(elapsed, 6),
    }


def run_probe(tokenizer: Any, model: Any, model_info: dict[str, Any]):
    import torch

    synchronize = torch.cuda.synchronize
    results = []
    pair_outputs = {}
    for case_id, case in TEST_CASES.items():
        for variant in ("full", "no_prediction"):
            prompt = build_prompt(case, variant)
            pair_key = f"{case_id}:{variant}"
            pair_outputs[pair_key] = []
            pair_rows = []
            for run_number in (1, 2):
                generated = generate_once(tokenizer, model, prompt, synchronize)
                parsed = parse_native_output(
                    generated["new_tokens_decoded_output"], variant
                )
                row = {
                    "test_case_id": case_id,
                    "case_direction": case["expected_direction"],
                    "variant": variant,
                    "run_number": run_number,
                    "synthetic_fixture": True,
                    "full_prompt": prompt,
                    **generated,
                    "base_model": BASE_MODEL,
                    "adapter_model": ADAPTER_MODEL,
                    "active_adapter": model_info["active_adapters"],
                    **parsed,
                    "empty_output": not generated["new_tokens_decoded_output"].strip(),
                    "consistent": None,
                }
                pair_outputs[pair_key].append(generated["new_tokens_decoded_output"])
                pair_rows.append(row)
            consistent = pair_outputs[pair_key][0] == pair_outputs[pair_key][1]
            for row in pair_rows:
                row["consistent"] = consistent
                results.append(row)
    return results


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    full = [row for row in results if row["variant"] == "full"]
    ablation = [row for row in results if row["variant"] == "no_prediction"]
    pair_first_rows = [row for row in results if row["run_number"] == 1]
    total = len(results)

    def count(field: str, rows: list[dict[str, Any]]) -> int:
        return sum(bool(row[field]) for row in rows)

    by_case = {}
    for case_id in TEST_CASES:
        selected = [row for row in results if row["test_case_id"] == case_id]
        by_case[case_id] = {
            "generation_count": len(selected),
            "total_gpu_inference_seconds": round(
                sum(row["gpu_inference_seconds"] for row in selected), 6
            ),
            "average_gpu_inference_seconds": round(
                sum(row["gpu_inference_seconds"] for row in selected) / len(selected),
                6,
            ),
            "full_outputs_complete": count(
                "all_expected_blocks_found",
                [row for row in selected if row["variant"] == "full"],
            ),
            "ablation_outputs_complete": count(
                "all_expected_blocks_found",
                [row for row in selected if row["variant"] == "no_prediction"],
            ),
        }

    return {
        "synthetic_fixture": True,
        "prediction_accuracy_evaluated": False,
        "total_generations": total,
        "full_variant_generations": len(full),
        "ablation_variant_generations": len(ablation),
        "official_four_block_complete_count": count(
            "all_expected_blocks_found", full
        ),
        "official_four_block_complete_rate": _rate(
            count("all_expected_blocks_found", full), len(full)
        ),
        "ablation_three_block_complete_count": count(
            "all_expected_blocks_found", ablation
        ),
        "ablation_three_block_complete_rate": _rate(
            count("all_expected_blocks_found", ablation), len(ablation)
        ),
        "positive_developments_parse_rate": _rate(
            count("found_positive_developments", results), total
        ),
        "potential_concerns_parse_rate": _rate(
            count("found_potential_concerns", results), total
        ),
        "prediction_parse_rate_full_only": _rate(
            count("found_prediction", full), len(full)
        ),
        "analysis_parse_rate": _rate(count("found_analysis", results), total),
        "consistent_pair_count": count("consistent", pair_first_rows),
        "total_prompt_pairs": len(pair_first_rows),
        "identical_output_consistency_rate": _rate(
            count("consistent", pair_first_rows), len(pair_first_rows)
        ),
        "empty_output_count": count("empty_output", results),
        "empty_output_rate": _rate(count("empty_output", results), total),
        "average_generated_token_count": _rate(
            sum(row["generated_token_count"] for row in results), total
        ),
        "degenerated_short_outputs": [
            {
                "test_case_id": row["test_case_id"],
                "variant": row["variant"],
                "run_number": row["run_number"],
                "raw_output": row["raw_output"],
            }
            for row in results
            if row["raw_output"].strip().casefold() in {"yes", "no", "no no"}
        ],
        "parse_failures": [
            {
                "test_case_id": row["test_case_id"],
                "variant": row["variant"],
                "run_number": row["run_number"],
                "parse_error": row["parse_error"],
            }
            for row in results
            if row["parse_error"]
        ],
        "per_case_generation_time": by_case,
        "acceptance_thresholds": {
            "official_four_block_complete_rate": 0.90,
            "empty_output_rate": 0.0,
            "identical_output_consistency_rate": 0.95,
        },
        "meets_format_threshold": (
            _rate(count("all_expected_blocks_found", full), len(full)) or 0
        ) >= 0.90,
        "meets_empty_output_threshold": count("empty_output", results) == 0,
        "meets_consistency_threshold": (
            _rate(count("consistent", pair_first_rows), len(pair_first_rows)) or 0
        ) >= 0.95,
    }


def _csv_value(field: str, value: Any) -> Any:
    if field in {
        "generation_kwargs",
        "active_adapter",
        "parsed_positive_developments",
        "parsed_potential_concerns",
    }:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_reports(
    results: list[dict[str, Any]],
    model_info: dict[str, Any],
    json_path: str | Path,
    csv_path: str | Path,
) -> dict[str, Any]:
    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "purpose": (
            "Native FinGPT Forecaster format and narrative-direction diagnostic; "
            "not a measurement of forecast accuracy."
        ),
        "fixture_notice": (
            "All company news, weekly prices, and basic financials are synthetic "
            "fixtures. There are no real future labels, so prediction accuracy "
            "cannot be evaluated."
        ),
        "official_reference": (
            "AI4Finance-Foundation/FinGPT/fingpt/FinGPT_Forecaster/app.py"
        ),
        "model": model_info,
        "summary": summarize(results),
        "results": results,
    }
    Path(json_path).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with Path(csv_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    field: _csv_value(field, row[field])
                    for field in CSV_FIELDS
                }
            )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run native-format FinGPT Forecaster and no-prediction ablation "
            "diagnostics without modifying the production pipeline."
        )
    )
    parser.add_argument("--json-output", default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--csv-output", default=DEFAULT_CSV_OUTPUT)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Refuse network access and require both model repositories in cache.",
    )
    args = parser.parse_args()

    try:
        tokenizer, model, model_info = load_forecaster(
            local_files_only=args.local_files_only
        )
        results = run_probe(tokenizer, model, model_info)
        report = write_reports(
            results, model_info, args.json_output, args.csv_output
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
