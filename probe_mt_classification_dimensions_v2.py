"""Second MT 7B multidimensional classification probe.

Compares deterministic free generation with candidate-label conditional
log-likelihood. This file is isolated from every production pipeline.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Any, Callable

from fingpt_sentiment import load_fingpt_model

MODEL_PROFILE = "mt-llama2-7b"
BASE_MODEL = "meta-llama/Llama-2-7b-hf"
ADAPTER_MODEL = "FinGPT/fingpt-mt_llama2-7b_lora"
DEFAULT_JSON_OUTPUT = "probe_mt_classification_dimensions_v2_results.json"
DEFAULT_CSV_OUTPUT = "probe_mt_classification_dimensions_v2_results.csv"
MAX_NEW_TOKENS = 8
LOW_MARGIN_THRESHOLD = 0.10
TERNARY_LABELS = ["negative", "neutral", "positive"]
BINARY_LABELS = ["Yes", "No"]

DIMENSIONS = [
    {"name": "sentiment", "label_space": TERNARY_LABELS,
     "definition": "overall financial sentiment toward NVIDIA",
     "instruction": "What is the overall financial sentiment toward NVIDIA expressed by the text? Please choose exactly one answer from {negative/neutral/positive}."},
    {"name": "growth_outlook", "label_space": TERNARY_LABELS,
     "definition": "effect on future revenue, demand, or business growth",
     "instruction": "What effect does the text indicate for NVIDIA's future revenue, demand, or business growth outlook? Please choose exactly one answer from {negative/neutral/positive}."},
    {"name": "financial_strength", "label_space": TERNARY_LABELS,
     "definition": "effect on profitability, cash flow, balance sheet, or financial resilience",
     "instruction": "What effect does the text indicate for NVIDIA's profitability, cash flow, balance sheet, or financial resilience? Please choose exactly one answer from {negative/neutral/positive}."},
    {"name": "fundamental_impact", "label_space": TERNARY_LABELS,
     "definition": "effect on actual operations and business fundamentals",
     "instruction": "What effect does the text indicate for NVIDIA's actual operations and business fundamentals? Please choose exactly one answer from {negative/neutral/positive}."},
    {"name": "short_term_impact", "label_space": TERNARY_LABELS,
     "definition": "company or market impact over the next days to weeks",
     "instruction": "What company or market impact does the text indicate for NVIDIA over the next several days to weeks? Please choose exactly one answer from {negative/neutral/positive}."},
    {"name": "long_term_impact", "label_space": TERNARY_LABELS,
     "definition": "effect on competitiveness and growth over coming months or longer",
     "instruction": "What effect does the text indicate for NVIDIA's competitiveness and growth over the coming months or longer? Please choose exactly one answer from {negative/neutral/positive}."},
    {"name": "risk_presence", "label_space": BINARY_LABELS,
     "definition": "presence of a material downside business, financial, or outlook risk",
     "instruction": "Does the text describe a material downside risk to NVIDIA's business, financial performance, or future outlook? Please choose exactly one answer from {Yes/No}."},
    {"name": "uncertainty_presence", "label_space": BINARY_LABELS,
     "definition": "presence of material uncertainty about future business or financial performance",
     "instruction": "Does the text express material uncertainty about NVIDIA's future business or financial performance? Please choose exactly one answer from {Yes/No}."},
    {"name": "relevance", "label_space": BINARY_LABELS,
     "definition": "material relevance to business, financial performance, or future outlook",
     "instruction": "Is the information materially relevant to NVIDIA's business, financial performance, or future outlook? Please choose exactly one answer from {Yes/No}."},
    {"name": "price_narrative", "label_space": BINARY_LABELS,
     "definition": "primarily price movement without a material fundamental change",
     "instruction": "Does the text mainly describe NVIDIA's past or current stock-price movement without providing a material change in business fundamentals? Please choose exactly one answer from {Yes/No}."},
]

FIXTURES = [
    {
        "case_id": "positive_fundamental",
        "text": "NVIDIA reported strong demand for AI chips, raised its revenue and profit outlook, and received new long-term customer commitments. Management reported no material new business risks.",
        "expected_labels": {
            "sentiment": "positive", "growth_outlook": "positive",
            "financial_strength": "positive", "fundamental_impact": "positive",
            "short_term_impact": "positive", "long_term_impact": "positive",
            "risk_presence": "No", "uncertainty_presence": "No",
            "relevance": "Yes", "price_narrative": "No",
        },
        "ambiguous_dimensions": [],
    },
    {
        "case_id": "negative_fundamental",
        "text": "NVIDIA customers delayed orders, channel inventory increased, and component costs rose. The company lowered its revenue outlook, while export restrictions are expected to reduce future sales.",
        "expected_labels": {
            "sentiment": "negative", "growth_outlook": "negative",
            "financial_strength": "negative", "fundamental_impact": "negative",
            "short_term_impact": "negative", "long_term_impact": "negative",
            "risk_presence": "Yes", "uncertainty_presence": "Yes",
            "relevance": "Yes", "price_narrative": "No",
        },
        "ambiguous_dimensions": [],
    },
    {
        "case_id": "mixed_fundamental",
        "text": "NVIDIA raised its overall revenue outlook as AI demand increased, but China export restrictions may reduce part of its sales. Approval timing for compliant products remains uncertain.",
        "expected_labels": {
            "sentiment": "neutral", "growth_outlook": "positive",
            "financial_strength": "positive", "fundamental_impact": None,
            "short_term_impact": None, "long_term_impact": None,
            "risk_presence": "Yes", "uncertainty_presence": "Yes",
            "relevance": "Yes", "price_narrative": "No",
        },
        "ambiguous_dimensions": ["fundamental_impact", "short_term_impact", "long_term_impact"],
    },
    {
        "case_id": "price_only_narrative",
        "text": "NVIDIA shares rose for a fifth session, reached a new record high, and were described by market commentators as having strong price momentum. No new revenue, demand, product, cost, or business fundamental information was reported.",
        "expected_labels": {
            "sentiment": "positive", "growth_outlook": "neutral",
            "financial_strength": "neutral", "fundamental_impact": "neutral",
            "short_term_impact": None, "long_term_impact": "neutral",
            "risk_presence": "No", "uncertainty_presence": "No",
            "relevance": "No", "price_narrative": "Yes",
        },
        "ambiguous_dimensions": ["short_term_impact"],
    },
    {
        "case_id": "irrelevant_news",
        "text": "A regional food retailer opened a new distribution center and appointed a new store operations manager. The event has no stated connection to NVIDIA, semiconductors, AI computing, or technology markets.",
        "expected_labels": {
            "sentiment": "neutral", "growth_outlook": "neutral",
            "financial_strength": "neutral", "fundamental_impact": "neutral",
            "short_term_impact": "neutral", "long_term_impact": "neutral",
            "risk_presence": "No", "uncertainty_presence": "No",
            "relevance": "No", "price_narrative": "No",
        },
        "ambiguous_dimensions": [],
    },
]

CSV_FIELDS = [
    "case_id", "dimension", "label_space", "expected_label", "run_id", "prompt",
    "raw_generated_output", "parsed_free_label", "free_label_valid",
    "free_matches_expected", "candidate_selected_label",
    "candidate_matches_expected", "candidate_scores",
    "candidate_log_likelihoods", "candidate_normalized_log_likelihoods",
    "score_margin", "free_candidate_agreement", "input_tokens",
    "generated_tokens", "inference_seconds", "error",
]


def build_prompt(dimension: dict[str, Any], text: str) -> str:
    return f"Instruction: {dimension['instruction']}\nInput: {text}\nAnswer: "


def parse_free_label(output: str, labels: list[str]) -> tuple[str | None, str]:
    cleaned = output.strip()
    if cleaned.endswith("."):
        cleaned = cleaned[:-1].rstrip()
    canonical = {label.casefold(): label for label in labels}
    return canonical.get(cleaned.casefold()), cleaned


def _input_device(model: Any):
    return model.get_input_embeddings().weight.device


def generate_free(tokenizer: Any, model: Any, prompt: str,
                  synchronize: Callable[[], None]) -> dict[str, Any]:
    import torch
    tokens = tokenizer(prompt, return_tensors="pt", truncation=False)
    input_tokens = int(tokens["input_ids"].shape[1])
    tokens = {key: value.to(_input_device(model)) for key, value in tokens.items()}
    kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS, "do_sample": False, "num_beams": 1,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**tokens, **kwargs)
    synchronize()
    elapsed = time.perf_counter() - started
    new_ids = generated[:, input_tokens:]
    return {
        "full_decoded_sequence": tokenizer.batch_decode(
            generated, skip_special_tokens=True)[0],
        "raw_generated_output": tokenizer.batch_decode(
            new_ids, skip_special_tokens=True)[0],
        "input_tokens": input_tokens,
        "generated_tokens": int(new_ids.shape[1]),
        "inference_seconds": round(elapsed, 6),
        "generation_kwargs": kwargs,
    }


def score_candidate_token_ids(model: Any, prompt_ids: Any,
                              candidate_ids: Any) -> dict[str, Any]:
    """Score answer targets only; prompt-token targets are never accumulated."""
    import torch
    if prompt_ids.ndim != 2 or candidate_ids.ndim != 2:
        raise ValueError("prompt_ids and candidate_ids must be rank-2 tensors")
    if prompt_ids.shape[0] != 1 or candidate_ids.shape[0] != 1:
        raise ValueError("candidate scoring requires batch size 1")
    if candidate_ids.shape[1] < 1:
        raise ValueError("candidate must contain at least one token")

    input_ids = torch.cat([prompt_ids, candidate_ids], dim=1)
    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
        ).logits[0].float()
    prompt_length = int(prompt_ids.shape[1])
    candidate_length = int(candidate_ids.shape[1])
    # Position n-1 predicts target token n. The first selected position predicts
    # the first candidate token immediately after the complete prompt.
    positions = torch.arange(
        prompt_length - 1,
        prompt_length + candidate_length - 1,
        device=logits.device,
    )
    targets = candidate_ids[0].to(logits.device)
    token_logs = torch.log_softmax(
        logits[positions], dim=-1
    ).gather(1, targets.unsqueeze(1)).squeeze(1)
    raw_sum = float(token_logs.sum().item())
    return {
        "raw_log_likelihood": raw_sum,
        "normalized_log_likelihood": raw_sum / candidate_length,
        "scored_token_count": candidate_length,
        "token_log_likelihoods": [
            float(value) for value in token_logs.detach().cpu().tolist()
        ],
        "prediction_positions": [
            int(value) for value in positions.detach().cpu().tolist()
        ],
    }


def score_candidates(tokenizer: Any, model: Any, prompt: str,
                     labels: list[str],
                     synchronize: Callable[[], None]) -> dict[str, Any]:
    import torch
    prompt_ids = tokenizer(
        prompt, return_tensors="pt", truncation=False
    )["input_ids"].to(_input_device(model))
    details = {}
    synchronize()
    started = time.perf_counter()
    for label in labels:
        # Prompt already ends in one literal space. Llama tokenizes label and
        # " "+label identically, so no extra prefix is injected here.
        candidate_ids = tokenizer(
            label, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(_input_device(model))
        details[label] = {
            **score_candidate_token_ids(model, prompt_ids, candidate_ids),
            "candidate_token_ids": [
                int(value) for value in candidate_ids[0].detach().cpu().tolist()
            ],
            "candidate_decoded": tokenizer.decode(
                candidate_ids[0], skip_special_tokens=True
            ),
        }
    synchronize()
    elapsed = time.perf_counter() - started
    normalized = torch.tensor(
        [details[label]["normalized_log_likelihood"] for label in labels],
        dtype=torch.float64,
    )
    relative_values = torch.softmax(normalized, dim=0).tolist()
    relative = {
        label: float(value) for label, value in zip(labels, relative_values)
    }
    ranked = sorted(labels, key=lambda label: relative[label], reverse=True)
    return {
        "selected_label": ranked[0],
        "candidate_scores": relative,
        "candidate_log_likelihoods": {
            label: details[label]["raw_log_likelihood"] for label in labels
        },
        "candidate_normalized_log_likelihoods": {
            label: details[label]["normalized_log_likelihood"] for label in labels
        },
        "candidate_details": details,
        "score_margin": relative[ranked[0]] - relative[ranked[1]],
        "scoring_seconds": round(elapsed, 6),
        "relative_score_basis": (
            "softmax over length-normalized conditional log-likelihoods"
        ),
        "scores_are_calibrated_probabilities": False,
    }


def _matches(actual: str | None, expected: str | None) -> bool | None:
    if expected is None:
        return None
    return actual is not None and actual.casefold() == expected.casefold()


def _active_adapters(model: Any) -> list[str]:
    active = getattr(model, "active_adapters", None)
    if callable(active):
        active = active()
    if active is None:
        active = getattr(model, "active_adapter", None)
    if active is None:
        return []
    return list(active) if isinstance(active, (list, tuple, set)) else [str(active)]


def run_probe(tokenizer: Any, model: Any) -> list[dict[str, Any]]:
    import torch
    synchronize = torch.cuda.synchronize
    rows = []
    for fixture in FIXTURES:
        for dimension in DIMENSIONS:
            prompt = build_prompt(dimension, fixture["text"])
            expected = fixture["expected_labels"][dimension["name"]]
            try:
                candidate = score_candidates(
                    tokenizer, model, prompt, dimension["label_space"], synchronize
                )
                pair = []
                for run_id in (1, 2):
                    free = generate_free(tokenizer, model, prompt, synchronize)
                    parsed, cleaned = parse_free_label(
                        free["raw_generated_output"], dimension["label_space"]
                    )
                    row = {
                        "case_id": fixture["case_id"],
                        "dimension": dimension["name"],
                        "label_space": dimension["label_space"],
                        "expected_label": expected,
                        "ambiguous_expected_label": expected is None,
                        "run_id": run_id,
                        "prompt": prompt,
                        **free,
                        "cleaned_free_answer": cleaned,
                        "parsed_free_label": parsed,
                        "free_label_valid": parsed is not None,
                        "free_matches_expected": _matches(parsed, expected),
                        "candidate_selected_label": candidate["selected_label"],
                        "candidate_matches_expected": _matches(
                            candidate["selected_label"], expected),
                        "candidate_scores": candidate["candidate_scores"],
                        "candidate_log_likelihoods": candidate[
                            "candidate_log_likelihoods"],
                        "candidate_normalized_log_likelihoods": candidate[
                            "candidate_normalized_log_likelihoods"],
                        "candidate_details": candidate["candidate_details"],
                        "score_margin": candidate["score_margin"],
                        "candidate_scoring_seconds": candidate["scoring_seconds"],
                        "free_candidate_agreement": (
                            parsed is not None
                            and parsed.casefold()
                            == candidate["selected_label"].casefold()
                        ),
                        "error": "",
                    }
                    pair.append(row)
                consistent = (
                    pair[0]["cleaned_free_answer"].casefold()
                    == pair[1]["cleaned_free_answer"].casefold()
                )
                for row in pair:
                    row["free_generation_consistent"] = consistent
                    rows.append(row)
            except Exception as exc:
                traceback.print_exc()
                for run_id in (1, 2):
                    rows.append(error_row(
                        fixture, dimension, expected, prompt, run_id, exc))
    return rows


def error_row(fixture: dict[str, Any], dimension: dict[str, Any],
              expected: str | None, prompt: str, run_id: int,
              exc: Exception) -> dict[str, Any]:
    return {
        "case_id": fixture["case_id"], "dimension": dimension["name"],
        "label_space": dimension["label_space"], "expected_label": expected,
        "ambiguous_expected_label": expected is None, "run_id": run_id,
        "prompt": prompt, "raw_generated_output": "",
        "cleaned_free_answer": "", "parsed_free_label": None,
        "free_label_valid": False,
        "free_matches_expected": None if expected is None else False,
        "candidate_selected_label": None,
        "candidate_matches_expected": None if expected is None else False,
        "candidate_scores": {}, "candidate_log_likelihoods": {},
        "candidate_normalized_log_likelihoods": {}, "candidate_details": {},
        "score_margin": None, "candidate_scoring_seconds": None,
        "free_candidate_agreement": False, "input_tokens": None,
        "generated_tokens": None, "inference_seconds": None,
        "full_decoded_sequence": "", "generation_kwargs": {},
        "free_generation_consistent": False,
        "error": f"{type(exc).__name__}: {exc}",
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _accuracy(rows: list[dict[str, Any]], field: str):
    eligible = [row for row in rows if row[field] is not None]
    correct = sum(bool(row[field]) for row in eligible)
    return correct, len(eligible), _rate(correct, len(eligible))


def summarize_dimension(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    selected = [row for row in rows if row["dimension"] == name]
    first_runs = [row for row in selected if row["run_id"] == 1]
    free_correct, free_total, free_accuracy = _accuracy(
        selected, "free_matches_expected")
    candidate_correct, candidate_total, candidate_accuracy = _accuracy(
        first_runs, "candidate_matches_expected")
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for row in selected:
        if row["expected_label"] is not None:
            confusion[row["expected_label"]][
                row["parsed_free_label"] or "<invalid>"
            ] += 1
    candidate_counts = Counter(
        row["candidate_selected_label"] or "<error>" for row in first_runs)
    labels = [
        row["candidate_selected_label"] for row in first_runs
        if row["candidate_selected_label"] is not None
    ]
    margins = [
        row["score_margin"] for row in first_runs
        if row["score_margin"] is not None
    ]
    valid = sum(bool(row["free_label_valid"]) for row in selected)
    consistent = sum(
        bool(row["free_generation_consistent"]) for row in first_runs)
    return {
        "free_valid_count": valid,
        "free_generation_count": len(selected),
        "free_valid_rate": _rate(valid, len(selected)),
        "free_expected_correct": free_correct,
        "free_expected_total": free_total,
        "free_expected_accuracy": free_accuracy,
        "free_consistent_pairs": consistent,
        "free_total_pairs": len(first_runs),
        "free_consistency_rate": _rate(consistent, len(first_runs)),
        "free_confusion": {
            expected: dict(counts) for expected, counts in confusion.items()
        },
        "candidate_expected_correct": candidate_correct,
        "candidate_expected_total": candidate_total,
        "candidate_expected_accuracy": candidate_accuracy,
        "candidate_selection_counts": dict(candidate_counts),
        "candidate_average_score_margin": (
            sum(margins) / len(margins) if margins else None),
        "candidate_label_collapse": (
            len(labels) > 1 and len(set(labels)) == 1),
        "free_candidate_agreement_rate": _rate(
            sum(bool(row["free_candidate_agreement"]) for row in selected),
            len(selected)),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first_runs = [row for row in rows if row["run_id"] == 1]
    free_correct, free_total, free_accuracy = _accuracy(
        rows, "free_matches_expected")
    candidate_correct, candidate_total, candidate_accuracy = _accuracy(
        first_runs, "candidate_matches_expected")
    valid = sum(bool(row["free_label_valid"]) for row in rows)
    consistent = sum(
        bool(row["free_generation_consistent"]) for row in first_runs)
    margins = [
        row["score_margin"] for row in first_runs
        if row["score_margin"] is not None
    ]
    complete_scores = sum(
        len(row["candidate_scores"]) == len(row["label_space"])
        and len(row["candidate_log_likelihoods"]) == len(row["label_space"])
        and len(row["candidate_normalized_log_likelihoods"])
        == len(row["label_space"])
        for row in rows
    )
    by_dimension = {
        dimension["name"]: summarize_dimension(rows, dimension["name"])
        for dimension in DIMENSIONS
    }
    low_margin = [
        {
            "case_id": row["case_id"],
            "dimension": row["dimension"],
            "selected_label": row["candidate_selected_label"],
            "score_margin": row["score_margin"],
            "candidate_scores": row["candidate_scores"],
        }
        for row in first_runs
        if row["score_margin"] is not None
        and row["score_margin"] < LOW_MARGIN_THRESHOLD
    ]
    return {
        "fixture_count": len(FIXTURES),
        "dimension_count": len(DIMENSIONS),
        "free_generation_count": len(rows),
        "free_valid_count": valid,
        "free_valid_rate": _rate(valid, len(rows)),
        "free_expected_correct": free_correct,
        "free_expected_total": free_total,
        "free_expected_accuracy": free_accuracy,
        "free_consistent_pairs": consistent,
        "free_total_pairs": len(first_runs),
        "free_consistency_rate": _rate(consistent, len(first_runs)),
        "free_empty_output_count": sum(
            not row["raw_generated_output"].strip() for row in rows),
        "free_empty_output_rate": _rate(
            sum(not row["raw_generated_output"].strip() for row in rows),
            len(rows)),
        "free_no_no_output_count": sum(
            row["cleaned_free_answer"].casefold() == "no no" for row in rows),
        "free_invalid_outputs": [
            {
                "case_id": row["case_id"], "dimension": row["dimension"],
                "run_id": row["run_id"],
                "raw_generated_output": row["raw_generated_output"],
            }
            for row in rows if not row["free_label_valid"]
        ],
        "candidate_expected_correct": candidate_correct,
        "candidate_expected_total": candidate_total,
        "candidate_expected_accuracy": candidate_accuracy,
        "candidate_complete_score_rows": complete_scores,
        "candidate_total_rows": len(rows),
        "candidate_complete_score_rate": _rate(complete_scores, len(rows)),
        "candidate_average_score_margin": (
            sum(margins) / len(margins) if margins else None),
        "candidate_low_margin_threshold": LOW_MARGIN_THRESHOLD,
        "candidate_low_margin_cases": low_margin,
        "candidate_collapsed_dimensions": [
            name for name, stats in by_dimension.items()
            if stats["candidate_label_collapse"]
        ],
        "free_candidate_agreement_count": sum(
            bool(row["free_candidate_agreement"]) for row in rows),
        "free_candidate_agreement_rate": _rate(
            sum(bool(row["free_candidate_agreement"]) for row in rows),
            len(rows)),
        "by_dimension": by_dimension,
        "special_dimensions": {
            name: by_dimension[name]
            for name in ("risk_presence", "uncertainty_presence", "relevance")
        },
        "acceptance_thresholds": {
            "free_valid_rate": 0.90,
            "free_consistency_rate": 0.95,
            "free_empty_output_rate": 0.0,
            "minimum_expected_label_accuracy_for_pipeline": 0.80,
        },
        "meets_free_valid_threshold": (
            _rate(valid, len(rows)) or 0) >= 0.90,
        "meets_free_consistency_threshold": (
            _rate(consistent, len(first_runs)) or 0) >= 0.95,
        "meets_empty_output_threshold": all(
            row["raw_generated_output"].strip() for row in rows),
        "meets_candidate_completeness_threshold": complete_scores == len(rows),
        "free_meets_expected_accuracy_for_pipeline": (
            free_accuracy is not None and free_accuracy >= 0.80),
        "candidate_meets_expected_accuracy_for_pipeline": (
            candidate_accuracy is not None and candidate_accuracy >= 0.80),
    }


def _csv_value(field: str, value: Any) -> Any:
    if field in {
        "label_space", "candidate_scores", "candidate_log_likelihoods",
        "candidate_normalized_log_likelihoods",
    }:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "" if value is None else value


def write_reports(rows: list[dict[str, Any]], model_info: dict[str, Any],
                  json_path: str | Path, csv_path: str | Path):
    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "purpose": (
            "Second independent MT multidimensional classification diagnostic; "
            "not connected to the production pipeline."),
        "score_notice": (
            "Candidate relative scores are softmax values over candidate "
            "length-normalized conditional log-likelihoods, not calibrated "
            "probabilities."),
        "expected_label_notice": (
            "Expected labels and ambiguous nulls were fixed before inference "
            "and were not altered after seeing results."),
        "model": model_info,
        "fixtures": FIXTURES,
        "dimensions": DIMENSIONS,
        "generation_kwargs": {
            "max_new_tokens": MAX_NEW_TOKENS, "do_sample": False,
            "num_beams": 1, "batch_size": 1,
            "pad_token_id": model_info["pad_token_id"],
            "eos_token_id": model_info["eos_token_id"],
        },
        "candidate_scoring": {
            "objective": "log P(candidate | prompt)",
            "raw_sum_saved": True, "length_normalized_saved": True,
            "relative_score_basis": (
                "softmax within candidate set over length-normalized "
                "log-likelihood"),
            "scores_are_calibrated_probabilities": False,
            "prompt_ends_with_literal_space": True,
            "only_answer_tokens_scored": True,
        },
        "summary": summarize(rows),
        "results": rows,
    }
    Path(json_path).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    with Path(csv_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: _csv_value(field, row[field]) for field in CSV_FIELDS
            })
    return report


def model_metadata(model: Any, tokenizer: Any, load_info: Any) -> dict[str, Any]:
    import torch
    active = _active_adapters(model)
    configs = getattr(model, "peft_config", {})
    declared = [
        str(getattr(config, "base_model_name_or_path", ""))
        for config in configs.values()
    ]
    if not active:
        raise RuntimeError("MT LoRA adapter is not active.")
    return {
        "base_model": BASE_MODEL, "adapter_model": ADAPTER_MODEL,
        "active_adapters": active, "adapter_is_active": True,
        "adapter_declared_base_models": declared,
        "base_checkpoint_name_match": any(
            Path(value).name.casefold() == Path(BASE_MODEL).name.casefold()
            for value in declared),
        "quantization": "4bit", "bnb_4bit_quant_type": "nf4",
        "compute_dtype": "torch.bfloat16", "device_map": "auto",
        "gpu_name": torch.cuda.get_device_name(0),
        "device": load_info.device,
        "model_load_seconds": round(load_info.load_seconds, 6),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "peak_allocated_vram_gb": None,
        "peak_reserved_vram_gb": None,
        "cache_behavior": (
            "Hugging Face from_pretrained cache reuse; no force_download"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run MT classification v2 with free generation and candidate "
            "conditional log-likelihood."))
    parser.add_argument("--json-output", default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--csv-output", default=DEFAULT_CSV_OUTPUT)
    args = parser.parse_args()
    try:
        import torch
        tokenizer, model, load_info = load_fingpt_model(
            model_profile=MODEL_PROFILE, quantization="4bit")
        info = model_metadata(model, tokenizer, load_info)
        rows = run_probe(tokenizer, model)
        info["peak_allocated_vram_gb"] = round(
            torch.cuda.max_memory_allocated(0) / (1024 ** 3), 2)
        info["peak_reserved_vram_gb"] = round(
            torch.cuda.max_memory_reserved(0) / (1024 ** 3), 2)
        report = write_reports(
            rows, info, args.json_output, args.csv_output)
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
