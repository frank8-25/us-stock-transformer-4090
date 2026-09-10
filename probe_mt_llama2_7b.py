"""Isolated diagnostic probe for the FinGPT MT Llama-2 7B adapter.

This module does not participate in the production pipeline. It prints raw
generation evidence without changing the production schema or parser.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Callable

from config import FINGPT_MODEL_PROFILES
from fingpt_multidimensional import (
    build_multidimensional_prompt,
    parse_multidimensional_output,
)
from fingpt_sentiment import load_fingpt_model


MODEL_PROFILE = "mt-llama2-7b"
DEFAULT_OUTPUT = "probe_mt_llama2_7b_results.json"
SIMPLE_INPUT = "NVIDIA reported strong demand for AI chips and raised its revenue outlook."
FULL_INPUT = (
    "NVIDIA reported strong demand for AI chips while warning that export "
    "restrictions may reduce sales in China."
)


def _simple_json_prompt(expected: dict[str, int]) -> str:
    expected_json = json.dumps(expected, separators=(",", ":"))
    return (
        "Instruction: Return exactly the requested JSON object and no other text.\n"
        f"Requested JSON: {expected_json}\n"
        f"Input: {SIMPLE_INPUT}\n"
        "Answer: "
    )


def build_probe_cases() -> list[dict[str, Any]]:
    """Return the six fixed prompts in the requested diagnostic order."""
    return [
        {
            "id": "official_sentiment",
            "description": "Official sentiment prompt format",
            "prompt": (
                "Instruction: What is the sentiment of this news? Please choose an "
                "answer from {negative/neutral/positive}.\n"
                f"Input: {SIMPLE_INPUT}\nAnswer: "
            ),
            "max_new_tokens": 32,
            "expected_json_keys": None,
        },
        {
            "id": "official_headline_classification",
            "description": "Official headline classification prompt format",
            "prompt": (
                "Instruction: Does the news headline talk about price going up? "
                "Please choose an answer from {Yes/No}.\n"
                "Input: NVIDIA shares rose after the company raised its revenue outlook.\n"
                "Answer: "
            ),
            "max_new_tokens": 32,
            "expected_json_keys": None,
        },
        {
            "id": "official_ner",
            "description": "Official NER prompt format",
            "prompt": (
                "Instruction: Please extract entities and their types from the input "
                "sentence, entity types should be chosen from "
                "{person/organization/location}.\n"
                "Input: NVIDIA warned that export restrictions may reduce sales in China.\n"
                "Answer: "
            ),
            "max_new_tokens": 96,
            "expected_json_keys": None,
        },
        {
            "id": "single_field_json",
            "description": "Exact one-field JSON instruction",
            "prompt": _simple_json_prompt({"sentiment": 1}),
            "max_new_tokens": 48,
            "expected_json_keys": ["sentiment"],
        },
        {
            "id": "three_field_json",
            "description": "Exact three-field JSON instruction",
            "prompt": _simple_json_prompt(
                {"sentiment": 1, "risk": 1, "growth_outlook": 1}
            ),
            "max_new_tokens": 96,
            "expected_json_keys": ["sentiment", "risk", "growth_outlook"],
        },
        {
            "id": "current_multidimensional",
            "description": "Current experimental multidimensional prompt",
            "prompt": build_multidimensional_prompt(FULL_INPUT, "news"),
            "max_new_tokens": 1024,
            "expected_json_keys": "experimental_multidimensional_schema",
        },
    ]


def parse_probe_json(
    output: str, expected_keys: list[str] | str | None
) -> tuple[bool, Any | None, str]:
    """Parse diagnostic JSON strictly without changing the production parser."""
    if expected_keys == "experimental_multidimensional_schema":
        try:
            value = parse_multidimensional_output(output, FULL_INPUT)
        except Exception as exc:
            return False, None, f"{type(exc).__name__}: {exc}"
        return True, value, ""

    try:
        value = json.loads(output.strip())
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"
    if not isinstance(value, dict):
        return False, None, "ValueError: decoded JSON is not an object"
    if expected_keys is not None and list(value) != expected_keys:
        return (
            False,
            None,
            f"ValueError: expected keys in order {expected_keys!r}, got {list(value)!r}",
        )
    return True, value, ""


def _active_adapters(model: Any) -> list[str]:
    value = getattr(model, "active_adapters", None)
    if callable(value):
        value = value()
    if value is None:
        value = getattr(model, "active_adapter", None)
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple, set)) else [str(value)]


def inspect_adapter(model: Any, configured_base: str) -> dict[str, Any]:
    peft_configs = getattr(model, "peft_config", {})
    details: dict[str, Any] = {}
    declared_bases: list[str] = []
    for name, adapter_config in peft_configs.items():
        declared = str(getattr(adapter_config, "base_model_name_or_path", ""))
        declared_bases.append(declared)
        details[str(name)] = {
            "peft_type": str(getattr(adapter_config, "peft_type", "")),
            "task_type": str(getattr(adapter_config, "task_type", "")),
            "base_model_name_or_path": declared,
            "inference_mode": getattr(adapter_config, "inference_mode", None),
        }

    def checkpoint_name(value: str) -> str:
        return Path(value.rstrip("/")).name.casefold()

    active = _active_adapters(model)
    return {
        "model_is_peft_model": bool(peft_configs),
        "available_adapters": sorted(str(name) for name in peft_configs),
        "active_adapters": active,
        "adapter_is_active": bool(active),
        "configured_base_model": configured_base,
        "adapter_declared_base_models": declared_bases,
        "base_identifier_exact_match": configured_base in declared_bases,
        "base_checkpoint_name_match": any(
            checkpoint_name(configured_base) == checkpoint_name(item)
            for item in declared_bases
        ),
        "adapter_configs": details,
    }


def _model_input_device(model: Any):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return model.device


def run_case(
    tokenizer: Any,
    model: Any,
    case: dict[str, Any],
    synchronize: Callable[[], None],
) -> dict[str, Any]:
    import torch

    prompt = case["prompt"]
    tokens = tokenizer(prompt, return_tensors="pt", truncation=False)
    input_token_count = int(tokens["input_ids"].shape[1])
    input_device = _model_input_device(model)
    tokens = {name: value.to(input_device) for name, value in tokens.items()}
    generation_kwargs = {
        "max_new_tokens": case["max_new_tokens"],
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

    # Causal LM output is input + continuation. Decode each view once and never
    # concatenate decoded strings.
    new_token_ids = generated[:, input_token_count:]
    full_decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    new_decoded = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)[0]
    json_ok, parsed_json, json_error = parse_probe_json(
        new_decoded, case["expected_json_keys"]
    )
    new_ids = new_token_ids[0].detach().cpu().tolist()
    return {
        "id": case["id"],
        "description": case["description"],
        "prompt": prompt,
        "input_token_count": input_token_count,
        "generated_token_count": len(new_ids),
        "full_decoded_text": full_decoded,
        "new_tokens_decoded_text": new_decoded,
        "generation_kwargs": generation_kwargs,
        "max_length_explicitly_passed": "max_length" in generation_kwargs,
        "json_parse_success": json_ok,
        "parsed_json": parsed_json,
        "json_parse_error": json_error,
        "gpu_inference_seconds": round(elapsed, 6),
        "generated_contains_eos": tokenizer.eos_token_id in new_ids,
        "generated_eos_positions": [
            index
            for index, token_id in enumerate(new_ids)
            if token_id == tokenizer.eos_token_id
        ],
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Probe six MT Llama-2 7B prompts without touching the production pipeline."
        )
    )
    parser.add_argument(
        "--quantization", choices=["4bit", "8bit", "fp16"], default="4bit"
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="JSON result path; use '-' to print only to stdout.",
    )
    args = parser.parse_args()

    import torch

    profile = FINGPT_MODEL_PROFILES[MODEL_PROFILE]
    result: dict[str, Any] = {
        "model_profile": MODEL_PROFILE,
        "model_name": profile["model_name"],
        "configured_base_model": profile["base_model"],
        "quantization": args.quantization,
        "batch_size": 1,
        "load": {},
        "tokenizer": {},
        "adapter": {},
        "generation_defaults": {},
        "tests": [],
        "fatal_error": "",
    }
    exit_code = 0
    try:
        tokenizer, model, load_info = load_fingpt_model(
            model_profile=MODEL_PROFILE,
            quantization=args.quantization,
        )
        result["load"] = _json_safe(vars(load_info))
        result["tokenizer"] = {
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token": tokenizer.pad_token,
            "pad_token_id": tokenizer.pad_token_id,
            "pad_equals_eos": tokenizer.pad_token_id == tokenizer.eos_token_id,
            "padding_side": tokenizer.padding_side,
        }
        result["adapter"] = inspect_adapter(model, profile["base_model"])
        generation_config = getattr(model, "generation_config", None)
        result["generation_defaults"] = {
            "max_length": getattr(generation_config, "max_length", None),
            "max_new_tokens": getattr(generation_config, "max_new_tokens", None),
            "eos_token_id": getattr(generation_config, "eos_token_id", None),
            "pad_token_id": getattr(generation_config, "pad_token_id", None),
            "note": (
                "The probe passes max_new_tokens only. max_length shown here is an "
                "inherited GenerationConfig default and is overridden when "
                "max_new_tokens is supplied."
            ),
        }

        synchronize = (
            torch.cuda.synchronize if torch.cuda.is_available() else lambda: None
        )
        for case in build_probe_cases():
            try:
                result["tests"].append(
                    run_case(tokenizer, model, case, synchronize)
                )
            except Exception as exc:
                traceback.print_exc()
                exit_code = 1
                result["tests"].append(
                    {
                        "id": case["id"],
                        "description": case["description"],
                        "prompt": case["prompt"],
                        "generation_kwargs": {
                            "max_new_tokens": case["max_new_tokens"],
                            "do_sample": False,
                            "num_beams": 1,
                            "pad_token_id": tokenizer.pad_token_id,
                            "eos_token_id": tokenizer.eos_token_id,
                        },
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
    except Exception as exc:
        traceback.print_exc()
        result["fatal_error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        exit_code = 1

    serialized = json.dumps(_json_safe(result), ensure_ascii=False, indent=2)
    print(serialized)
    if args.output != "-":
        Path(args.output).write_text(serialized + "\n", encoding="utf-8")
        print(
            f"\nSaved diagnostic report: {Path(args.output).resolve()}",
            file=sys.stderr,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
