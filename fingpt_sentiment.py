import re
import time
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from config import (
    FINGPT_DEFAULT_MODEL_PROFILE, FINGPT_DEFAULT_QUANTIZATION,
    FINGPT_SENTIMENT_MODEL, FINGPT_BASE_MODEL, FINGPT_ADAPTER_TYPE, FINGPT_MODEL_PROFILES,
)


FINGPT_PROMPT = (
    "Instruction: What is the sentiment of this news? "
    "Please choose an answer from {{negative/neutral/positive}}\n"
    "Input: {text}\n"
    "Answer: "
)
FINGPT_SENTIMENT_COLUMNS = [
    "sentiment_label",
    "sentiment_score",
    "model_name",
    "model_profile",
    "fingpt_error",
    "fingpt_raw_output",
    "fingpt_inference_seconds",
]


@dataclass
class FinGPTLoadInfo:
    device: str
    load_seconds: float
    cuda_available: bool
    gpu_name: str | None
    gpu_total_vram_gb: float | None
    quantization: str
    model_name: str
    base_model: str
    peak_allocated_vram_gb: float | None
    peak_reserved_vram_gb: float | None


def score_label(label: str) -> int:
    label = str(label).lower()
    if label == "positive":
        return 1
    if label == "negative":
        return -1
    return 0


def build_fingpt_prompt(text: str, max_chars: int = 1800) -> str:
    clean_text = " ".join(str(text).split())[:max_chars]
    return FINGPT_PROMPT.format(text=clean_text)


def parse_sentiment_label(output: str) -> str | None:
    text = str(output).lower()
    answer_match = re.search(r"answer:\s*([a-z]+)", text)
    candidates = [answer_match.group(1)] if answer_match else []
    candidates.extend(re.findall(r"\b(positive|negative|neutral)\b", text))
    for candidate in candidates:
        if candidate in {"positive", "negative", "neutral"}:
            return candidate
    return None


def get_torch_hardware() -> dict:
    import torch

    info = {
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "gpu_name": None,
        "gpu_total_vram_gb": None,
        "gpu_allocated_vram_gb": None,
        "gpu_reserved_vram_gb": None,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_total_vram_gb"] = round(props.total_memory / (1024**3), 2)
        info["gpu_allocated_vram_gb"] = round(torch.cuda.memory_allocated(0) / (1024**3), 2)
        info["gpu_reserved_vram_gb"] = round(torch.cuda.memory_reserved(0) / (1024**3), 2)
    return info


def load_fingpt_model(
    allow_cpu: bool = False,
    model_profile: str = FINGPT_DEFAULT_MODEL_PROFILE,
    quantization: str = FINGPT_DEFAULT_QUANTIZATION,
):
    import torch

    hardware = get_torch_hardware()
    if not hardware["cuda_available"] and not allow_cpu:
        raise RuntimeError(
            "CUDA is not available. FinGPT v3.3 uses a 13B Llama2 base model; "
            "CPU inference is expected to be impractical. Pass allow_cpu=True only for explicit experiments."
        )

    profile = FINGPT_MODEL_PROFILES[model_profile]

    from peft import PeftModel
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    start = time.perf_counter()
    if hardware["cuda_available"]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)

    if quantization not in {"4bit", "8bit", "fp16"}:
        raise ValueError(f"Unsupported quantization: {quantization}")
    if not hardware["cuda_available"] and quantization != "fp16":
        raise ValueError("CPU experiments require --quantization fp16 (loaded as float32).")
    remote_code = profile["architecture"] == "chatglm"
    tokenizer = AutoTokenizer.from_pretrained(profile["base_model"], trust_remote_code=remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    compute_dtype = (
        torch.bfloat16 if hardware["cuda_available"] and torch.cuda.is_bf16_supported()
        else torch.float16 if hardware["cuda_available"] else torch.float32
    )
    model_kwargs = {
        "trust_remote_code": remote_code,
        "device_map": "auto",
        "dtype": compute_dtype if quantization != "fp16" else (
            torch.float16 if hardware["cuda_available"] else torch.float32
        ),
    }
    if quantization == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True,
        )
    elif quantization == "8bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model_class = AutoModelForCausalLM
    if remote_code:
        # ChatGLM2 depends on legacy remote code; compatibility errors propagate.
        model_class = AutoModel
        model_kwargs["config"] = AutoConfig.from_pretrained(profile["base_model"], trust_remote_code=True)
    base_model = model_class.from_pretrained(profile["base_model"], **model_kwargs)
    model = PeftModel.from_pretrained(base_model, profile["model_name"], is_trainable=False)
    model = model.eval()
    peak_allocated = None
    peak_reserved = None
    if hardware["cuda_available"]:
        peak_allocated = round(torch.cuda.max_memory_allocated(0) / (1024**3), 2)
        peak_reserved = round(torch.cuda.max_memory_reserved(0) / (1024**3), 2)
    load_info = FinGPTLoadInfo(
        device=hardware["device"],
        load_seconds=time.perf_counter() - start,
        cuda_available=hardware["cuda_available"],
        gpu_name=hardware["gpu_name"],
        gpu_total_vram_gb=hardware["gpu_total_vram_gb"],
        quantization=quantization,
        model_name=profile["model_name"],
        base_model=profile["base_model"],
        peak_allocated_vram_gb=peak_allocated,
        peak_reserved_vram_gb=peak_reserved,
    )
    return tokenizer, model, load_info


def run_fingpt_batch(
    tokenizer,
    model,
    texts: Iterable[str],
    max_new_tokens: int = 8,
    model_name: str = FINGPT_SENTIMENT_MODEL,
) -> tuple[list[dict], float]:
    import torch

    prompts = [build_fingpt_prompt(text) for text in texts]
    start = time.perf_counter()

    def fail_result(raw_output: str, error: str, label: str = "") -> dict:
        cleaned = str(raw_output or "").strip()
        return {
            "sentiment_label": label,
            "sentiment_score": None,
            "model_name": model_name,
            "fingpt_raw_output": cleaned,
            "fingpt_error": error,
        }

    if hasattr(model, "chat"):
        decoded = []
        for prompt in prompts:
            response, _history = model.chat(tokenizer, prompt, history=[])
            decoded.append(str(response))
        seconds = time.perf_counter() - start
        results = []
        for output in decoded:
            cleaned = output.strip()
            if not cleaned:
                results.append(fail_result(output, "FinGPT generated an empty response."))
                continue
            label = parse_sentiment_label(cleaned)
            if label is None:
                results.append(fail_result(cleaned, "FinGPT returned no parseable sentiment label.", label="unparsed"))
                continue
            results.append(
                {
                    "sentiment_label": label,
                    "sentiment_score": score_label(label),
                    "model_name": model_name,
                    "fingpt_raw_output": cleaned,
                    "fingpt_error": "",
                }
            )
        return results, seconds

    tokens = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    if getattr(model, "device", None) is not None:
        tokens = {key: value.to(model.device) for key, value in tokens.items()}
    with torch.no_grad():
        generated = model.generate(
            **tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    seconds = time.perf_counter() - start
    decoded = tokenizer.batch_decode(
        generated[:, tokens["input_ids"].shape[1]:], skip_special_tokens=True,
    )
    results = []
    for output in decoded:
        raw_answer = output.strip()
        if not raw_answer:
            results.append(fail_result(output, "FinGPT generated an empty response."))
            continue
        label = parse_sentiment_label(raw_answer)
        if label is None:
            results.append(fail_result(raw_answer, "FinGPT returned no parseable sentiment label.", label="unparsed"))
            continue
        results.append(
            {
                "sentiment_label": label,
                "sentiment_score": score_label(label),
                "model_name": model_name,
                "fingpt_raw_output": raw_answer,
                "fingpt_error": "",
            }
        )
    return results, seconds


def analyze_texts_with_fingpt(
    df: pd.DataFrame,
    text_column: str,
    batch_size: int = 1,
    allow_cpu: bool = False,
    model_profile: str = FINGPT_DEFAULT_MODEL_PROFILE,
    quantization: str = FINGPT_DEFAULT_QUANTIZATION,
) -> pd.DataFrame:
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    df = df.copy()
    if df.empty:
        for column in FINGPT_SENTIMENT_COLUMNS:
            df[column] = []
        return df

    tokenizer, model, _load_info = load_fingpt_model(
        allow_cpu=allow_cpu,
        model_profile=model_profile,
        quantization=quantization,
    )
    all_results = []
    texts = df[text_column].fillna("").astype(str).tolist()
    for start_idx in range(0, len(texts), batch_size):
        batch = texts[start_idx : start_idx + batch_size]
        batch_results, seconds = run_fingpt_batch(
            tokenizer, model, batch, model_name=FINGPT_MODEL_PROFILES[model_profile]["model_name"],
        )
        per_item_seconds = seconds / max(len(batch), 1)
        for result in batch_results:
            result["fingpt_inference_seconds"] = per_item_seconds
            result["model_profile"] = model_profile
        all_results.extend(batch_results)

    for column in FINGPT_SENTIMENT_COLUMNS:
        df[column] = [result.get(column) for result in all_results]
    return df
