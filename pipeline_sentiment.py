"""Resumable sentiment inference, document aggregation and point-in-time features."""
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import re
import traceback

import numpy as np
import pandas as pd

from pipeline_documents import SOURCES, chunk_text, digest, read_csv, save_json, write_csv

LABEL_SCORES = {"negative": -1, "neutral": 0, "positive": 1}


def infer_documents(documents, output, model_profile, quantization, retry_errors=False):
    from fingpt_sentiment import FINGPT_MODEL_PROFILES, load_fingpt_model, run_fingpt_batch
    from fingpt_report import write_html_report
    output = Path(output)
    started_at = datetime.now().astimezone().isoformat()
    model_name = FINGPT_MODEL_PROFILES[model_profile]["model_name"]
    signature = json.dumps({"profile": model_profile, "quantization": quantization,
                            "pipeline": 1, "max_tokens": 512, "max_chars": 1800,
                            "packages": {p: importlib.metadata.version(p)
                                         for p in ["torch", "transformers", "peft"]}}, sort_keys=True)
    save_json(output / "inference_config.json", json.loads(signature))
    results, all_chunks = [], []
    tokenizer = model = None
    load_error = ""
    for doc in documents.to_dict("records"):
        row = {**doc, "model_profile": model_profile, "model_name": model_name,
               "sentiment_label": "", "sentiment_score": None, "fingpt_raw_output": "",
               "fingpt_error": doc["preparation_error"], "fingpt_inference_seconds": 0.0,
               "chunk_count": 0, "successful_chunks": 0}
        doc_chunks = []
        if not row["fingpt_error"]:
            cache = output / "inference_cache" / (digest(signature + doc["document_id"] + doc["text"]) + ".json")
            try:
                if cache.exists():
                    saved = json.loads(cache.read_text(encoding="utf-8"))
                    if not saved["result"]["fingpt_error"] or not retry_errors:
                        # Refresh provenance/date review metadata even when inference is cached.
                        results.append({**saved["result"], **doc})
                        all_chunks.extend(saved["chunks"])
                        continue
                if load_error:
                    raise RuntimeError(load_error)
                if model is None:
                    try:
                        tokenizer, model, _ = load_fingpt_model(model_profile=model_profile, quantization=quantization)
                    except Exception as exc:
                        load_error = f"{type(exc).__name__}: {exc}"
                        raise
                chunks = chunk_text(doc["text"], tokenizer)
                row["chunk_count"] = len(chunks)
                for index, text in enumerate(chunks):
                    chunk = {"document_id": doc["document_id"], "chunk_index": index, "text": text}
                    checkpoint = output / "chunk_cache" / (digest(signature + text) + ".json")
                    saved_chunk = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else None
                    if saved_chunk is not None and (not saved_chunk["fingpt_error"] or not retry_errors):
                        chunk.update(saved_chunk)
                    else:
                        try:
                            batch, seconds = run_fingpt_batch(tokenizer, model, [text], model_name=model_name)
                            if len(batch) != 1:
                                raise ValueError("Unexpected inference batch size.")
                            prediction = batch[0]
                            labels = set(re.findall(r"\b(positive|neutral|negative)\b",
                                                    prediction["fingpt_raw_output"].lower()))
                            error = prediction.get("fingpt_error", "")
                            if len(labels) != 1 or prediction.get("sentiment_label") not in labels:
                                error = error or "Unparsed or ambiguous sentiment output."
                            prediction["fingpt_error"] = error
                            prediction["fingpt_inference_seconds"] = seconds
                            prediction["sentiment_score"] = (
                                LABEL_SCORES[prediction["sentiment_label"]] if not error else None)
                        except Exception as exc:
                            traceback.print_exc()
                            prediction = {"sentiment_label": "", "sentiment_score": None,
                                          "fingpt_raw_output": "", "fingpt_inference_seconds": 0.0,
                                          "fingpt_error": f"{type(exc).__name__}: {exc}"}
                        save_json(checkpoint, prediction)
                        chunk.update(prediction)
                    doc_chunks.append(chunk)
                valid = [c for c in doc_chunks if not c["fingpt_error"]]
                row["successful_chunks"] = len(valid)
                row["fingpt_inference_seconds"] = sum(c["fingpt_inference_seconds"] for c in doc_chunks)
                row["fingpt_raw_output"] = "\n".join(c["fingpt_raw_output"] for c in doc_chunks)
                errors = [f"chunk {c['chunk_index']}: {c['fingpt_error']}" for c in doc_chunks if c["fingpt_error"]]
                if errors:
                    row["fingpt_error"] = "; ".join(errors)
                else:
                    # Equal chunk mean, then equal document weighting downstream.
                    score = sum(c["sentiment_score"] for c in valid) / len(valid)
                    row["sentiment_score"] = score
                    row["sentiment_label"] = "positive" if score > 0 else "negative" if score < 0 else "neutral"
                save_json(cache, {"result": row, "chunks": doc_chunks})
            except Exception as exc:
                traceback.print_exc()
                row["fingpt_error"] = f"{type(exc).__name__}: {exc}"
        results.append(row)
        all_chunks.extend(doc_chunks)
        # Per-document/chunk JSON caches preserve resume state without quadratic CSV I/O.
        if len(results) % 100 == 0:
            write_csv(pd.DataFrame(results), output / "sentiment_documents.csv")
    columns = list(documents.columns) + ["model_profile", "model_name", "sentiment_label", "sentiment_score",
                                        "fingpt_raw_output", "fingpt_error", "fingpt_inference_seconds",
                                        "chunk_count", "successful_chunks"]
    frame = pd.DataFrame(results, columns=columns)
    write_csv(frame, output / "sentiment_documents.csv")
    write_csv(pd.DataFrame(all_chunks, columns=["document_id", "chunk_index", "text", "sentiment_label",
                                               "sentiment_score", "model_name", "fingpt_raw_output",
                                               "fingpt_error", "fingpt_inference_seconds"]),
              output / "sentiment_chunks.csv")
    errors = int(frame.fingpt_error.ne("").sum())
    write_html_report({"run_id": output.name, "started_at": started_at,
                       "finished_at": datetime.now().astimezone().isoformat(),
                       "model_profile": model_profile, "model_name": model_name,
                       "quantization": quantization, "status": "failed" if errors or frame.empty else "completed",
                       "fingpt_error": f"{errors} documents failed or excluded." if errors else ""},
                      frame, output / "sentiment_report.html")
    return frame


def build_features(sentiments, trends, start, end, frequency="weekly"):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    cutoffs = pd.date_range(start, end, freq="W-SUN" if frequency == "weekly" else "D")
    if cutoffs.empty:
        raise ValueError("No complete feature periods in the selected range.")
    data = sentiments.copy()
    if data.empty or not (data.fingpt_error.eq("") & data.sentiment_label.isin(LABEL_SCORES)).any():
        raise ValueError("No successful sentiment documents available for features.")
    data["available_date"] = pd.to_datetime(data["available_date"], errors="raise")
    data = data.loc[data.available_date.between(start, end)]
    if frequency == "weekly":
        data["date"] = data.available_date + pd.to_timedelta(6 - data.available_date.dt.weekday, unit="D")
    else:
        data["date"] = data.available_date
    features = pd.DataFrame({"date": cutoffs})
    feature_columns = []
    for source in SOURCES:
        source_data = data.loc[data.source_type.eq(source)]
        for cutoff in cutoffs:
            rows = source_data.loc[source_data.date.eq(cutoff)]
            valid = rows.loc[rows.fingpt_error.eq("") & rows.sentiment_label.isin(LABEL_SCORES)]
            scores = pd.to_numeric(valid.sentiment_score, errors="raise")
            values = {"document_count": len(valid), "error_count": len(rows)-len(valid),
                      "missing": int(valid.empty), "sentiment_mean": float(scores.mean()) if len(valid) else 0.0}
            for label in LABEL_SCORES:
                count = int(valid.sentiment_label.eq(label).sum())
                values[label + "_count"] = count
                values[label + "_ratio"] = count / len(valid) if len(valid) else 0.0
            for key, value in values.items():
                name = source + "_" + key
                features.loc[features.date.eq(cutoff), name] = value
                if name not in feature_columns:
                    feature_columns.append(name)
    # Monthly trends become available only after their month ends.
    # Historical normalized Trends are not point-in-time snapshots; document this limitation.
    if not trends.empty:
        trends = trends.copy()
        trends["date"] = pd.to_datetime(trends.date, errors="raise")
        if not trends.date.dt.day.eq(1).all():
            raise ValueError("Trends input must be monthly observations dated on month start.")
        if trends.duplicated(["date", "trend_keyword"]).any():
            raise ValueError("Conflicting monthly Trends rows.")
        trends = trends.loc[~trends.is_partial.astype(str).str.lower().isin(["true", "1"])]
        trends["available_date"] = trends.date + pd.offsets.MonthBegin(1)
        trends["trend_value"] = pd.to_numeric(trends.trend_value, errors="raise")
        for keyword, values in trends.groupby("trend_keyword"):
            name = "trend_" + digest(str(keyword))[:8]
            aligned = pd.merge_asof(features[["date"]], values.sort_values("available_date"),
                                    left_on="date", right_on="available_date", direction="backward")
            features[name] = aligned.trend_value.fillna(0).to_numpy()
            features[name + "_missing"] = aligned.trend_value.isna().astype(int).to_numpy()
            feature_columns.extend([name, name + "_missing"])
    return features, feature_columns


def build_dataset(features, prices, horizon=5):
    """Use first close strictly after feature cutoff, then horizon trading observations."""
    if horizon < 1:
        raise ValueError("horizon must be positive.")
    prices = prices.copy()
    if not {"date", "close"}.issubset(prices.columns):
        raise ValueError("Prices CSV requires date,close (split-adjusted close).")
    prices["date"] = pd.to_datetime(prices.date, errors="raise")
    prices["close"] = pd.to_numeric(prices.close, errors="raise")
    prices = prices.sort_values("date").reset_index(drop=True)
    if prices.date.duplicated().any() or not np.isfinite(prices.close).all() or prices.close.le(0).any():
        raise ValueError("Prices must have unique dates and finite positive closes.")
    dataset = features.copy()
    dataset["date"] = pd.to_datetime(dataset.date)
    entries = prices.date.searchsorted(dataset.date, side="right")
    for column in ["target_start", "target_end"]:
        dataset[column] = pd.Series(pd.NaT, index=dataset.index, dtype="datetime64[ns]")
    for column in ["target_start_close", "target_end_close", "future_return", "label"]:
        dataset[column] = np.nan
    for index, entry in zip(dataset.index, entries):
        exit_index = entry + horizon
        if exit_index >= len(prices):
            continue
        # Refuse to map missing historical prices to an entry months later.
        if (prices.date.iloc[entry] - dataset.loc[index, "date"]).days > 7:
            continue
        start_price, end_price = prices.close.iloc[entry], prices.close.iloc[exit_index]
        dataset.loc[index, ["target_start", "target_end"]] = [prices.date.iloc[entry], prices.date.iloc[exit_index]]
        dataset.loc[index, ["target_start_close", "target_end_close", "future_return", "label"]] = [
            start_price, end_price, end_price/start_price-1, int(end_price > start_price)]
    return dataset


def download_prices(ticker, start, end, destination):
    import yfinance as yf
    frame = yf.download(ticker, start=start, end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                        auto_adjust=True, progress=False, multi_level_index=False)
    if frame is None or frame.empty:
        raise ValueError("Price download returned no rows.")
    output = frame.reset_index().rename(columns={"Date": "date", "Close": "close"})
    output["date"] = pd.to_datetime(output.date).dt.tz_localize(None).dt.strftime("%Y-%m-%d")
    write_csv(output[["date", "close"]], destination)
    save_json(Path(destination).with_suffix(".metadata.json"),
              {"ticker": ticker, "provider": "yfinance", "auto_adjust": True,
               "start": start, "end": end})
