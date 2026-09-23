"""Offline pipeline tests. FinGPT and all external data retrieval are mocked."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from pipeline_documents import BodyParser, chunk_text, prepare_documents, read_csv
from pipeline_sentiment import build_dataset, build_features, infer_documents
from pipeline_transformer import temporal_samples
import run_pipeline


class Tokenizer:
    def encode(self, text, add_special_tokens=True):
        return list(range(len(text.encode("utf-8")) + int(add_special_tokens)))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        torch.set_num_threads(1)

    def make_raw(self, count=90):
        dates = pd.date_range("2020-01-06", periods=count, freq="W-MON")
        for source, suffix in [("news", "news"), ("earnings_call", "earnings_call"), ("ten_k", "10k")]:
            folder = self.root / "data/raw" / source
            folder.mkdir(parents=True)
            selected = dates if source == "news" else dates[:2]
            pd.DataFrame([dict(date=d.strftime("%Y-%m-%d"), title="Public document", url=f"https://example.test/{source}/{i}",
                               content=("positive earnings " if i % 2 else "negative earnings ") * 25)
                          for i, d in enumerate(selected)]).to_csv(folder / f"NVDA_{suffix}_raw.csv", index=False)
        folder = self.root / "data/raw/google_trends"
        folder.mkdir(parents=True)
        pd.DataFrame([dict(date=d.strftime("%Y-%m-%d"), trend_keyword="NVDA", trend_value=i+1, is_partial=False)
                      for i, d in enumerate(pd.date_range("2020-01-01", periods=24, freq="MS"))]).to_csv(
                          folder / "NVDA_google_trends_raw.csv", index=False)

    def prediction(self, tokenizer, model, texts, **kwargs):
        label = "positive" if "positive" in texts[0] else "negative"
        return [dict(sentiment_label=label, sentiment_score=1 if label == "positive" else -1,
                     fingpt_raw_output=label, fingpt_error="", model_name="mock")], 0.01

    def test_parser_removes_hidden_content(self):
        parser = BodyParser()
        parser.feed("<html><head>bad</head><nav>menu</nav><article><p>" + "Valid prose " * 30 +
                    "</p><script>bad code</script><ix:hidden>us-gaap:secret</ix:hidden></article></html>")
        self.assertIn("Valid prose", parser.text())
        for word in ["bad", "menu", "secret"]:
            self.assertNotIn(word, parser.text())

    def test_chunking_preserves_body_and_exact_token_budget(self):
        text = "word " * 2000 + "ending"
        chunks = chunk_text(text, Tokenizer())
        self.assertEqual(" ".join(chunks), text)
        from fingpt_sentiment import build_fingpt_prompt
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 1800)
            self.assertLessEqual(len(Tokenizer().encode(build_fingpt_prompt(chunk))), 512)
        self.assertTrue(chunks[-1].endswith("ending"))

    def test_preparation_excludes_incomplete_and_unverified(self):
        self.make_raw(2)
        p = self.root / "data/raw/news/NVDA_news_raw.csv"
        frame = read_csv(p).drop(columns="content")
        frame.to_csv(p, index=False)
        ec = self.root / "data/raw/earnings_call/NVDA_earnings_call_raw.csv"
        frame = read_csv(ec)
        frame["title"] = "CFO Commentary"
        frame["content"] = "a"*20000
        frame.to_csv(ec, index=False)
        output = self.root / "prepared"
        docs = prepare_documents(self.root, "NVDA", output, "2020-01-01", "2021-12-31")
        self.assertTrue(docs.loc[docs.source_type.eq("news"), "preparation_error"].ne("").all())
        self.assertTrue(docs.loc[docs.source_type.eq("earnings_call"), "preparation_error"].str.contains("date needs review").all())
        self.assertTrue((output / "preparation_errors.csv").exists())

    def test_failed_and_unparsed_inference_are_not_neutral(self):
        self.make_raw(2)
        output = self.root / "infer"
        docs = prepare_documents(self.root, "NVDA", output, "2020-01-01", "2021-12-31")
        with patch("fingpt_sentiment.load_fingpt_model", return_value=(Tokenizer(), object(), None)), \
             patch("fingpt_sentiment.run_fingpt_batch", return_value=(
                 [dict(sentiment_label="unparsed", fingpt_raw_output="unknown", fingpt_error="")], .1)):
            results = infer_documents(docs, output, "sentiment-llama2-13b", "4bit")
        self.assertTrue(results.fingpt_error.ne("").all())
        self.assertTrue(results.sentiment_score.isna().all())
        with patch("fingpt_sentiment.load_fingpt_model", side_effect=AssertionError("cached results should not load")):
            again = infer_documents(docs, output, "sentiment-llama2-13b", "4bit")
        self.assertEqual(again.fingpt_error.tolist(), results.fingpt_error.tolist())

    def test_run_fingpt_batch_handles_empty_generation(self):
        class EmptyTokenizer:
            eos_token_id = 2
            pad_token_id = 2

            def __call__(self, prompts, return_tensors=None, padding=None, truncation=None, max_length=None):
                return {"input_ids": torch.tensor([[1, 2, 3]])}

            def batch_decode(self, pieces, skip_special_tokens=None):
                return [""]

        class EmptyModel:
            device = "cpu"

            def generate(self, **kwargs):
                return torch.tensor([[1, 2, 3]])

        rows, seconds = __import__("fingpt_sentiment").run_fingpt_batch(EmptyTokenizer(), EmptyModel(), ["example text"], model_name="mock")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sentiment_label"], "")
        self.assertIsNone(rows[0]["sentiment_score"])
        self.assertIn("empty response", rows[0]["fingpt_error"])

    def test_unparsed_chat_generation_preserves_missing_score(self):
        class ChatModel:
            def chat(self, tokenizer, prompt, history=None):
                return "unrecognizable output", None

        rows, _ = __import__("fingpt_sentiment").run_fingpt_batch(
            object(), ChatModel(), ["example"], model_name="mock")
        self.assertEqual(rows[0]["sentiment_label"], "unparsed")
        self.assertIsNone(rows[0]["sentiment_score"])
        self.assertTrue(rows[0]["fingpt_error"])

    def test_model_load_failure_is_recorded_for_all_documents(self):
        self.make_raw(2)
        output = self.root / "failed_load"
        docs = prepare_documents(self.root, "NVDA", output, "2020-01-01", "2021-12-31")
        with patch("fingpt_sentiment.load_fingpt_model", side_effect=RuntimeError("mock load failure")) as loader, \
             contextlib.redirect_stderr(io.StringIO()):
            result = infer_documents(docs, output, "sentiment-llama2-13b", "4bit")
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(len(result), len(docs))
        self.assertTrue(result.fingpt_error.str.contains("mock load failure").all())
        self.assertTrue((output / "sentiment_report.html").exists())

    def test_cached_success_does_not_reload_model(self):
        self.make_raw(2)
        output = self.root / "cached"
        docs = prepare_documents(self.root, "NVDA", output, "2020-01-01", "2021-12-31")
        with patch("fingpt_sentiment.load_fingpt_model", return_value=(Tokenizer(), object(), None)), \
             patch("fingpt_sentiment.run_fingpt_batch", side_effect=self.prediction):
            first = infer_documents(docs, output, "sentiment-llama2-13b", "4bit")
        with patch("fingpt_sentiment.load_fingpt_model", side_effect=AssertionError("must use cache")) as loader:
            second = infer_documents(docs, output, "sentiment-llama2-13b", "4bit")
        loader.assert_not_called()
        self.assertEqual(first.sentiment_score.tolist(), second.sentiment_score.tolist())

    def test_preparation_errors_stop_cli_before_inference(self):
        self.make_raw(2)
        path = self.root / "data/raw/news/NVDA_news_raw.csv"
        read_csv(path).drop(columns="content").to_csv(path, index=False)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = run_pipeline.main(["prepare", "--root", str(self.root)])
        self.assertEqual(code, 1)
        run = next((self.root / "data/pipeline_runs").iterdir())
        with patch("fingpt_sentiment.load_fingpt_model") as loader, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = run_pipeline.main(["sentiment", "--root", str(self.root), "--run-dir", str(run)])
        self.assertEqual(code, 1)
        loader.assert_not_called()

    def test_trends_lag_and_error_exclusion(self):
        documents = pd.DataFrame([
            dict(available_date="2020-01-02", source_type="news", sentiment_label="positive", sentiment_score=1, fingpt_error=""),
            dict(available_date="2020-01-03", source_type="news", sentiment_label="neutral", sentiment_score=0, fingpt_error="failed")])
        trends = pd.DataFrame([dict(date="2020-01-01", trend_keyword="NVDA", trend_value=70, is_partial=False)])
        features, columns = build_features(documents, trends, "2020-01-01", "2020-02-09")
        name = next(c for c in columns if c.startswith("trend_") and not c.endswith("_missing"))
        self.assertTrue(features.loc[features.date.lt("2020-02-01"), name].eq(0).all())
        self.assertEqual(features.loc[features.date.eq("2020-02-02"), name].iloc[0], 70)
        first = features.iloc[0]
        self.assertEqual(first.news_document_count, 1)
        self.assertEqual(first.news_error_count, 1)
        self.assertEqual(first.news_neutral_count, 0)

    def test_targets_and_scaling_do_not_leak(self):
        dates = pd.date_range("2020-01-01", periods=90)
        features = pd.DataFrame({"date": dates, "signal": np.arange(90, dtype=float)})
        prices = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=100), "close": 100+np.arange(100)})
        dataset = build_dataset(features, prices, 5)
        labeled = dataset.loc[dataset.label.notna()]
        self.assertTrue((labeled.target_start > labeled.date).all())
        data, groups, mean, std = temporal_samples(dataset, ["signal"], 4)
        train, validation, test = (groups[k][0] for k in ["train", "validation", "test"])
        self.assertLess(data.target_end.iloc[train].max(), data.date.iloc[validation].min())
        self.assertLess(data.target_end.iloc[validation].max(), data.date.iloc[test].min())
        changed = dataset.copy()
        changed.loc[test, "signal"] = 1e9
        _, _, mean2, std2 = temporal_samples(changed, ["signal"], 4)
        np.testing.assert_array_equal(mean, mean2)
        np.testing.assert_array_equal(std, std2)
        with self.assertRaises(ValueError):
            temporal_samples(dataset, ["future_return"], 4)

    def test_end_to_end_cli_with_mock_fingpt(self):
        # Exercise CLI orchestration without training any Transformer.
        def fake_train(dataset, columns, output, **kwargs):
            self.assertNotIn("future_return", columns)
            torch.save({"feature_columns": columns}, output / "transformer.pt")
            for name in ("transformer_predictions.csv", "training_history.csv"):
                (output / name).write_text("mock\n")
            (output / "metrics.json").write_text(json.dumps({"test": {"majority_baseline_accuracy": 0.5}}))
            (output / "training_config.json").write_text("{}")

        self.make_raw()
        prices = self.root / "fixture_prices.csv"
        dates = pd.bdate_range("2020-01-01", "2022-02-01")
        pd.DataFrame({"date": dates, "close": 100 + np.sin(np.arange(len(dates))/8)*5}).to_csv(prices, index=False)
        with patch("fingpt_sentiment.load_fingpt_model", return_value=(Tokenizer(), object(), None)), \
             patch("fingpt_sentiment.run_fingpt_batch", side_effect=self.prediction), \
             patch("pipeline_transformer.train_transformer", side_effect=fake_train) as trainer, \
             contextlib.redirect_stdout(io.StringIO()):
            code = run_pipeline.main(["all", "--root", str(self.root), "--start", "2020-01-01",
                                      "--end", "2021-12-31", "--prices", str(prices),
                                      "--epochs", "1", "--lookback", "4"])
        self.assertEqual(code, 0)
        trainer.assert_called_once()
        run = next((self.root / "data/pipeline_runs").iterdir())
        manifest = json.loads((run / "manifest.json").read_text())
        self.assertTrue(all(record["status"] == "completed" for record in manifest["stages"].values()))
        for filename in ["documents.csv", "sentiment_chunks.csv", "sentiment_documents.csv",
                         "features.csv", "dataset.csv", "transformer.pt", "metrics.json"]:
            self.assertTrue((run / filename).exists(), filename)
        metrics = json.loads((run / "metrics.json").read_text())
        self.assertIn("majority_baseline_accuracy", metrics["test"])
        checkpoint = torch.load(run / "transformer.pt", map_location="cpu", weights_only=True)
        self.assertNotIn("future_return", checkpoint["feature_columns"])
        # Editing an artifact invalidates downstream use.
        with (run / "dataset.csv").open("a") as handle:
            handle.write("\n")
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            code = run_pipeline.main(["train", "--root", str(self.root), "--run-dir", str(run), "--epochs", "1"])
        self.assertEqual(code, 1)
        manifest = json.loads((run / "manifest.json").read_text())
        self.assertEqual(manifest["stages"]["train"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
