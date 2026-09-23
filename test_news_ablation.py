import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from news_ablation import (
    ExperimentSpec,
    NewsInferenceCache,
    align_news_to_trading_date,
    build_cache_key,
    build_daily_dissemination_features,
    build_daily_news_features,
    build_dissemination_features,
    build_stage2_rows,
    choose_cluster_representative,
    deduplicate_news,
    make_event_cluster_manifest,
    write_offline_validation_report,
)


class NewsAblationTests(unittest.TestCase):
    def test_cache_key_includes_model_and_text_hash(self):
        key = build_cache_key(
            "NVIDIA hits record revenue",
            base_model="NousResearch/Llama-2-13b-hf",
            adapter_name="FinGPT/fingpt-sentiment_llama2-13b_lora",
            model_revision="abc123",
            quantization="4bit",
            prompt_version="v1",
            generation_settings={"do_sample": False, "max_new_tokens": 8},
        )
        self.assertIsInstance(key, str)
        self.assertEqual(len(key), 64)
        self.assertEqual(key, hashlib.sha256(
            json.dumps({
                "input_sha256": hashlib.sha256("NVIDIA hits record revenue".encode("utf-8")).hexdigest(),
                "base_model": "NousResearch/Llama-2-13b-hf",
                "adapter_name": "FinGPT/fingpt-sentiment_llama2-13b_lora",
                "model_revision": "abc123",
                "quantization": "4bit",
                "prompt_version": "v1",
                "generation_settings": {"do_sample": False, "max_new_tokens": 8},
            }, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest())

    def test_inference_cache_resume_and_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            cache = NewsInferenceCache(cache_dir)
            key = build_cache_key(
                "NVIDIA gains",
                base_model="NousResearch/Llama-2-13b-hf",
                adapter_name="FinGPT/fingpt-sentiment_llama2-13b_lora",
                model_revision="rev1",
                quantization="4bit",
                prompt_version="v1",
                generation_settings={"do_sample": False},
            )
            record = {
                "original_row_id": "r1",
                "published_at": "2024-01-02T12:00:00Z",
                "effective_trading_date": "2024-01-02",
                "title": "NVIDIA gains",
                "normalized_title": "nvidia gains",
                "text_sha256": hashlib.sha256("NVIDIA gains".encode("utf-8")).hexdigest(),
                "sentiment_label": "positive",
                "sentiment_score": 1,
                "fingpt_raw_output": "positive",
                "model_name": "mock-model",
                "base_model": "NousResearch/Llama-2-13b-hf",
                "adapter_name": "FinGPT/fingpt-sentiment_llama2-13b_lora",
                "prompt_version": "v1",
                "inference_seconds": 1.2,
                "fingpt_error": "",
                "processed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            cache.save(key, record)
            self.assertEqual(cache.load(key)["sentiment_label"], "positive")
            self.assertTrue(cache.exists(key))

            error_key = build_cache_key(
                "NVIDIA misses",
                base_model="NousResearch/Llama-2-13b-hf",
                adapter_name="FinGPT/fingpt-sentiment_llama2-13b_lora",
                model_revision="rev1",
                quantization="4bit",
                prompt_version="v1",
                generation_settings={"do_sample": False},
            )
            cache.save_error(error_key, {"original_row_id": "r2", "fingpt_error": "timeout"})
            self.assertIn("timeout", cache.load(error_key)["fingpt_error"])

    def test_dedup_and_cluster_manifest(self):
        rows = pd.DataFrame([
            {"original_row_id": "1", "published_at": "2024-01-02T09:00:00Z", "title": "NVIDIA gains outlook", "url": "u1", "domain": "example.com", "source": "news"},
            {"original_row_id": "2", "published_at": "2024-01-02T09:15:00Z", "title": "NVIDIA gains outlook", "url": "u2", "domain": "example.com", "source": "news"},
            {"original_row_id": "3", "published_at": "2024-01-04T10:00:00Z", "title": "NVIDIA reports record revenue", "url": "u3", "domain": "newswire.com", "source": "news"},
        ])
        dedup = deduplicate_news(rows)
        self.assertEqual(len(dedup["raw"]), 3)
        self.assertEqual(len(dedup["canonical"]), 2)
        self.assertTrue(dedup["raw"].loc[dedup["raw"].original_row_id.eq("2"), "exact_duplicate"].iloc[0])

        cluster = make_event_cluster_manifest(dedup["canonical"], similarity_threshold=0.9, time_window_hours=72)
        self.assertIn("cluster_size_final", cluster.columns)
        self.assertIn("cluster_representative_row_id", cluster.columns)
        self.assertTrue(cluster["cluster_size_final"].gt(0).all())

    def test_same_url_far_apart_different_titles_is_retained(self):
        rows = pd.DataFrame([
            {"original_row_id": "u1-a", "published_at": "2020-01-01T09:00:00Z", "title": "NVIDIA stock gains after AI demand", "url": "https://example.com/nvda", "domain": "example.com", "source": "news"},
            {"original_row_id": "u1-b", "published_at": "2020-01-31T09:00:00Z", "title": "NVIDIA stock drops after earnings miss", "url": "https://example.com/nvda", "domain": "example.com", "source": "news"},
        ])
        dedup = deduplicate_news(rows)
        self.assertEqual(len(dedup["canonical"]), 2)
        self.assertFalse(dedup["raw"].loc[dedup["raw"].original_row_id.eq("u1-b"), "exact_duplicate"].iloc[0])

    def test_same_title_on_same_day_is_deduped_even_with_different_url(self):
        rows = pd.DataFrame([
            {"original_row_id": "n1", "published_at": "2024-01-02T09:00:00Z", "title": "Could NVIDIA Be a Millionaire Maker Stock?", "url": "https://a.example/1", "domain": "example.com", "source": "news"},
            {"original_row_id": "n2", "published_at": "2024-01-02T09:15:00Z", "title": "Could NVIDIA Be a Millionaire Maker Stock?", "url": "https://b.example/2", "domain": "example.com", "source": "news"},
        ])
        dedup = deduplicate_news(rows)
        self.assertEqual(len(dedup["canonical"]), 1)
        self.assertTrue(dedup["raw"].loc[dedup["raw"].original_row_id.eq("n2"), "exact_duplicate"].iloc[0])

    def test_cluster_manifest_is_required_for_c_and_d(self):
        rows = pd.DataFrame([
            {"original_row_id": "1", "published_at": "2024-01-02T09:00:00Z", "title": "NVIDIA gains outlook", "url": "u1", "domain": "example.com", "source": "news"},
            {"original_row_id": "2", "published_at": "2024-01-02T09:15:00Z", "title": "NVIDIA gains outlook", "url": "u2", "domain": "example.com", "source": "news"},
        ])
        with self.assertRaisesRegex(ValueError, "cluster manifest"):
            from news_ablation import run_experiment_matrix
            run_experiment_matrix(rows, cluster_manifest=None)

    def test_unique_domain_count_uses_as_of_domain_set(self):
        rows = pd.DataFrame([
            {"original_row_id": "a", "published_at": "2024-01-02T09:00:00Z", "domain": "a.com", "event_cluster_id": "cluster-1", "cluster_size_asof_t": 1},
            {"original_row_id": "b", "published_at": "2024-01-02T10:00:00Z", "domain": "b.com", "event_cluster_id": "cluster-1", "cluster_size_asof_t": 2},
            {"original_row_id": "c", "published_at": "2024-01-02T11:00:00Z", "domain": "a.com", "event_cluster_id": "cluster-1", "cluster_size_asof_t": 3},
        ])
        metrics = build_dissemination_features(rows, as_of_date="2024-01-02")
        self.assertEqual(metrics["unique_domain_count_asof_t"].iloc[0], 1)
        self.assertEqual(metrics["unique_domain_count_asof_t"].iloc[1], 2)
        self.assertEqual(metrics["unique_domain_count_asof_t"].iloc[2], 2)

    def test_as_of_cutoff_uses_exact_timestamp_ordering(self):
        rows = pd.DataFrame([
            {"original_row_id": "a", "published_at": "2024-01-02T10:00:00Z", "domain": "a.com", "event_cluster_id": "cluster-1"},
            {"original_row_id": "b", "published_at": "2024-01-02T15:00:00Z", "domain": "b.com", "event_cluster_id": "cluster-1"},
        ])
        features = build_dissemination_features(rows, as_of_date="2024-01-02")
        self.assertEqual(features["cluster_size_asof_t"].iloc[0], 1)
        self.assertEqual(features["cluster_size_asof_t"].iloc[1], 2)
        self.assertEqual(features["unique_domain_count_asof_t"].iloc[0], 1)
        self.assertEqual(features["unique_domain_count_asof_t"].iloc[1], 2)
        self.assertEqual(features["dissemination_span_hours_asof_t"].iloc[1], 5.0)

    def test_choose_cluster_representative_is_deterministic(self):
        rows = pd.DataFrame([
            {"original_row_id": "a", "published_at": "2024-01-02T06:00:00Z", "title": "NVIDIA exceeds forecasts", "domain": "reuters.com", "url": "u1"},
            {"original_row_id": "b", "published_at": "2024-01-02T07:00:00Z", "title": "NVIDIA tops estimates", "domain": "bloomberg.com", "url": "u2"},
            {"original_row_id": "c", "published_at": "2024-01-02T08:00:00Z", "title": "NVIDIA shares climb", "domain": "marketwatch.com", "url": "u3"},
        ])
        rep = choose_cluster_representative(rows)
        self.assertEqual(rep["original_row_id"], "a")

    def test_choose_cluster_representative_handles_missing_domain(self):
        rows = pd.DataFrame([
            {"original_row_id": "a", "published_at": "2024-01-02T06:00:00Z", "title": "NVIDIA exceeds forecasts"},
            {"original_row_id": "b", "published_at": "2024-01-02T07:00:00Z", "title": "NVIDIA tops estimates"},
        ])
        rep = choose_cluster_representative(rows)
        self.assertIn(rep["original_row_id"], {"a", "b"})

    def test_time_alignment_uses_market_time_and_no_future_leakage(self):
        df = pd.DataFrame([
            {"published_at": "2024-01-04T15:30:00Z", "title": "NVIDIA beats"},
            {"published_at": "2024-01-04T20:30:00Z", "title": "NVIDIA after hours"},
            {"published_at": "2024-01-06T09:00:00Z", "title": "Weekend article"},
        ])
        aligned = align_news_to_trading_date(df)
        self.assertEqual(aligned["effective_trading_date"].tolist(), ["2024-01-04", "2024-01-04", "2024-01-08"])

    def test_authoritative_calendar_market_rules(self):
        prices = pd.DataFrame({
            "date": [
                "2024-03-28", "2024-04-01", "2024-10-14", "2024-10-15",
                "2024-11-11", "2024-11-12", "2025-01-06", "2025-01-07",
            ]
        })
        calendar = pd.DatetimeIndex(pd.to_datetime(prices["date"]).dt.normalize())
        rows = pd.DataFrame([
            {"published_at": "2024-03-29T05:00:00-04:00", "title": "Good Friday article"},
            {"published_at": "2024-10-14T15:59:59-04:00", "title": "Columbus Day before close"},
            {"published_at": "2024-10-14T16:00:00-04:00", "title": "Columbus Day at close"},
            {"published_at": "2024-11-11T15:59:59-05:00", "title": "Veterans Day evening"},
            {"published_at": "2024-11-11T16:00:00-05:00", "title": "Veterans Day at close"},
            {"published_at": "2025-01-04T18:00:00-05:00", "title": "Weekend after hours"},
        ])
        aligned = align_news_to_trading_date(rows, price_calendar=calendar)
        self.assertEqual(aligned["effective_trading_date"].tolist(), [
            "2024-04-01", "2024-10-14", "2024-10-15", "2024-11-11", "2024-11-12", "2025-01-06",
        ])
        self.assertNotIn("2024-03-29", aligned["effective_trading_date"].tolist())
        self.assertIn("2024-10-14", aligned["effective_trading_date"].tolist())
        self.assertIn("2024-11-11", aligned["effective_trading_date"].tolist())

    def test_authoritative_calendar_requires_price_dates_and_rejects_missing_range(self):
        calendar = pd.Index(pd.to_datetime(["2024-03-28", "2024-04-01"]).normalize())
        rows = pd.DataFrame([
            {"published_at": "2024-04-02T09:00:00-04:00", "title": "After last trading day"},
        ])
        with self.assertRaisesRegex(ValueError, "outside the price calendar|missing"):
            align_news_to_trading_date(rows, price_calendar=calendar)

    def test_price_calendar_dates_are_unique_and_sorted(self):
        calendar = pd.DatetimeIndex(pd.to_datetime([
            "2020-01-09", "2020-01-10", "2020-01-13",
            "2020-01-14", "2020-01-14",
        ]).normalize())
        with self.assertRaisesRegex(ValueError, "unique|sorted"):
            align_news_to_trading_date(pd.DataFrame([{"published_at": "2020-01-10T09:00:00Z", "title": "x"}]), price_calendar=calendar)

    def test_effective_trading_dates_always_belong_to_price_calendar(self):
        calendar = pd.DatetimeIndex(pd.to_datetime([
            "2024-10-14", "2024-10-15", "2024-11-11", "2024-11-12",
        ]).normalize())
        rows = pd.DataFrame([
            {"published_at": "2024-10-14T15:59:59-04:00", "title": "same day"},
            {"published_at": "2024-10-14T16:00:00-04:00", "title": "next day"},
            {"published_at": "2024-11-11T15:59:59-05:00", "title": "veterans"},
        ])
        aligned = align_news_to_trading_date(rows, price_calendar=calendar)
        self.assertTrue(set(aligned["effective_trading_date"]).issubset(set(calendar.strftime("%Y-%m-%d"))))

    def test_stage2_uses_manifest_retained_rows_for_b(self):
        raw = pd.DataFrame([
            {"original_row_id": "a1", "published_at": "2024-01-02T09:00:00Z", "title": "NVIDIA gains", "url": "u1", "domain": "a.com", "source": "news"},
            {"original_row_id": "a2", "published_at": "2024-01-02T09:15:00Z", "title": "NVIDIA gains", "url": "u2", "domain": "b.com", "source": "news"},
            {"original_row_id": "a3", "published_at": "2024-01-02T09:30:00Z", "title": "NVIDIA misses", "url": "u3", "domain": "c.com", "source": "news"},
        ])
        manifest = pd.DataFrame([
            {"original_row_id": "a1", "published_at": "2024-01-02T09:00:00Z", "title": "NVIDIA gains", "url": "u1", "domain": "a.com", "normalized_title": "nvidia gains", "preliminary_cluster_id": "cluster_1", "exact_duplicate": False, "duplicate_of_row_id": ""},
            {"original_row_id": "a2", "published_at": "2024-01-02T09:15:00Z", "title": "NVIDIA gains", "url": "u2", "domain": "b.com", "normalized_title": "nvidia gains", "preliminary_cluster_id": "cluster_1", "exact_duplicate": True, "duplicate_of_row_id": "a1"},
            {"original_row_id": "a3", "published_at": "2024-01-02T09:30:00Z", "title": "NVIDIA misses", "url": "u3", "domain": "c.com", "normalized_title": "nvidia misses", "preliminary_cluster_id": "cluster_2", "exact_duplicate": False, "duplicate_of_row_id": ""},
        ])
        stage2 = build_stage2_rows(raw, manifest, price_calendar=pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03"]).normalize()))
        self.assertEqual(stage2["B_count"], 2)
        self.assertTrue(stage2["B"]["original_row_id"].isin(["a2"]).sum() == 0)
        self.assertTrue((stage2["B"]["retained_in_B"] == True).all())
        self.assertEqual(stage2["D"]["cluster_id"].nunique(), 2)
        self.assertEqual((stage2["D"]["cluster_id"].map(str) == "cluster_1").sum(), 1)

    def test_same_title_same_day_cross_domain_keeps_one_b_but_increases_d_domain_count(self):
        raw = pd.DataFrame([
            {"original_row_id": "m1", "published_at": "2020-01-13T09:00:00Z", "title": "Could NVIDIA Be a Millionaire Maker Stock?", "url": "https://a.example/1", "domain": "a.example", "source": "news"},
            {"original_row_id": "m2", "published_at": "2020-01-13T09:05:00Z", "title": "Could NVIDIA Be a Millionaire Maker Stock?", "url": "https://b.example/2", "domain": "b.example", "source": "news"},
            {"original_row_id": "m3", "published_at": "2020-01-13T09:10:00Z", "title": "Could NVIDIA Be a Millionaire Maker Stock?", "url": "https://a.example/1", "domain": "a.example", "source": "news"},
        ])
        manifest = pd.DataFrame([
            {"original_row_id": "m1", "published_at": "2020-01-13T09:00:00Z", "title": "Could NVIDIA Be a Millionaire Maker Stock?", "url": "https://a.example/1", "domain": "a.example", "normalized_title": "could nvidia be a millionaire maker stock?", "preliminary_cluster_id": "cluster_millionaire", "exact_duplicate": False, "duplicate_of_row_id": ""},
            {"original_row_id": "m2", "published_at": "2020-01-13T09:05:00Z", "title": "Could NVIDIA Be a Millionaire Maker Stock?", "url": "https://b.example/2", "domain": "b.example", "normalized_title": "could nvidia be a millionaire maker stock?", "preliminary_cluster_id": "cluster_millionaire", "exact_duplicate": True, "duplicate_of_row_id": "m1"},
            {"original_row_id": "m3", "published_at": "2020-01-13T09:10:00Z", "title": "Could NVIDIA Be a Millionaire Maker Stock?", "url": "https://a.example/1", "domain": "a.example", "normalized_title": "could nvidia be a millionaire maker stock?", "preliminary_cluster_id": "cluster_millionaire", "exact_duplicate": True, "duplicate_of_row_id": "m1"},
        ])
        calendar = pd.DatetimeIndex(pd.to_datetime(["2020-01-13"]).normalize())
        stage2 = build_stage2_rows(raw, manifest, price_calendar=calendar)
        self.assertEqual(stage2["B_count"], 1)
        self.assertEqual(stage2["D"]["cluster_id"].nunique(), 1)
        self.assertEqual(stage2["D"].loc[stage2["D"]["cluster_id"].eq("cluster_millionaire"), "cluster_size_asof_t"].iloc[0], 3)
        self.assertEqual(stage2["D"].loc[stage2["D"]["cluster_id"].eq("cluster_millionaire"), "unique_domain_count_asof_t"].iloc[0], 2)

    def test_d_snapshot_keys_are_unique_and_every_cluster_has_snapshot(self):
        raw = pd.DataFrame([
            {"original_row_id": "c1", "published_at": "2024-01-02T09:00:00Z", "title": "NVIDIA gains", "url": "u1", "domain": "a.com", "source": "news"},
            {"original_row_id": "c2", "published_at": "2024-01-03T09:00:00Z", "title": "NVIDIA surprises", "url": "u2", "domain": "b.com", "source": "news"},
        ])
        manifest = pd.DataFrame([
            {"original_row_id": "c1", "published_at": "2024-01-02T09:00:00Z", "title": "NVIDIA gains", "url": "u1", "domain": "a.com", "normalized_title": "nvidia gains", "preliminary_cluster_id": "cluster_alpha", "exact_duplicate": False, "duplicate_of_row_id": ""},
            {"original_row_id": "c2", "published_at": "2024-01-03T09:00:00Z", "title": "NVIDIA surprises", "url": "u2", "domain": "b.com", "normalized_title": "nvidia surprises", "preliminary_cluster_id": "cluster_beta", "exact_duplicate": False, "duplicate_of_row_id": ""},
        ])
        calendar = pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03"]).normalize())
        stage2 = build_stage2_rows(raw, manifest, price_calendar=calendar)
        self.assertEqual(stage2["D"].duplicated(subset=["cluster_id", "effective_trading_date"]).sum(), 0)
        self.assertGreaterEqual(len(stage2["D"]), stage2["C_count"])
        self.assertTrue((stage2["D"]["cluster_size_asof_t"] >= 1).all())
        self.assertTrue((stage2["D"]["unique_domain_count_asof_t"] >= 1).all())
        self.assertTrue((stage2["D"]["dissemination_span_hours_asof_t"] >= 0).all())

    def test_report_numbers_match_generated_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            raw = pd.DataFrame([
                {"original_row_id": "r1", "published_at": "2024-01-02T09:00:00Z", "title": "NVIDIA gains", "url": "u1", "domain": "a.com", "source": "news"},
                {"original_row_id": "r2", "published_at": "2024-01-02T09:05:00Z", "title": "NVIDIA gains", "url": "u2", "domain": "b.com", "source": "news"},
            ])
            manifest = pd.DataFrame([
                {"original_row_id": "r1", "published_at": "2024-01-02T09:00:00Z", "title": "NVIDIA gains", "url": "u1", "domain": "a.com", "normalized_title": "nvidia gains", "preliminary_cluster_id": "cluster_1", "exact_duplicate": False, "duplicate_of_row_id": ""},
                {"original_row_id": "r2", "published_at": "2024-01-02T09:05:00Z", "title": "NVIDIA gains", "url": "u2", "domain": "b.com", "normalized_title": "nvidia gains", "preliminary_cluster_id": "cluster_1", "exact_duplicate": True, "duplicate_of_row_id": "r1"},
            ])
            calendar = pd.DatetimeIndex(pd.to_datetime(["2024-01-02"]).normalize())
            stage2 = build_stage2_rows(raw, manifest, price_calendar=calendar)
            (tmpdir / "raw.csv").write_text(raw.to_csv(index=False), encoding="utf-8")
            pd.DataFrame({"date": calendar.strftime("%Y-%m-%d"), "close": [1.0]}).to_csv(tmpdir / "prices.csv", index=False)
            manifest.to_csv(tmpdir / "manifest.csv", index=False)
            stage2_dir = tmpdir / "stage2"
            stage2_dir.mkdir(parents=True, exist_ok=True)
            stage2["D"].to_csv(stage2_dir / "d_snapshot_manifest.csv", index=False)
            stage2_summary = {"A": {"rows": len(raw)}, "B": {"rows": stage2["B_count"]}, "C": {"rows": stage2["C_count"]}, "D": {"rows": stage2["D_count"]}}
            (stage2_dir / "experiment_summary.json").write_text(json.dumps(stage2_summary, indent=2), encoding="utf-8")
            write_offline_validation_report(tmpdir / "report.md", raw_file=tmpdir / "raw.csv", price_file=tmpdir / "prices.csv", manifest_file=tmpdir / "manifest.csv", stage2_dir=stage2_dir)
            self.assertTrue((tmpdir / "report.md").exists())
            text = (tmpdir / "report.md").read_text(encoding="utf-8")
            self.assertIn("A raw rows", text)
            self.assertIn("D daily event snapshots", text)

    def test_known_row_pairs_do_not_share_cluster(self):
        raw = pd.DataFrame([
            {"original_row_id": "news_f2d2f861dcaeaaa9751f", "published_at": "2020-01-09T12:00:00Z", "title": "Alpha title", "url": "https://a.example/alpha", "domain": "a.example", "source": "news"},
            {"original_row_id": "news_aacdbcbb84c1ca29e1e5", "published_at": "2020-01-09T20:00:00Z", "title": "Beta title", "url": "https://b.example/beta", "domain": "b.example", "source": "news"},
        ])
        manifest = pd.DataFrame([
            {"original_row_id": "news_f2d2f861dcaeaaa9751f", "published_at": "2020-01-09T12:00:00Z", "title": "Alpha title", "url": "https://a.example/alpha", "domain": "a.example", "normalized_title": "alpha title", "preliminary_cluster_id": "cluster_alpha", "exact_duplicate": False, "duplicate_of_row_id": ""},
            {"original_row_id": "news_aacdbcbb84c1ca29e1e5", "published_at": "2020-01-09T20:00:00Z", "title": "Beta title", "url": "https://b.example/beta", "domain": "b.example", "normalized_title": "beta title", "preliminary_cluster_id": "cluster_beta", "exact_duplicate": False, "duplicate_of_row_id": ""},
        ])
        stage2 = build_stage2_rows(raw, manifest, price_calendar=pd.DatetimeIndex(pd.to_datetime(["2020-01-09"]).normalize()))
        self.assertNotEqual(stage2["D"].loc[stage2["D"]["original_row_id"] == "news_f2d2f861dcaeaaa9751f", "cluster_id"].iloc[0], stage2["D"].loc[stage2["D"]["original_row_id"] == "news_aacdbcbb84c1ca29e1e5", "cluster_id"].iloc[0])

    def test_daily_news_features_ratio_and_failures(self):
        rows = pd.DataFrame([
            {"effective_trading_date": "2024-01-02", "sentiment_label": "positive", "sentiment_score": 1.0, "fingpt_error": ""},
            {"effective_trading_date": "2024-01-02", "sentiment_label": "neutral", "sentiment_score": 0.0, "fingpt_error": ""},
            {"effective_trading_date": "2024-01-02", "sentiment_label": "negative", "sentiment_score": -1.0, "fingpt_error": ""},
            {"effective_trading_date": "2024-01-02", "sentiment_label": "", "sentiment_score": None, "fingpt_error": "timeout"},
            {"effective_trading_date": "2024-01-03", "sentiment_label": "positive", "sentiment_score": 1.0, "fingpt_error": ""},
        ])
        features = build_daily_news_features(rows)
        row = features.loc[features["effective_trading_date"] == "2024-01-02"].iloc[0]
        self.assertEqual(row["news_positive_count"], 1)
        self.assertEqual(row["news_negative_count"], 1)
        self.assertEqual(row["news_neutral_count"], 1)
        self.assertEqual(row["news_inference_failure_count"], 1)
        self.assertEqual(row["news_positive_ratio"], 1 / 3)
        self.assertEqual(row["news_negative_ratio"], 1 / 3)
        self.assertEqual(row["news_has_data"], 1)
        self.assertEqual(row["news_item_or_event_count"], 4)

    def test_dissemination_features_are_causal(self):
        rows = pd.DataFrame([
            {"original_row_id": "a", "published_at": "2024-01-02T09:00:00Z", "domain": "a.com", "event_cluster_id": "cluster-1"},
            {"original_row_id": "b", "published_at": "2024-01-02T10:00:00Z", "domain": "b.com", "event_cluster_id": "cluster-1"},
            {"original_row_id": "c", "published_at": "2024-01-02T17:00:00Z", "domain": "c.com", "event_cluster_id": "cluster-1"},
            {"original_row_id": "d", "published_at": "2024-01-03T08:00:00Z", "domain": "d.com", "event_cluster_id": "cluster-1"},
        ])
        features = build_dissemination_features(rows, as_of_date="2024-01-02")
        self.assertEqual(features["cluster_size_asof_t"].iloc[0], 1)
        self.assertEqual(features["cluster_size_asof_t"].iloc[1], 2)
        self.assertEqual(features["cluster_size_asof_t"].iloc[2], 3)
        self.assertEqual(features["unique_domain_count_asof_t"].iloc[0], 1)
        self.assertEqual(features["unique_domain_count_asof_t"].iloc[1], 2)
        self.assertEqual(features["unique_domain_count_asof_t"].iloc[2], 3)
        self.assertGreater(features["dissemination_span_hours_asof_t"].iloc[2], 0)

    def test_daily_dissemination_schema_has_missing_values_for_non_d(self):
        rows = pd.DataFrame([
            {"effective_trading_date": "2024-01-02", "sentiment_label": "positive", "sentiment_score": 1.0, "fingpt_error": ""},
            {"effective_trading_date": "2024-01-02", "sentiment_label": "negative", "sentiment_score": -1.0, "fingpt_error": ""},
            {"effective_trading_date": "2024-01-03", "sentiment_label": "neutral", "sentiment_score": 0.0, "fingpt_error": ""},
        ])
        daily = build_daily_news_features(rows)
        self.assertEqual(set(daily.columns), {
            "effective_trading_date",
            "news_sentiment_mean",
            "news_sentiment_sum",
            "news_sentiment_std",
            "news_positive_count",
            "news_neutral_count",
            "news_negative_count",
            "news_positive_ratio",
            "news_neutral_ratio",
            "news_negative_ratio",
            "news_item_or_event_count",
            "news_has_data",
            "news_inference_failure_count",
        })
        d_rows = build_daily_dissemination_features(pd.DataFrame([
            {"effective_trading_date": "2024-01-02", "cluster_size_asof_t": 2, "unique_domain_count_asof_t": 2, "dissemination_span_hours_asof_t": 12.0},
            {"effective_trading_date": "2024-01-03", "cluster_size_asof_t": 5, "unique_domain_count_asof_t": 3, "dissemination_span_hours_asof_t": 24.0},
        ]))
        self.assertEqual(list(d_rows.columns), [
            "effective_trading_date",
            "cluster_size_asof_t",
            "unique_domain_count_asof_t",
            "dissemination_span_hours_asof_t",
        ])
        self.assertEqual(d_rows["cluster_size_asof_t"].tolist(), [2, 5])

    def test_experiment_names_are_stable(self):
        self.assertEqual(ExperimentSpec.A.value, "A")
        self.assertEqual(ExperimentSpec.B.value, "B")
        self.assertEqual(ExperimentSpec.C.value, "C")
        self.assertEqual(ExperimentSpec.D.value, "D")


if __name__ == "__main__":
    unittest.main()
