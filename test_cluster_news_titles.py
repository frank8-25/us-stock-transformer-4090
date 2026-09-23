import argparse
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

import cluster_news_titles as module


def frame(rows):
    return pd.DataFrame(rows, columns=[
        "date", "title", "url", "source", "domain", "seendate"
    ])


def row(title, seen, url="", domain="example.com", source="example.com"):
    return {
        "date": seen[:4] + "-" + seen[4:6] + "-" + seen[6:8],
        "title": title, "url": url, "source": source, "domain": domain,
        "seendate": seen,
    }


class NormalizationTests(unittest.TestCase):
    def test_normalization_is_conservative(self):
        value = module.normalize_title(
            "  NVIDIA UP 10% — Yahoo Finance  ", "finance.yahoo.com", "finance.yahoo.com"
        )
        self.assertEqual(value, "nvidia up 10%")
        self.assertIn("up", value)
        self.assertIn("10", value)

    def test_unverified_suffix_is_retained(self):
        value = module.normalize_title("NVIDIA is not down - analyst note", "example.com", "example.com")
        self.assertIn("not", value)
        self.assertIn("down", value)
        self.assertIn("analyst note", value)


class DeduplicationTests(unittest.TestCase):
    def test_exact_duplicate_marking_and_canonical(self):
        data = frame([
            row("NVIDIA rises - Example", "20240101T010000Z", "https://a", "example.com"),
            row("NVIDIA rises", "20240101T020000Z", "https://b", "example.com"),
            row("Different wording", "20240102T010000Z", "https://a", "example.com"),
        ])
        rows, canonical = module.prepare_rows(data)
        self.assertEqual(int(rows.exact_duplicate.sum()), 1)
        self.assertEqual(len(canonical), 2)
        self.assertEqual(rows.iloc[0].duplicate_of_row_id, "")
        self.assertTrue(rows.iloc[1].duplicate_of_row_id)
        self.assertFalse(rows.iloc[2].exact_duplicate)

    def test_stable_row_id_is_reproducible_and_duplicate_safe(self):
        data = frame([
            row("Same", "20240101T010000Z", "https://a"),
            row("Same", "20240101T010000Z", "https://a"),
        ])
        first = module.stable_row_ids(data)
        second = module.stable_row_ids(data.copy())
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 2)

    def test_same_url_different_titles_are_not_grouped_as_exact_duplicates(self):
        data = frame([
            row("NVIDIA rises as demand surges", "20240102T090000Z", "https://example.com/nvda"),
            row("NVIDIA falls as demand cools", "20240102T091500Z", "https://example.com/nvda"),
        ])
        rows, canonical = module.prepare_rows(data)
        self.assertEqual(len(canonical), 2)
        self.assertFalse(rows.iloc[1].exact_duplicate)

    def test_empty_url_is_allowed(self):
        rows, canonical = module.prepare_rows(frame([row("Valid title", "20240101T010000Z")]))
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(canonical), 1)

    def test_missing_title_and_bad_date_fail(self):
        with self.assertRaisesRegex(ValueError, "blank"):
            module.prepare_rows(frame([row(" ", "20240101T010000Z")]))
        with self.assertRaisesRegex(ValueError, "publication"):
            module.prepare_rows(frame([{
                "date": "bad", "title": "Valid", "url": "", "source": "x",
                "domain": "x", "seendate": "bad"
            }]))


class CausalClusteringTests(unittest.TestCase):
    def cluster(self, hours, titles=None, threshold=0.9):
        titles = titles or ["nvidia raises outlook " + str(i) for i in range(len(hours))]
        embeddings = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(hours), 1))
        times = pd.to_datetime(["2024-01-01T00:00:00Z"] * len(hours), utc=True) + pd.to_timedelta(hours, unit="h")
        ids = ["id" + str(i) for i in range(len(hours))]
        return module.cluster_causal(embeddings, times, ids, titles, threshold, 72)

    def test_only_past_window_and_72_hour_boundary(self):
        result = self.cluster([0, 72, 145])
        self.assertEqual(result["assignments"][0], result["assignments"][1])
        self.assertNotEqual(result["assignments"][1], result["assignments"][2])

    def test_direction_conflict_does_not_merge(self):
        result = self.cluster([0, 1], ["nvidia raises outlook", "nvidia cuts outlook"])
        self.assertNotEqual(result["assignments"][0], result["assignments"][1])

    def test_deterministic_output(self):
        first = self.cluster([0, 1, 2])
        second = self.cluster([0, 1, 2])
        self.assertEqual(first["assignments"], second["assignments"])
        self.assertEqual(first["link_similarity"], second["link_similarity"])

    def test_future_row_does_not_change_as_of_size(self):
        data = frame([
            row("NVIDIA raises outlook", "20240101T000000Z", "https://a"),
            row("NVIDIA raises revenue outlook", "20240101T010000Z", "https://b"),
            row("NVIDIA raises annual outlook", "20240101T020000Z", "https://c"),
        ])
        rows, canonical = module.prepare_rows(data)
        embeddings = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (3, 1))
        clustered = module.cluster_causal(
            embeddings, canonical._published, canonical.original_row_id.tolist(),
            canonical.normalized_title.tolist(), 0.9, 72
        )
        manifest, _, _ = module.materialize_manifest(rows, canonical, clustered, 0.9)
        self.assertEqual(manifest.cluster_size_as_of_publication.tolist(), [1, 2, 3])
        self.assertEqual(manifest.cluster_size_final.tolist(), [3, 3, 3])

    def test_no_full_n_squared_matrix(self):
        count = 100
        result = self.cluster(list(range(count)), ["neutral headline " + str(i) for i in range(count)])
        self.assertFalse(result["full_matrix_allocated"])
        self.assertLessEqual(result["max_active_window"], 72)
        self.assertLess(result["candidate_comparisons"], count * count)


class CliTests(unittest.TestCase):
    def test_valid_embedding_cache_avoids_model_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            ids = ["a", "b"]
            canonical_hash = hashlib.sha256(chr(10).join(ids).encode("utf-8")).hexdigest()
            values = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            np.save(output / "canonical_title_embeddings.npy", values)
            (output / "embedding_cache_metadata.json").write_text(__import__("json").dumps({
                "model": "cached", "input_sha256": "input", "canonical_ids_sha256": canonical_hash,
                "embedding_rows": 2, "embedding_dimension": 2, "normalized": True, "dtype": "float32",
                "device": "cuda", "embedding_seconds": 1.0,
            }))
            prior = "sentence_transformers" in sys.modules
            loaded, metadata = module.load_embeddings(
                ["one", "two"], "cached", "cuda", 2, 42, output, "input", ids
            )
            np.testing.assert_array_equal(loaded, values)
            self.assertTrue(metadata["cache_hit"])
            self.assertEqual("sentence_transformers" in sys.modules, prior)

    def test_help_does_not_import_sentence_transformers(self):
        prior = "sentence_transformers" in sys.modules
        with self.assertRaises(SystemExit) as context:
            module.build_parser().parse_args(["--help"])
        self.assertEqual(context.exception.code, 0)
        self.assertEqual("sentence_transformers" in sys.modules, prior)

    def test_dry_run_preserves_input_and_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "news.csv"
            output = root / "run"
            data = frame([
                row("NVIDIA raises outlook", "20240101T000000Z", "https://a", "a.com"),
                row("NVIDIA raises revenue outlook", "20240101T010000Z", "https://b", "b.com"),
            ])
            data.to_csv(input_path, index=False, encoding="utf-8-sig")
            before = hashlib.sha256(input_path.read_bytes()).hexdigest()

            def fake_loader(titles, model, device, batch, seed, output_dir, input_sha, ids):
                values = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(titles), 1))
                return values, {
                    "model": model, "device": "cpu-test", "embedding_rows": len(titles),
                    "embedding_dimension": 2, "embedding_seconds": 0.0,
                }

            args = argparse.Namespace(
                input=input_path, output_dir=output, model="fake", time_window_hours=72,
                similarity_threshold=0.90, thresholds=[0.85, 0.90, 0.93, 0.95],
                batch_size=2, device="cpu", seed=42, limit=None, dry_run=True,
            )
            report = module.run(args, embedding_loader=fake_loader)
            after = hashlib.sha256(input_path.read_bytes()).hexdigest()
            self.assertEqual(before, after)
            self.assertTrue(report["input_unchanged"])
            manifest = pd.read_csv(output / "title_clustering_manifest.csv")
            self.assertEqual(list(manifest.columns), module.REQUIRED_OUTPUT_COLUMNS)
            self.assertFalse(report["safeguards"]["urls_fetched"])


if __name__ == "__main__":
    unittest.main()
