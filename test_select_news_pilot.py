import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

import select_news_pilot as module


def manifest_row(cluster, row_id, published, title, domain, url, size=1):
    return {
        "original_row_id": row_id, "published_at": published, "title": title,
        "normalized_title": title.casefold(), "url": url, "domain": domain,
        "preliminary_cluster_id": cluster, "cluster_size_final": str(size),
    }


def representative(cluster, year, title):
    published = str(year) + "-06-01T00:00:00Z"
    return {
        "preliminary_cluster_id": cluster, "cluster_size_final": 1,
        "representative_original_row_id": "row_" + cluster, "published_at": published,
        "title": title, "representative_normalized_title": title.casefold(),
        "representative_domain": "example.com", "representative_url": "https://example.com/" + cluster,
        "candidate_url_1": "https://example.com/" + cluster, "candidate_url_2": "",
        "candidate_url_3": "", "cluster_first_published_at": published,
        "cluster_last_published_at": published, "dissemination_span_hours": 0.0,
        "unique_domain_count": 1, "year": year, "split": module.split_for_year(year),
    }


class RepresentativeTests(unittest.TestCase):
    def test_deterministic_representative_priority(self):
        rows = pd.DataFrame([
            manifest_row("c", "z", "2024-01-01T00:00:00Z", "Short", "other.com", "https://z", 3),
            manifest_row("c", "b", "2024-01-01T00:00:00Z", "Longer Reuters title", "reuters.com", "https://b", 3),
            manifest_row("c", "a", "2024-01-01T01:00:00Z", "Later primary", "nvidia.com", "https://a", 3),
        ])
        parsed = module.parse_manifest(rows)
        first = module.choose_representative(parsed)
        second = module.choose_representative(parsed.sample(frac=1, random_state=9))
        self.assertEqual(first.original_row_id, "b")
        self.assertEqual(second.original_row_id, "b")

    def test_cluster_crossing_boundary_stays_with_first_split(self):
        rows = pd.DataFrame([
            manifest_row("c", "a", "2023-12-31T23:00:00Z", "First", "a.com", "https://a", 2),
            manifest_row("c", "b", "2024-01-01T01:00:00Z", "Second", "b.com", "https://b", 2),
        ])
        reps, _, boundary = module.make_representatives(module.parse_manifest(rows))
        self.assertEqual(reps.iloc[0].split, "train")
        self.assertEqual(boundary, 1)

    def test_url_limit_and_domain_diversity(self):
        rows = pd.DataFrame([
            manifest_row("c", "a", "2024-01-01T00:00:00Z", "First", "a.com", "https://a1", 5),
            manifest_row("c", "b", "2024-01-01T01:00:00Z", "Second", "a.com", "https://a2", 5),
            manifest_row("c", "c", "2024-01-01T02:00:00Z", "Third", "b.com", "https://b", 5),
            manifest_row("c", "d", "2024-01-01T03:00:00Z", "Fourth", "c.com", "https://c", 5),
            manifest_row("c", "e", "2024-01-01T04:00:00Z", "Fifth", "d.com", "https://d", 5),
        ])
        parsed = module.parse_manifest(rows)
        rep = module.choose_representative(parsed)
        slots, records = module.choose_url_candidates(parsed, rep)
        self.assertEqual(slots[0], "https://a1")
        self.assertEqual(len(records), 3)
        self.assertEqual(len({record["domain"] for record in records}), 3)

    def test_missing_representative_url_leaves_first_slot_blank(self):
        rows = pd.DataFrame([
            manifest_row("c", "a", "2024-01-01T00:00:00Z", "First", "a.com", "", 3),
            manifest_row("c", "b", "2024-01-01T01:00:00Z", "Second", "b.com", "https://b", 3),
            manifest_row("c", "c", "2024-01-01T02:00:00Z", "Third", "c.com", "https://c", 3),
        ])
        parsed = module.parse_manifest(rows)
        rep = module.choose_representative(parsed)
        slots, records = module.choose_url_candidates(parsed, rep)
        self.assertEqual(slots, ["", "https://b", "https://c"])
        self.assertEqual(len(records), 2)

    def test_invalid_time_fails(self):
        rows = pd.DataFrame([manifest_row("c", "a", "bad", "Title", "a.com", "https://a")])
        with self.assertRaisesRegex(ValueError, "published_at"):
            module.parse_manifest(rows)


class SamplingTests(unittest.TestCase):
    def complete_representatives(self):
        rows = []
        for year, quota in module.YEAR_QUOTAS.items():
            for index in range(quota + 5):
                digest = hashlib.sha256((str(year) + ":" + str(index)).encode()).hexdigest()
                rows.append(representative("c_" + str(year) + "_" + str(index), year, "event " + digest))
        return pd.DataFrame(rows)

    def test_fixed_seed_reproducibility_split_quotas_and_unique_clusters(self):
        reps = self.complete_representatives()
        first, shortages, _ = module.select_pilot(reps, seed=17)
        second, _, _ = module.select_pilot(reps, seed=17)
        self.assertFalse(shortages)
        self.assertEqual(first.sample_id.tolist(), second.sample_id.tolist())
        self.assertEqual(first.preliminary_cluster_id.nunique(), 260)
        self.assertEqual(first.split.value_counts().to_dict(), module.SPLIT_QUOTAS)
        self.assertEqual(first.published_at.str[:4].astype(int).value_counts().to_dict(), module.YEAR_QUOTAS)

    def test_high_similarity_does_not_cross_split(self):
        reps = pd.DataFrame([
            representative("train_a", 2023, "nvidia raises annual revenue outlook sharply"),
            representative("validation_a", 2024, "nvidia raises annual revenue outlook sharply"),
        ])
        selected, shortages, rejected = module.select_pilot(reps, {2023: 1, 2024: 1}, seed=1)
        self.assertEqual(len(selected), 1)
        self.assertIn("2024", shortages)
        self.assertEqual(rejected, 1)


class CliTests(unittest.TestCase):
    def test_help_does_not_process_data(self):
        with self.assertRaises(SystemExit) as context:
            module.build_parser().parse_args(["--help"])
        self.assertEqual(context.exception.code, 0)

    def test_run_does_not_modify_raw_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "news.csv"
            raw.write_text("date,title,url\n2024-01-01,Title,https://a\n", encoding="utf-8")
            raw_hash = hashlib.sha256(raw.read_bytes()).hexdigest()
            cluster = root / "clusters.csv"
            pd.DataFrame([manifest_row(
                "c", "a", "2024-01-01T00:00:00Z", "Unique title", "a.com", "https://a"
            )]).to_csv(cluster, index=False)
            report = root / "run.json"
            report.write_text(json.dumps({
                "selected_similarity_threshold": 0.9, "input_sha256_after": raw_hash
            }))
            args = type("Args", (), {
                "cluster_manifest": cluster, "clustering_report": report, "raw_news": raw,
                "output_dir": root / "output", "similarity_threshold": 0.9, "seed": 42,
            })()
            result = module.run(args)
            self.assertEqual(hashlib.sha256(raw.read_bytes()).hexdigest(), raw_hash)
            self.assertTrue(result["raw_news_unchanged"])
            self.assertEqual(result["status"], "quota_shortfall")


if __name__ == "__main__":
    unittest.main()
