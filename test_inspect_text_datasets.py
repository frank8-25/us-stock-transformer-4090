"""Synthetic, read-only inventory regressions; no real datasets or models required."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

import inspect_text_datasets as inspect


class InspectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = {}
        for source in inspect.SPECS:
            row = dict(date="2024-01-02", title="Synthetic title",
                       content="Item 1 business. Item 1A risks. Item 7 discussion. Item 7A risk.",
                       summary="Short summary", url="https://example.test", source="mock",
                       speaker="CEO", section="Prepared Remarks", quarter="Q1", fiscal_year="2024",
                       filing_date="2024-01-02", chunk_id="test")
            path = self.root / (source + ".csv")
            pd.DataFrame([row]).to_csv(path, index=False)
            self.paths[source] = path

    def cli(self, strict=False, paths=None, output=None):
        args = ["--output-dir", str(output or self.root / "reports")]
        for source, path in (paths or self.paths).items():
            args.extend(["--" + source.replace("_", "-") + "-file", str(path)])
        if strict:
            args.append("--strict")
        with contextlib.redirect_stdout(io.StringIO()):
            return inspect.main(args)

    def news(self, rows):
        pd.DataFrame(rows).to_csv(self.paths["news"], index=False)
        return inspect.inspect_dataset("news", self.paths["news"])

    def test_all_present_complete(self):
        self.assertEqual(self.cli(strict=True), 0)
        report = json.loads((self.root / "reports/dataset_inspection_report.json").read_text())
        self.assertEqual(report["summary"]["ok"], 4)
        for item in report["datasets"]:
            self.assertEqual(item["rows"], 1)
            self.assertEqual(item["dates"]["parse_success_ratio"], 1)
            self.assertEqual(len(item["sha256"]), 64)

    def test_missing_required_files_reported_without_crash(self):
        missing = {source: self.root / ("missing_" + source + ".csv") for source in inspect.SPECS}
        self.assertEqual(self.cli(paths=missing), 0)
        self.assertEqual(self.cli(paths=missing, strict=True), 1)
        report = json.loads((self.root / "reports/dataset_inspection_report.json").read_text())
        self.assertEqual(report["summary"]["missing"], 4)

    def test_missing_manual_supplement_is_optional(self):
        paths = {**self.paths, "manual_earnings_call": self.root / "absent.csv"}
        self.assertEqual(self.cli(paths=paths, strict=True), 0)
        report = inspect.inspect_dataset("manual_earnings_call", paths["manual_earnings_call"])
        self.assertEqual(report["status"], "missing")
        self.assertEqual(report["severity"], "warning")

    def test_required_column_missing(self):
        report = self.news([{"title": "A"}])
        self.assertIn("date", report["fields"]["missing_required"])
        self.assertEqual(report["status"], "error")
        self.assertEqual(self.cli(strict=True), 1)

    def test_recommended_columns_only_warning(self):
        report = self.news([{"date": "2024-01-02", "title": "A"}])
        self.assertEqual(report["status"], "warning")
        self.assertIn("content OR summary", report["fields"]["missing_recommended"])
        self.assertEqual(self.cli(strict=True), 0)

    def test_invalid_dates(self):
        report = self.news([{"date": "not-a-date", "title": "A"}, {"date": "2024-02-30", "title": "B"},
                            {"date": "2024-01-02", "title": "C"}])
        self.assertEqual(report["dates"]["parsed_failure"], 2)
        self.assertEqual(report["dates"]["parsed_success"], 1)
        self.assertAlmostEqual(report["dates"]["parse_success_ratio"], 1/3)
        self.assertEqual(self.cli(strict=True), 1)

    def test_blank_text_and_empty_cells(self):
        report = self.news([
            {"date": "2024-01-02", "title": " ", "content": "", "summary": ""},
            {"date": "2024-01-02", "title": "A", "content": " ", "summary": ""},
            {"date": "2024-01-02", "title": "B", "content": "", "summary": "body"},
            {"date": "2024-01-02", "title": "C", "content": "content", "summary": "ignored"}])
        text = report["text"]
        self.assertEqual(text["blank_title_rows"], 1)
        self.assertEqual(text["blank_content_rows"], 3)
        self.assertEqual(text["blank_summary_rows"], 2)
        self.assertEqual(text["blank_analysis_rows"], 1)
        self.assertEqual(text["title_only_rows"], 1)
        self.assertEqual(text["title_and_body_or_summary_rows"], 2)
        self.assertEqual(report["missing_values"]["content"]["empty_cells"], 2)
        self.assertEqual(report["missing_values"]["content"]["blank_including_whitespace"], 3)

    def test_exact_and_key_duplicates(self):
        row = {"date": "2024-01-02", "title": "A", "url": "https://example.test"}
        report = self.news([row, row, {**row, "url": "https://another.test"}])
        self.assertEqual(report["duplicates"]["all_columns_extra_rows"], 1)
        self.assertEqual(report["duplicates"]["date+title+url_extra_rows"], 1)
        self.assertEqual(report["duplicates"]["date+title_extra_rows"], 2)
        report = self.news([{"date": "2024-01-02", "title": "A"}])
        self.assertIsNone(report["duplicates"]["date+title+url_extra_rows"])

    def test_long_text_warning_and_per_row_content_lengths(self):
        path = self.paths["earnings_call"]
        pd.DataFrame([{"date": "2024-01-02", "content": "x"*9000},
                      {"date": "2024-01-03", "content": "y"*4500}]).to_csv(path, index=False)
        report = inspect.inspect_dataset("earnings_call", path)
        self.assertEqual(report["text"]["over_4000_characters"], 2)
        self.assertEqual(report["text"]["over_8000_characters"], 1)
        self.assertEqual(report["content_lengths_by_row"], [
            {"data_row": 1, "characters": 9000}, {"data_row": 2, "characters": 4500}])
        self.assertEqual(report["text"]["analysis_characters"]["maximum"], 9000)
        self.assertEqual(report["status"], "warning")

    def test_empty_header_only_and_zero_byte_csv(self):
        pd.DataFrame(columns=["date", "title"]).to_csv(self.paths["news"], index=False)
        report = inspect.inspect_dataset("news", self.paths["news"])
        self.assertEqual(report["rows"], 0)
        self.assertIsNone(report["dates"]["parse_success_ratio"])
        self.assertIsNone(report["text"]["analysis_characters"]["minimum"])
        self.assertEqual(report["status"], "warning")
        self.paths["news"].write_bytes(b"")
        report = inspect.inspect_dataset("news", self.paths["news"])
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["rows"], 0)

    def test_sha256_and_readonly_csv(self):
        before = {source: path.read_bytes() for source, path in self.paths.items()}
        hashes = {source: inspect.file_sha256(path) for source, path in self.paths.items()}
        self.cli()
        self.cli(strict=True)
        for source, path in self.paths.items():
            self.assertEqual(path.read_bytes(), before[source])
            self.assertEqual(inspect.file_sha256(path), hashes[source])
            self.assertEqual(hashes[source], hashlib.sha256(before[source]).hexdigest())

    def test_three_reports_without_raw_text(self):
        secret = "PRIVATE_SENTENCE_NEVER_RENDER_THIS"
        self.news([{"date": "2024-01-02", "title": secret, "content": secret*400,
                    "url": "https://example.test/?token=DO_NOT_RENDER"}])
        self.assertEqual(self.cli(), 0)
        for suffix in ("json", "md", "html"):
            output = (self.root / "reports" / (inspect.REPORT_STEM + "." + suffix)).read_text()
            self.assertNotIn(secret, output)
            self.assertNotIn("DO_NOT_RENDER", output)
        page = (self.root / "reports/dataset_inspection_report.html").read_text()
        self.assertIn('<meta charset="utf-8">', page)
        self.assertNotIn("<script", page)
        self.assertNotIn("<link", page)

    def test_observed_date_counts_no_zero_fill(self):
        report = self.news([{"date": d, "title": "A"} for d in
                            ["2024-01-01", "2024-01-01", "2024-01-03", "2024-01-22"]])
        self.assertEqual(report["daily_counts"], {"2024-01-01": 2, "2024-01-03": 1, "2024-01-22": 1})
        self.assertEqual(report["weekly_counts"], {"2024-01-01": 3, "2024-01-22": 1})
        self.assertEqual(report["dates"]["daily_volume"]["median"], 1)
        self.assertEqual(report["dates"]["weekly_volume"]["mean"], 2)

    def test_mixed_dates_and_timezone(self):
        report = self.news([{"date": d, "title": "A"} for d in
                            ["2024/01/02", "2024-01-03T00:30:00+08:00", "January 4, 2024"]])
        self.assertEqual(report["dates"]["parsed_success"], 3)
        self.assertEqual(report["daily_counts"]["2024-01-02"], 2)

    def test_tenk_item_mentions_and_labels(self):
        path = self.paths["ten_k"]
        pd.DataFrame([{"date": "2024-01-02", "content": "Item 1A risk. Item 7A risk.", "section": "Item 7"},
                      {"date": "2024-01-02", "content": "text", "section": "1"}]).to_csv(path, index=False)
        report = inspect.inspect_dataset("ten_k", path)
        self.assertEqual(report["ten_k_items"]["Item 1"]["content_mention_rows"], 0)
        self.assertEqual(report["ten_k_items"]["Item 1"]["section_label_rows"], 1)
        self.assertEqual(report["ten_k_items"]["Item 1A"]["content_mention_rows"], 1)
        self.assertEqual(report["ten_k_items"]["Item 7"]["section_label_rows"], 1)
        self.assertEqual(report["ten_k_items"]["Item 7A"]["content_mention_rows"], 1)

    def test_malformed_csv_safe_error(self):
        self.paths["news"].write_text('date,title\n"PRIVATE_INVALID_CONTENT', encoding="utf-8")
        report = inspect.inspect_dataset("news", self.paths["news"])
        self.assertEqual(report["status"], "error")
        self.assertNotIn("PRIVATE_INVALID_CONTENT", json.dumps(report))
        self.assertEqual(self.cli(), 0)

    def test_report_output_cannot_overwrite_input(self):
        destination = self.root / (inspect.REPORT_STEM + ".json")
        destination.write_bytes(self.paths["news"].read_bytes())
        before = destination.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.cli(paths={**self.paths, "news": destination}, output=self.root)
        self.assertEqual(destination.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
