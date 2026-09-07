"""Offline report regressions: no model imports, downloads or inference."""
import contextlib
from datetime import datetime, timezone
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

import fingpt_report as html
import run_fingpt_smoke_test as smoke


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def test_html_summary_escape_and_preview(self):
        rows = [
            {"date": "2026-09-07", "source_type": "news", "text": "<script>alert(1)</script>" + "x" * 500,
             "sentiment_label": label, "sentiment_score": score,
             "fingpt_raw_output": label, "fingpt_error": "", "fingpt_inference_seconds": seconds}
            for label, score, seconds in [("positive", 1, 1), ("neutral", 0, 2), ("negative", -1, 3)]
        ]
        rows.append({"text": "failure", "sentiment_label": "neutral", "fingpt_error": "<b>failed</b>"})
        rows.append({"text": "unknown", "sentiment_label": "unparsed", "fingpt_raw_output": "unknown",
                     "fingpt_error": pd.NA, "fingpt_inference_seconds": float("nan")})
        frame = pd.DataFrame(rows)
        original = frame.copy(deep=True)
        summary = html.summarize_results(frame)
        self.assertEqual((summary["total"], summary["success"], summary["failed"]), (5, 4, 1))
        self.assertEqual(summary["counts"], dict(positive=1, neutral=1, negative=1))
        self.assertEqual(summary["unparsed"], 1)
        self.assertEqual(summary["average_seconds"], 2)
        path = self.directory / "mock.html"
        html.write_html_report({"model_name": "<script>bad</script>"}, frame, path)
        page = path.read_text()
        self.assertNotIn("<script>", page)
        self.assertIn("&lt;script&gt;", page)
        self.assertIn("&lt;b&gt;failed&lt;/b&gt;", page)
        self.assertIn("33.3%", page)
        self.assertNotIn("x" * 500, page)
        self.assertIn("row-error", page)
        for column in html.TABLE_COLUMNS:
            self.assertIn(f' scope="col">{column}</th>', page)
        self.assertNotIn("<script src=", page)
        self.assertNotIn("<link", page)
        pd.testing.assert_frame_equal(frame, original)

    def test_empty_report(self):
        path = self.directory / "empty.html"
        html.write_html_report({"fingpt_error": "No samples"}, pd.DataFrame(), path)
        page = path.read_text()
        self.assertIn("No rows available.", page)
        self.assertIn("N/A", page)
        self.assertIn("0 (0.0%)", page)
        self.assertIn("No samples", page)

    def test_identical_timestamps_get_independent_directories(self):
        with patch.object(smoke, "ARCHIVE_ROOT", self.directory):
            stamp = datetime(2026, 9, 7, tzinfo=timezone.utc)
            first = smoke.create_run_directory(stamp)
            second = smoke.create_run_directory(stamp)
        self.assertNotEqual(first, second)
        self.assertTrue(first.is_dir() and second.is_dir())

    def test_main_success_failure_empty_and_partial(self):
        previous = Path.cwd()
        os.chdir(self.directory)
        self.addCleanup(os.chdir, previous)
        info = SimpleNamespace(device="mock", model_name="mock adapter", base_model="mock base",
                               quantization="4bit", peak_allocated_vram_gb=0, peak_reserved_vram_gb=0)
        result = ([dict(sentiment_label="positive", sentiment_score=1, fingpt_raw_output="positive",
                        fingpt_error="", model_name="mock adapter")], 0.5)
        long_text = "Full original text " * 300
        earlier_archives = {}
        for case in ("success", "failure", "empty", "partial"):
            count = 0 if case == "empty" else 2 if case == "partial" else 1
            frame = pd.DataFrame([dict(date="2026-09-07", source_type="news", text=long_text)] * count,
                                 columns=["date", "source_type", "text"])
            stderr = io.StringIO()
            with patch("sys.argv", ["smoke", "--news-limit", "2"]), \
                 patch.object(smoke, "build_sample", return_value=frame), \
                 patch.object(smoke, "get_torch_hardware", return_value={"cuda_available": True}), \
                 patch.object(smoke, "dependency_status", return_value={}), \
                 patch.object(smoke, "load_fingpt_model", return_value=(None, None, info)) as load, \
                 patch.object(smoke, "run_fingpt_batch", return_value=result) as run, \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                if case == "failure":
                    load.side_effect = RuntimeError("mock load failure")
                if case == "partial":
                    run.side_effect = [result, RuntimeError("mock inference failure")]
                code = smoke.main()
            self.assertEqual(code, 0 if case == "success" else 1)
            if case == "empty":
                load.assert_not_called()
            if case != "success":
                self.assertIn("Traceback", stderr.getvalue())
            directories = set(Path("data/archive").iterdir())
            current = (directories - earlier_archives.keys()).pop()
            page = (current / html.HTML_REPORT_FILE).read_text()
            self.assertEqual(page, Path(html.HTML_REPORT_FILE).read_text())
            self.assertEqual((current / smoke.REPORT_FILE).read_text(), Path(smoke.REPORT_FILE).read_text())
            self.assertTrue((current / "fingpt_model_config.json").exists())
            csv = pd.read_csv(next(current.glob("processed_sample/*.csv")), keep_default_na=False)
            self.assertNotIn("text_preview", csv.columns)
            self.assertEqual(len(csv), count)
            if count:
                self.assertEqual(csv.text.iloc[0], long_text)
                self.assertEqual(bool(csv.fingpt_error.iloc[-1]), case != "success")
            if case == "partial":
                self.assertEqual(csv.fingpt_error.iloc[0], "")
                self.assertIn("0.5000 s", page)
            for archived, content in earlier_archives.items():
                self.assertEqual((archived / html.HTML_REPORT_FILE).read_text(), content)
            earlier_archives[current] = page


if __name__ == "__main__":
    unittest.main()
