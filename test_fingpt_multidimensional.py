"""Mock-only multidimensional extraction tests: never load pretrained models."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
import torch

import config
import fingpt_multidimensional as multi
import fingpt_report as html
import fingpt_sentiment as sentiment
import run_fingpt_smoke_test as smoke

TEXT = "NVIDIA reported strong demand for AI chips while warning about export restrictions."
VALID = {
    **multi.SCHEMA_EXAMPLE,
    "sentiment": 1, "risk": 1, "uncertainty": 1, "growth_outlook": 1,
    "financial_strength": 0, "fundamental_impact": 1, "price_narrative": 0, "relevance": 2,
    "short_term_impact": 0, "long_term_impact": 1, "event_type": "demand",
    "positive_factors": ["Strong AI chip demand"], "potential_concerns": ["Export restrictions"],
    "summary": "AI chip demand is strong. Export restrictions present a concern.",
    "evidence": ["strong demand for AI chips", "export restrictions"],
}


class Tokenizer:
    eos_token_id = 2

    def __init__(self, outputs, length=2):
        self.outputs = iter(outputs)
        self.length = length
        self.prompts = []

    def __call__(self, prompt, **kwargs):
        assert kwargs["truncation"] is False
        self.prompts.append(prompt)
        return {"input_ids": torch.ones((1, self.length), dtype=torch.long)}

    def batch_decode(self, tokens, **kwargs):
        assert tokens.shape[1] == 2, "Must decode generated tokens only."
        return [next(self.outputs)]


class Model:
    device = torch.device("cpu")
    config = SimpleNamespace(max_position_embeddings=4096)

    def generate(self, **kwargs):
        assert kwargs["do_sample"] is False
        assert kwargs["num_beams"] == 1
        assert kwargs["max_new_tokens"] == config.FINGPT_MULTIDIM_MAX_NEW_TOKENS
        assert "temperature" not in kwargs
        return torch.ones((1, kwargs["input_ids"].shape[1] + 2), dtype=torch.long)


class ParserTests(unittest.TestCase):
    def parse(self, value):
        return multi.parse_multidimensional_output(json.dumps(value), TEXT)

    def test_valid_complete_json(self):
        self.assertEqual(self.parse(VALID), VALID)

    def test_code_fence(self):
        raw = "```json\n" + json.dumps(VALID) + "\n```"
        self.assertEqual(multi.parse_multidimensional_output(raw, TEXT), VALID)

    def test_surrounding_text(self):
        raw = "Here is the analysis: " + json.dumps(VALID) + " End of analysis."
        self.assertEqual(multi.parse_multidimensional_output(raw, TEXT), VALID)

    def test_first_object_is_used(self):
        self.assertEqual(multi.parse_multidimensional_output(json.dumps(VALID) + json.dumps({}), TEXT), VALID)
        with self.assertRaisesRegex(ValueError, "Schema keys"):
            multi.parse_multidimensional_output("{}" + json.dumps(VALID), TEXT)

    def test_braces_and_escaped_quotes_inside_strings(self):
        value = copy.deepcopy(VALID)
        value["summary"] = 'A {braced} "quoted" comment.'
        self.assertEqual(self.parse(value), value)

    def test_missing_field(self):
        value = {k: v for k, v in VALID.items() if k != "risk"}
        with self.assertRaisesRegex(ValueError, "missing=.*risk"):
            self.parse(value)

    def test_out_of_range(self):
        for field, (low, high) in multi.NUMERIC_RANGES.items():
            for bad in (low-1, high+1):
                with self.subTest(field=field, value=bad), self.assertRaisesRegex(ValueError, field):
                    self.parse({**VALID, field: bad})

    def test_noninteger(self):
        for value in (1.5, 1.0, "1", True, None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "integer"):
                self.parse({**VALID, "sentiment": value})

    def test_invalid_event(self):
        with self.assertRaisesRegex(ValueError, "event_type"):
            self.parse({**VALID, "event_type": "stock_prediction"})

    def test_list_too_long(self):
        for field in multi.LIST_FIELDS:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "at most 4"):
                self.parse({**VALID, field: ["demand"]*5})

    def test_invalid_list_type_or_items(self):
        for bad in ("demand", [1], [""], ["x"*241], None):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                self.parse({**VALID, "positive_factors": bad})

    def test_evidence_absent(self):
        with self.assertRaisesRegex(ValueError, "not present"):
            self.parse({**VALID, "evidence": ["record-breaking profit"]})

    def test_evidence_case_insensitive(self):
        self.parse({**VALID, "evidence": ["STRONG DEMAND FOR AI CHIPS"]})

    def test_unparseable(self):
        for raw in ("positive", "", "null", '{"risk":', "{bad json}"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                multi.parse_multidimensional_output(raw, TEXT)

    def test_duplicate_and_nonfinite_json(self):
        raw = json.dumps(VALID)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            multi.parse_multidimensional_output(raw.replace('"risk": 1', '"risk": 1, "risk": 0'), TEXT)
        with self.assertRaisesRegex(ValueError, "Invalid JSON"):
            multi.parse_multidimensional_output(raw.replace('"risk": 1', '"risk": NaN'), TEXT)

    def test_extra_priced_in_forbidden(self):
        with self.assertRaisesRegex(ValueError, "extra=.*priced_in"):
            self.parse({**VALID, "priced_in": 1})

    def test_summary_limit(self):
        for bad in ("One. Two. Three.", "x"*601, ["not a string"]):
            with self.subTest(value=bad), self.assertRaisesRegex(ValueError, "summary"):
                self.parse({**VALID, "summary": bad})

    def test_source_aware_prompt(self):
        for source, instruction in multi.SOURCE_INSTRUCTIONS.items():
            prompt = multi.build_multidimensional_prompt(TEXT, source)
            self.assertIn(instruction, prompt)
            self.assertIn(TEXT, prompt)
            self.assertIn(config.COMPANY_NAME, prompt)
            self.assertIn(json.dumps(multi.SCHEMA_EXAMPLE), prompt)
        with self.assertRaisesRegex(ValueError, "source_type"):
            multi.build_multidimensional_prompt(TEXT, "unknown")

    def test_model_pairs_are_centralized(self):
        self.assertIs(sentiment.FINGPT_MODEL_PROFILES, config.FINGPT_MODEL_PROFILES)
        self.assertEqual(config.FINGPT_MODEL_PROFILES["sentiment-llama2-13b"]["base_model"],
                         "NousResearch/Llama-2-13b-hf")
        self.assertEqual(config.FINGPT_MODEL_PROFILES["mt-llama2-7b"]["base_model"],
                         "meta-llama/Llama-2-7b-hf")
        self.assertEqual(config.FINGPT_MODEL_PROFILES["mt-llama2-7b"]["model_name"],
                         "FinGPT/fingpt-mt_llama2-7b_lora")

    def test_generation_and_mixed_rows(self):
        raws = [json.dumps(VALID), "bad output", json.dumps(VALID)]
        with contextlib.redirect_stderr(io.StringIO()):
            results, seconds = multi.run_multidimensional_batch(Tokenizer(raws), Model(),
                                                               [TEXT]*3, ["news", "ten_k", "earnings_call"], "mock")
        self.assertEqual(len(results), 3)
        self.assertEqual([bool(row["fingpt_error"]) for row in results], [False, True, False])
        self.assertEqual(results[1]["fingpt_raw_output"], "bad output")
        self.assertTrue(all(results[1][key] is None for key in multi.SCHEMA_FIELDS))
        self.assertGreaterEqual(seconds, 0)

    def test_overlength_is_not_truncated(self):
        with contextlib.redirect_stderr(io.StringIO()):
            rows, _ = multi.run_multidimensional_batch(Tokenizer([], length=4000), Model(), [TEXT], ["news"], "mock")
        self.assertIn("context budget", rows[0]["fingpt_error"])
        self.assertTrue(all(rows[0][key] is None for key in multi.SCHEMA_FIELDS))

    def test_empty_batch(self):
        rows, _ = multi.run_multidimensional_batch(None, None, [], [], "mock")
        self.assertEqual(rows, [])


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        previous = Path.cwd()
        os.chdir(self.temp.name)
        self.addCleanup(os.chdir, previous)

    def invoke(self, raws, samples=None, extras=None, load_error=None):
        if samples is None:
            samples = pd.DataFrame([dict(source_type="news", date="2026-09-09", text=TEXT)]*len(raws))
        profile = config.FINGPT_MODEL_PROFILES["mt-llama2-7b"]
        info = SimpleNamespace(device="mock", model_name=profile["model_name"], base_model=profile["base_model"],
                               quantization="4bit", peak_allocated_vram_gb=0, peak_reserved_vram_gb=0)
        stderr = io.StringIO()
        with patch("sys.argv", ["smoke", "--analysis-mode", "multidimensional", "--model-profile", "mt-llama2-7b"] + (extras or [])), \
             patch.object(smoke, "get_torch_hardware", return_value={"cuda_available": True}), \
             patch.object(smoke, "dependency_status", return_value={}), \
             patch.object(smoke, "build_sample", return_value=samples), \
             patch.object(smoke, "load_fingpt_model", return_value=(Tokenizer(raws), Model(), info)) as loader, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
            if load_error:
                loader.side_effect = RuntimeError(load_error)
            code = smoke.main()
        archive = max(Path("data/archive").iterdir(), key=lambda p: p.stat().st_mtime_ns)
        csv = pd.read_csv(next(archive.glob("processed_sample/*.csv")), keep_default_na=False)
        return code, csv, archive, stderr.getvalue()

    def test_mixed_success_failure_csv_html_markdown(self):
        code, csv, archive, stderr = self.invoke([json.dumps(VALID), "bad output", json.dumps(VALID)])
        self.assertEqual(code, 1)
        self.assertEqual(len(csv), 3)
        self.assertEqual(csv.fingpt_error.ne("").tolist(), [False, True, False])
        self.assertEqual(csv.fingpt_raw_output.iloc[1], "bad output")
        for field in multi.SCHEMA_FIELDS:
            self.assertEqual(csv[field].iloc[1], "", field)
        for field in multi.LIST_FIELDS:
            self.assertEqual(json.loads(csv[field].iloc[0]), VALID[field])
        self.assertEqual(csv.text.iloc[0], TEXT)
        self.assertNotIn("text_preview", csv)
        page = (archive / html.HTML_REPORT_FILE).read_text()
        self.assertIn("Mean dimension values", page)
        self.assertIn("Event counts", page)
        self.assertIn("Experimental multidimensional extraction", page)
        self.assertIn("not calibrated probabilities", page)
        self.assertIn('class="error"', page)
        self.assertIn('<meta charset="utf-8">', page)
        self.assertNotIn("<script", page)
        self.assertNotIn("<link", page)
        self.assertEqual(page, Path(html.HTML_REPORT_FILE).read_text())
        report = (archive / smoke.REPORT_FILE).read_text()
        self.assertIn("risk", report)
        self.assertIn("Traceback", report)
        self.assertIn("Traceback", stderr)
        self.assertNotIn("NaN", report)

    def test_direct_text_success(self):
        code, csv, archive, _ = self.invoke([json.dumps(VALID)], extras=["--text", TEXT])
        self.assertEqual(code, 0)
        self.assertEqual(csv.source_type.iloc[0], "direct_text")
        self.assertEqual(csv.sentiment.iloc[0], 1)
        saved = json.loads((archive / "fingpt_model_config.json").read_text())
        self.assertEqual(saved["analysis_mode"], "multidimensional")

    def test_load_failure_outputs_null_fields(self):
        code, csv, _, stderr = self.invoke([json.dumps(VALID)], load_error="mock download failure")
        self.assertEqual(code, 1)
        self.assertIn("mock download failure", stderr)
        self.assertTrue(all(csv[field].iloc[0] == "" for field in multi.SCHEMA_FIELDS))
        self.assertIn("mock download failure", csv.fingpt_error.iloc[0])

    def test_empty_data_generates_reports(self):
        empty = pd.DataFrame(columns=["source_type", "date", "text"])
        code, csv, archive, _ = self.invoke([], samples=empty)
        self.assertEqual(code, 1)
        self.assertTrue(csv.empty)
        page = (archive / html.HTML_REPORT_FILE).read_text()
        self.assertIn("No rows available.", page)
        self.assertIn("N/A", page)

    def test_html_escaping_and_preview_only(self):
        row = {**VALID, "text": "<script>bad</script>"+"z"*600, "fingpt_raw_output": "<script>bad</script>",
               "fingpt_error": "", "fingpt_inference_seconds": .1}
        frame = pd.DataFrame([row])
        html.write_html_report({"analysis_mode": "multidimensional"}, frame, "test.html")
        page = Path("test.html").read_text()
        self.assertNotIn("<script>", page)
        self.assertIn("&lt;script&gt;", page)
        self.assertNotIn("z"*600, page)
        self.assertEqual(frame.text.iloc[0], row["text"])

    def test_profile_mode_incompatibility_before_loading(self):
        for profile in ("sentiment-llama2-13b", "sentiment-chatglm2-6b"):
            with patch("sys.argv", ["smoke", "--analysis-mode", "multidimensional", "--model-profile", profile]), \
                 patch.object(smoke, "load_fingpt_model") as load, contextlib.redirect_stderr(io.StringIO()) as err:
                with self.assertRaises(SystemExit) as raised:
                    smoke.main()
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("requires --model-profile mt-llama2-7b", err.getvalue())
            load.assert_not_called()

    def test_sentiment_default_remains_legacy(self):
        info = SimpleNamespace(device="mock", model_name=config.FINGPT_SENTIMENT_MODEL,
                               base_model=config.FINGPT_BASE_MODEL, quantization="4bit",
                               peak_allocated_vram_gb=0, peak_reserved_vram_gb=0)
        for label, score in (("positive", 1), ("neutral", 0), ("negative", -1)):
            with patch("sys.argv", ["smoke", "--model-profile", "sentiment-llama2-13b",
                                    "--quantization", "4bit", "--text", TEXT]), \
                 patch.object(smoke, "get_torch_hardware", return_value={"cuda_available": True}), \
                 patch.object(smoke, "dependency_status", return_value={}), \
                 patch.object(smoke, "load_fingpt_model", return_value=(None, None, info)), \
                 patch.object(smoke, "run_fingpt_batch", return_value=(
                     [dict(sentiment_label=label, sentiment_score=score, fingpt_raw_output=label,
                           fingpt_error="", model_name=info.model_name)], .1)), \
                 patch.object(smoke, "run_multidimensional_batch") as multi_run, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(smoke.main(), 0)
            multi_run.assert_not_called()
            self.assertIn("Sentiment distribution", Path(html.HTML_REPORT_FILE).read_text())
            self.assertNotIn("Mean dimension values", Path(html.HTML_REPORT_FILE).read_text())


if __name__ == "__main__":
    unittest.main()
