"""Standard-library-only tests for annotation validation."""
import ast
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import validate_multidimensional_labels as validator

TEXT = "NVIDIA raised its revenue outlook while regulatory approval remains uncertain."


def evidence(quote, source=TEXT):
    start = source.index(quote)
    return [{"quote": quote, "start": start, "end": start + len(quote)}]


def valid_record(sample_id="sample-1", split="train"):
    directional = {
        "sentiment": "positive", "growth_outlook": "positive",
        "financial_strength": "insufficient_information", "fundamental_impact": "positive",
        "short_term_impact": "neutral", "long_term_impact": "positive",
    }
    labels = {**directional, "risk_presence": "yes", "uncertainty_presence": "yes",
              "relevance": "yes", "price_narrative": "no"}
    ev = {
        "sentiment": evidence("raised its revenue outlook"),
        "growth_outlook": evidence("raised its revenue outlook"),
        "fundamental_impact": evidence("raised its revenue outlook"),
        "short_term_impact": evidence("while regulatory approval remains uncertain"),
        "long_term_impact": evidence("raised its revenue outlook"),
        "risk_presence": evidence("regulatory approval remains uncertain"),
        "uncertainty_presence": evidence("remains uncertain"),
        "relevance": evidence("NVIDIA"),
    }
    return {
        "sample_id": sample_id, "ticker": "NVDA", "source_type": "news",
        "published_at": "2025-01-01T08:00:00Z", "as_of_date": "2025-01-01",
        "title": "", "text": TEXT, "labels": labels, "evidence": ev,
        "annotator": "synthetic_test", "annotation_version": "v1", "split": split,
    }


class ValidatorTests(unittest.TestCase):
    def validate_records(self, records):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "items.jsonl"
            path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
            return validator.validate_jsonl(path)

    def test_valid_record_passes(self):
        result = self.validate_records([valid_record()])
        self.assertTrue(result["valid"], result["errors"])

    def test_missing_dimension_fails(self):
        record = valid_record()
        del record["labels"]["sentiment"]
        self.assertFalse(self.validate_records([record])["valid"])

    def test_invalid_label_fails(self):
        record = valid_record()
        record["labels"]["risk_presence"] = "maybe"
        self.assertFalse(self.validate_records([record])["valid"])

    def test_duplicate_sample_id_fails(self):
        self.assertFalse(self.validate_records([valid_record(), valid_record()])["valid"])

    def test_evidence_quote_not_present_fails(self):
        record = valid_record()
        record["evidence"]["sentiment"][0]["quote"] = "invented quote"
        self.assertFalse(self.validate_records([record])["valid"])

    def test_evidence_offset_mismatch_fails(self):
        record = valid_record()
        record["evidence"]["sentiment"][0]["start"] += 1
        self.assertFalse(self.validate_records([record])["valid"])

    def test_cross_split_duplicate_text_fails(self):
        second = valid_record("sample-2", "test")
        self.assertFalse(self.validate_records([valid_record(), second])["valid"])

    def test_high_similarity_across_split_warns(self):
        second = valid_record("sample-2", "validation")
        second["text"] = TEXT + " Today."
        for items in second["evidence"].values():
            for item in items:
                item["start"] = second["text"].index(item["quote"])
                item["end"] = item["start"] + len(item["quote"])
        result = self.validate_records([valid_record(), second])
        self.assertTrue(result["valid"])
        self.assertTrue(any("highly similar" in warning for warning in result["warnings"]))

    def test_label_collapse_warns(self):
        records = []
        for index in range(3):
            record = valid_record(f"sample-{index}")
            record["text"] += f" Item {index}."
            records.append(record)
        result = self.validate_records(records)
        self.assertTrue(any("label collapse" in warning for warning in result["warnings"]))

    def test_neutral_requires_evidence_but_insufficient_forbids_it(self):
        neutral = valid_record()
        del neutral["evidence"]["short_term_impact"]
        self.assertFalse(self.validate_records([neutral])["valid"])
        insufficient = valid_record("sample-2")
        insufficient["labels"]["financial_strength"] = "insufficient_information"
        insufficient["evidence"]["financial_strength"] = evidence("NVIDIA")
        self.assertFalse(self.validate_records([insufficient])["valid"])

    def test_title_offsets_use_canonical_title_newline_text(self):
        record = valid_record()
        record["title"] = "Company update"
        shifted = len(record["title"]) + 1
        for items in record["evidence"].values():
            for item in items:
                item["start"] += shifted
                item["end"] += shifted
        result = self.validate_records([record])
        self.assertTrue(result["valid"], result["errors"])


    def test_cli_returns_nonzero_for_invalid_data(self):
        record = valid_record()
        del record["labels"]["sentiment"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(validator.main([str(path)]), 1)

    def test_validator_has_no_ml_or_cuda_imports(self):
        tree = ast.parse(Path(validator.__file__).read_text(encoding="utf-8"))
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".")[0])
        self.assertTrue({"torch", "transformers", "peft", "accelerate", "bitsandbytes"}.isdisjoint(roots))


    def test_json_schema_structure_and_target_specific_sentiment(self):
        schema = json.loads(Path("multidimensional_label_schema.json").read_text(encoding="utf-8"))
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertIn("directionalLabel", schema["$defs"])
        sentiment = schema["properties"]["labels"]["properties"]["sentiment"]
        self.assertEqual(sentiment["$ref"], "#/$defs/directionalLabel")
        self.assertIn("specifically toward NVDA", sentiment["description"])

    def test_price_narrative_examples_have_four_yes_without_fundamental_labels(self):
        rows = [json.loads(line) for line in Path("multidimensional_annotation_examples.jsonl").read_text(encoding="utf-8").splitlines()]
        price_only = [row for row in rows if row["labels"]["price_narrative"] == "yes"]
        self.assertGreaterEqual(len(price_only), 4)
        for row in price_only:
            for dimension in ("growth_outlook", "financial_strength", "fundamental_impact"):
                self.assertEqual(row["labels"][dimension], "insufficient_information")
            self.assertEqual(row["labels"]["relevance"], "no")

    def test_unrelated_negative_event_is_not_nvda_sentiment(self):
        rows = [json.loads(line) for line in Path("multidimensional_annotation_examples.jsonl").read_text(encoding="utf-8").splitlines()]
        unrelated = next(row for row in rows if row["sample_id"] == "synthetic-08")
        self.assertEqual(unrelated["labels"]["relevance"], "no")
        self.assertEqual(unrelated["labels"]["sentiment"], "insufficient_information")


if __name__ == "__main__":
    unittest.main()
