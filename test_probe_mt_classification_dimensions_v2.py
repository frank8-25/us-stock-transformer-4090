"""Model-free tests for MT classification v2 scoring."""
import math
from types import SimpleNamespace
import unittest
import torch
import probe_mt_classification_dimensions_v2 as probe


class UniformLogitModel:
    def __init__(self, vocab_size=7):
        self.vocab_size = vocab_size

    def __call__(self, input_ids, attention_mask, use_cache):
        self.assertions = (attention_mask.shape == input_ids.shape, use_cache)
        batch, length = input_ids.shape
        return SimpleNamespace(
            logits=torch.zeros((batch, length, self.vocab_size)))


class CandidateScoringTests(unittest.TestCase):
    def test_only_answer_tokens_contribute(self):
        model = UniformLogitModel(7)
        candidate = torch.tensor([[4, 5]])
        short = probe.score_candidate_token_ids(
            model, torch.tensor([[1, 2, 3]]), candidate)
        long = probe.score_candidate_token_ids(
            model, torch.tensor([[1, 2, 3, 1, 2, 3]]), candidate)
        expected = -2 * math.log(7)
        self.assertEqual(short["scored_token_count"], 2)
        self.assertEqual(long["scored_token_count"], 2)
        self.assertAlmostEqual(short["raw_log_likelihood"], expected, places=5)
        self.assertAlmostEqual(long["raw_log_likelihood"], expected, places=5)
        self.assertAlmostEqual(
            short["normalized_log_likelihood"], -math.log(7), places=5)
        self.assertEqual(short["prediction_positions"], [2, 3])
        self.assertEqual(long["prediction_positions"], [5, 6])

    def test_prompt_shape_and_answer_space(self):
        for fixture in probe.FIXTURES:
            for dimension in probe.DIMENSIONS:
                prompt = probe.build_prompt(dimension, fixture["text"])
                self.assertTrue(prompt.startswith("Instruction: "))
                self.assertIn("\nInput: ", prompt)
                self.assertTrue(prompt.endswith("\nAnswer: "))

    def test_parser_cleanup_is_narrow(self):
        self.assertEqual(
            probe.parse_free_label(" Positive.\n", probe.TERNARY_LABELS),
            ("positive", "Positive"))
        self.assertEqual(
            probe.parse_free_label("yes", probe.BINARY_LABELS),
            ("Yes", "yes"))
        for invalid in ("No No", "negative because risk increased", ""):
            parsed, _ = probe.parse_free_label(
                invalid, probe.TERNARY_LABELS)
            self.assertIsNone(parsed)

    def test_expectations_and_ambiguity_are_explicit(self):
        names = {item["name"] for item in probe.DIMENSIONS}
        self.assertEqual(len(names), 10)
        self.assertEqual(len(probe.FIXTURES), 5)
        for fixture in probe.FIXTURES:
            self.assertEqual(set(fixture["expected_labels"]), names)
            nulls = {
                name for name, value in fixture["expected_labels"].items()
                if value is None}
            self.assertEqual(nulls, set(fixture["ambiguous_dimensions"]))


if __name__ == "__main__":
    unittest.main()
