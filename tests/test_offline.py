"""Offline regression checks; no model downloads, database writes or API calls."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.data_utils import read_kb_chunks
from src.build_multihoprag_mini import build_corpus
from src.generate_multihoprag_english import calculate_item_metrics
from src.project_paths import PROJECT_ROOT, configured_path
from src.run_multihoprag_llm_resume_eval import is_complete, parse_order


class OfflineRegressionTests(unittest.TestCase):
    def test_corpus_chunk_ids_round_trip_through_kb_parser(self):
        records, _ = build_corpus([{"title": "Example", "body": "Evidence text", "url": "https://example.org"}])
        self.assertEqual(records[0]["id"], "题_0_块_0")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kb.md"
            path.write_text(f"[{records[0]['id']}] {records[0]['text']}", encoding="utf-8")
            self.assertEqual(read_kb_chunks(path), ([records[0]["id"]], [records[0]["text"]]))

    def test_kb_preserves_unicode_order_and_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kb.md"
            path.write_text("intro\n---\n[题_7_块_2] 中文证据\nline two\n---\n[题_1_块_0] English", encoding="utf-8")
            self.assertEqual(read_kb_chunks(path), (["题_7_块_2", "题_1_块_0"], ["中文证据\nline two", "English"]))

    def test_duplicate_chunks_are_not_silently_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kb.md"
            path.write_text("[题_0_块_0] A\n---\n[题_0_块_0] B", encoding="utf-8")
            self.assertEqual(read_kb_chunks(path), (["题_0_块_0", "题_0_块_0"], ["A", "B"]))

    def test_llm_order_keeps_valid_unique_ids_then_fills(self):
        order, ok, error = parse_order("3, 3, 99, 1", 6)
        self.assertEqual(order, [3, 1, 0, 2, 4])
        self.assertTrue(ok)
        self.assertEqual(error, "")

    def test_invalid_llm_response_is_not_complete(self):
        order, ok, error = parse_order("I cannot rank these passages.", 6)
        self.assertEqual(order, [])
        self.assertFalse(ok)
        self.assertTrue(error)
        self.assertFalse(is_complete({"ids": list(range(5)), "parse_ok": False}))
        self.assertFalse(is_complete({"ids": [0], "parse_ok": True}))

    def test_generation_normalization_and_mismatch(self):
        self.assertEqual(calculate_item_metrics("The Paris.", "Paris"), (1.0, 1.0, 1.0, 1))
        self.assertEqual(calculate_item_metrics("London", "Paris"), (0.0, 0.0, 0.0, 0))

    def test_relative_model_override_uses_project_root(self):
        with patch.dict(os.environ, {"TEST_MODEL_DIR": "custom/model"}):
            self.assertEqual(configured_path("TEST_MODEL_DIR", Path("fallback")), PROJECT_ROOT / "custom/model")

    def test_empty_model_override_preserves_default(self):
        with patch.dict(os.environ, {"TEST_MODEL_DIR": ""}):
            self.assertEqual(configured_path("TEST_MODEL_DIR", Path("fallback")), Path("fallback"))

    def test_launcher_lists_without_site_packages(self):
        result = subprocess.run([sys.executable, "-S", "-m", "src", "--list"], cwd=PROJECT_ROOT,
                                text=True, capture_output=True, check=True)
        self.assertIn("rerank-multihoprag", result.stdout)
        self.assertIn("build-multihoprag", result.stdout)


if __name__ == "__main__":
    unittest.main()
