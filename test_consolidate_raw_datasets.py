"""Offline CLI-safety tests for raw dataset consolidation."""
import contextlib
import io
import unittest
from unittest.mock import patch

import consolidate_raw_datasets as consolidate


class ConsolidationCliTests(unittest.TestCase):
    def test_help_exits_before_any_filesystem_write(self):
        with patch.object(consolidate.Path, "mkdir") as mkdir, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                consolidate.main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        mkdir.assert_not_called()

    def test_apply_is_required_before_any_filesystem_write(self):
        with patch.object(consolidate.Path, "mkdir") as mkdir, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                consolidate.main([])
        self.assertNotEqual(raised.exception.code, 0)
        mkdir.assert_not_called()


if __name__ == "__main__":
    unittest.main()
