import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import claude_mcp_server as server


class EnvHelpersTestCase(unittest.TestCase):
    def test_get_env_prefers_primary_even_if_empty(self):
        with mock.patch.dict(os.environ, {"PRIMARY": "", "FALLBACK": "value"}, clear=True):
            value = server._get_env("PRIMARY", "FALLBACK", "default")
            self.assertEqual(value, "")

    def test_get_env_uses_fallback_when_primary_missing(self):
        with mock.patch.dict(os.environ, {"FALLBACK": "value"}, clear=True):
            value = server._get_env("PRIMARY", "FALLBACK", "default")
            self.assertEqual(value, "value")

    def test_get_env_args_split_string(self):
        with mock.patch.dict(os.environ, {"PRIMARY": "--one two"}, clear=True):
            args = server._get_env_args("PRIMARY", None, "")
            self.assertEqual(args, ["--one", "two"])

    def test_build_claude_env_resolves_relative_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rel_dir = "config"
            with mock.patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": rel_dir}, clear=False):
                env = server._build_claude_env(tmpdir)
                resolved = Path(env["CLAUDE_CONFIG_DIR"])
                self.assertTrue(resolved.is_absolute())
                base = Path(tmpdir).resolve()
                self.assertTrue(resolved.is_relative_to(base))
                self.assertTrue(resolved.exists())

    def test_build_claude_env_suppresses_invalid_dir(self):
        with mock.patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": "/root/invalid/path"}, clear=False):
            with mock.patch("claude_mcp_server.Path.mkdir", side_effect=OSError("fail")):
                env = server._build_claude_env()
                self.assertNotIn("CLAUDE_CONFIG_DIR", env)

    def test_sanitize_notification_masks_home_directory(self):
        home = str(Path.home())
        text = f"{home}/secrets/token"
        masked = server._sanitize_notification_text(text)
        self.assertNotIn(home, masked)
        self.assertIn("~", masked)


if __name__ == "__main__":
    unittest.main()
