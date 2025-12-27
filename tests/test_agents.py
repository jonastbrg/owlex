"""
Tests for agent CLI command construction.
"""

from unittest.mock import patch

import pytest
from owlex.agents.aider import AiderRunner
from owlex.agents.codex import CodexRunner
from owlex.agents.gemini import GeminiRunner


class TestCodexRunner:
    """Tests for Codex CLI command construction."""

    @pytest.fixture
    def runner(self):
        return CodexRunner()

    def test_exec_basic_command(self, runner):
        """Should build basic exec command."""
        with patch("owlex.agents.codex.config") as mock_config:
            mock_config.codex.bypass_approvals = False

            cmd = runner.build_exec_command(prompt="Hello")

            assert cmd.command[0] == "codex"
            assert cmd.command[1] == "exec"
            assert "--skip-git-repo-check" in cmd.command
            assert "--full-auto" in cmd.command
            assert "-" in cmd.command  # stdin marker
            assert cmd.prompt == "Hello"
            assert cmd.output_prefix == "Codex Output"
            assert cmd.stream is True

    def test_exec_with_working_directory(self, runner):
        """Should add --cd flag for working directory."""
        with patch("owlex.agents.codex.config") as mock_config:
            mock_config.codex.bypass_approvals = False

            cmd = runner.build_exec_command(
                prompt="Hello",
                working_directory="/path/to/dir"
            )

            assert "--cd" in cmd.command
            idx = cmd.command.index("--cd")
            assert cmd.command[idx + 1] == "/path/to/dir"

    def test_exec_with_search_enabled(self, runner):
        """Should add --enable web_search_request when search enabled."""
        with patch("owlex.agents.codex.config") as mock_config:
            mock_config.codex.bypass_approvals = False

            cmd = runner.build_exec_command(
                prompt="Hello",
                enable_search=True
            )

            assert "--enable" in cmd.command
            idx = cmd.command.index("--enable")
            assert cmd.command[idx + 1] == "web_search_request"

    def test_exec_without_search(self, runner):
        """Should not include search flag when disabled."""
        with patch("owlex.agents.codex.config") as mock_config:
            mock_config.codex.bypass_approvals = False

            cmd = runner.build_exec_command(
                prompt="Hello",
                enable_search=False
            )

            assert "--enable" not in cmd.command

    def test_exec_bypass_approvals(self, runner):
        """Should use bypass flag when config enables it."""
        with patch("owlex.agents.codex.config") as mock_config:
            mock_config.codex.bypass_approvals = True

            cmd = runner.build_exec_command(prompt="Hello")

            assert "--dangerously-bypass-approvals-and-sandbox" in cmd.command
            assert "--full-auto" not in cmd.command

    def test_resume_basic_command(self, runner):
        """Should build basic resume command."""
        with patch("owlex.agents.codex.config") as mock_config:
            mock_config.codex.bypass_approvals = False

            cmd = runner.build_resume_command(
                session_ref="abc123",
                prompt="Continue"
            )

            assert "resume" in cmd.command
            assert "abc123" in cmd.command
            assert "-" in cmd.command
            assert cmd.prompt == "Continue"
            assert cmd.stream is False  # Resume uses non-streaming

    def test_resume_last_session(self, runner):
        """Should use --last flag for last session."""
        with patch("owlex.agents.codex.config") as mock_config:
            mock_config.codex.bypass_approvals = False

            cmd = runner.build_resume_command(
                session_ref="--last",
                prompt="Continue"
            )

            assert "--last" in cmd.command

    def test_resume_with_search(self, runner):
        """Resume should also support search flag."""
        with patch("owlex.agents.codex.config") as mock_config:
            mock_config.codex.bypass_approvals = False

            cmd = runner.build_resume_command(
                session_ref="abc123",
                prompt="Continue",
                enable_search=True
            )

            assert "--enable" in cmd.command

    def test_not_found_hint(self, runner):
        """Should include helpful installation hint."""
        with patch("owlex.agents.codex.config") as mock_config:
            mock_config.codex.bypass_approvals = False

            cmd = runner.build_exec_command(prompt="Hello")

            assert "Codex CLI" in cmd.not_found_hint

    def test_resume_rejects_flag_injection(self, runner):
        """Should reject session_ref starting with dash to prevent flag injection."""
        with patch("owlex.agents.codex.config") as mock_config:
            mock_config.codex.bypass_approvals = False

            with pytest.raises(ValueError) as exc_info:
                runner.build_resume_command(
                    session_ref="--malicious-flag",
                    prompt="Hello"
                )

            assert "cannot start with '-'" in str(exc_info.value)

    def test_resume_rejects_single_dash(self, runner):
        """Should reject session_ref that is just a dash."""
        with patch("owlex.agents.codex.config") as mock_config:
            mock_config.codex.bypass_approvals = False

            with pytest.raises(ValueError) as exc_info:
                runner.build_resume_command(
                    session_ref="-",
                    prompt="Hello"
                )

            assert "cannot start with '-'" in str(exc_info.value)

    def test_resume_accepts_valid_uuid(self, runner):
        """Should accept valid UUID-like session IDs."""
        with patch("owlex.agents.codex.config") as mock_config:
            mock_config.codex.bypass_approvals = False

            cmd = runner.build_resume_command(
                session_ref="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                prompt="Hello"
            )

            assert "a1b2c3d4-e5f6-7890-abcd-ef1234567890" in cmd.command


class TestGeminiRunner:
    """Tests for Gemini CLI command construction."""

    @pytest.fixture
    def runner(self):
        return GeminiRunner()

    def test_exec_basic_command(self, runner):
        """Should build basic exec command."""
        with patch("owlex.agents.gemini.config") as mock_config:
            mock_config.gemini.yolo_mode = False

            cmd = runner.build_exec_command(prompt="Hello")

            assert cmd.command[0] == "gemini"
            assert "Hello" in cmd.command  # Prompt is CLI arg, not stdin
            assert cmd.prompt == ""  # Empty because prompt is in command
            assert cmd.output_prefix == "Gemini Output"
            assert cmd.stream is True

    def test_exec_with_working_directory(self, runner):
        """Should add --include-directories flag for working directory."""
        with patch("owlex.agents.gemini.config") as mock_config:
            mock_config.gemini.yolo_mode = False

            cmd = runner.build_exec_command(
                prompt="Hello",
                working_directory="/path/to/dir"
            )

            assert "--include-directories" in cmd.command
            idx = cmd.command.index("--include-directories")
            assert cmd.command[idx + 1] == "/path/to/dir"
            assert cmd.cwd == "/path/to/dir"

    def test_exec_yolo_mode(self, runner):
        """Should add yolo approval mode when enabled."""
        with patch("owlex.agents.gemini.config") as mock_config:
            mock_config.gemini.yolo_mode = True

            cmd = runner.build_exec_command(prompt="Hello")

            assert "--approval-mode" in cmd.command
            idx = cmd.command.index("--approval-mode")
            assert cmd.command[idx + 1] == "yolo"

    def test_exec_without_yolo(self, runner):
        """Should not include yolo flag when disabled."""
        with patch("owlex.agents.gemini.config") as mock_config:
            mock_config.gemini.yolo_mode = False

            cmd = runner.build_exec_command(prompt="Hello")

            assert "--approval-mode" not in cmd.command

    def test_resume_basic_command(self, runner):
        """Should build basic resume command."""
        with patch("owlex.agents.gemini.config") as mock_config:
            mock_config.gemini.yolo_mode = False

            cmd = runner.build_resume_command(
                session_ref="latest",
                prompt="Continue"
            )

            assert "-r" in cmd.command
            assert "latest" in cmd.command
            assert "Continue" in cmd.command
            assert cmd.stream is False  # Resume uses non-streaming

    def test_resume_with_yolo(self, runner):
        """Resume should also use yolo mode when enabled."""
        with patch("owlex.agents.gemini.config") as mock_config:
            mock_config.gemini.yolo_mode = True

            cmd = runner.build_resume_command(
                session_ref="latest",
                prompt="Continue"
            )

            assert "--approval-mode" in cmd.command

    def test_not_found_hint(self, runner):
        """Should include helpful installation hint."""
        with patch("owlex.agents.gemini.config") as mock_config:
            mock_config.gemini.yolo_mode = False

            cmd = runner.build_exec_command(prompt="Hello")

            assert "npm install" in cmd.not_found_hint
            assert "@google/gemini-cli" in cmd.not_found_hint

    def test_search_ignored(self, runner):
        """Gemini should accept but ignore enable_search parameter."""
        with patch("owlex.agents.gemini.config") as mock_config:
            mock_config.gemini.yolo_mode = False

            # Should not raise an error
            cmd = runner.build_exec_command(
                prompt="Hello",
                enable_search=True
            )

            # Search is not applicable to Gemini
            assert "--search" not in cmd.command
            assert "--enable" not in cmd.command

    def test_exec_handles_dash_prompt(self, runner):
        """Should handle prompts starting with dash (no -- separator needed for Gemini)."""
        with patch("owlex.agents.gemini.config") as mock_config:
            mock_config.gemini.yolo_mode = False

            cmd = runner.build_exec_command(prompt="-malicious prompt")

            # Gemini doesn't use -- separator (causes stdin wait mode)
            # The prompt is passed as positional argument directly
            assert "-malicious prompt" in cmd.command
            # -- should NOT be in the command (it causes Gemini to wait for stdin)
            assert "--" not in cmd.command

    def test_resume_handles_dash_prompt(self, runner):
        """Resume should handle prompts starting with dash."""
        with patch("owlex.agents.gemini.config") as mock_config:
            mock_config.gemini.yolo_mode = False

            cmd = runner.build_resume_command(
                session_ref="latest",
                prompt="--dangerous"
            )

            # Prompt is passed directly, no -- separator
            assert "--dangerous" in cmd.command


class TestAiderRunner:
    """Tests for Aider CLI command construction."""

    @pytest.fixture
    def runner(self):
        return AiderRunner()

    def test_exec_basic_command(self, runner):
        """Should build basic exec command."""
        with patch("owlex.agents.aider.config") as mock_config:
            mock_config.aider.model = None
            mock_config.aider.dry_run = True
            mock_config.aider.yes_always = True
            mock_config.aider.no_git = False
            mock_config.aider.auto_commits = False

            cmd = runner.build_exec_command(prompt="Fix the bug")

            assert cmd.command[0] == "aider"
            assert "--message" in cmd.command
            assert "Fix the bug" in cmd.command
            assert "--dry-run" in cmd.command  # Read-only by default
            assert "--yes-always" in cmd.command
            assert cmd.prompt == ""  # Prompt is in command
            assert cmd.output_prefix == "Aider Output"
            assert cmd.stream is True

    def test_exec_with_working_directory(self, runner):
        """Should set cwd for working directory."""
        with patch("owlex.agents.aider.config") as mock_config:
            mock_config.aider.model = None
            mock_config.aider.dry_run = True
            mock_config.aider.yes_always = True
            mock_config.aider.no_git = False
            mock_config.aider.auto_commits = False

            cmd = runner.build_exec_command(
                prompt="Hello",
                working_directory="/path/to/dir"
            )

            assert cmd.cwd == "/path/to/dir"

    def test_exec_with_model(self, runner):
        """Should add model flag when configured."""
        with patch("owlex.agents.aider.config") as mock_config:
            mock_config.aider.model = "claude-3-sonnet"
            mock_config.aider.dry_run = True
            mock_config.aider.yes_always = True
            mock_config.aider.no_git = False
            mock_config.aider.auto_commits = False

            cmd = runner.build_exec_command(prompt="Hello")

            assert "--model" in cmd.command
            idx = cmd.command.index("--model")
            assert cmd.command[idx + 1] == "claude-3-sonnet"

    def test_exec_with_no_git(self, runner):
        """Should add --no-git flag when configured."""
        with patch("owlex.agents.aider.config") as mock_config:
            mock_config.aider.model = None
            mock_config.aider.dry_run = True
            mock_config.aider.yes_always = True
            mock_config.aider.no_git = True
            mock_config.aider.auto_commits = False

            cmd = runner.build_exec_command(prompt="Hello")

            assert "--no-git" in cmd.command

    def test_exec_without_yes_always(self, runner):
        """Should not include --yes-always when disabled."""
        with patch("owlex.agents.aider.config") as mock_config:
            mock_config.aider.model = None
            mock_config.aider.dry_run = True
            mock_config.aider.yes_always = False
            mock_config.aider.no_git = False
            mock_config.aider.auto_commits = False

            cmd = runner.build_exec_command(prompt="Hello")

            assert "--yes-always" not in cmd.command

    def test_exec_without_dry_run(self, runner):
        """Should not include --dry-run when disabled."""
        with patch("owlex.agents.aider.config") as mock_config:
            mock_config.aider.model = None
            mock_config.aider.dry_run = False
            mock_config.aider.yes_always = True
            mock_config.aider.no_git = False
            mock_config.aider.auto_commits = False

            cmd = runner.build_exec_command(prompt="Hello")

            assert "--dry-run" not in cmd.command

    def test_exec_with_auto_commits(self, runner):
        """Should not add --no-auto-commits when auto_commits is True."""
        with patch("owlex.agents.aider.config") as mock_config:
            mock_config.aider.model = None
            mock_config.aider.dry_run = True
            mock_config.aider.yes_always = True
            mock_config.aider.no_git = False
            mock_config.aider.auto_commits = True

            cmd = runner.build_exec_command(prompt="Hello")

            assert "--no-auto-commits" not in cmd.command

    def test_resume_delegates_to_exec(self, runner):
        """Resume should delegate to exec (Aider uses history file for context)."""
        with patch("owlex.agents.aider.config") as mock_config:
            mock_config.aider.model = None
            mock_config.aider.dry_run = True
            mock_config.aider.yes_always = True
            mock_config.aider.no_git = False
            mock_config.aider.auto_commits = False

            cmd = runner.build_resume_command(
                session_ref="ignored",
                prompt="Continue working"
            )

            # Resume just calls exec since Aider uses history files
            assert "--message" in cmd.command
            assert "Continue working" in cmd.command

    def test_not_found_hint(self, runner):
        """Should include helpful installation hint."""
        with patch("owlex.agents.aider.config") as mock_config:
            mock_config.aider.model = None
            mock_config.aider.dry_run = True
            mock_config.aider.yes_always = True
            mock_config.aider.no_git = False
            mock_config.aider.auto_commits = False

            cmd = runner.build_exec_command(prompt="Hello")

            assert "aider-chat" in cmd.not_found_hint


class TestAgentInterface:
    """Tests for AgentRunner interface compliance."""

    def test_aider_has_name(self):
        """Aider runner should have name property."""
        runner = AiderRunner()
        assert runner.name == "aider"

    def test_codex_has_name(self):
        """Codex runner should have name property."""
        runner = CodexRunner()
        assert runner.name == "codex"

    def test_gemini_has_name(self):
        """Gemini runner should have name property."""
        runner = GeminiRunner()
        assert runner.name == "gemini"

    def test_aider_has_output_cleaner(self):
        """Aider runner should provide output cleaner."""
        runner = AiderRunner()
        cleaner = runner.get_output_cleaner()
        assert callable(cleaner)

    def test_codex_has_output_cleaner(self):
        """Codex runner should provide output cleaner."""
        runner = CodexRunner()
        cleaner = runner.get_output_cleaner()
        assert callable(cleaner)

    def test_gemini_has_output_cleaner(self):
        """Gemini runner should provide output cleaner."""
        runner = GeminiRunner()
        cleaner = runner.get_output_cleaner()
        assert callable(cleaner)
