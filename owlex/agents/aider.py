"""
Aider CLI agent runner.
"""

import re
from typing import Callable

from ..config import config
from .base import AgentRunner, AgentCommand


def clean_aider_output(raw_output: str, original_prompt: str = "") -> str:
    """Clean Aider CLI output by removing noise."""
    if not config.aider.clean_output:
        return raw_output
    cleaned = raw_output
    # Remove ANSI escape codes
    cleaned = re.sub(r'\x1b\[[0-9;]*m', '', cleaned)
    # Remove aider startup messages
    cleaned = re.sub(r'^Aider v[\d.]+.*\n', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'^Model:.*\n', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'^Git repo:.*\n', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'^Repo-map:.*\n', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'^Use /help.*\n', '', cleaned, flags=re.MULTILINE)
    # Collapse multiple newlines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


class AiderRunner(AgentRunner):
    """Runner for Aider AI pair programming CLI."""

    @property
    def name(self) -> str:
        return "aider"

    def build_exec_command(
        self,
        prompt: str,
        working_directory: str | None = None,
        enable_search: bool = False,  # Aider doesn't have web search
        **kwargs,
    ) -> AgentCommand:
        """Build command for running Aider with a single message."""
        full_command = ["aider"]

        # Model configuration
        if config.aider.model:
            full_command.extend(["--model", config.aider.model])

        # Read-only mode - don't modify files
        if config.aider.dry_run:
            full_command.append("--dry-run")

        # Auto-accept all changes (only relevant if not dry-run)
        if config.aider.yes_always:
            full_command.append("--yes-always")

        # Disable git operations if configured
        if config.aider.no_git:
            full_command.append("--no-git")

        # Disable auto-commits if configured
        if not config.aider.auto_commits:
            full_command.append("--no-auto-commits")

        # Non-interactive mode with message
        full_command.extend(["--message", prompt])

        return AgentCommand(
            command=full_command,
            prompt="",  # Prompt is in command via --message
            cwd=working_directory,
            output_prefix="Aider Output",
            not_found_hint="Please ensure Aider is installed (uv tool install aider-chat or pip install aider-chat).",
            stream=True,
        )

    def build_resume_command(
        self,
        session_ref: str,
        prompt: str,
        working_directory: str | None = None,
        enable_search: bool = False,
        **kwargs,
    ) -> AgentCommand:
        """Build command for continuing an Aider session.

        Note: Aider doesn't have explicit session resume like Codex/Gemini.
        It uses chat history files. For now, we just run a new message
        in the same working directory (which preserves context via .aider.chat.history.md).
        """
        # Aider maintains context through its chat history file in the working directory
        # So "resuming" is effectively just running another message in the same dir
        return self.build_exec_command(
            prompt=prompt,
            working_directory=working_directory,
            enable_search=enable_search,
            **kwargs,
        )

    def get_output_cleaner(self) -> Callable[[str, str], str]:
        return clean_aider_output
