#!/usr/bin/env python3
"""
MCP Server for Claude CLI Integration
Allows Claude Code to send plans to the Claude CLI for review
"""

import asyncio
import json
import os
import re
import shlex
import sys
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession


# Configuration helpers
def _get_env(primary: str, fallback: Optional[str] = None, default: str = "") -> str:
    """
    Return the value of an environment variable, preferring the Claude-prefixed
    key while still honoring legacy Codex names. Empty strings are treated as
    intentional values and therefore bypass the fallback.
    """
    value = os.environ.get(primary)
    if value is not None:
        return value
    if fallback:
        fallback_value = os.environ.get(fallback)
        if fallback_value is not None:
            return fallback_value
    return default


def _get_env_bool(primary: str, fallback: Optional[str], default: str) -> bool:
    return _get_env(primary, fallback, default).lower() == "true"


def _get_env_args(primary: str, fallback: Optional[str], default: str) -> list[str]:
    raw_value = _get_env(primary, fallback, default)
    return shlex.split(raw_value)


# Configuration
# Set to False to require approval for Claude operations (more secure)
# Set to True for automated execution without approval (less secure)
BYPASS_APPROVALS = _get_env_bool(
    "CLAUDE_BYPASS_APPROVALS", "CODEX_BYPASS_APPROVALS", "false"
)

# Set to False to show full Claude output including prompt templates
# Set to True to clean output and show only Claude's actual response (default)
CLEAN_OUTPUT = _get_env_bool("CLAUDE_CLEAN_OUTPUT", "CODEX_CLEAN_OUTPUT", "true")

# Base command configuration
CLAUDE_BINARY = _get_env("CLAUDE_CLI_BINARY", "CODEX_CLI_BINARY", "claude")
CLAUDE_AUTO_SUBCOMMAND = _get_env("CLAUDE_AUTO_SUBCOMMAND", "CODEX_AUTO_SUBCOMMAND", "auto")
CLAUDE_AUTO_ARGS = _get_env_args("CLAUDE_AUTO_ARGS", "CODEX_AUTO_ARGS", "")
CLAUDE_AUTO_STANDARD_ARGS = _get_env_args(
    "CLAUDE_AUTO_STANDARD_ARGS",
    "CODEX_AUTO_STANDARD_ARGS",
    "",
)
CLAUDE_AUTO_BYPASS_ARGS = _get_env_args(
    "CLAUDE_AUTO_BYPASS_ARGS",
    "CODEX_AUTO_BYPASS_ARGS",
    "--dangerously-skip-permissions",
)
CLAUDE_AUTO_STDIN_ARG = _get_env("CLAUDE_AUTO_STDIN_ARG", "CODEX_AUTO_STDIN_ARG", "-")
NOTIFICATION_PREVIEW_CHARS = int(
    _get_env("CLAUDE_NOTIFICATION_PREVIEW_CHARS", "CODEX_NOTIFICATION_PREVIEW_CHARS", "600")
)

# Review Memory - stores past findings to avoid repeating the same issues
REVIEW_MEMORY_FILE = Path("data/review_memory.jsonl")
REVIEW_MEMORY_MAX_ENTRIES = 100  # Keep only the most recent 100 reviews to prevent endless growth


# Task Management
@dataclass
class Task:
    """Represents a background Claude task"""
    task_id: str
    status: str  # pending, running, completed, failed, cancelled
    command: str
    args: dict
    start_time: datetime
    context: Optional[Context[ServerSession, None]] = field(default=None, repr=False)
    completion_time: Optional[datetime] = None
    result: Optional[str] = None
    error: Optional[str] = None
    async_task: Optional[asyncio.Task] = field(default=None, repr=False)
    process: Optional[asyncio.subprocess.Process] = field(default=None, repr=False)


# Global task storage
tasks: dict[str, Task] = {}


def _task_display_name(task: Task) -> str:
    if task.command == "review":
        return "Claude review"
    if isinstance(task.args, dict):
        cmd = task.args.get("command")
        if cmd:
            return f"Claude command '{cmd}'"
    return "Claude task"


def _summarize_error(message: Optional[str], limit: int = 200) -> str:
    if not message:
        return "See get_task_result for details."
    cleaned = message.strip().splitlines()[0]
    if len(cleaned) > limit:
        return cleaned[:limit - 3] + "..."
    return cleaned


def _truncate_output(output: str, limit: int) -> str:
    """
    Return a truncated version of output suitable for inline notifications.
    """
    if not output:
        return ""
    sanitized = output.strip()
    if len(sanitized) <= limit:
        return sanitized
    return sanitized[:limit - 3] + "..."


def _sanitize_notification_text(text: str) -> str:
    """
    Best-effort scrubbing of sensitive paths or tokens before including text
    in inline notifications.
    """
    if not text:
        return ""
    sanitized = text.replace(str(Path.home()), "~")
    return sanitized


async def _send_context_message(task: Task, level: str, message: str):
    if not task.context:
        return

    handler = getattr(task.context, level, None)
    if not callable(handler):
        handler = task.context.info

    try:
        await asyncio.shield(handler(message))
    except Exception as e:
        print(f"[ERROR] Failed to send {level} notification: {e}", file=sys.stderr, flush=True)


async def _emit_task_notification(task: Task):
    if not task.context:
        return

    prefix = "[claude-async]"
    label = _task_display_name(task)

    if task.status == "completed":
        message = f"{prefix} {label} finished (task {task.task_id})."
        preview = _truncate_output(_sanitize_notification_text(task.result or ""), NOTIFICATION_PREVIEW_CHARS)
        if preview:
            message += f"\n\nPreview:\n{preview}"
        message += "\n\nRun get_task_result to view the full Claude output."
        level = "info"
        await _send_context_message(task, level, message)
    elif task.status == "failed":
        summary = _summarize_error(task.error)
        message = f"{prefix} {label} failed (task {task.task_id}). {summary}"
        preview_source = task.error or task.result or ""
        preview = _truncate_output(_sanitize_notification_text(preview_source), NOTIFICATION_PREVIEW_CHARS)
        if preview:
            message += f"\n\nError preview:\n{preview}"
        message += "\n\nRun get_task_result to inspect the full error output."
        level = "error"
        await _send_context_message(task, level, message)
    elif task.status == "cancelled":
        message = f"{prefix} {label} was cancelled (task {task.task_id})."
        level = "warning"
        await _send_context_message(task, level, message)


async def cleanup_old_tasks():
    """Background task to clean up completed tasks after 5 minutes"""
    while True:
        try:
            await asyncio.sleep(60)  # Check every minute
            now = datetime.now()
            tasks_to_remove = []

            for task_id, task in tasks.items():
                if task.completion_time and (now - task.completion_time) > timedelta(minutes=5):
                    tasks_to_remove.append(task_id)

            for task_id in tasks_to_remove:
                del tasks[task_id]

        except Exception as e:
            # Log but don't crash the cleanup task
            print(f"Error in cleanup_old_tasks: {e}", flush=True)


# Initialize FastMCP server
mcp = FastMCP("claude-cli-server")


def clean_claude_output(raw_output: str, original_prompt: str = "") -> str:
    """
    Clean Claude CLI output by removing echoed prompt templates.

    Args:
        raw_output: Raw output from Claude CLI
        original_prompt: The prompt we sent to Claude (optional)

    Returns:
        Cleaned output with only the actual response
    """
    if not CLEAN_OUTPUT:
        return raw_output

    cleaned = raw_output

    # Remove the exact prompt template if it appears at the start of output
    # Only remove exact matches to avoid accidentally stripping valid responses
    if original_prompt and cleaned.startswith(original_prompt):
        cleaned = cleaned[len(original_prompt):].lstrip()

    # Remove extra blank lines (3+ consecutive newlines)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    # Remove leading/trailing whitespace
    cleaned = cleaned.strip()

    return cleaned


def _build_claude_env(base_cwd: Optional[str] = None) -> dict[str, str]:
    """
    Construct the environment for Claude CLI invocations. Unless
    `CLAUDE_CONFIG_DIR` is explicitly provided, the CLI will fall back to its
    default configuration directory (e.g., `~/.claude`).
    """
    env = os.environ.copy()
    config_dir = env.get("CLAUDE_CONFIG_DIR")

    if config_dir:
        try:
            config_path = Path(config_dir).expanduser()
            if not config_path.is_absolute():
                resolved_base = Path(base_cwd) if base_cwd else Path.cwd()
                config_path = resolved_base / config_path
            config_path = config_path.resolve()
            config_path.mkdir(parents=True, exist_ok=True)
            env["CLAUDE_CONFIG_DIR"] = str(config_path)
        except Exception as exc:
            print(
                f"[WARNING] Unable to prepare CLAUDE_CONFIG_DIR '{config_dir}': {exc}",
                file=sys.stderr,
                flush=True,
            )
            env.pop("CLAUDE_CONFIG_DIR", None)

    return env


def _append_review_memory(review_type: str, files: list[str], result: str, working_directory: str = None):
    """
    Append a review finding to the review memory JSONL file.
    Automatically trims old entries to keep file size bounded.

    Args:
        review_type: Type of review (plan, code, architecture, security)
        files: List of files reviewed
        result: The Claude review output
        working_directory: Working directory of the review
    """
    try:
        # Ensure data directory exists
        REVIEW_MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Create memory entry
        entry = {
            "timestamp": datetime.now().isoformat(),
            "review_type": review_type,
            "files": files or [],
            "working_directory": working_directory or os.getcwd(),
            "result_summary": result[:500] if result else "",  # Store first 500 chars as summary
            "resolved": False
        }

        # Read existing entries (if any) and keep only the most recent ones
        existing_entries = deque(maxlen=REVIEW_MEMORY_MAX_ENTRIES - 1)  # -1 to make room for new entry
        if REVIEW_MEMORY_FILE.exists():
            with open(REVIEW_MEMORY_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            existing_entries.append(line)
                        except Exception:
                            continue

        # Write back the trimmed entries plus the new one
        with open(REVIEW_MEMORY_FILE, 'w', encoding='utf-8') as f:
            for line in existing_entries:
                f.write(line + '\n')
            f.write(json.dumps(entry) + '\n')

    except Exception as e:
        # Don't fail the review if memory logging fails
        print(f"[WARNING] Failed to append to review memory: {e}", file=sys.stderr, flush=True)


def _read_review_memory(file_filter: str = None, review_type: str = None, limit: int = 50) -> list[dict]:
    """
    Read review memory entries from JSONL file efficiently using deque.

    Args:
        file_filter: Optional substring to filter by file path
        review_type: Optional review type to filter by
        limit: Maximum number of entries to return (most recent first)

    Returns:
        List of review memory entries
    """
    try:
        if not REVIEW_MEMORY_FILE.exists():
            return []

        # Use deque with maxlen to efficiently keep only the most recent entries
        entries = deque(maxlen=limit)
        with open(REVIEW_MEMORY_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)

                    # Apply filters
                    if file_filter and not any(file_filter in f for f in entry.get("files", [])):
                        continue
                    if review_type and entry.get("review_type") != review_type:
                        continue

                    entries.append(entry)
                except json.JSONDecodeError:
                    continue

        # Convert to list and return most recent first
        return list(reversed(entries))

    except Exception as e:
        print(f"[WARNING] Failed to read review memory: {e}", file=sys.stderr, flush=True)
        return []


async def _run_claude_review(task_id: str, plan: str, context: str, review_type: str,
                             working_directory: str = None, files: list[str] = None):
    """
    Background coroutine that runs Claude review and updates task status
    """
    task = tasks[task_id]

    try:
        task.status = "running"

        # Construct the review prompt
        if working_directory and files:
            files_list = ", ".join(files)
            review_prompt = f"""{review_type.capitalize()} review of {files_list}:

{plan}

{f"Context: {context}" if context else ""}"""
        elif working_directory:
            review_prompt = f"""{review_type.capitalize()} review:

{plan}

{f"Context: {context}" if context else ""}"""
        else:
            review_prompt = f"""{review_type.capitalize()} review:

{plan}

{f"Context: {context}" if context else ""}"""

        # Build command (without prompt to avoid ARG_MAX)
        cmd = [CLAUDE_BINARY]
        if CLAUDE_AUTO_SUBCOMMAND:
            cmd.append(CLAUDE_AUTO_SUBCOMMAND)

        extra_args = CLAUDE_AUTO_BYPASS_ARGS if BYPASS_APPROVALS else CLAUDE_AUTO_STANDARD_ARGS
        if extra_args:
            cmd.extend(extra_args)

        if CLAUDE_AUTO_ARGS:
            cmd.extend(CLAUDE_AUTO_ARGS)

        if CLAUDE_AUTO_STDIN_ARG:
            cmd.append(CLAUDE_AUTO_STDIN_ARG)

        # Execute Claude CLI with prompt via stdin
        env = _build_claude_env(working_directory)
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_directory if working_directory else None,
            env=env
        )

        task.process = process

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=review_prompt.encode('utf-8')),
                timeout=300
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            task.status = "failed"
            task.error = "Claude CLI review timed out after 300 seconds"
            task.completion_time = datetime.now()
            await _emit_task_notification(task)
            return

        stdout_text = stdout.decode('utf-8') if stdout else ""
        stderr_text = stderr.decode('utf-8') if stderr else ""

        if process.returncode == 0:
            cleaned_output = clean_claude_output(stdout_text, review_prompt)
            task.status = "completed"
            task.result = f"Claude CLI Review Results:\n\n{cleaned_output}"
            task.completion_time = datetime.now()

            # Log to review memory
            _append_review_memory(review_type, files, cleaned_output, working_directory)

            await _emit_task_notification(task)

        else:
            error_output = []
            if stdout_text.strip():
                error_output.append(f"stdout:\n{stdout_text}")
            if stderr_text.strip():
                error_output.append(f"stderr:\n{stderr_text}")

            combined_error = "\n\n".join(error_output) if error_output else "No output"
            task.status = "failed"
            task.error = f"Claude CLI returned an error (exit code {process.returncode}):\n\n{combined_error}"
            task.completion_time = datetime.now()
            await _emit_task_notification(task)

    except asyncio.CancelledError:
        # Task was cancelled
        task.status = "cancelled"
        task.completion_time = datetime.now()
        if task.process:
            try:
                task.process.kill()
                await task.process.wait()
            except:
                pass
        await _emit_task_notification(task)
        raise

    except FileNotFoundError as e:
        if e.filename == CLAUDE_BINARY:
            task.error = f"Error: '{CLAUDE_BINARY}' command not found. Please ensure Claude CLI is installed and in your PATH."
        else:
            task.error = f"Error: Path not found or inaccessible: {e.filename or working_directory or 'unknown'}"
        task.status = "failed"
        task.completion_time = datetime.now()
        await _emit_task_notification(task)

    except Exception as e:
        task.status = "failed"
        task.error = f"Error executing Claude CLI: {str(e)}"
        task.completion_time = datetime.now()
        await _emit_task_notification(task)


@mcp.tool()
async def start_review(
    plan: str,
    ctx: Context[ServerSession, None],
    context: str = "",
    review_type: str = "plan",
    working_directory: str = None,
    files: list[str] = None
) -> str:
    """
    Start a background Claude review and return task ID immediately.

    Args:
        plan: The plan, code, or changes to review
        context: Additional context about the task or goal (optional)
        review_type: Type of review to perform (plan, code, architecture, security)
        working_directory: Working directory for Claude CLI to use (optional)
        files: Specific files to review (optional)
    """
    # Validate inputs
    if not plan or not plan.strip():
        return "Error: 'plan' parameter is required and cannot be empty."

    if working_directory and not working_directory.strip():
        return "Error: 'working_directory' cannot be empty if provided."

    # Create task
    task_id = str(uuid.uuid4())
    task = Task(
        task_id=task_id,
        status="pending",
        command="review",
        args={
            "plan": plan,
            "context": context,
            "review_type": review_type,
            "working_directory": working_directory,
            "files": files
        },
        start_time=datetime.now(),
        context=ctx  # Store context for notifications
    )

    tasks[task_id] = task

    # Launch background task
    task.async_task = asyncio.create_task(_run_claude_review(
        task_id, plan, context, review_type, working_directory, files
    ))

    return f"Claude review started. Task ID: {task_id}\n\nUse get_task_status or get_task_result to check progress."


async def _run_claude_command(task_id: str, command: str, args: list[str]):
    """
    Background coroutine that runs Claude CLI command and updates task status
    """
    task = tasks[task_id]

    try:
        task.status = "running"

        # Track whether this is a prompt
        is_prompt = False
        prompt_text = ""

        # Normalize args
        args = list(args or [])

        if args and not BYPASS_APPROVALS:
            # Reject attempts to inject bypass flags when approvals must remain enforced
            blocked_flag_names = {
                "--dangerously-bypass-approvals-and-sandbox",
                "--dangerously-skip-permissions",
            }
            sanitized_args: list[str] = []
            rejected_args: list[str] = []

            for arg in args:
                flag_name = arg.split("=", 1)[0]
                if flag_name in blocked_flag_names:
                    rejected_args.append(arg)
                else:
                    sanitized_args.append(arg)

            if rejected_args:
                task.status = "failed"
                task.error = (
                    "Security restriction: the following flags are not permitted when "
                    "CLAUDE_BYPASS_APPROVALS is disabled: " + ", ".join(rejected_args)
                )
                task.completion_time = datetime.now()
                await _emit_task_notification(task)
                return

            args = sanitized_args

        # Build command
        stdin_input = None
        # Check if command looks like a Claude subcommand (single word, no spaces)
        # If it has spaces or looks like natural language, treat as a prompt
        if command and not ' ' in command and command.isascii() and command.strip() == command:
            # Likely a subcommand - pass through to Claude CLI
            full_command = [CLAUDE_BINARY, command] + args
        else:
            # Treat as a prompt for the auto workflow (use stdin to avoid ARG_MAX)
            is_prompt = True
            prompt_text = command
            stdin_input = command.encode('utf-8')

            full_command = [CLAUDE_BINARY]
            if CLAUDE_AUTO_SUBCOMMAND:
                full_command.append(CLAUDE_AUTO_SUBCOMMAND)

            extra_args = CLAUDE_AUTO_BYPASS_ARGS if BYPASS_APPROVALS else CLAUDE_AUTO_STANDARD_ARGS
            if extra_args:
                full_command.extend(extra_args)

            if CLAUDE_AUTO_ARGS:
                full_command.extend(CLAUDE_AUTO_ARGS)

            if CLAUDE_AUTO_STDIN_ARG:
                full_command.append(CLAUDE_AUTO_STDIN_ARG)

            full_command.extend(args)

        # Execute Claude CLI
        env = _build_claude_env()
        process = await asyncio.create_subprocess_exec(
            *full_command,
            stdin=asyncio.subprocess.PIPE if stdin_input else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )

        task.process = process

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=stdin_input),
                timeout=300
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            task.status = "failed"
            task.error = "Claude CLI command timed out after 300 seconds"
            task.completion_time = datetime.now()
            await _emit_task_notification(task)
            return

        stdout_text = stdout.decode('utf-8') if stdout else ""
        stderr_text = stderr.decode('utf-8') if stderr else ""

        if process.returncode == 0:
            # Clean the output if this was a prompt
            if is_prompt:
                cleaned_output = clean_claude_output(stdout_text, prompt_text)
                task.result = f"Claude CLI Output:\n\n{cleaned_output}"
            else:
                task.result = f"Claude CLI Output:\n\n{stdout_text}"

            task.status = "completed"
            task.completion_time = datetime.now()
            await _emit_task_notification(task)

        else:
            error_output = []
            if stdout_text.strip():
                error_output.append(f"stdout:\n{stdout_text}")
            if stderr_text.strip():
                error_output.append(f"stderr:\n{stderr_text}")

            combined_error = "\n\n".join(error_output) if error_output else "No output"
            task.status = "failed"
            task.error = f"Claude CLI Error (exit code {process.returncode}):\n\n{combined_error}"
            task.completion_time = datetime.now()
            await _emit_task_notification(task)

    except asyncio.CancelledError:
        # Task was cancelled
        task.status = "cancelled"
        task.completion_time = datetime.now()
        if task.process:
            try:
                task.process.kill()
                await task.process.wait()
            except:
                pass
        await _emit_task_notification(task)
        raise

    except FileNotFoundError as e:
        if e.filename == CLAUDE_BINARY:
            task.error = f"Error: '{CLAUDE_BINARY}' command not found. Please ensure Claude CLI is installed and in your PATH."
        else:
            task.error = f"Error: Path not found or inaccessible: {e.filename or 'unknown'}"
        task.status = "failed"
        task.completion_time = datetime.now()
        await _emit_task_notification(task)

    except Exception as e:
        task.status = "failed"
        task.error = f"Error executing Claude CLI: {str(e)}"
        task.completion_time = datetime.now()
        await _emit_task_notification(task)


@mcp.tool()
async def start_claude_command(
    command: str,
    ctx: Context[ServerSession, None],
    args: list[str] = None
) -> str:
    """
    Start a background Claude CLI command and return task ID immediately.

    Args:
        command: The Claude CLI command to execute
        args: Arguments to pass to the command
    """
    # Validate inputs
    if not command or not command.strip():
        return "Error: 'command' parameter is required and cannot be empty."

    # Create task
    task_id = str(uuid.uuid4())
    task = Task(
        task_id=task_id,
        status="pending",
        command="execute",
        args={
            "command": command,
            "args": args or []
        },
        start_time=datetime.now(),
        context=ctx  # Store context for notifications
    )

    tasks[task_id] = task

    # Launch background task
    task.async_task = asyncio.create_task(_run_claude_command(
        task_id, command, args or []
    ))

    return f"Claude command started. Task ID: {task_id}\n\nUse get_task_status or get_task_result to check progress."


@mcp.tool()
async def get_task_status(task_id: str) -> str:
    """
    Get the status of a background task.

    Args:
        task_id: The task ID returned by start_review or start_claude_command
    """
    if not task_id or task_id not in tasks:
        return f"Error: Task '{task_id}' not found. It may have expired (tasks are kept for 5 minutes after completion)."

    task = tasks[task_id]

    status_info = f"""Task ID: {task_id}
Status: {task.status}
Command: {task.command}
Started: {task.start_time.strftime('%Y-%m-%d %H:%M:%S')}"""

    if task.completion_time:
        duration = (task.completion_time - task.start_time).total_seconds()
        status_info += f"\nCompleted: {task.completion_time.strftime('%Y-%m-%d %H:%M:%S')}"
        status_info += f"\nDuration: {duration:.1f} seconds"

    if task.status == "pending":
        status_info += "\n\nTask is queued and will start soon."
    elif task.status == "running":
        elapsed = (datetime.now() - task.start_time).total_seconds()
        status_info += f"\n\nTask is running (elapsed: {elapsed:.1f} seconds)."
    elif task.status == "completed":
        status_info += "\n\nTask completed successfully. Use get_task_result to retrieve the output."
    elif task.status == "failed":
        status_info += f"\n\nTask failed: {task.error}"
    elif task.status == "cancelled":
        status_info += "\n\nTask was cancelled."

    return status_info


@mcp.tool()
async def get_task_result(task_id: str) -> str:
    """
    Get the result of a completed background task.

    Args:
        task_id: The task ID returned by start_review or start_claude_command
    """
    if not task_id or task_id not in tasks:
        return f"Error: Task '{task_id}' not found. It may have expired (tasks are kept for 5 minutes after completion)."

    task = tasks[task_id]

    if task.status == "pending":
        return f"Task {task_id} is still pending. Use get_task_status to check its status."
    elif task.status == "running":
        elapsed = (datetime.now() - task.start_time).total_seconds()
        return f"Task {task_id} is still running (elapsed: {elapsed:.1f} seconds). Use get_task_status to check its status."
    elif task.status == "completed":
        return task.result or "Task completed but no output was captured."
    elif task.status == "failed":
        return f"Task {task_id} failed:\n\n{task.error}"
    elif task.status == "cancelled":
        return f"Task {task_id} was cancelled."
    else:
        return f"Task {task_id} has unknown status: {task.status}"


@mcp.tool()
async def wait_for_task(task_id: str, timeout: int = 300) -> str:
    """
    Wait for a background task to complete and return its result.
    Blocks until the task finishes or timeout is reached.

    Args:
        task_id: The task ID returned by start_review or start_claude_command
        timeout: Maximum seconds to wait (default: 300)
    """
    if not task_id or task_id not in tasks:
        return f"Error: Task '{task_id}' not found. It may have expired (tasks are kept for 5 minutes after completion)."

    task = tasks[task_id]

    # If task is already done, return immediately
    if task.status in ["completed", "failed", "cancelled"]:
        if task.status == "completed":
            return f"Task completed!\n\n{task.result or 'No output was captured.'}"
        elif task.status == "failed":
            return f"Task failed:\n\n{task.error}"
        else:
            return f"Task was cancelled."

    # Wait for the asyncio task to complete
    # Use shield to prevent timeout from cancelling the background task
    if task.async_task:
        try:
            await asyncio.wait_for(asyncio.shield(task.async_task), timeout=timeout)
        except asyncio.TimeoutError:
            # Task keeps running in background despite timeout
            elapsed = (datetime.now() - task.start_time).total_seconds()
            return f"Task is still running after {timeout}s timeout (total elapsed: {elapsed:.1f}s). Use get_task_result to check later."
        except asyncio.CancelledError:
            return f"Task was cancelled."
        except Exception as e:
            return f"Task failed with exception: {str(e)}"

    # Return the result based on final status
    if task.status == "completed":
        return f"Task completed!\n\n{task.result or 'No output was captured.'}"
    elif task.status == "failed":
        return f"Task failed:\n\n{task.error}"
    elif task.status == "cancelled":
        return f"Task was cancelled."
    else:
        return f"Task ended with unexpected status: {task.status}"


@mcp.tool()
async def cancel_task(task_id: str) -> str:
    """
    Cancel a running background task.

    Args:
        task_id: The task ID to cancel
    """
    if not task_id or task_id not in tasks:
        return f"Error: Task '{task_id}' not found. It may have expired (tasks are kept for 5 minutes after completion)."

    task = tasks[task_id]

    if task.status in ["completed", "failed", "cancelled"]:
        return f"Task {task_id} is already {task.status} and cannot be cancelled."

    try:
        # Kill the subprocess if it exists
        if task.process:
            try:
                task.process.kill()
                await task.process.wait()
            except Exception as e:
                # Process might already be dead
                pass

        # Cancel the asyncio task
        if task.async_task:
            task.async_task.cancel()
            try:
                await task.async_task
            except asyncio.CancelledError:
                pass

        # Update task status
        task.status = "cancelled"
        task.completion_time = datetime.now()

        # Emit notification for cancellation
        await _emit_task_notification(task)

        return f"Task {task_id} has been cancelled."

    except Exception as e:
        return f"Error cancelling task {task_id}: {str(e)}"


@mcp.tool()
async def query_review_memory(
    file_filter: str = None,
    review_type: str = None,
    limit: int = 10
) -> str:
    """
    Query past Claude review findings from review memory.

    Args:
        file_filter: Optional substring to filter by file path (e.g., "server.py")
        review_type: Optional review type to filter by (plan, code, architecture, security)
        limit: Maximum number of entries to return (default: 10, max: 50)

    Returns:
        Formatted list of past review findings
    """
    # Clamp limit to reasonable range
    limit = max(1, min(limit, 50))

    entries = _read_review_memory(file_filter, review_type, limit)

    if not entries:
        filters = []
        if file_filter:
            filters.append(f"file containing '{file_filter}'")
        if review_type:
            filters.append(f"type '{review_type}'")

        filter_text = " and ".join(filters) if filters else "any criteria"
        return f"No review memory entries found matching {filter_text}."

    # Format results
    result_lines = [f"Found {len(entries)} review memory entries:\n"]

    for i, entry in enumerate(entries, 1):
        timestamp = entry.get("timestamp", "unknown")
        review_type = entry.get("review_type", "unknown")
        files = entry.get("files", [])
        files_text = ", ".join(files) if files else "general review"
        summary = entry.get("result_summary", "")
        resolved = entry.get("resolved", False)
        status = "✓ resolved" if resolved else "○ unresolved"

        result_lines.append(f"\n{i}. [{timestamp}] {review_type.upper()} review - {status}")
        result_lines.append(f"   Files: {files_text}")
        if summary:
            # Truncate summary to first 200 chars for display
            display_summary = summary[:200]
            if len(summary) > 200:
                display_summary += "..."
            result_lines.append(f"   Summary: {display_summary}")

    return "\n".join(result_lines)


@mcp.tool()
async def clear_review_memory() -> str:
    """
    Clear all review memory entries with timestamped backup.

    Returns:
        Confirmation message
    """
    try:
        if REVIEW_MEMORY_FILE.exists():
            # Backup with timestamp to avoid overwriting previous backups
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = REVIEW_MEMORY_FILE.with_suffix(f'.jsonl.backup.{timestamp}')
            REVIEW_MEMORY_FILE.rename(backup_path)
            return f"Review memory cleared. Backup saved to: {backup_path}"
        else:
            return "Review memory is already empty (file doesn't exist)."
    except Exception as e:
        return f"Error clearing review memory: {str(e)}"


# FastMCP auto-discovers tools via @mcp.tool() decorators
# No need for manual list_tools() or call_tool() handlers


if __name__ == "__main__":
    # Start cleanup task in background
    async def run_with_cleanup():
        cleanup_task = asyncio.create_task(cleanup_old_tasks())
        try:
            # Use run_stdio_async to avoid double event loop
            await mcp.run_stdio_async()
        finally:
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass

    asyncio.run(run_with_cleanup())
