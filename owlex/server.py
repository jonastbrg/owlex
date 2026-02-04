#!/usr/bin/env python3
"""
MCP Server for Codex CLI and Gemini CLI Integration
Allows Claude Code to start/resume sessions with Codex or Gemini for advice
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime

from pydantic import Field
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession

from .models import TaskResponse, ErrorCode, Agent, Task
from .engine import engine, DEFAULT_TIMEOUT, codex_runner, gemini_runner, opencode_runner, claudeor_runner, grok_runner
from .council import Council
from .config import config
from .roles import get_resolver


# Initialize FastMCP server
mcp = FastMCP("owlex-server")


# === Resources ===

async def _get_cli_version(cmd: str) -> str:
    """Get version string from a CLI tool."""
    try:
        proc = await asyncio.create_subprocess_exec(
            cmd, "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        return stdout.decode().strip().split('\n')[0]
    except Exception:
        return "unknown"


def _get_codex_model() -> str:
    """Get Codex model from config file."""
    import pathlib
    config_path = pathlib.Path.home() / ".codex" / "config.toml"
    try:
        if config_path.exists():
            content = config_path.read_text()
            for line in content.split('\n'):
                if line.startswith('model ='):
                    return line.split('=')[1].strip().strip('"\'')
    except Exception:
        pass
    return "default"


def _get_gemini_model() -> str:
    """Get Gemini model from config."""
    return config.gemini.model


def _get_opencode_model() -> str:
    """Get OpenCode model from env or default."""
    model = os.environ.get("OPENCODE_MODEL", "").strip()
    return model if model else "openrouter/anthropic/claude-sonnet-4"


def _get_grok_model() -> str:
    """Get Grok model from config."""
    return config.grok.model


@mcp.resource("owlex://agents")
async def get_agents() -> str:
    """List available agents and their configuration."""
    excluded = config.council.exclude_agents

    # Query CLI versions in parallel
    codex_ver, gemini_ver, opencode_ver = await asyncio.gather(
        _get_cli_version("codex"),
        _get_cli_version("gemini"),
        _get_cli_version("opencode"),
    )

    agents = {
        "codex": {
            "available": "codex" not in excluded,
            "cli_version": codex_ver,
            "model": _get_codex_model(),
            "description": "Deep reasoning, code review, bug finding",
            "config": {
                "enable_search": config.codex.enable_search,
                "bypass_approvals": config.codex.bypass_approvals,
            }
        },
        "gemini": {
            "available": "gemini" not in excluded,
            "cli_version": gemini_ver,
            "model": config.gemini.model,
            "fallback_model": config.gemini.fallback_model,
            "description": "1M context window, multimodal, large codebases",
            "config": {
                "yolo_mode": config.gemini.yolo_mode,
            }
        },
        "opencode": {
            "available": "opencode" not in excluded,
            "cli_version": opencode_ver,
            "model": _get_opencode_model(),
            "description": "Alternative perspective, configurable models",
            "config": {
                "agent_mode": config.opencode.agent,
            }
        },
        "grok": {
            "available": "grok" not in excluded,
            "cli_version": opencode_ver,  # Uses OpenCode CLI
            "model": _get_grok_model(),
            "code_model": config.grok.code_model,
            "description": "Deliberate contrarian via xAI/Grok, less aligned perspective",
            "config": {
                "agent_mode": config.grok.agent,
            }
        },
    }

    return json.dumps({
        "agents": agents,
        "excluded": list(excluded),
        "default_timeout": config.default_timeout,
    }, indent=2)


@mcp.resource("owlex://council/status")
def get_council_status() -> str:
    """Get status of running council deliberations."""
    council_tasks = []

    for task_id, task in engine.tasks.items():
        if task.command == "council_ask":
            elapsed = (datetime.now() - task.start_time).total_seconds()
            council_tasks.append({
                "task_id": task_id,
                "status": task.status,
                "elapsed_seconds": round(elapsed, 1),
                "prompt": task.args.get("prompt", "")[:100] + "..." if len(task.args.get("prompt", "")) > 100 else task.args.get("prompt", ""),
                "deliberate": task.args.get("deliberate", True),
                "critique": task.args.get("critique", False),
            })

    # Sort by most recent first
    council_tasks.sort(key=lambda x: x["elapsed_seconds"])

    running = [t for t in council_tasks if t["status"] == "running"]
    pending = [t for t in council_tasks if t["status"] == "pending"]

    return json.dumps({
        "running_count": len(running),
        "pending_count": len(pending),
        "total_count": len(council_tasks),
        "running": running,
        "pending": pending,
        "recent": council_tasks[:5],
    }, indent=2)


def _log(msg: str):
    """Log progress to stderr for CLI visibility."""
    print(msg, file=sys.stderr, flush=True)


def _validate_working_directory(working_directory: str | None) -> tuple[str | None, str | None]:
    """Validate and expand working directory. Returns (expanded_path, error_message)."""
    if not working_directory:
        return None, None
    expanded = os.path.expanduser(working_directory)
    if not os.path.isdir(expanded):
        return None, f"working_directory '{working_directory}' does not exist or is not a directory."
    return expanded, None


# === Codex Tools ===

@mcp.tool()
async def start_codex_session(
    ctx: Context[ServerSession, None],
    prompt: str = Field(description="The question or request to send"),
    working_directory: str | None = Field(default=None, description="Working directory for Codex (--cd flag)"),
    enable_search: bool = Field(default=True, description="Enable web search (--search flag)")
) -> dict:
    """Start a new Codex session (no prior context)."""
    if not prompt or not prompt.strip():
        return TaskResponse(success=False, error="'prompt' parameter is required.", error_code=ErrorCode.INVALID_ARGS).model_dump()

    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return TaskResponse(success=False, error=error, error_code=ErrorCode.INVALID_ARGS).model_dump()

    task = engine.create_task(
        command=f"{Agent.CODEX.value}_exec",
        args={"prompt": prompt.strip(), "working_directory": working_directory, "enable_search": enable_search},
        context=ctx,
    )

    task.async_task = asyncio.create_task(engine.run_agent(
        task, codex_runner, mode="exec",
        prompt=prompt.strip(), working_directory=working_directory, enable_search=enable_search
    ))

    return TaskResponse(
        success=True,
        task_id=task.task_id,
        status=task.status,
        message="Codex session started. Use wait_for_task to get result.",
    ).model_dump()


@mcp.tool()
async def resume_codex_session(
    ctx: Context[ServerSession, None],
    prompt: str = Field(description="The question or request to send to the resumed session"),
    session_id: str | None = Field(default=None, description="Session ID to resume (uses --last if not provided)"),
    working_directory: str | None = Field(default=None, description="Working directory for Codex (--cd flag)"),
    enable_search: bool = Field(default=True, description="Enable web search (--search flag)")
) -> dict:
    """Resume an existing Codex session and ask for advice."""
    if not prompt or not prompt.strip():
        return TaskResponse(success=False, error="'prompt' parameter is required.", error_code=ErrorCode.INVALID_ARGS).model_dump()

    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return TaskResponse(success=False, error=error, error_code=ErrorCode.INVALID_ARGS).model_dump()

    use_last = not session_id or not session_id.strip()
    session_ref = "--last" if use_last else session_id.strip()

    # Validate session ID if provided (not using --last)
    if not use_last and not codex_runner.validate_session_id(session_ref):
        return TaskResponse(
            success=False,
            error=f"Invalid session_id: '{session_id}' - contains disallowed characters",
            error_code=ErrorCode.INVALID_ARGS
        ).model_dump()

    task = engine.create_task(
        command=f"{Agent.CODEX.value}_resume",
        args={"session_id": session_ref, "prompt": prompt.strip(), "working_directory": working_directory, "enable_search": enable_search},
        context=ctx,
    )

    task.async_task = asyncio.create_task(engine.run_agent(
        task, codex_runner, mode="resume",
        prompt=prompt.strip(), session_ref=session_ref, working_directory=working_directory, enable_search=enable_search
    ))

    return TaskResponse(
        success=True,
        task_id=task.task_id,
        status=task.status,
        message=f"Codex resume started{' (last session)' if use_last else f' for session {session_id}'}. Use wait_for_task to get result.",
    ).model_dump()


# === Gemini Tools ===

def _is_rate_limit_error(error: str | None) -> bool:
    """Check if an error indicates rate limiting."""
    if not error:
        return False
    error_lower = error.lower()
    rate_limit_indicators = [
        "429",
        "rate limit",
        "rate_limit",
        "ratelimit",
        "quota exceeded",
        "quota_exceeded",
        "resource exhausted",
        "resource_exhausted",
        "too many requests",
    ]
    return any(indicator in error_lower for indicator in rate_limit_indicators)


async def _run_gemini_with_fallback(
    task: Task,
    mode: str,
    prompt: str,
    working_directory: str | None,
    session_ref: str | None = None,
) -> tuple[bool, str | None]:
    """
    Run Gemini agent with automatic fallback to secondary model on rate limit.

    Returns (used_fallback, fallback_model) tuple.
    """
    # Try primary model first
    await engine.run_agent(
        task, gemini_runner, mode=mode,
        prompt=prompt, working_directory=working_directory,
        session_ref=session_ref,
    )

    # Check if we should fallback
    if task.status == "failed" and _is_rate_limit_error(task.error):
        fallback_model = config.gemini.fallback_model
        if fallback_model:
            _log(f"Rate limit detected, falling back to {fallback_model}")

            # Reset task state for retry
            task.status = "pending"
            task.error = None
            task.result = None
            task.warnings = None
            task.completion_time = None

            # Retry with fallback model
            await engine.run_agent(
                task, gemini_runner, mode=mode,
                prompt=prompt, working_directory=working_directory,
                session_ref=session_ref,
                model=fallback_model,  # Override to fallback model
            )
            return (True, fallback_model)

    return (False, None)


@mcp.tool()
async def start_gemini_session(
    ctx: Context[ServerSession, None],
    prompt: str = Field(description="The question or request to send"),
    working_directory: str | None = Field(default=None, description="Working directory for Gemini context"),
) -> dict:
    """Start a new Gemini CLI session (no prior context)."""
    if not prompt or not prompt.strip():
        return TaskResponse(success=False, error="'prompt' parameter is required.", error_code=ErrorCode.INVALID_ARGS).model_dump()

    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return TaskResponse(success=False, error=error, error_code=ErrorCode.INVALID_ARGS).model_dump()

    task = engine.create_task(
        command=f"{Agent.GEMINI.value}_exec",
        args={"prompt": prompt.strip(), "working_directory": working_directory},
        context=ctx,
    )

    async def run_with_fallback():
        used_fallback, fallback_model = await _run_gemini_with_fallback(
            task, mode="exec",
            prompt=prompt.strip(), working_directory=working_directory,
        )
        if used_fallback:
            task.args["used_fallback"] = True
            task.args["fallback_model"] = fallback_model

    task.async_task = asyncio.create_task(run_with_fallback())

    return TaskResponse(
        success=True,
        task_id=task.task_id,
        status=task.status,
        message=f"Gemini session started ({config.gemini.model}, fallback: {config.gemini.fallback_model}). Use wait_for_task to get result.",
    ).model_dump()


@mcp.tool()
async def resume_gemini_session(
    ctx: Context[ServerSession, None],
    prompt: str = Field(description="The question or request to send to the resumed session"),
    session_ref: str = Field(default="latest", description="Session to resume: 'latest' for most recent, or index number"),
    working_directory: str | None = Field(default=None, description="Working directory for Gemini context"),
) -> dict:
    """Resume an existing Gemini CLI session with full conversation history."""
    if not prompt or not prompt.strip():
        return TaskResponse(success=False, error="'prompt' parameter is required.", error_code=ErrorCode.INVALID_ARGS).model_dump()

    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return TaskResponse(success=False, error=error, error_code=ErrorCode.INVALID_ARGS).model_dump()

    # Validate session reference (must be numeric index or "latest")
    if not gemini_runner.validate_session_id(session_ref):
        return TaskResponse(
            success=False,
            error=f"Invalid session_ref: '{session_ref}' - must be 'latest' or a numeric index",
            error_code=ErrorCode.INVALID_ARGS
        ).model_dump()

    task = engine.create_task(
        command=f"{Agent.GEMINI.value}_resume",
        args={"session_ref": session_ref, "prompt": prompt.strip(), "working_directory": working_directory},
        context=ctx,
    )

    async def run_with_fallback():
        used_fallback, fallback_model = await _run_gemini_with_fallback(
            task, mode="resume",
            prompt=prompt.strip(), working_directory=working_directory,
            session_ref=session_ref,
        )
        if used_fallback:
            task.args["used_fallback"] = True
            task.args["fallback_model"] = fallback_model

    task.async_task = asyncio.create_task(run_with_fallback())

    return TaskResponse(
        success=True,
        task_id=task.task_id,
        status=task.status,
        message=f"Gemini resume started ({config.gemini.model}, fallback: {config.gemini.fallback_model}, session: {session_ref}). Use wait_for_task to get result.",
    ).model_dump()


# === OpenCode Tools ===

@mcp.tool()
async def start_opencode_session(
    ctx: Context[ServerSession, None],
    prompt: str = Field(description="The question or request to send"),
    working_directory: str | None = Field(default=None, description="Working directory for OpenCode context"),
) -> dict:
    """Start a new OpenCode session (no prior context)."""
    if not prompt or not prompt.strip():
        return TaskResponse(success=False, error="'prompt' parameter is required.", error_code=ErrorCode.INVALID_ARGS).model_dump()

    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return TaskResponse(success=False, error=error, error_code=ErrorCode.INVALID_ARGS).model_dump()

    task = engine.create_task(
        command=f"{Agent.OPENCODE.value}_exec",
        args={"prompt": prompt.strip(), "working_directory": working_directory},
        context=ctx,
    )

    task.async_task = asyncio.create_task(engine.run_agent(
        task, opencode_runner, mode="exec",
        prompt=prompt.strip(), working_directory=working_directory
    ))

    return TaskResponse(
        success=True,
        task_id=task.task_id,
        status=task.status,
        message="OpenCode session started. Use wait_for_task to get result.",
    ).model_dump()


@mcp.tool()
async def resume_opencode_session(
    ctx: Context[ServerSession, None],
    prompt: str = Field(description="The question or request to send to the resumed session"),
    session_id: str | None = Field(default=None, description="Session ID to resume (uses --continue if not provided)"),
    working_directory: str | None = Field(default=None, description="Working directory for OpenCode context"),
) -> dict:
    """Resume an existing OpenCode session with full conversation history."""
    if not prompt or not prompt.strip():
        return TaskResponse(success=False, error="'prompt' parameter is required.", error_code=ErrorCode.INVALID_ARGS).model_dump()

    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return TaskResponse(success=False, error=error, error_code=ErrorCode.INVALID_ARGS).model_dump()

    use_continue = not session_id or not session_id.strip()
    session_ref = "--continue" if use_continue else session_id.strip()

    # Validate session ID if provided (not using --continue)
    if not use_continue and not opencode_runner.validate_session_id(session_ref):
        return TaskResponse(
            success=False,
            error=f"Invalid session_id: '{session_id}' - contains disallowed characters",
            error_code=ErrorCode.INVALID_ARGS
        ).model_dump()

    task = engine.create_task(
        command=f"{Agent.OPENCODE.value}_resume",
        args={"session_id": session_ref, "prompt": prompt.strip(), "working_directory": working_directory},
        context=ctx,
    )

    task.async_task = asyncio.create_task(engine.run_agent(
        task, opencode_runner, mode="resume",
        prompt=prompt.strip(), session_ref=session_ref, working_directory=working_directory
    ))

    return TaskResponse(
        success=True,
        task_id=task.task_id,
        status=task.status,
        message=f"OpenCode resume started{' (continuing last session)' if use_continue else f' for session {session_id}'}. Use wait_for_task to get result.",
    ).model_dump()


# === Claude OpenRouter Session Tools ===

@mcp.tool()
async def start_claudeor_session(
    ctx: Context[ServerSession, None],
    prompt: str = Field(description="The question or request to send"),
    working_directory: str | None = Field(default=None, description="Working directory for Claude context"),
) -> dict:
    """
    Start a new Claude Code session via OpenRouter.

    Uses Claude CLI with OpenRouter backend, allowing alternative models
    like DeepSeek, GPT-4o, Gemini, etc. Configure model via CLAUDEOR_MODEL env var.
    """
    if not prompt or not prompt.strip():
        return TaskResponse(success=False, error="'prompt' parameter is required.", error_code=ErrorCode.INVALID_ARGS).model_dump()

    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return TaskResponse(success=False, error=error, error_code=ErrorCode.INVALID_ARGS).model_dump()

    # Check if API key is configured
    if not config.claudeor.api_key:
        return TaskResponse(
            success=False,
            error="OPENROUTER_API_KEY or CLAUDEOR_API_KEY environment variable not set",
            error_code=ErrorCode.INVALID_ARGS
        ).model_dump()

    task = engine.create_task(
        command=f"{Agent.CLAUDEOR.value}_exec",
        args={"prompt": prompt.strip(), "working_directory": working_directory},
        context=ctx,
    )

    task.async_task = asyncio.create_task(engine.run_agent(
        task, claudeor_runner, mode="exec",
        prompt=prompt.strip(), working_directory=working_directory
    ))

    model_info = f" ({config.claudeor.model})" if config.claudeor.model else ""
    return TaskResponse(
        success=True,
        task_id=task.task_id,
        status=task.status,
        message=f"Claude OpenRouter{model_info} session started. Use wait_for_task to get result.",
    ).model_dump()


@mcp.tool()
async def resume_claudeor_session(
    ctx: Context[ServerSession, None],
    prompt: str = Field(description="The question or request to send to the resumed session"),
    session_id: str | None = Field(default=None, description="Session ID to resume (uses --continue if not provided)"),
    working_directory: str | None = Field(default=None, description="Working directory for Claude context"),
) -> dict:
    """Resume an existing Claude OpenRouter session with full conversation history."""
    if not prompt or not prompt.strip():
        return TaskResponse(success=False, error="'prompt' parameter is required.", error_code=ErrorCode.INVALID_ARGS).model_dump()

    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return TaskResponse(success=False, error=error, error_code=ErrorCode.INVALID_ARGS).model_dump()

    # Check if API key is configured
    if not config.claudeor.api_key:
        return TaskResponse(
            success=False,
            error="OPENROUTER_API_KEY or CLAUDEOR_API_KEY environment variable not set",
            error_code=ErrorCode.INVALID_ARGS
        ).model_dump()

    use_continue = not session_id or not session_id.strip()
    session_ref = "continue" if use_continue else session_id.strip()

    # Validate session ID if provided
    if not use_continue and not claudeor_runner.validate_session_id(session_ref):
        return TaskResponse(
            success=False,
            error=f"Invalid session_id: '{session_id}' - contains disallowed characters",
            error_code=ErrorCode.INVALID_ARGS
        ).model_dump()

    task = engine.create_task(
        command=f"{Agent.CLAUDEOR.value}_resume",
        args={"session_id": session_ref, "prompt": prompt.strip(), "working_directory": working_directory},
        context=ctx,
    )

    task.async_task = asyncio.create_task(engine.run_agent(
        task, claudeor_runner, mode="resume",
        prompt=prompt.strip(), session_ref=session_ref, working_directory=working_directory
    ))

    model_info = f" ({config.claudeor.model})" if config.claudeor.model else ""
    return TaskResponse(
        success=True,
        task_id=task.task_id,
        status=task.status,
        message=f"Claude OpenRouter{model_info} resume started{' (continuing last session)' if use_continue else f' for session {session_id}'}. Use wait_for_task to get result.",
    ).model_dump()


# === Grok Session Tools ===

@mcp.tool()
async def start_grok_session(
    ctx: Context[ServerSession, None],
    prompt: str = Field(description="The question or request to send"),
    working_directory: str | None = Field(default=None, description="Working directory for Grok context"),
    for_coding: bool = Field(default=False, description="Use coding model (grok-code-fast-1) instead of reasoning model"),
) -> dict:
    """
    Start a new Grok session via OpenCode with xAI backend.

    Uses OpenCode CLI with xAI/Grok models. Requires XAI_API_KEY environment variable.

    Two model options:
    - for_coding=False (default): Uses GROK_MODEL (default: xai/grok-4-1-fast) for reasoning/deliberation
    - for_coding=True: Uses GROK_CODE_MODEL (default: xai/grok-code-fast-1) for coding tasks
    """
    if not prompt or not prompt.strip():
        return TaskResponse(success=False, error="'prompt' parameter is required.", error_code=ErrorCode.INVALID_ARGS).model_dump()

    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return TaskResponse(success=False, error=error, error_code=ErrorCode.INVALID_ARGS).model_dump()

    task = engine.create_task(
        command=f"{Agent.GROK.value}_exec",
        args={"prompt": prompt.strip(), "working_directory": working_directory, "for_coding": for_coding},
        context=ctx,
    )

    task.async_task = asyncio.create_task(engine.run_agent(
        task, grok_runner, mode="exec",
        prompt=prompt.strip(), working_directory=working_directory, for_coding=for_coding
    ))

    model = config.grok.code_model if for_coding else config.grok.model
    return TaskResponse(
        success=True,
        task_id=task.task_id,
        status=task.status,
        message=f"Grok session started ({model}). Use wait_for_task to get result.",
    ).model_dump()


@mcp.tool()
async def resume_grok_session(
    ctx: Context[ServerSession, None],
    prompt: str = Field(description="The question or request to send to the resumed session"),
    session_id: str | None = Field(default=None, description="Session ID to resume (uses --continue if not provided)"),
    working_directory: str | None = Field(default=None, description="Working directory for Grok context"),
    for_coding: bool = Field(default=False, description="Use coding model (grok-code-fast-1) instead of reasoning model"),
) -> dict:
    """Resume an existing Grok session with full conversation history."""
    if not prompt or not prompt.strip():
        return TaskResponse(success=False, error="'prompt' parameter is required.", error_code=ErrorCode.INVALID_ARGS).model_dump()

    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return TaskResponse(success=False, error=error, error_code=ErrorCode.INVALID_ARGS).model_dump()

    use_continue = not session_id or not session_id.strip()
    session_ref = "--continue" if use_continue else session_id.strip()

    # Validate session ID if provided
    if not use_continue and not grok_runner.validate_session_id(session_ref):
        return TaskResponse(
            success=False,
            error=f"Invalid session_id: '{session_id}' - contains disallowed characters",
            error_code=ErrorCode.INVALID_ARGS
        ).model_dump()

    task = engine.create_task(
        command=f"{Agent.GROK.value}_resume",
        args={"session_id": session_ref, "prompt": prompt.strip(), "working_directory": working_directory, "for_coding": for_coding},
        context=ctx,
    )

    task.async_task = asyncio.create_task(engine.run_agent(
        task, grok_runner, mode="resume",
        prompt=prompt.strip(), session_ref=session_ref, working_directory=working_directory, for_coding=for_coding
    ))

    model = config.grok.code_model if for_coding else config.grok.model
    return TaskResponse(
        success=True,
        task_id=task.task_id,
        status=task.status,
        message=f"Grok resume started ({model}){' (continuing last session)' if use_continue else f' for session {session_id}'}. Use wait_for_task to get result.",
    ).model_dump()


# === Task Management Tools ===

@mcp.tool()
async def get_task_result(task_id: str) -> dict:
    """
    Get the result of a task (Codex, Gemini, or OpenCode).

    Args:
        task_id: The task ID returned by start/resume session
    """
    task = engine.get_task(task_id)
    if not task:
        return TaskResponse(success=False, error=f"Task '{task_id}' not found.", error_code=ErrorCode.NOT_FOUND).model_dump()

    if task.status == "pending":
        return TaskResponse(
            success=True,
            task_id=task_id,
            status=task.status,
            message="Task is still pending.",
        ).model_dump()
    elif task.status == "running":
        elapsed = (datetime.now() - task.start_time).total_seconds()
        return TaskResponse(
            success=True,
            task_id=task_id,
            status=task.status,
            message=f"Task is still running ({elapsed:.1f}s elapsed).",
        ).model_dump()
    elif task.status == "completed":
        return TaskResponse(
            success=True,
            task_id=task_id,
            status=task.status,
            content=task.result,
            warnings=task.warnings,
            duration_seconds=(task.completion_time - task.start_time).total_seconds() if task.completion_time else None,
        ).model_dump()
    elif task.status == "failed":
        return TaskResponse(
            success=False,
            task_id=task_id,
            status=task.status,
            error=task.error,
            error_code=ErrorCode.EXECUTION_FAILED,
            duration_seconds=(task.completion_time - task.start_time).total_seconds() if task.completion_time else None,
        ).model_dump()
    elif task.status == "cancelled":
        return TaskResponse(
            success=False,
            task_id=task_id,
            status=task.status,
            error=task.error or "Task was cancelled.",
            error_code=ErrorCode.CANCELLED,
            duration_seconds=(task.completion_time - task.start_time).total_seconds() if task.completion_time else None,
        ).model_dump()
    else:
        # Unknown status - should not happen but handle gracefully
        return TaskResponse(
            success=False,
            task_id=task_id,
            status=task.status,
            error=f"Unexpected task status: {task.status}",
            error_code=ErrorCode.INTERNAL_ERROR,
        ).model_dump()


@mcp.tool()
async def wait_for_task(task_id: str, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """
    Wait for a task to complete and return its result.

    Args:
        task_id: The task ID to wait for
        timeout: Maximum seconds to wait (default: 300)
    """
    task = engine.get_task(task_id)
    if not task:
        return TaskResponse(success=False, error=f"Task '{task_id}' not found.", error_code=ErrorCode.NOT_FOUND).model_dump()

    if task.status in ["completed", "failed", "cancelled"]:
        if task.status == "completed":
            return TaskResponse(
                success=True,
                task_id=task_id,
                status=task.status,
                content=task.result,
                warnings=task.warnings,
                duration_seconds=(task.completion_time - task.start_time).total_seconds() if task.completion_time else None,
            ).model_dump()
        error_code = ErrorCode.EXECUTION_FAILED if task.status == "failed" else ErrorCode.CANCELLED
        return TaskResponse(
            success=False,
            task_id=task_id,
            status=task.status,
            error=task.error,
            error_code=error_code,
        ).model_dump()

    if task.async_task:
        # Check if task already completed (e.g., between abort and re-wait)
        if task.async_task.done():
            try:
                task.async_task.result()  # Re-raise any exception from the task
            except asyncio.CancelledError:
                # Task was cancelled - mark appropriately
                if task.status not in ["completed", "failed", "cancelled"]:
                    task.status = "cancelled"
                    task.error = "Task was cancelled"
                    task.completion_time = datetime.now()
            except BaseException as e:
                # Catch BaseException to handle all exceptions including SystemExit, KeyboardInterrupt
                if task.status not in ["completed", "failed", "cancelled"]:
                    task.status = "failed"
                    task.error = f"Task failed: {str(e)}"
                    task.completion_time = datetime.now()
            # Fall through to return result below
        else:
            try:
                await asyncio.wait_for(asyncio.shield(task.async_task), timeout=timeout)
            except asyncio.TimeoutError:
                return TaskResponse(
                    success=False,
                    task_id=task_id,
                    status="timeout",
                    error=f"Task still running after {timeout}s. Use get_task_result to check later.",
                    error_code=ErrorCode.TIMEOUT,
                ).model_dump()
            except asyncio.CancelledError:
                # User aborted the wait (e.g., pressed ESC) - task keeps running
                return TaskResponse(
                    success=True,
                    task_id=task_id,
                    status=task.status,
                    message="Wait aborted. Task still running. Use get_task_result or wait_for_task later.",
                ).model_dump()
            except Exception as e:
                # Bug fix: Set task.status and task.error so subsequent calls are consistent
                task.status = "failed"
                task.error = f"Task failed: {str(e)}"
                task.completion_time = datetime.now()
                return TaskResponse(
                    success=False,
                    task_id=task_id,
                    status=task.status,
                    error=task.error,
                    error_code=ErrorCode.INTERNAL_ERROR,
                ).model_dump()
    else:
        # Bug fix: No async_task means task may have failed before launch or is in unexpected state
        # Return actual status instead of assuming CANCELLED
        return TaskResponse(
            success=False,
            task_id=task_id,
            status=task.status,
            error=task.error or f"Task has no async handler (status: {task.status})",
            error_code=ErrorCode.INTERNAL_ERROR,
        ).model_dump()

    if task.status == "completed":
        return TaskResponse(
            success=True,
            task_id=task_id,
            status=task.status,
            content=task.result,
            warnings=task.warnings,
            duration_seconds=(task.completion_time - task.start_time).total_seconds() if task.completion_time else None,
        ).model_dump()

    error_code = ErrorCode.EXECUTION_FAILED if task.status == "failed" else ErrorCode.CANCELLED
    return TaskResponse(
        success=False,
        task_id=task_id,
        status=task.status,
        error=task.error,
        error_code=error_code,
    ).model_dump()


@mcp.tool()
async def list_tasks(
    status_filter: str | None = Field(default=None, description="Filter by status: pending, running, completed, failed, cancelled"),
    limit: int = Field(default=20, description="Maximum number of tasks to return"),
) -> dict:
    """
    List all tracked tasks with their current status.

    Args:
        status_filter: Optional filter by task status
        limit: Maximum number of tasks to return (default: 20)
    """
    tasks_list = []
    for task_id, task in list(engine.tasks.items())[-limit:]:
        if status_filter and task.status != status_filter:
            continue
        elapsed = (datetime.now() - task.start_time).total_seconds()
        tasks_list.append({
            "task_id": task_id,
            "command": task.command,
            "status": task.status,
            "elapsed_seconds": round(elapsed, 1),
            "has_result": task.result is not None,
            "has_error": task.error is not None,
        })

    return {
        "success": True,
        "count": len(tasks_list),
        "tasks": tasks_list,
    }


@mcp.tool()
async def cancel_task(task_id: str) -> dict:
    """
    Cancel a running task and kill its subprocess.

    Args:
        task_id: The task ID to cancel
    """
    task = engine.get_task(task_id)
    if not task:
        return TaskResponse(success=False, error=f"Task '{task_id}' not found.", error_code=ErrorCode.NOT_FOUND).model_dump()

    if task.status in ["completed", "failed", "cancelled"]:
        return TaskResponse(
            success=False,
            task_id=task_id,
            status=task.status,
            error=f"Task already {task.status}, cannot cancel.",
            error_code=ErrorCode.INVALID_ARGS,
        ).model_dump()

    # Kill the subprocess and cancel the async task
    await engine.kill_task_subprocess(task)
    task.status = "cancelled"
    task.error = "Cancelled by user"
    task.completion_time = datetime.now()

    return TaskResponse(
        success=True,
        task_id=task_id,
        status=task.status,
        message="Task cancelled successfully.",
    ).model_dump()


# === Council Tool ===

async def _run_council_deliberation(
    task,
    prompt: str,
    working_directory: str | None,
    claude_opinion: str | None,
    deliberate: bool,
    critique: bool,
    timeout: int,
    roles: dict[str, str] | list[str] | None = None,
    team: str | None = None,
    context_id: str | None = None,
):
    """Run council deliberation and update task with result."""
    from .models import TaskStatus
    try:
        task.status = TaskStatus.RUNNING.value
        council = Council(context=task.context)
        response = await council.deliberate(
            prompt=prompt,
            working_directory=working_directory,
            claude_opinion=claude_opinion,
            deliberate=deliberate,
            critique=critique,
            timeout=timeout,
            roles=roles,
            team=team,
            context_id=context_id,
        )
        task.result = response.model_dump_json(indent=2)
        task.status = TaskStatus.COMPLETED.value
        task.end_time = datetime.now()
    except ValueError as e:
        # Role/team validation errors
        task.error = str(e)
        task.status = TaskStatus.FAILED.value
        task.end_time = datetime.now()
    except Exception as e:
        task.error = str(e)
        task.status = TaskStatus.FAILED.value
        task.end_time = datetime.now()


@mcp.tool()
async def council_ask(
    ctx: Context[ServerSession, None],
    prompt: str = Field(description="The question or task to send to the council"),
    claude_opinion: str | None = Field(default=None, description="Claude's initial opinion to share with the council"),
    working_directory: str | None = Field(default=None, description="Working directory for context"),
    deliberate: bool = Field(default=True, description="If true, share answers between agents for a second round of deliberation"),
    critique: bool = Field(default=False, description="If true, round 2 asks agents to critique/find flaws instead of revise"),
    timeout: int = Field(default=DEFAULT_TIMEOUT, description="Timeout per agent in seconds"),
    roles: dict[str, str] | list[str] | None = Field(
        default=None,
        description=(
            "Role assignments for agents. Can be:\n"
            "- Dict mapping agent to role: {\"codex\": \"security\", \"gemini\": \"perf\"}\n"
            "- List of roles (auto-assigned in order): [\"security\", \"perf\", \"skeptic\"]\n"
            "Built-in roles: security, perf, skeptic, architect, maintainer, dx, testing"
        )
    ),
    team: str | None = Field(
        default=None,
        description=(
            "Team preset name (alternative to roles). "
            "Built-in teams: security_audit, code_review, architecture_review, devil_advocate, balanced"
        )
    ),
    context_id: str | None = Field(
        default=None,
        description=(
            "Shared context ID for pre-hydration. If provided, context data is loaded "
            "and made available to agents, reducing token duplication."
        )
    ),
) -> dict:
    """
    Ask the council (Codex, Gemini, and OpenCode) a question and collect their answers.

    Sends the prompt to all three agents in parallel, waits for responses,
    and returns all answers for the MCP client (Claude Code) to synthesize.

    Supports specialist roles ("hats") for agents to operate with specific perspectives.

    Role specification (mutually exclusive, priority order):
    1. roles parameter - explicit mapping or auto-assign list
    2. team parameter - use a predefined team preset
    3. Neither - all agents operate without special roles

    Examples:
    - roles={"codex": "security", "gemini": "perf"} - explicit assignment
    - roles=["security", "perf", "skeptic"] - auto-assign to codex, gemini, opencode
    - team="security_audit" - use the security audit team preset

    If claude_opinion is provided, it will be shared with other council members
    during deliberation so they can consider Claude's perspective.

    If deliberate=True, shares all answers (including Claude's) with each agent
    for a second round, allowing them to revise after seeing others' responses.

    If critique=True, round 2 asks agents to find bugs, security issues, and
    architectural flaws instead of politely revising their answers.
    """
    if not prompt or not prompt.strip():
        return TaskResponse(success=False, error="'prompt' parameter is required.", error_code=ErrorCode.INVALID_ARGS).model_dump()

    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return TaskResponse(success=False, error=error, error_code=ErrorCode.INVALID_ARGS).model_dump()

    # Validate that roles and team are not both specified
    if roles is not None and team is not None:
        return TaskResponse(
            success=False,
            error="Cannot specify both 'roles' and 'team' parameters. Use one or the other.",
            error_code=ErrorCode.INVALID_ARGS
        ).model_dump()

    # Early validation of roles/team to return proper error codes
    excluded = config.council.exclude_agents
    active_agents = [a for a in ["codex", "gemini", "opencode", "claudeor"] if a not in excluded]

    # Use default team from config if no roles/team specified
    effective_team = team if team is not None else config.council.default_team
    role_spec = roles if roles is not None else effective_team
    try:
        resolver = get_resolver()
        resolved_roles = resolver.resolve(role_spec, active_agents)
    except ValueError as e:
        return TaskResponse(
            success=False,
            error=str(e),
            error_code=ErrorCode.INVALID_ARGS
        ).model_dump()

    # Build role summary for initial response
    role_assignments = {agent: role.id for agent, role in resolved_roles.items()}
    role_names = {agent: role.name for agent, role in resolved_roles.items()}

    # Create task and run council deliberation asynchronously
    task = engine.create_task(
        command="council_ask",
        args={
            "prompt": prompt.strip(),
            "working_directory": working_directory,
            "claude_opinion": claude_opinion,
            "deliberate": deliberate,
            "critique": critique,
            "timeout": timeout,
            "roles": roles,
            "team": effective_team,
            "context_id": context_id,
        },
        context=ctx,
    )

    task.async_task = asyncio.create_task(_run_council_deliberation(
        task,
        prompt=prompt.strip(),
        working_directory=working_directory,
        claude_opinion=claude_opinion,
        deliberate=deliberate,
        critique=critique,
        timeout=timeout,
        roles=roles,
        team=effective_team,
        context_id=context_id,
    ))

    # Return with role information
    response = TaskResponse(
        success=True,
        task_id=task.task_id,
        status=task.status,
        message="Council deliberation started. Use wait_for_task to get result.",
    ).model_dump()

    # Add role details to response
    response["council"] = {
        "agents": active_agents,
        "excluded": list(excluded),
        "team": effective_team,
        "roles": role_assignments,
        "role_names": role_names,
        "include_claude_opinion": config.council.include_claude_opinion,
        "context_id": context_id,
    }

    return response


# === Liza Tools (Peer-Supervised Coding) ===

# Module-level Liza orchestrators (per working directory)
_liza_orchestrators: dict[str, "LizaOrchestrator"] = {}


def _get_liza_orchestrator(working_directory: str | None = None) -> "LizaOrchestrator":
    """Get or create a Liza orchestrator for the working directory."""
    from .liza import LizaOrchestrator, LizaConfig

    wd = working_directory or os.getcwd()
    if wd not in _liza_orchestrators:
        config_obj = LizaConfig(
            working_directory=wd,
            reviewers=["codex", "gemini"],  # Default reviewers
        )
        orchestrator = LizaOrchestrator(config=config_obj)

        # Set up the reviewer runner to use owlex agents with session persistence
        async def run_reviewer(
            agent: str,
            prompt: str,
            working_dir: str | None,
            timeout: int,
            task_id: str,
        ) -> str | None:
            """Run a reviewer agent via owlex, with session persistence for iterations."""
            runner_map = {
                "codex": codex_runner,
                "gemini": gemini_runner,
                "opencode": opencode_runner,
                "grok": grok_runner,
            }
            runner = runner_map.get(agent)
            if runner is None:
                return None

            # Get existing session ID from blackboard (for resume on iterations 2+)
            liza_task = orchestrator.blackboard.get_task(task_id)
            existing_session = liza_task.reviewer_sessions.get(agent) if liza_task else None

            # Record start time for session ID parsing
            start_mtime = time.time()

            engine_task = engine.create_task(
                command=f"liza_review_{agent}",
                args={"prompt": prompt, "working_directory": working_dir},
                context=None,
            )

            if existing_session:
                # Resume existing session (iteration 2+)
                _log(f"Resuming {agent} session: {existing_session}")
                await engine.run_agent(
                    engine_task, runner, mode="resume",
                    session_ref=existing_session,
                    prompt=prompt, working_directory=working_dir,
                    **({"enable_search": False} if agent == "codex" else {})
                )
            else:
                # Start new session (iteration 1)
                await engine.run_agent(
                    engine_task, runner, mode="exec",
                    prompt=prompt, working_directory=working_dir,
                    **({"enable_search": False} if agent == "codex" else {})
                )

            # Capture session ID for next iteration
            if engine_task.status == "completed" and liza_task and not existing_session:
                try:
                    session_id = await runner.parse_session_id(
                        engine_task.result or "",
                        since_mtime=start_mtime,
                        working_directory=working_dir,
                    )
                    if session_id:
                        liza_task.reviewer_sessions[agent] = session_id
                        orchestrator.blackboard.update_task(liza_task)
                        _log(f"Captured {agent} session: {session_id}")
                except Exception as e:
                    _log(f"Warning: Failed to parse session ID for {agent}: {e}")

            if engine_task.status == "completed":
                return engine_task.result
            return None

        orchestrator.set_reviewer_runner(run_reviewer)
        _liza_orchestrators[wd] = orchestrator

    return _liza_orchestrators[wd]


@mcp.tool()
async def liza_start(
    ctx: Context[ServerSession, None],
    task_description: str = Field(description="Description of what Claude should implement"),
    reviewers: list[str] | None = Field(default=None, description="Reviewer agents (default: ['codex', 'gemini'])"),
    max_iterations: int = Field(default=5, description="Maximum coder-reviewer cycles"),
    done_when: str | None = Field(default=None, description="Optional completion criteria"),
    working_directory: str | None = Field(default=None, description="Working directory for the task"),
) -> dict:
    """
    Start a Liza peer-reviewed coding task.

    Creates a task for Claude (the coder) to implement. After Claude implements,
    use liza_submit to send the implementation for review by Codex/Gemini.

    Architecture:
    - Claude Code = Coder (trusted, actually writes code)
    - Codex/Gemini/OpenCode/Grok = Reviewers (examine and provide feedback)

    Flow:
    1. liza_start → Creates task, returns task_id
    2. Claude implements the task (using Write/Edit/Bash tools)
    3. liza_submit → Sends to reviewers, returns verdicts
    4. If REJECT: Claude fixes based on feedback, goto step 3
    5. If APPROVE: Done!
    """
    if not task_description or not task_description.strip():
        return {"success": False, "error": "'task_description' is required."}

    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return {"success": False, "error": error}

    try:
        orchestrator = _get_liza_orchestrator(working_directory)

        # Update reviewers if specified
        if reviewers:
            orchestrator.config.reviewers = reviewers

        task = orchestrator.create_task(
            description=task_description.strip(),
            reviewers=reviewers,
            max_iterations=max_iterations,
            done_when=done_when,
        )

        return {
            "success": True,
            "task_id": task.id,
            "message": f"Task created. Claude should now implement: {task_description[:100]}...",
            "instructions": (
                "1. Implement the task using Write/Edit/Bash tools\n"
                "2. When done, call liza_submit with your implementation summary\n"
                "3. Reviewers will examine and provide feedback\n"
                "4. If rejected, fix issues and submit again"
            ),
            "task": {
                "id": task.id,
                "description": task.description,
                "reviewers": task.reviewers,
                "max_iterations": task.max_iterations,
                "done_when": task.done_when,
            },
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def liza_submit(
    ctx: Context[ServerSession, None],
    task_id: str = Field(description="Task ID from liza_start"),
    implementation_summary: str = Field(description="Summary of what was implemented (for reviewers)"),
    working_directory: str | None = Field(default=None, description="Working directory"),
) -> dict:
    """
    Submit Claude's implementation for review by Codex/Gemini.

    After Claude implements the task, call this to send it for review.
    Reviewers will examine the code and return APPROVE or REJECT with feedback.

    If REJECT: Claude should fix the issues and call liza_submit again.
    If APPROVE: The task is complete!
    """
    if not task_id or not task_id.strip():
        return {"success": False, "error": "'task_id' is required."}
    if not implementation_summary or not implementation_summary.strip():
        return {"success": False, "error": "'implementation_summary' is required."}

    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return {"success": False, "error": error}

    try:
        orchestrator = _get_liza_orchestrator(working_directory)

        # Check task exists and is in right state
        task = orchestrator.blackboard.get_task(task_id.strip())
        if task is None:
            return {"success": False, "error": f"Task '{task_id}' not found."}

        # Check iteration limit
        if task.iteration >= task.max_iterations:
            return {
                "success": False,
                "error": f"Max iterations ({task.max_iterations}) reached. Task blocked.",
                "task_status": orchestrator.get_task_status(task_id),
            }

        # Submit for review
        result = await orchestrator.submit_for_review(
            task_id=task_id.strip(),
            implementation_summary=implementation_summary.strip(),
        )

        if result.all_approved:
            orchestrator.mark_approved(task_id)
            return {
                "success": True,
                "approved": True,
                "message": "All reviewers APPROVED! Task complete.",
                "review_summary": {
                    "iteration": result.iteration,
                    "reviews": {
                        reviewer: {
                            "status": v.status.value,
                            "confidence": v.confidence,
                        }
                        for reviewer, v in result.reviews.items()
                    },
                },
            }
        else:
            # Prepare for next iteration
            orchestrator.prepare_for_iteration(task_id, result.merged_feedback)
            task = orchestrator.blackboard.get_task(task_id)

            return {
                "success": True,
                "approved": False,
                "message": "Review REJECTED. Please address the feedback and submit again.",
                "iteration": task.iteration,
                "remaining_iterations": task.max_iterations - task.iteration,
                "feedback": result.merged_feedback,
                "issues_found": result.issues_found,
                "review_summary": {
                    "reviews": {
                        reviewer: {
                            "status": v.status.value,
                            "confidence": v.confidence,
                            "feedback_preview": (v.feedback[:200] + "...") if v.feedback and len(v.feedback) > 200 else v.feedback,
                        }
                        for reviewer, v in result.reviews.items()
                    },
                },
                "instructions": (
                    "1. Read the feedback above carefully\n"
                    "2. Fix the identified issues\n"
                    "3. Do NOT introduce unrelated changes\n"
                    "4. Call liza_submit again with updated summary"
                ),
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def liza_status(
    task_id: str = Field(description="Task ID to check"),
    working_directory: str | None = Field(default=None, description="Working directory"),
) -> dict:
    """
    Get the current status of a Liza task.

    Returns task details including status, iteration count, reviewers, and any feedback.
    """
    if not task_id or not task_id.strip():
        return {"success": False, "error": "'task_id' is required."}

    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return {"success": False, "error": error}

    try:
        orchestrator = _get_liza_orchestrator(working_directory)
        status = orchestrator.get_task_status(task_id.strip())

        if status is None:
            return {"success": False, "error": f"Task '{task_id}' not found."}

        return {
            "success": True,
            "task": status,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def liza_feedback(
    task_id: str = Field(description="Task ID to get feedback for"),
    working_directory: str | None = Field(default=None, description="Working directory"),
) -> dict:
    """
    Get the latest reviewer feedback for a Liza task.

    Returns formatted feedback that Claude should address before resubmitting.
    """
    if not task_id or not task_id.strip():
        return {"success": False, "error": "'task_id' is required."}

    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return {"success": False, "error": error}

    try:
        orchestrator = _get_liza_orchestrator(working_directory)
        feedback = orchestrator.get_feedback_for_claude(task_id.strip())

        if feedback is None:
            return {
                "success": True,
                "has_feedback": False,
                "message": "No feedback available (task may be approved or not yet reviewed).",
            }

        return {
            "success": True,
            "has_feedback": True,
            "feedback": feedback,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.resource("owlex://liza/blackboard")
def liza_blackboard_resource() -> str:
    """View the current Liza blackboard state."""
    try:
        orchestrator = _get_liza_orchestrator()
        if not orchestrator.blackboard.exists():
            return "No Liza blackboard found. Use liza_start to create a task."

        state = orchestrator.blackboard.read()
        lines = [
            "# Liza Blackboard",
            f"**Goal:** {state.goal}",
            f"**Created:** {state.created_at}",
            f"**Updated:** {state.updated_at}",
            "",
            "## Tasks",
        ]

        for task in state.tasks:
            status_emoji = {
                "APPROVED": "✅",
                "REJECTED": "❌",
                "WORKING": "🔨",
                "IN_REVIEW": "👀",
                "BLOCKED": "🚫",
            }.get(task.status.value, "📋")

            lines.append(f"### {task.id} {status_emoji} {task.status.value}")
            lines.append(f"**Description:** {task.description[:100]}...")
            lines.append(f"**Coder:** {task.coder} | **Reviewers:** {', '.join(task.reviewers)}")
            lines.append(f"**Iteration:** {task.iteration}/{task.max_iterations}")
            if task.merged_feedback:
                lines.append(f"**Has Feedback:** Yes")
            lines.append("")

        if state.log:
            lines.append("## Recent Log")
            for entry in state.log[-10:]:
                lines.append(f"- {entry}")

        return "\n".join(lines)
    except Exception as e:
        return f"Error reading blackboard: {e}"


# === Context Tools (Shared Agent Context) ===

# Module-level ContextManager instances (per working directory)
_context_managers: dict[str, "ContextManager"] = {}


def _get_context_manager(working_directory: str | None = None) -> "ContextManager":
    """Get or create a ContextManager for the working directory."""
    from .context_manager import ContextManager, ContextManagerConfig

    wd = working_directory or os.getcwd()
    if wd not in _context_managers:
        config = ContextManagerConfig(working_directory=wd)
        _context_managers[wd] = ContextManager(config)

    return _context_managers[wd]


@mcp.tool()
async def create_context(
    ctx: Context[ServerSession, None],
    namespace: str = Field(description="Namespace for isolation (e.g., 'task:123', 'worktree:feature-x', 'council:session-1')"),
    data: dict = Field(description="Context data to store (JSON object)"),
    ttl_days: int | None = Field(default=None, description="Time-to-live in days (default: 7)"),
    working_directory: str | None = Field(default=None, description="Working directory for context storage"),
) -> dict:
    """
    Create a new shared context that agents can read/write.

    Use this to share context between agents without duplicating it in prompts.
    For example, create a context with task requirements that multiple agents
    can reference during a council deliberation.

    Returns context_id and version for future updates.
    """
    if not namespace or not namespace.strip():
        return {"success": False, "error": "'namespace' is required."}
    if not isinstance(data, dict):
        return {"success": False, "error": "'data' must be a JSON object."}

    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return {"success": False, "error": error}

    try:
        manager = _get_context_manager(working_directory)
        context = manager.create(
            namespace=namespace.strip(),
            data=data,
            ttl_days=ttl_days,
        )

        return {
            "success": True,
            "context_id": context.id,
            "version": context.version,
            "namespace": context.namespace,
            "size_bytes": context.size_bytes,
            "expires_at": context.expires_at.isoformat() if context.expires_at else None,
            "message": f"Context created. Use context_id '{context.id}' and version {context.version} for updates.",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def get_context(
    ctx: Context[ServerSession, None],
    context_id: str = Field(description="Context ID to retrieve"),
    working_directory: str | None = Field(default=None, description="Working directory for context storage"),
) -> dict:
    """
    Get a shared context by ID.

    Returns the context data and current version. The version is needed
    for updates to ensure optimistic concurrency.
    """
    if not context_id or not context_id.strip():
        return {"success": False, "error": "'context_id' is required."}

    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return {"success": False, "error": error}

    try:
        from .context_manager import ContextNotFoundError

        manager = _get_context_manager(working_directory)
        context = manager.get(context_id.strip())

        return {
            "success": True,
            "context_id": context.id,
            "namespace": context.namespace,
            "data": context.data,
            "version": context.version,
            "size_bytes": context.size_bytes,
            "created_at": context.created_at.isoformat(),
            "updated_at": context.updated_at.isoformat(),
            "expires_at": context.expires_at.isoformat() if context.expires_at else None,
        }
    except ContextNotFoundError:
        return {"success": False, "error": f"Context '{context_id}' not found."}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def update_context(
    ctx: Context[ServerSession, None],
    context_id: str = Field(description="Context ID to update"),
    updates: dict = Field(description="Data updates to apply (merged with existing data)"),
    version: int = Field(description="Expected current version (for optimistic locking)"),
    working_directory: str | None = Field(default=None, description="Working directory for context storage"),
) -> dict:
    """
    Update a shared context with optimistic concurrency.

    The version parameter must match the current version in the database.
    If another agent has updated the context since you read it, this will
    fail with a version conflict error. In that case, re-read the context
    to get the latest version and try again.

    Updates are merged with existing data (not replaced).
    """
    if not context_id or not context_id.strip():
        return {"success": False, "error": "'context_id' is required."}
    if not isinstance(updates, dict):
        return {"success": False, "error": "'updates' must be a JSON object."}
    if version is None:
        return {"success": False, "error": "'version' is required for optimistic locking."}

    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return {"success": False, "error": error}

    try:
        from .context_manager import ContextNotFoundError, VersionConflictError

        manager = _get_context_manager(working_directory)
        context = manager.update(
            context_id=context_id.strip(),
            updates=updates,
            version=version,
        )

        return {
            "success": True,
            "context_id": context.id,
            "new_version": context.version,
            "size_bytes": context.size_bytes,
            "message": f"Context updated. New version is {context.version}.",
        }
    except ContextNotFoundError:
        return {"success": False, "error": f"Context '{context_id}' not found."}
    except VersionConflictError as e:
        return {
            "success": False,
            "error": "version_conflict",
            "message": str(e),
            "expected_version": e.expected_version,
            "actual_version": e.actual_version,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def list_contexts(
    ctx: Context[ServerSession, None],
    namespace: str | None = Field(default=None, description="Filter by namespace (exact match)"),
    limit: int = Field(default=20, description="Maximum number of contexts to return"),
    working_directory: str | None = Field(default=None, description="Working directory for context storage"),
) -> dict:
    """
    List shared contexts, optionally filtered by namespace.

    Returns summaries (without full data) for efficiency.
    Use get_context to retrieve full data for a specific context.
    """
    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return {"success": False, "error": error}

    try:
        manager = _get_context_manager(working_directory)
        contexts = manager.list(
            namespace=namespace.strip() if namespace else None,
            limit=limit,
        )

        return {
            "success": True,
            "count": len(contexts),
            "contexts": [c.to_dict() for c in contexts],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def delete_context(
    ctx: Context[ServerSession, None],
    context_id: str = Field(description="Context ID to delete"),
    working_directory: str | None = Field(default=None, description="Working directory for context storage"),
) -> dict:
    """
    Delete a shared context.

    This permanently removes the context. Use with caution.
    """
    if not context_id or not context_id.strip():
        return {"success": False, "error": "'context_id' is required."}

    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return {"success": False, "error": error}

    try:
        manager = _get_context_manager(working_directory)
        deleted = manager.delete(context_id.strip())

        if deleted:
            return {
                "success": True,
                "message": f"Context '{context_id}' deleted.",
            }
        else:
            return {
                "success": False,
                "error": f"Context '{context_id}' not found.",
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def vacuum_contexts(
    ctx: Context[ServerSession, None],
    working_directory: str | None = Field(default=None, description="Working directory for context storage"),
) -> dict:
    """
    Clean up expired and over-quota contexts.

    Performs:
    1. Delete all expired contexts (past TTL)
    2. If over size quota (500MB), delete oldest by access time until under quota

    Returns statistics about cleanup.
    """
    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return {"success": False, "error": error}

    try:
        manager = _get_context_manager(working_directory)
        stats = manager.vacuum()

        return {
            "success": True,
            **stats,
            "message": f"Cleaned up {stats['total_deleted']} contexts.",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.resource("owlex://contexts/stats")
def context_stats_resource() -> str:
    """View context storage statistics."""
    try:
        manager = _get_context_manager()
        stats = manager.get_stats()

        lines = [
            "# Context Storage Statistics",
            "",
            f"**Total Contexts:** {stats['context_count']}",
            f"**Namespaces:** {stats['namespace_count']}",
            f"**Total Size:** {stats['total_size_bytes'] / 1024 / 1024:.2f} MB",
            f"**Quota Usage:** {stats['quota_usage_percent']:.1f}%",
            f"**Max Size:** {stats['max_size_bytes'] / 1024 / 1024:.0f} MB",
            f"**Max Age:** {stats['max_age_days']} days",
            f"**Expired (pending cleanup):** {stats['expired_count']}",
            "",
            f"**Oldest Created:** {stats['oldest_created'] or 'N/A'}",
            f"**Newest Updated:** {stats['newest_updated'] or 'N/A'}",
        ]

        return "\n".join(lines)
    except Exception as e:
        return f"Error getting context stats: {e}"


# === Git Worktree Tools ===

import subprocess


def _run_git_command(args: list[str], cwd: str | None = None) -> tuple[bool, str, str]:
    """Run a git command and return (success, stdout, stderr)."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


@mcp.tool()
async def create_worktree(
    ctx: Context[ServerSession, None],
    branch: str = Field(description="Branch name for the worktree (created if doesn't exist)"),
    path: str | None = Field(default=None, description="Path for worktree (default: worktrees/<branch>)"),
    base_branch: str = Field(default="HEAD", description="Base branch to create from (if creating new branch)"),
    working_directory: str | None = Field(default=None, description="Git repository directory"),
) -> dict:
    """
    Create a git worktree for parallel task isolation.

    Each worktree has its own working directory and can be on a different branch,
    enabling parallel development without conflicts. The Liza blackboard is
    automatically isolated per worktree.

    Example:
        create_worktree(branch="feature-auth")
        # Creates worktrees/feature-auth/ with its own .owlex/feature-auth/liza-state.yaml
    """
    if not branch or not branch.strip():
        return {"success": False, "error": "'branch' is required."}

    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return {"success": False, "error": error}

    wd = working_directory or os.getcwd()

    # Validate git repository
    success, _, stderr = _run_git_command(["rev-parse", "--git-dir"], cwd=wd)
    if not success:
        return {"success": False, "error": f"Not a git repository: {stderr}"}

    branch = branch.strip()
    # Default path: worktrees/<branch>
    worktree_path = path if path else f"worktrees/{branch}"
    full_path = os.path.join(wd, worktree_path)

    # Check if worktree already exists
    if os.path.exists(full_path):
        return {"success": False, "error": f"Path already exists: {worktree_path}"}

    # Check if branch exists
    success, _, _ = _run_git_command(["rev-parse", "--verify", f"refs/heads/{branch}"], cwd=wd)
    branch_exists = success

    if branch_exists:
        # Worktree with existing branch
        success, stdout, stderr = _run_git_command(
            ["worktree", "add", worktree_path, branch],
            cwd=wd,
        )
    else:
        # Create new branch from base
        success, stdout, stderr = _run_git_command(
            ["worktree", "add", "-b", branch, worktree_path, base_branch],
            cwd=wd,
        )

    if not success:
        return {"success": False, "error": f"Failed to create worktree: {stderr}"}

    return {
        "success": True,
        "branch": branch,
        "path": worktree_path,
        "full_path": full_path,
        "created_branch": not branch_exists,
        "message": f"Worktree created at {worktree_path}. Liza state will be isolated to .owlex/{branch}/",
    }


@mcp.tool()
async def list_worktrees(
    ctx: Context[ServerSession, None],
    working_directory: str | None = Field(default=None, description="Git repository directory"),
) -> dict:
    """
    List all git worktrees in the repository.

    Returns information about each worktree including path, branch, and commit.
    """
    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return {"success": False, "error": error}

    wd = working_directory or os.getcwd()

    # Validate git repository
    success, _, stderr = _run_git_command(["rev-parse", "--git-dir"], cwd=wd)
    if not success:
        return {"success": False, "error": f"Not a git repository: {stderr}"}

    # List worktrees in porcelain format
    success, stdout, stderr = _run_git_command(["worktree", "list", "--porcelain"], cwd=wd)
    if not success:
        return {"success": False, "error": f"Failed to list worktrees: {stderr}"}

    worktrees = []
    current = {}

    for line in stdout.split("\n"):
        if not line:
            if current:
                worktrees.append(current)
                current = {}
            continue

        if line.startswith("worktree "):
            current["path"] = line[9:]
        elif line.startswith("HEAD "):
            current["commit"] = line[5:]
        elif line.startswith("branch "):
            current["branch"] = line[7:].replace("refs/heads/", "")
        elif line == "detached":
            current["detached"] = True
        elif line == "bare":
            current["bare"] = True

    if current:
        worktrees.append(current)

    return {
        "success": True,
        "count": len(worktrees),
        "worktrees": worktrees,
    }


@mcp.tool()
async def prune_worktrees(
    ctx: Context[ServerSession, None],
    working_directory: str | None = Field(default=None, description="Git repository directory"),
    dry_run: bool = Field(default=True, description="If true, only report what would be pruned"),
) -> dict:
    """
    Prune stale git worktree entries.

    Removes worktree entries for paths that no longer exist.
    Use dry_run=True (default) to preview before actually pruning.
    """
    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return {"success": False, "error": error}

    wd = working_directory or os.getcwd()

    # Validate git repository
    success, _, stderr = _run_git_command(["rev-parse", "--git-dir"], cwd=wd)
    if not success:
        return {"success": False, "error": f"Not a git repository: {stderr}"}

    args = ["worktree", "prune"]
    if dry_run:
        args.append("--dry-run")
    args.append("-v")

    success, stdout, stderr = _run_git_command(args, cwd=wd)
    if not success:
        return {"success": False, "error": f"Failed to prune worktrees: {stderr}"}

    # Parse output for pruned entries
    pruned = []
    for line in (stdout + "\n" + stderr).split("\n"):
        if "Removing" in line or "pruning" in line.lower():
            pruned.append(line.strip())

    return {
        "success": True,
        "dry_run": dry_run,
        "pruned_count": len(pruned),
        "pruned": pruned,
        "message": f"{'Would prune' if dry_run else 'Pruned'} {len(pruned)} stale worktree(s).",
    }


@mcp.resource("owlex://worktrees")
def worktrees_resource() -> str:
    """View git worktrees in current repository."""
    try:
        wd = os.getcwd()
        success, _, stderr = _run_git_command(["rev-parse", "--git-dir"], cwd=wd)
        if not success:
            return f"Not a git repository: {stderr}"

        success, stdout, stderr = _run_git_command(["worktree", "list"], cwd=wd)
        if not success:
            return f"Failed to list worktrees: {stderr}"

        lines = [
            "# Git Worktrees",
            "",
            "```",
            stdout,
            "```",
        ]

        return "\n".join(lines)
    except Exception as e:
        return f"Error listing worktrees: {e}"


def main():
    """Entry point for owlex-server command."""
    import argparse
    import signal

    from . import __version__

    parser = argparse.ArgumentParser(
        prog="owlex-server",
        description="MCP server for multi-agent CLI orchestration"
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"owlex {__version__}"
    )
    parser.parse_args()

    async def run_with_cleanup():
        loop = asyncio.get_running_loop()
        shutdown_event = asyncio.Event()

        def signal_handler(sig):
            _log(f"Received signal {sig}, shutting down...")
            shutdown_event.set()

        # Register signal handlers for graceful shutdown
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda s=sig: signal_handler(s))
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass

        engine.start_cleanup_loop()
        try:
            # Run MCP server with shutdown monitoring
            server_task = asyncio.create_task(mcp.run_stdio_async())
            shutdown_task = asyncio.create_task(shutdown_event.wait())

            done, pending = await asyncio.wait(
                [server_task, shutdown_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            # If shutdown was triggered, cancel the server
            if shutdown_task in done:
                server_task.cancel()
                try:
                    await server_task
                except asyncio.CancelledError:
                    pass
            # If server task completed (possibly due to client disconnect), log it
            if server_task in done:
                try:
                    server_task.result()  # Re-raise any exception
                except Exception as e:
                    _log(f"Server task ended: {e}")
        except asyncio.CancelledError:
            _log("Server cancelled")
        except Exception as e:
            _log(f"Server error: {e}")
        finally:
            # Kill all running tasks before exit
            await engine.kill_all_tasks()
            engine.stop_cleanup_loop()
            _log("Server shutdown complete.")

    asyncio.run(run_with_cleanup())


if __name__ == "__main__":
    main()
