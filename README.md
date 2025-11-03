# Claude CLI MCP Server

An MCP server that integrates the Claude CLI with Claude Code so you can hand off
reviews, audits, or ad-hoc commands to the local Claude tooling without leaving
your editor.

## Features
- **Async Claude Runs** – kick off Claude CLI sessions and get notified when they finish.
- **Workspace Awareness** – run reviews or commands from any directory or subset of files.
- **Review Memory** – optional JSONL log to avoid repeating identical findings.
- **Configurable Flags** – tune the underlying `claude auto` invocation through environment variables.

## Installation

1. **Create and activate a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify the Claude CLI is available**
   ```bash
   which claude
   ```
   Follow Anthropic’s installation instructions if the binary is missing.

4. **Configure Claude Code**

   Add the server to `~/.claude.json` (adjust the paths to your checkout):
   ```json
   {
     "mcpServers": {
       "claude-cli": {
         "type": "stdio",
         "command": "/path/to/claude-cli-mcp/venv/bin/python3",
         "args": ["/path/to/claude-cli-mcp/claude_mcp_server.py"],
         "env": {
           "CLAUDE_CLEAN_OUTPUT": "true"
         }
       }
     }
   }
   ```

5. **Restart Claude Code** so the new server is discovered.

## Slash Commands

You can mirror your existing commands or introduce new ones, for example:

```
/review-with-claude Investigate the flaky login tests
/ask-claude Summarise src/auth/*
```

Claude Code will route the request through this MCP server, which in turn calls
the Claude CLI.

## Environment Variables

All settings accept string booleans (`true` / `false`) and fall back to the Codex
equivalents for backward compatibility.

- `CLAUDE_BYPASS_APPROVALS` – default `false`; set to `true` to run without sandbox/approval guards.
- `CLAUDE_CLEAN_OUTPUT` – default `true`; set to `false` to keep Claude CLI prompt scaffolding.
- `CLAUDE_CLI_BINARY` – override the `claude` executable path.
- `CLAUDE_AUTO_SUBCOMMAND` – defaults to `auto`; change if you want to call a different workflow.
- `CLAUDE_AUTO_ARGS` – extra arguments appended to every auto invocation (parsed with `shlex`).
- `CLAUDE_AUTO_STANDARD_ARGS` / `CLAUDE_AUTO_BYPASS_ARGS` – arguments appended depending on the approval mode (defaults: `""` and `--dangerously-skip-permissions` respectively).
- `CLAUDE_AUTO_STDIN_ARG` – optional flag that tells the CLI to read the task description from stdin.
- `CLAUDE_NOTIFICATION_PREVIEW_CHARS` – characters to echo in async completion previews (default `600`).

By default the MCP server reuses whatever `CLAUDE_CONFIG_DIR` the CLI already
uses (typically `~/.claude`). Set the variable yourself if you want to isolate
credentials or logs for MCP sessions.

## MCP Tools

The server exposes these tools to Claude Code (and any MCP client):

- `start_review` – launch a background Claude review of a plan or code snippet.
- `start_claude_command` – run any Claude CLI subcommand or natural-language task.
- `get_task_status` / `get_task_result` – poll task progress and retrieve stdout.
- `cancel_task` – stop an in-flight task.
- `query_review_memory` – inspect the optional review memory log.

When a background task finishes, the server now emits an async `[claude-async]` notification with
an inline preview of the output so you can stay in the prompt without immediately calling
`get_task_result`.
## Troubleshooting

```bash
# Run the MCP server directly
./venv/bin/python3 claude_mcp_server.py

# Inspect Claude Code logs
claude config logs
```

If the server reports `Error: 'claude' command not found`, ensure the CLI is in
your `PATH` or set `CLAUDE_CLI_BINARY` to the full path.
