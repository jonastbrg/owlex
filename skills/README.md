# Owlex Skills for Claude Code

These skills provide slash commands for invoking owlex agents from Claude Code.

## Installation

Copy the skill directories to your Claude Code skills folder:

```bash
cp -r skills/* ~/.claude/skills/
```

## Available Skills

| Skill | Description |
|-------|-------------|
| `/work` | Orchestrator mode - analyzes tasks and delegates to optimal agents via Task tool |
| `/council` | Consult the AI council (Codex + Gemini + OpenCode + Grok) with 2-round deliberation |
| `/codex` | Start an OpenAI Codex CLI session for deep reasoning |
| `/gemini` | Start a Google Gemini CLI session with 1M context |
| `/critique` | Council critique mode - agents find bugs and flaws |
| `/liza` | Peer-reviewed coding - Claude implements, Codex/Gemini/Grok review with binding verdicts |

## Usage

### Council Skills

All council skills are **non-blocking** - they start the task and return immediately with a task_id.

```bash
/council How should I structure this auth system?
# Returns: Council started (task: abc-123...)

# Check results later:
mcp__owlex__get_task_result task_id=abc-123...
```

### Liza Skill

Liza is a peer-review loop:

```bash
/liza Add rate limiting to the login endpoint

# Claude implements, then:
liza_submit(task_id, "Added rate limiting with Redis...")

# Reviewers examine, provide feedback
# Loop until all approve
```

### Work Skill (Orchestrator Mode)

The `/work` skill activates Orchestrator Mode with Task tool delegation:

```bash
/work Implement user authentication

# Claude analyzes the task:
# 1. Loads context from Memora
# 2. Spawns Explore agents to find existing patterns
# 3. Routes to /liza for implementation with review
# 4. Persists learnings at checkpoints
```

Key feature: Uses the **Task tool** to delegate to specialized agents:

```python
# Parallel exploration
Task(subagent_type="Explore", prompt="Find auth files")
Task(subagent_type="feature-dev:code-explorer", prompt="Analyze security patterns")
```

## Requirements

- Claude Code with owlex MCP server configured
- Codex CLI, Gemini CLI, and/or OpenCode installed (depending on which agents you want)
- For Grok: `XAI_API_KEY` environment variable
