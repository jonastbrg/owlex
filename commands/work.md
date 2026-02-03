# /work - Orchestrator Mode

Activates Orchestrator Mode with intelligent task routing and agent delegation.

## Usage

```bash
/work <task description>
```

## What It Does

1. **Context Loading**: Searches Memora for relevant project memories
2. **Task Analysis**: Determines optimal routing based on task characteristics
3. **Agent Delegation**: Spawns specialized agents via Task tool for exploration
4. **Execution**: Routes to appropriate mode (Liza, Council, or direct agent)
5. **Checkpoint**: Persists learnings at completion

## Routing Logic

| Task Type | Route |
|-----------|-------|
| Implementation | `/liza` (peer-reviewed coding) |
| Architecture/security decision | `/council` (deliberation) |
| Code exploration | Task tool with `Explore` agent |
| Code review | Task tool with `feature-dev:code-reviewer` |
| Research/analysis | Task tool with `general-purpose` agent |

## Task Tool Delegation

The `/work` skill uses the Task tool to spawn specialized agents:

```python
# Parallel codebase exploration
Task(subagent_type="Explore", prompt="Find authentication files")
Task(subagent_type="Explore", prompt="Find API endpoint definitions")

# Code analysis
Task(subagent_type="feature-dev:code-explorer", prompt="Analyze error handling patterns")
```

### Available Agent Types

| Agent | Purpose |
|-------|---------|
| `Explore` | Fast codebase exploration |
| `Plan` | Implementation strategy design |
| `general-purpose` | Multi-step autonomous research |
| `feature-dev:code-reviewer` | Bug/security/quality review |
| `feature-dev:code-explorer` | Architecture mapping |
| `feature-dev:code-architect` | Feature design blueprints |

## Examples

```bash
# Implementation task → routes to /liza
/work Add rate limiting to the API endpoints

# Architecture question → routes to /council
/work Should we use microservices or a monolith?

# Exploration task → uses Task tool with Explore agent
/work Understand how authentication works in this codebase

# Code review → uses Task tool with code-reviewer agent
/work Review the recent changes for security issues
```

## Memory Integration

The `/work` skill integrates with Memora:

- **Start**: Loads relevant memories for context
- **End**: Creates checkpoint memory with task outcome

```python
# Load context
mcp__memora__memory_hybrid_search(
    query="[task keywords]",
    limit=5,
    tags_all=["project:current"]
)

# Save checkpoint
mcp__memora__memory_create(
    content="COMPLETED: [summary]",
    tags=["project:X", "checkpoint:final"]
)
```

## Related Commands

- `/liza` - Direct access to peer-reviewed coding mode
- `/council` - Direct access to council deliberation
- `/cr` - Quick code review via Codex
