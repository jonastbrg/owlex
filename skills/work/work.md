# Work Skill - Orchestrator Mode with Task Delegation

<skill-context>
Orchestrator mode that intelligently routes tasks to specialized agents using the Task tool. This enables multi-agent collaboration for complex implementations.

Use this skill when:
- User says "/work" or "start working on"
- User wants a task implemented with proper routing
- Complex tasks requiring multiple agents
- User wants the orchestrator to decide the best approach
</skill-context>

## Overview

The `/work` skill activates Orchestrator Mode, which:
1. Analyzes the task to determine optimal routing
2. Loads relevant context from Memora (if available)
3. Delegates to specialized agents via the Task tool
4. Persists learnings at checkpoints

## Task Delegation via Task Tool

**CRITICAL**: For complex exploration, research, or multi-step tasks, use the Task tool to spawn specialized agents instead of doing everything directly.

### When to Use Task Tool

| Scenario | Agent Type | Why |
|----------|------------|-----|
| Codebase exploration | `Explore` | Fast, specialized for finding patterns |
| Code review | `feature-dev:code-reviewer` | Deep analysis with confidence scoring |
| Architecture analysis | `feature-dev:code-explorer` | Traces execution paths, maps layers |
| Implementation planning | `Plan` | Designs step-by-step approaches |
| General research | `general-purpose` | Multi-step autonomous research |

### Parallel Agent Execution

Launch multiple agents in parallel when tasks are independent:

```
Task tool call 1: Explore agent - "Find all authentication files"
Task tool call 2: Explore agent - "Find all API endpoint definitions"
Task tool call 3: feature-dev:code-explorer - "Analyze error handling patterns"
```

All three run concurrently, maximizing efficiency.

## Workflow

### Phase 1: Context Loading

```python
# Load relevant memories
mcp__memora__memory_hybrid_search(
    query="[task keywords]",
    limit=5,
    tags_all=["project:current"]
)
```

### Phase 2: Task Analysis & Routing

Analyze the task and choose the best approach:

| Task Type | Route To |
|-----------|----------|
| Implementation requiring review | `/liza` (Coder Mode) |
| Architecture/security decisions | `/council` (Deliberation) |
| Deep code analysis (< 5 files) | Codex via Task or direct |
| Large codebase exploration | Gemini via Task or direct |
| Multi-file refactoring | Plan agent first, then implement |

### Phase 3: Execution

**For Liza tasks:**
```
liza_start → Implement → liza_submit → Review Loop → Approval
```

**For Council tasks:**
```
council_ask(prompt, team="...", deliberate=True)
```

**For Agent delegation:**
```
Task tool with appropriate subagent_type
```

### Phase 4: Checkpoint & Persist

Only create memories at checkpoints:
- Task completion
- Council decision reached
- Error/failure (for learning)
- Session end

```python
mcp__memora__memory_create(
    content="COMPLETED: [summary]",
    tags=["project:X", "checkpoint:final"],
    metadata={"route": "liza|council|codex|gemini"}
)
```

## Agent Types for Task Tool

### Built-in Agents

| Agent | Use For |
|-------|---------|
| `Explore` | Quick codebase exploration, file finding |
| `Plan` | Implementation strategy design |
| `general-purpose` | Complex multi-step research |
| `Bash` | Git operations, command execution |

### Feature Development Agents

| Agent | Use For |
|-------|---------|
| `feature-dev:code-reviewer` | Bug/security/quality review |
| `feature-dev:code-explorer` | Architecture mapping, pattern analysis |
| `feature-dev:code-architect` | Feature design blueprints |

### PR Review Agents

| Agent | Use For |
|-------|---------|
| `pr-review-toolkit:code-reviewer` | Full PR review |
| `pr-review-toolkit:silent-failure-hunter` | Find silent failures in error handling |
| `pr-review-toolkit:type-design-analyzer` | Type invariant analysis |

## Example Session

```
User: /work Implement user session management

Claude: I'll analyze this task and set up the work.

1. Loading context from Memora...
   [Found 3 relevant memories about auth patterns]

2. This is an implementation task requiring review.
   Routing to Liza (Coder Mode).

3. Creating Liza task...
   [liza_start("Implement user session management")]

4. Exploring codebase for existing patterns...
   [Task tool: Explore agent - "Find authentication and session files"]

5. Based on exploration, implementing...
   [Write/Edit operations]

6. Submitting for review...
   [liza_submit with implementation summary]

7. Review feedback received, iterating...
   [Fix issues, resubmit]

8. Approved! Creating checkpoint memory.
   [memory_create with completion summary]
```

## Key Principles

1. **Use agents for exploration** - Don't search manually, spawn Explore agents
2. **Parallelize when possible** - Launch independent agents together
3. **Checkpoint, not N+1** - Only persist at meaningful milestones
4. **Route to specialists** - Match task type to agent strengths
5. **Liza for implementations** - Code changes need external review
