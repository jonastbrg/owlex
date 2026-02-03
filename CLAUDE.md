# Owlex - Claude Code Configuration

## Overview

Owlex is an MCP server that provides access to multiple AI agents (Codex, Gemini, OpenCode) from within Claude Code. It enables:

1. **Council Deliberation** - Get multiple AI perspectives with optional revision rounds
2. **Liza Peer Review** - Claude implements, external agents review with binding verdicts
3. **Individual Sessions** - Direct agent access with session persistence

## Quick Reference

### Council (Consultation)

```
/council "Your question here"
```

Or with tools:
```python
council_ask(
    prompt="Your question",
    team="security_audit",  # or roles=["security", "perf", "skeptic"]
    deliberate=True,
    critique=True,
)
```

### Liza (Peer-Reviewed Implementation)

```
/liza "Implement feature X"
```

**Flow:**
1. `liza_start` - Create task
2. Claude implements (Write/Edit/Bash)
3. `liza_submit` - Send for review
4. If REJECT → fix and resubmit
5. If ALL APPROVE → done

**Tools:**
| Tool | Purpose |
|------|---------|
| `liza_start` | Create implementation task |
| `liza_submit` | Submit for review |
| `liza_status` | Check task status |
| `liza_feedback` | Get reviewer feedback |

**Blackboard:** `.owlex/liza-state.yaml`

### Individual Agents

```python
# Codex
start_codex_session(prompt="...")
resume_codex_session(prompt="...", session_id="...")

# Gemini
start_gemini_session(prompt="...")
resume_gemini_session(prompt="...", session_ref="latest")

# OpenCode
start_opencode_session(prompt="...")
resume_opencode_session(prompt="...")
```

## When to Use What

| Scenario | Tool |
|----------|------|
| Need multiple opinions | `/council` |
| Production code with review | `/liza` |
| Deep code analysis | `start_codex_session` |
| Large codebase (1M tokens) | `start_gemini_session` |
| Alternative perspective | `start_opencode_session` |
| Architecture decisions | `council_ask team="architecture_review"` |
| Security review | `council_ask team="security_audit"` |

## Hooks

The plugin includes hooks that inject framework guidance at key moments:

| Hook | Trigger | Injects |
|------|---------|---------|
| `PreToolUse` | `liza_start` | Framework protocol (Task delegation, Coder Contract) |
| `PreToolUse` | `EnterPlanMode` | Planning reminder (if .owlex/ or framework/ exists) |
| `PreToolUse` | `Edit\|Write` | Liza task check (asks if no active task) |

These hooks ensure you don't forget to:
- Use Task tool for codebase exploration instead of manual search
- Parallelize independent agent calls
- Submit for review via `liza_submit`

## Liza Architecture

```
Claude Code = Orchestrator + Coder (trusted)
Codex/Gemini/OpenCode = Reviewers (external validation)
```

**Key Principles:**
- Claude implements, cannot self-approve
- Reviewers provide binding APPROVE/REJECT verdicts
- Multiple reviewers catch different issues
- Iterate until all approve or max iterations

**Contract Rules (from Liza):**
- No fabrication - don't claim to have done something you haven't
- No test corruption - never modify tests to make failing code pass
- No scope creep - only implement what was requested
- Be honest about assumptions and limitations

## Configuration

Environment variables:
```bash
COUNCIL_EXCLUDE_AGENTS=""        # Skip agents: "opencode,gemini"
COUNCIL_DEFAULT_TEAM=""          # Default team preset
OWLEX_DEFAULT_TIMEOUT=300        # Timeout in seconds
CODEX_BYPASS_APPROVALS=false     # Bypass sandbox
GEMINI_YOLO_MODE=false           # Auto-approve actions
```

## Development

```bash
# Run tests
pytest tests/test_liza.py -v

# Install locally
pip install -e .

# Check module
python -c "from owlex.liza import LizaOrchestrator; print('OK')"
```

## File Structure

```
owlex/
├── server.py          # MCP server with all tools
├── council.py         # Council deliberation logic
├── engine.py          # Agent execution engine
├── liza/              # Liza peer-review system
│   ├── blackboard.py  # State management
│   ├── contracts.py   # Behavioral rules
│   ├── orchestrator.py # Review loop coordination
│   └── protocol.py    # Verdict parsing
├── skills/            # Slash command skills
│   ├── liza/          # /liza skill
│   └── council/       # /council skill
└── commands/          # Command documentation
    ├── liza.md
    └── council.md
```

---

## Memora Integration (Persistent Memory)

### Automatic Memory Protocol

After significant events, Claude MUST create memora memories:

| Event | Memory Action |
|-------|---------------|
| Council deliberation | `memory_create` with all perspectives + consensus |
| Liza APPROVE | `memory_create` with implementation summary |
| Liza REJECT | `memory_create` with feedback for learning |
| Design decision | `memory_create` with rationale |

### Integration Flow

```
Linear Issue → liza_start → Claude implements → liza_submit
                                                    ↓
                                          Council reviews
                                                    ↓
                              APPROVE → linear_update + memory_create
                              REJECT  → memory_create(feedback) → fix → resubmit
```

### Memory Templates

**After Council:**
```python
mcp__memora__memory_create(
    content="COUNCIL: [Question]\n\nPerspectives:\n- Codex: [view]\n- Gemini: [view]\n\nConsensus: [outcome]",
    tags=["council", "decision"],
    metadata={"participants": ["codex", "gemini"]}
)
```

**After Liza Approval:**
```python
mcp__memora__memory_create(
    content="LIZA: [Task]\nIterations: N\nVerdict: APPROVED\nKey feedback: [points]",
    tags=["liza", "approved"],
    metadata={"iterations": N, "reviewers": ["codex", "gemini"]}
)
```

### Knowledge Graph

View at `http://localhost:8765/graph`

---

## Task Tool Delegation

**Use the Task tool** to spawn specialized agents for exploration and analysis. Don't manually search when agents can do it better.

### Built-in Agent Types

| Agent | Use For | Example |
|-------|---------|---------|
| `Explore` | Fast codebase exploration | "Find all MCP tool definitions" |
| `Plan` | Implementation strategy | "Plan the rate limiting feature" |
| `general-purpose` | Complex multi-step research | "Analyze how errors propagate" |
| `Bash` | Git/command operations | "Check recent commits" |

### Feature Development Agents

| Agent | Use For |
|-------|---------|
| `feature-dev:code-reviewer` | Bug/security/quality review |
| `feature-dev:code-explorer` | Architecture mapping, pattern analysis |
| `feature-dev:code-architect` | Feature design blueprints |

### Parallel Execution

Launch independent agents in a single message for maximum efficiency:

```
Task(subagent_type="Explore", prompt="Find auth files")
Task(subagent_type="Explore", prompt="Find API endpoints")
# Both run concurrently
```

### When NOT to Delegate

- Simple file reads → use `Read` tool
- Specific class lookup → use `Grep/Glob`
- Known file edits → do directly

---

## Framework Task Claimer

The `framework/` subproject provides automated task dispatch:

### Architecture

```
Linear Issue → Webhook → liza_start → Blackboard
                                         ↓
                              Task Claimer (polls)
                                         ↓
                              Claim + Dispatch Agent
                                         ↓
                              Agent implements → liza_submit
                                         ↓
                              Council reviews
                                         ↓
                              APPROVE → PR created
```

### Task Claimer Service

Location: `framework/src/linear_bridge/task_claimer.py`

**What it does:**
1. Polls `.owlex/liza-state.yaml` for UNCLAIMED tasks
2. Claims task (updates status to WORKING, sets coder)
3. Dispatches to configured agent (claude/codex/gemini)
4. Monitors agent process
5. On failure: unclaims task for retry

**Configuration:**
```bash
LIZA_WORKING_DIRECTORY=/path/to/project
CLAIMER_POLL_INTERVAL=5        # seconds
CLAIMER_MAX_CONCURRENT=1       # parallel tasks
CLAIMER_DISPATCH_AGENT=claude  # claude|codex|gemini
CLAIMER_DRY_RUN=false         # log only, no dispatch
```

**Run:**
```bash
# As entry point
task-claimer

# As module
python -m linear_bridge.task_claimer
```

### Blackboard States

| Status | Meaning |
|--------|---------|
| UNCLAIMED | Task waiting to be picked up |
| WORKING | Agent is implementing |
| IN_REVIEW | Submitted, awaiting council verdict |
| DONE | All reviewers approved |
| REJECTED | Reviewers rejected, needs fixes |

### Agent Dispatch

The claimer builds a prompt instructing the agent to:
1. Implement the task
2. Call `liza_submit` when done
3. Handle rejections and resubmit
4. Continue until approved or max iterations

### Failure Handling

- **Agent exits non-zero**: Task reverted to UNCLAIMED
- **Agent not found**: Task reverted, error logged
- **Shutdown signal**: Active processes terminated, tasks unclaimed

---

## Linear Integration

The framework provides full Linear integration for issue-driven development.

### Agent Assignment

Assign issues to AI agents directly in Linear:
- `@Gemini Researcher` - Large codebase exploration, research tasks
- `@Claude Orchestrator` - General implementation with peer review
- `@Codex` - Deep code analysis, debugging

When an issue is assigned to an agent:
1. Linear sends `AgentSession` webhook
2. Webhook acknowledges within 10 seconds (required by Linear)
3. Task is created via `liza_start`
4. Agent implements and submits for review
5. Linear status updated throughout

### Status Mapping

| Liza Status | Linear Status |
|-------------|---------------|
| CLAIMED | In Progress |
| WORKING | In Progress |
| READY_FOR_REVIEW | In Review |
| IN_REVIEW | In Review |
| APPROVED | Done |
| REJECTED | In Progress |
| MERGED | Done |
| SUPERSEDED | Canceled |

### Webhook Configuration

```bash
# Required
LINEAR_WEBHOOK_SECRET=your-webhook-secret
LINEAR_API_KEY=lin_api_...

# Per-agent secrets (multi-agent setup)
LINEAR_WEBHOOK_SECRET_GEMINI=secret-for-gemini
LINEAR_WEBHOOK_SECRET_CLAUDE=secret-for-claude

# Per-agent OAuth tokens
LINEAR_OAUTH_TOKEN_GEMINI=oauth-token
LINEAR_OAUTH_TOKEN_CLAUDE=oauth-token

# Agent ID mapping
LINEAR_AGENT_GEMINI=linear-agent-uuid
LINEAR_AGENT_CLAUDE=linear-agent-uuid

# Optional
LINEAR_ALLOWED_ORGS=org-id-1,org-id-2
```

### Workflow

```
Linear Issue Created
        ↓
Assign to @Gemini Researcher (or other agent)
        ↓
AgentSession webhook received
        ↓
Acknowledge session (within 10s)
        ↓
liza_start(issue description)
        ↓
Agent implements in worktree
        ↓
liza_submit → Council reviews
        ↓
APPROVE → Linear status = Done
REJECT  → Linear status = In Progress, iterate
```

### Follow-up Messages

Users can send follow-up messages in Linear to refine the task:
- Comments mentioning the agent trigger follow-up processing
- The existing session is resumed with the new context
- Agent continues work based on the additional guidance

### Running the Webhook Server

```bash
# Start webhook receiver
uvicorn linear_bridge.webhook:app --host 0.0.0.0 --port 8080

# Or with the entry point
linear-webhook

# Start status updater (watches blackboard → updates Linear)
python -m linear_bridge.updater
```
