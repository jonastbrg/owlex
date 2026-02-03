# Liza Skill - Peer-Supervised Coding

<skill-context>
Liza is a peer-supervised coding system where Claude (coder) implements tasks and external agents (Codex, Gemini) review with binding verdicts. Based on https://github.com/liza-mas/liza.

Use this skill when:
- User wants rigorous code review before completion
- User says "liza", "peer review", "external review", "adversarial review"
- User wants multiple AI perspectives on implementation quality
- User wants to ensure code is production-ready
</skill-context>

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Claude Code                               │
│         Orchestrator + CODER (trusted implementer)           │
└─────────────────────────────────────────────────────────────┘
                              │
                    [implementation]
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
    ┌───────────┐        ┌──────────┐        ┌──────────┐
    │   Codex   │        │  Gemini  │        │ OpenCode │
    │ (Reviewer)│        │(Reviewer)│        │(Reviewer)│
    └───────────┘        └──────────┘        └──────────┘
```

## Workflow

### Step 1: Start Task

```
Use mcp__owlex__liza_start to create the task:

Parameters:
- task_description: What to implement
- reviewers: ["codex", "gemini"] (default)
- max_iterations: 5 (default)
- done_when: Optional completion criteria
```

### Step 2: Implement

Claude implements the task using standard tools (Write, Edit, Bash).
Follow the Coder Contract:
- No fabrication - don't claim to have done something you haven't
- No test corruption - never modify tests to make failing code pass
- No scope creep - only implement what was requested
- Be honest about assumptions and limitations

### Step 3: Submit for Review

```
Use mcp__owlex__liza_submit:

Parameters:
- task_id: From liza_start
- implementation_summary: Description of what you implemented
```

Reviewers will examine and return:
- **APPROVE**: Code meets requirements, no blocking issues
- **REJECT**: Issues found that must be fixed + specific feedback

### Step 4: Handle Verdict

**If ALL APPROVE:**
- Task complete! Celebrate.

**If ANY REJECT:**
1. Read the merged feedback carefully
2. Fix the identified issues
3. Do NOT introduce unrelated changes
4. Call liza_submit again with updated summary
5. Repeat until approved or max iterations

## Key Principles

### External Validation
Claude cannot self-approve. Reviewers provide binding verdicts.
This prevents:
- Sycophancy
- Phantom fixes
- Hallucinated completions
- Self-certification bias

### Critique Mode
Reviewers actively look for:
- Security vulnerabilities
- Race conditions
- Resource leaks
- Unhandled edge cases
- Missing validation
- Incorrect error handling

### Multi-Reviewer Benefit
Different reviewers catch different issues:
- Codex: Deep code analysis, bug finding
- Gemini: Architecture, patterns, large-scale issues
- OpenCode: Alternative perspective, edge cases

## Example Session

```
User: /liza Add rate limiting to the login endpoint

Claude: I'll create a Liza task for this.

[Calls liza_start with task_description]
→ Returns task_id: "task-1"

Claude: Now implementing rate limiting...

[Uses Write/Edit to add rate limiting code]

Claude: Implementation complete. Submitting for review.

[Calls liza_submit with implementation_summary]
→ Codex: REJECT - "Missing IP-based limiting, only user-based"
→ Gemini: REJECT - "No Redis backend for distributed systems"

Claude: I see the issues. Let me fix them.

[Fixes the implementation]

[Calls liza_submit again]
→ Codex: APPROVE
→ Gemini: APPROVE

Claude: All reviewers approved! Rate limiting is now implemented with:
- User-based rate limiting
- IP-based rate limiting
- Redis backend for distributed systems
```

## MCP Tools Reference

| Tool | Description |
|------|-------------|
| `liza_start` | Create a task for Claude to implement |
| `liza_submit` | Submit implementation for review |
| `liza_status` | Get current task status |
| `liza_feedback` | Get formatted feedback to address |

## Blackboard

State persisted in `.owlex/liza-state.yaml`:
- Task lifecycle tracking
- Review history
- Iteration count
- Merged feedback

View with: `owlex://liza/blackboard` resource

## Linear Integration

Liza integrates with Linear for issue-driven development. Tasks can originate from Linear issues assigned to AI agents.

### Architecture

```
Linear Issue → @mention Agent → Webhook
                                    ↓
                         ┌──────────────────┐
                         │  Webhook Server  │ ← Must ACK within 10s
                         └────────┬─────────┘
                                  ↓
                            liza_start
                                  ↓
                    ┌─────────────────────────┐
                    │  Blackboard (.owlex/)   │
                    └─────────────────────────┘
                         ↑              ↓
              ┌──────────┴──┐    ┌──────┴───────┐
              │   Updater   │    │ Task Claimer │
              │ (→ Linear)  │    │ (→ Agent)    │
              └─────────────┘    └──────────────┘
```

### Assigning Issues to Agents

In Linear, assign issues via @mention to AI agents:

| @mention | Internal Name | CLI | Use For |
|----------|---------------|-----|---------|
| `@Claude Coder` | `claude` | `claude --print --dangerously-skip-permissions` | General implementation |
| `@Gemini Researcher` | `gemini` | `gemini` | Large codebase exploration |
| `@Codex` | `codex` | `codex --approval-mode full-auto` | Deep code analysis |
| `@Grok Coder` | `grok` | `opencode` | Alternative perspective |

**Auto-assignment:** Issues are assigned to both Jonathan Steinberg (human, for oversight) and the AI agent.

### Creating Issues via MCP

Use `mcp__plugin_linear_linear__create_issue`:
- `assignee`: Human reviewer (e.g., "Jonathan Steinberg" or "me")
- `delegate`: AI agent (e.g., "Claude Coder")
- `state`: Initial status (e.g., "Backlog", "Todo")
- `parentId`: Parent issue ID for sub-issues

### Delegation Strategies

**Single agent:** Assign one issue to one agent
```
Issue: "Add rate limiting" → @Claude Coder
```

**Multi-agent collaboration:** Assign one issue to multiple agents
```
Issue: "Review security" → @Claude Coder @Gemini Researcher
```

**Divide and conquer:** Create sub-issues for parallel work
```
Parent: "Implement auth system"
├── Sub-issue: "Design auth flow"       → @Gemini Researcher
├── Sub-issue: "Implement JWT"          → @Claude Coder
├── Sub-issue: "Add OAuth providers"    → @Claude Coder
└── Sub-issue: "Security audit"         → @Codex
```

### Workflow

1. **Create issue** in Linear (assign to human + AI agent)
2. **Webhook fires** → AgentSession created
3. **Webhook ACKs** within 10 seconds (required)
4. **liza_start** creates task on blackboard
5. **Task Claimer** claims and dispatches to agent
6. **Agent implements** the task
7. **liza_submit** sends to council
8. **Council reviews** → APPROVE or REJECT
9. **Updater** syncs status to Linear

### Linear Issue Statuses

Standard workflow states:
- **Backlog** / **Todo** - Not started
- **In Progress** - Agent working
- **In Review** - Council reviewing
- **Done** - Approved and complete
- **Canceled** - Superseded or abandoned

### Status Mapping (Liza → Linear)

| Liza Status | Linear Status |
|-------------|---------------|
| UNCLAIMED | *(no update)* |
| WORKING | In Progress |
| READY_FOR_REVIEW | In Review |
| IN_REVIEW | In Review |
| APPROVED | Done |
| REJECTED | In Progress |
| MERGED | Done |
| SUPERSEDED | Canceled |

### Follow-up Messages

@mention the agent in comments to continue the conversation:
- Triggers `PROMPTED` webhook action
- Existing session resumed
- Agent continues based on feedback

## Task Claimer (Automated Dispatch)

The Task Claimer (`framework/src/linear_bridge/task_claimer.py`) polls the blackboard and dispatches tasks.

### Polling Cycle (every 5s)

```
1. Release tasks from cooldown (if expired)
2. Skip if agent in cooldown
3. Get UNCLAIMED tasks
4. For each: claim → dispatch → monitor
```

### Dispatch Prompt

```
You have been assigned Liza task {task_id}.
Task: {description}
Instructions:
1. Implement using appropriate tools
2. Call liza_submit when done
3. If rejected, fix and resubmit
4. Continue until approved
```

### Error Handling

| Error Type | Detection | Action |
|------------|-----------|--------|
| Quota/rate limit | `usage limit`, `rate limit exceeded`, `429` | 1hr cooldown |
| Timeout | `timed out`, `command timed out` | 5min cooldown |
| Other failure | Non-zero exit | Unclaim for retry |

### Running the Services

```bash
# 1. Webhook server
uvicorn linear_bridge.webhook:app --host 0.0.0.0 --port 8080

# 2. Task Claimer
task-claimer

# 3. Status Updater
python -m linear_bridge.updater -d /path/to/project
```

## Configuration

Default reviewers: `["codex", "gemini"]`
Default max iterations: 5
Critique mode: enabled (reviewers actively find issues)
Require all approve: true (all reviewers must approve)
