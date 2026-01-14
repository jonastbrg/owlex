---
name: council
description: Get multiple AI perspectives on a question via Owlex Council
user_invocable: true
---

# Council Skill

Get perspectives from multiple AI agents (Codex, Gemini, OpenCode) with optional deliberation.

## Usage

```
/council <your question>
```

## Examples

```
/council Should we use microservices or a monolith for this project?
/council Review this architecture for potential issues
/council What's the best approach for handling auth in this codebase?
```

## Instructions

When invoked, you should:

1. **Parse the question** from the user's prompt
2. **Form your initial opinion** (recommended for richer deliberation)
3. **Call the council** using the Owlex MCP:

```python
mcp__owlex__council_ask(
    prompt="<user's question>",
    claude_opinion="<your initial take>",
    working_directory="<current project path>",
    deliberate=True,  # Agents refine after seeing each other
    timeout=300
)
```

4. **Wait for results**:
```python
mcp__owlex__wait_for_task(task_id="<from above>", timeout=300)
```

5. **Synthesize and present**:
   - Areas of agreement
   - Notable disagreements
   - Unique insights from each agent
   - Your recommendation based on the council

## When to Use

- Architectural decisions with trade-offs
- Complex debugging needing multiple perspectives
- Design reviews
- Security audits (multiple eyes)
- Any decision where you want diverse viewpoints
