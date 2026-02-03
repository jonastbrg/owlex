# Framework (Multi-Agent Orchestration)

**Last Updated**: 2025-02-02
**Status**: Maintenance

## Current Focus

Maintenance mode. Linear integration and Liza peer-review system are stable. Bug fixes only, no new features.

## Architecture

```
Linear -> Webhook -> Liza -> Council -> PR
```

## Components

| Component | Status | Purpose |
|-----------|--------|---------|
| Linear Bridge | Stable | Webhook receiver, status sync |
| Liza System | Stable | Peer-review orchestration |
| Task Claimer | Stable | Automated task dispatch |
| CUA Config | Stable | Sandbox isolation |

## Recent Progress

- 2025-01-29: Linear agent assignment integration complete
- Task claimer service working
- Multi-agent webhook support added

## Key Decisions

- Liza mandatory for all code changes (no self-certification)
- Multiple reviewers (Codex + Gemini) catch different issues
- Blackboard pattern for state (`.owlex/liza-state.yaml`)

## Blockers / Risks

- None currently (maintenance mode)

## Next Steps

1. Bug fixes as needed
2. Documentation improvements

## Context for Agents

**Repository**: `~/owlex/framework`

**Key files**:
- `src/linear_bridge/` - Webhook receiver, status sync
- `src/cua_config/` - Sandbox isolation
- `.owlex/liza-state.yaml` - Task state blackboard

**Commands**:
- Start webhook: `uvicorn linear_bridge.webhook:app --port 8080`
- Start task claimer: `python -m linear_bridge.task_claimer`

**Environment**:
- `LINEAR_WEBHOOK_SECRET` - Required
- `LINEAR_API_KEY` - Required for status updates

**MANDATORY**: ALL code changes must use Liza workflow (liza_start -> implement -> liza_submit -> review -> commit).
