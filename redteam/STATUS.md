# Redteam (Adversarial Attacks)

**Last Updated**: 2026-02-03
**Status**: Active
**Repository**: `~/embodied-safety/redteam/`

## Current Focus

Adversarial attack orchestrators for embodied AI systems. TAP, PAIR, Crescendo implementations.

## Progress

| Component | Progress | Status |
|-----------|----------|--------|
| TAP orchestrator | 90% | Complete |
| PAIR orchestrator | 85% | Active |
| Crescendo orchestrator | 80% | Active |
| Campaign runner | 85% | Active |
| LLM judge scorer | 90% | Complete |
| Probe scorer | 75% | Active |

## Recent Progress

- TAPOrchestrator with tree-based search + pruning
- CampaignRunner for multi-behavior testing
- HarmBench/BadRobot behavior sources integrated

## Key Decisions

- Focus on transferable attacks
- Prioritize safety-relevant failure modes
- Protocol-driven design (Target, Scorer, Orchestrator interfaces)

## Blockers / Risks

- None currently

## Next Steps

1. Expand attack surface coverage
2. Multi-modal attack support (Phase 3 Lightwheel)
3. Document attack effectiveness

## Context for Agents

**Location**: `~/embodied-safety/redteam/`
**Key Files**: `runner.py`, `orchestrators/tap.py`, `protocols.py`
**Tests**: `pytest redteam/tests/`
