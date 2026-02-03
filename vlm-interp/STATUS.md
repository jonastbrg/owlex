# VLM Interpretability

**Last Updated**: 2026-02-03
**Status**: Active
**Repository**: `~/embodied-safety/vlm_interp/`

## Current Focus

Understanding internal representations in vision-language models. Circuit breakers and safety probes.

## Progress

| Component | Progress | Status |
|-----------|----------|--------|
| Activation extraction | 80% | Active |
| Circuit breakers | 70% | Active |
| Safety probes | 50% | In Progress |

## Recent Progress

- Circuit breaker implementation complete
- SafetyMonitor aggregation working
- Activation extractor functional

## Key Decisions

- Use normalized directions with threshold=0 as decision boundary
- Linear probes for interpretability

## Blockers / Risks

- None currently

## Next Steps

1. Define interpretability metrics
2. Complete probe training pipeline
3. Integration with Lightwheel (Phase 2)

## Context for Agents

**Location**: `~/embodied-safety/vlm_interp/`
**Key Files**: `repe/circuit_breakers.py`, `activations/extractor.py`
**Tests**: `pytest vlm_interp/tests/`
