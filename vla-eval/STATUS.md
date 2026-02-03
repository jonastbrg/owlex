# VLA Evaluation

**Last Updated**: 2026-02-03
**Status**: Active
**Repository**: `~/embodied-safety/vla_eval/`

## Current Focus

Building evaluation framework for vision-language-action models. Adversarial attack pipeline integration.

## Progress

| Component | Progress | Status |
|-----------|----------|--------|
| Attack pipeline | 75% | Active |
| VLA target adapter | 80% | Active |
| SimplerEnv integration | 60% | In Progress |
| Action scorer | 70% | Active |

## Recent Progress

- VLAAttackPipeline implemented with TAP/PAIR/Crescendo support
- VLATarget wraps models for redteam compatibility
- Action harm classifier working

## Key Decisions

- Standardized evaluation protocol across models
- Modular metric system
- Integration with redteam orchestrators

## Blockers / Risks

- None currently

## Next Steps

1. Complete baseline evaluation runs
2. Document evaluation metrics
3. Isaac Lab integration (Lightwheel Phase 1)

## Context for Agents

**Location**: `~/embodied-safety/vla_eval/`
**Key Files**: `adversarial/pipeline.py`, `adversarial/vla_target.py`
**Tests**: `pytest vla_eval/tests/`
