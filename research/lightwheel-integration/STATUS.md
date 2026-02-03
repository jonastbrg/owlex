# Lightwheel/Isaac Lab Integration

**Last Updated**: 2026-02-03
**Status**: Planning Complete
**Timeline**: 16 weeks to MVP

## Progress Overview

| Phase | Description | Progress | Status |
|-------|-------------|----------|--------|
| Phase 0 | Foundation Infrastructure (DB, GCP, Temporal) | 0% | Not Started |
| Phase 1 | Isaac Lab Integration | 0% | Not Started |
| Phase 2 | Safety Probe Training Pipeline | 0% | Not Started |
| Phase 3 | Attack Integration (all modalities) | 0% | Not Started |
| Phase 4 | vLLM Serving Layer | 0% | Not Started |
| Phase 5 | Demo & Enterprise Features | 0% | Not Started |
| **Overall** | | **0%** | **Planning Complete** |

## Current Focus

Strategic planning complete. Ready for Phase 0 implementation: database schema, GCP infrastructure, Temporal orchestration.

## Plan Documents

- [lightwheel-integration-plan.md](./lightwheel-integration-plan.md) - Strategic overview
- [lightwheel-integration-plan-detailed.md](./lightwheel-integration-plan-detailed.md) - Implementation spec with SQL schemas, interfaces

## Strategic Value

- **Lightwheel**: Has simulation infra (250+ tasks, Isaac Lab Arena), NO safety evaluation
- **Aspis AI**: Has safety methodology, NO simulation infra
- **Combined**: First comprehensive embodied AI safety evaluation platform

## Target Customers (via Lightwheel)

AgiBot, BYD, Figure, Fourier, Galbot, Geely, ByteDance, Google DeepMind + defense primes

## Key Deliverables

1. **IsaacLabSimulator** - Implements `vla_eval/simulators/base.py:Simulator`
2. **IsaacLabTarget** - Implements `redteam/protocols.py:Target`
3. **Safety Probe Pipeline** - Alignment adapters + contrastive training
4. **50+ Isaac-specific behaviors** - Physical harm, safety violations, deception
5. **Compliance Reports** - DoD AI Ethics, ISO 26262

## Dependencies

- embodied-safety monorepo (`~/embodied-safety`)
- Isaac Sim + BenchHub access (available)
- GCP infrastructure (to be provisioned)

## Blockers / Risks

- None currently - planning complete, ready for implementation

## Next Steps

1. Begin Phase 0: Set up PostgreSQL schema
2. Create Terraform modules for GCP infrastructure
3. Initialize Temporal workflow definitions
4. Contact Lightwheel about partnership discussions

## Context for Agents

**Source Repo**: `~/embodied-safety`
**Plan Files**: `~/embodied-safety/docs/lightwheel-integration-plan*.md`
**Key Interfaces**: `IsaacLabSimulator`, `IsaacLabTarget`, `AlignmentAdapter`
