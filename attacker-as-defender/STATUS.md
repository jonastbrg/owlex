# Attacker-as-Defender

**Last Updated**: 2025-02-02
**Status**: Active

## Current Focus

Stage 1 dataset collection for adversarial competence training. Building the conceptual foundation through knowledge distillation from psychology, social engineering, and game theory sources.

## Core Thesis

> A model that learns **how** to attack will be better at **detecting** attacks than one trained only to refuse.

## 4-Stage Pipeline Overview

```
Stage 1: Conceptual Foundation (Knowledge Distillation) <- CURRENT
Stage 2: Pattern Recognition (Supervised Fine-Tuning)
Stage 3: Refinement (Game-Theoretic Self-Play)
Stage 4: Deployment as Safety Tool
```

## Recent Progress

- 2025-02-02: Documentation structure finalized
- Project thesis and methodology documented
- 4-stage pipeline design complete

## Key Decisions

- **Venue**: TMLR (Transactions on Machine Learning Research)
- **Models**: Qwen3-4B variants for training, distill to ~1B for deployment
- **Teachers**: Grok-4, Claude Opus 4.5, GPT-5.2 for distillation
- **Guards**: LlamaGuard-4, ShieldGemma, Qwen3Guard as baselines

## Blockers / Risks

- Stage 1 dataset collection requires careful curation

## Next Steps

1. Build Stage 1 curriculum datasets (Cialdini, social eng, game theory)
2. Define 5-level ladder training examples
3. Prepare SFT data format

## Context for Agents

**Repository**: `~/Desktop/Attacker-as-defender`

**Key files**:
- `docs/` - All documentation (see 00-index.md for map)
- `datasets/` - Dataset folders (to be populated)
- `docs/13-open-questions.md` - Research questions

**Workflow**:
- Research questions -> `docs/13-open-questions.md`
- New dataset source -> `docs/07-datasets.md`
- New task idea -> `docs/08-task-cards.md`
- Experiment design -> `docs/10-experiments.md`

**MANDATORY**: Use `/liza` for code, `/council` for design decisions, agents for all non-trivial work.
