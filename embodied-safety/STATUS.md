# Embodied Safety Monorepo

**Last Updated**: 2025-02-02
**Status**: Active

## Current Focus

VLA adversarial evaluation and attack development. Building robust evaluation frameworks for vision-language-action models.

## Sub-Projects

| Sub-Project | Status | Description |
|-------------|--------|-------------|
| [vlm_interp](./vlm_interp.md) | Active | VLM interpretability research |
| [vla_eval](./vla_eval.md) | Active | VLA evaluation framework |
| [redteam](./redteam.md) | Active | Adversarial attacks |
| [amber](./amber.md) | Planning | Benchmark generation |
| [typographic](./typographic.md) | Active | Typographic attacks |
| [robospy](./robospy.md) | Dormant | Privacy analysis |

## Recent Progress

- 2025-02-02: Continued VLA evaluation experiments
- 2025-01-31: GCP infrastructure setup for training
- 2025-01-26: Initial monorepo structure established

## Key Decisions

- Shared infrastructure in `core/` (modify with care)
- Use pytest for all testing
- GCP for compute-intensive experiments

## Blockers / Risks

- None currently

## Next Steps

1. Complete VLA evaluation baseline
2. Implement typographic attack variants
3. Document evaluation metrics

## Context for Agents

**Repository**: `~/embodied-safety`

**Key files**:
- `core/` - Shared infrastructure (be careful modifying)
- `pyproject.toml` - Project configuration
- `scripts/` - Utility scripts

**Tests**: `pytest [subproject]/tests/`

**GCP**: See `gcp/` for cloud infrastructure configs
