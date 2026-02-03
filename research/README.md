# Research Dashboard

Central status hub for research projects. Human-readable, git-versioned, agent-accessible.

## Active Projects

| Project | Status | Progress | Current Focus | Priority |
|---------|--------|----------|---------------|----------|
| [lightwheel-integration](./lightwheel-integration/STATUS.md) | Planning Complete | 0% | Ready for Phase 0 implementation | P0 |
| [redteam](./redteam/STATUS.md) | Active | 85% | TAP/PAIR/Crescendo orchestrators | P0 |
| [vla-eval](./vla-eval/STATUS.md) | Active | 70% | VLA attack pipeline | P0 |
| [vlm-interp](./vlm-interp/STATUS.md) | Active | 65% | Circuit breakers, safety probes | P1 |
| [attacker-as-defender](./attacker-as-defender/STATUS.md) | Active | - | Stage 1 dataset collection | P1 |
| [typographic](./typographic/STATUS.md) | Active | 30% | Visual text attacks | P2 |
| [amber](./amber/STATUS.md) | Planning | 20% | Benchmark generation | P2 |
| [framework](./framework/STATUS.md) | Maintenance | - | Linear integration stable | P3 |
| [robospy](./robospy/STATUS.md) | Dormant | 40% | Privacy analysis (on hold) | P3 |

## This Week's Priorities

1. **Lightwheel Integration**: Begin Phase 0 - DB schema, GCP infra, Temporal setup
2. **Redteam + VLA-Eval**: Continue attack orchestrator refinement
3. **Attacker-as-Defender**: Build Stage 1 curriculum datasets
4. **Framework**: Bug fixes only (no new features)

## Cross-Project Notes

- All projects use Linear for task tracking (team: JON)
- Use `/liza` for code changes, `/council` for design decisions
- Memora available for semantic search of past decisions

## Architecture

```
research-dashboard (this repo)     <- WHERE things stand
       |
       v
Linear (JON team)                  <- WHAT to do next
       |
       v
Local CLAUDE.md files              <- HOW to work locally
       |
       v
.owlex/liza-state.yaml             <- In-flight execution state
```

## Usage

### For Humans
- Edit STATUS.md files directly in VS Code or GitHub UI
- Push changes to trigger version history

### For Remote Agents
```python
# Clone and read
git clone https://github.com/jonastbrg/research-dashboard.git /tmp/dashboard
with open("/tmp/dashboard/PROJECT/STATUS.md") as f:
    context = f.read()

# Or via GitHub API (no clone needed)
import httpx
url = "https://raw.githubusercontent.com/jonastbrg/research-dashboard/main/PROJECT/STATUS.md"
resp = httpx.get(url, headers={"Authorization": f"token {GITHUB_TOKEN}"})
context = resp.text
```

## Update Cadence

- **Daily**: Update current focus after significant progress
- **Weekly**: Review all STATUS.md files, update priorities
- **Event-driven**: Update after major decisions or blockers

---

*Last portfolio review: 2026-02-03*
