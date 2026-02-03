# Lightwheel/Isaac Lab Integration for Aspis AI Safety Platform

> **Date**: 2026-01-26
> **Status**: Planning complete, ready for implementation
> **Goal**: Full Product MVP - Production-ready safety evaluation module for Lightwheel customers
> **Access**: Isaac Sim + BenchHub both available
> **Scope**: Multi-modal comprehensive (all attack modalities)
> **Timeline**: 16 weeks (not rushed - build it right)

---

## Executive Summary

Build a production-grade safety evaluation platform by integrating:
- **Aspis AI** safety methodology (introspection, interpretability, adversarial defense)
- **Lightwheel BenchHub** simulation infrastructure (250+ tasks, Isaac Lab Arena)
- **embodied-safety** attack protocols (TAP, PAIR, GCG, patch attacks)

### Strategic Value
- Lightwheel has simulation infra but NO safety evaluation
- Aspis has safety methodology but NO simulation infra
- Combined = first comprehensive embodied AI safety evaluation platform
- Target customers: Figure, BYD, ByteDance, DeepMind + defense primes

---

## Context: Aspis AI Startup

### Mission
Full-stack safety platform for embodied AI: Defense UGVs, Autonomous Vehicles, Humanoid Robots

### Three Pillars:
1. **Introspection Engine** - Real-time monitoring of model internal states, attention patterns, latent representations
2. **Interpretability Module** - Human-understandable explanations for autonomous decisions (DoD compliance, ISO 26262)
3. **Adversarial Defense Layer** - Detection of manipulation, jailbreaks, hostile intent; real-time blocking

### Target Markets (Citi 2035 projections):
- Autonomous Vehicles: $700B+, 58M units
- Humanoids & Caring Robots: $390B, 31M units
- Delivery & Service Robots: $470B, 39M units
- Defense UGVs & Security: $750B, <5M units
- **Total: $2.3T+, 1.3B robots**

### Team:
- **Jonathan Steinberg** - CEO, former DDR&D PM, M.Sc. Embodied AI Safety
- **Assaf Lahiany** - CTO, 20 years Israeli defense, OB1 lead researcher
- **Prof. Oren Gal** - Chief Scientist, SAIL Lab head, former gov CTO

---

## Context: Lightwheel AI Platform

### Company Overview
Physical AI infrastructure company delivering data and platforms for embodied AI. Co-developed Isaac Lab Arena with NVIDIA.

### Products:
1. **BenchHub** - Orchestration framework on Isaac Lab Arena
   - 250+ tasks: LW-RoboCasa-Tasks (138 kitchen), LW-LIBERO-Tasks (130 tabletop)
   - GPU-accelerated parallel execution (thousands of environments)
   - USD/MJCF assets, Isaac Sim integration

2. **RoboFinals** - Industrial-grade evaluation platform
   - Deterministic evaluation, generalization tests
   - On-premise deployment option (critical for security evals)
   - VLA and World Model evaluation

3. **EgoSuite** - 300k+ hours egocentric human data
   - Multi-modal: RGB, depth, motion
   - Diverse environments: homes, factories, warehouses

4. **SimReady Library** - Physics-validated assets
   - LW-YCB Benchmark (125 assets)
   - USD/MJCF formats

### Supported Robots:
Unitree G1, Figure 01, Franka Panda, PandaOmron, DoublePanda, LeRobot SO100/101, ARX-X7s, Agilex-Piper

### Customers:
AgiBot, BYD, Figure, Fourier, Galbot, Geely, ByteDance, Google DeepMind

### NVIDIA Partnership:
- Co-developed Isaac Lab Arena
- Powers GR00T initiative
- Contributed task chaining and scene randomization

---

## Context: embodied-safety Monorepo

### Core Modules:
- **core/** - Shared VLM clients (OpenRouter, Gemini, Veo, local), GCS, config
- **redteam/** - Adversarial testing (TAP, PAIR, Crescendo, GCG attacks)
- **vla_eval/** - VLA adversarial evaluation (patch attacks, action scoring)
- **vlm_interp/** - Mechanistic interpretability (activation analysis, probes)
- **amber/** - Multi-modal behavior benchmark (text, video, image generation)
- **robospy/** - Privacy risk analysis for household robots
- **typographic/** - Visual text attack analysis

### Attack Vectors Supported:
TEXT, IMAGE, AUDIO, VIDEO, ACTION_SEQUENCE, SCENE_DESCRIPTION

### Attack Methods:
- TAP (Tree of Attacks with Pruning)
- PAIR (Paired attacker-judge-target)
- Crescendo (Progressive refinement)
- GCG (Gradient-based)
- Typographic (OCR-based injection)
- Patch attacks (visual adversarial)

### Behavior Sources:
- HarmBench (~400 text + ~110 multimodal)
- BadRobot (embodied/robotic scenarios)
- ASIMOV (constraint violations)
- Custom VLA behaviors (12+ scenarios)

### Lines of Code: ~68,261 Python across 8 modules

---

## Production Architecture (Council Consensus)

### Infrastructure Overview
```
┌─────────────────────────────────────────────────────────────────────┐
│                         CONTROL PLANE (GKE)                         │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐        │
│  │ API Server│  │ Temporal  │  │  vLLM     │  │ Dashboard │        │
│  │           │  │ Workflow  │  │  Router   │  │    UI     │        │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│                      DATA PLANE (Compute Engine MIGs)               │
│  ┌───────────────────────────┐  ┌───────────────────────────┐      │
│  │   Isaac Sim Workers       │  │    vLLM Model Servers     │      │
│  │   (GPU VMs, golden imgs)  │  │  (per model_version_id)   │      │
│  └───────────────────────────┘  └───────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐        │
│  │ Cloud SQL │  │    GCS    │  │ BigQuery  │  │   Redis   │        │
│  │ (Postgres)│  │(artifacts)│  │ (logs)    │  │  (queue)  │        │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘        │
└─────────────────────────────────────────────────────────────────────┘
```

### Council Consensus (Codex + Gemini + OpenCode)

**Database (PostgreSQL + JSONB):**
- Core entities (normalized): tenants, projects, models, model_versions (immutable), datasets, sim_environments (immutable), attacks, attack_runs, campaigns, campaign_runs, episodes, artifacts, audit_events
- JSONB for flexibility: task/attack-specific payloads
- New tables: determinism_profiles, feature_schema_versions, compliance_reports
- High-volume logs: Offload to BigQuery/ClickHouse, link via artifacts

**vLLM Serving:**
- Phase 1: Compute Engine MIGs (not GKE) - one server per model version
- Routing layer: Maps model_version_id → endpoint
- Determinism: Disable dynamic batching, fixed preprocessing, pinned seeds
- Phase 2: GKE only when multi-tenant GPU sharing needed

**Isaac Sim Infrastructure:**
- Compute Engine GPU VMs with MIGs (not K8s for simulation)
- Golden images: Packer-built, pinned to driver/CUDA/Isaac Lab/Lightwheel versions
- Dedicated MIG pools: "deterministic" vs "throughput" modes
- Single-GPU, no live rendering for determinism

**Agent Orchestration:**
- Temporal for campaign lifecycle + audit (NOT Celery)
- Redis task queue only for bursty parallel tasks
- Agent roles: CampaignPlanner, AttackExecutor, SimRunner, Scorer, ReportBuilder

**Safety Probe Training:**
- Standardized feature interface: feature_vector, schema_version, model_family, layer_id
- Alignment adapters: Linear adapters per model family to shared latent space
- Training: Contrastive + supervised losses on (safe, unsafe) activation pairs
- Calibration: Per-family calibration stored in DB

**Determinism Profile:**
- Version lock: Isaac Sim, Isaac Lab, Lightwheel, CUDA, driver, OS image
- Fixed seeds: campaign + episode level
- Resource isolation: Pinned GPU, CPU affinity, dedicated MIG pools
- Store provenance: sim_environment_id, container_digest, driver_version, gpu_model

---

## Phase 0: Foundation Infrastructure (Weeks 1-3)

### 0.1 Database Schema (PostgreSQL + JSONB)

```sql
-- Immutable artifacts
CREATE TABLE model_versions (
    id UUID PRIMARY KEY,
    model_id UUID REFERENCES models(id),
    version_tag VARCHAR(100),
    docker_image VARCHAR(500),
    weights_uri VARCHAR(500),
    config_hash VARCHAR(64),
    build_git_sha VARCHAR(40),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE sim_environments (
    id UUID PRIMARY KEY,
    isaac_version VARCHAR(50),
    isaac_lab_sha VARCHAR(40),
    lightwheel_sha VARCHAR(40),
    os_image VARCHAR(200),
    driver_version VARCHAR(50),
    cuda_version VARCHAR(20),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Evaluation entities
CREATE TABLE campaigns (
    id UUID PRIMARY KEY,
    project_id UUID REFERENCES projects(id),
    name VARCHAR(200),
    description TEXT,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE campaign_runs (
    id UUID PRIMARY KEY,
    campaign_id UUID REFERENCES campaigns(id),
    status VARCHAR(50),
    seed BIGINT,
    sim_environment_id UUID REFERENCES sim_environments(id),
    model_version_id UUID REFERENCES model_versions(id),
    determinism_profile_id UUID REFERENCES determinism_profiles(id),
    config JSONB,
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE episodes (
    id UUID PRIMARY KEY,
    campaign_run_id UUID REFERENCES campaign_runs(id),
    task_id UUID REFERENCES tasks(id),
    status VARCHAR(50),
    seed BIGINT,
    metrics JSONB,
    failure_reason TEXT,
    start_ts TIMESTAMPTZ,
    end_ts TIMESTAMPTZ
);

CREATE TABLE attack_runs (
    id UUID PRIMARY KEY,
    episode_id UUID REFERENCES episodes(id),
    attack_id UUID REFERENCES attacks(id),
    status VARCHAR(50),
    seed BIGINT,
    params JSONB,
    results JSONB,
    judge_score FLOAT,
    probe_scores JSONB,
    circuit_breaker_triggered BOOLEAN,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Audit trail (append-only)
CREATE TABLE audit_events (
    id UUID PRIMARY KEY,
    actor_id UUID,
    action VARCHAR(100),
    entity_type VARCHAR(50),
    entity_id UUID,
    diff JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Determinism configuration
CREATE TABLE determinism_profiles (
    id UUID PRIMARY KEY,
    name VARCHAR(100),
    kernel_flags JSONB,
    batching_policy VARCHAR(50),
    seed_strategy VARCHAR(50),
    gpu_affinity_enabled BOOLEAN,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Safety probes
CREATE TABLE feature_schema_versions (
    id UUID PRIMARY KEY,
    model_family VARCHAR(100),
    layer_ids INTEGER[],
    preprocessing_config JSONB,
    adapter_weights_uri VARCHAR(500),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 0.2 GCP Infrastructure Setup

**Terraform modules:**
| Module | Purpose |
|--------|---------|
| `infra/gke/` | GKE cluster for control plane |
| `infra/mig/isaac/` | Isaac Sim worker MIGs |
| `infra/mig/vllm/` | vLLM model server MIGs |
| `infra/cloudsql/` | PostgreSQL instance |
| `infra/gcs/` | Storage buckets (artifacts, logs) |

### 0.3 Agent Orchestration (Temporal)

**Agent roles:**
| Agent | Responsibility |
|-------|---------------|
| `CampaignPlanner` | Decomposes campaign into tasks |
| `AttackExecutor` | Runs TAP/PAIR/GCG attacks |
| `SimRunner` | Executes Isaac Lab episodes |
| `Scorer` | Aggregates safety metrics |
| `ReportBuilder` | Generates compliance reports |

---

## Phase 1: Isaac Lab Integration (Weeks 4-6)

### Module Structure

```
lightwheel/
├── __init__.py
├── config.py
├── db/
│   ├── models.py
│   ├── repositories.py
│   └── migrations/
├── arena/
│   ├── simulator.py            # IsaacLabSimulator(Simulator)
│   ├── scene_loader.py
│   └── asset_registry.py
├── benchhub/
│   ├── client.py
│   └── task_registry.py
├── targets/
│   ├── isaac_target.py         # IsaacLabTarget(Target)
│   └── vllm_target.py
├── scenarios/
│   ├── safety_behaviors.py
│   └── harm_classifier.py
├── introspection/
│   ├── engine.py
│   └── safety_monitor.py
├── probes/
│   ├── training.py
│   ├── adapters.py
│   └── contracts.py
├── workflows/
│   └── ...
├── demo/
│   ├── safety_demo.py
│   └── gradio_ui.py
└── audit/
    ├── logger.py
    └── reports.py
```

---

## Phase 2: Safety Probe Training Pipeline (Weeks 7-9)

### Feature Interface Contract

```python
@dataclass
class ProbeInput:
    feature_vector: torch.Tensor
    feature_schema_version: UUID
    model_family: str
    layer_id: int

@dataclass
class ProbeOutput:
    risk_score: float
    violation_type: str | None
    confidence: float
```

### Alignment Adapter Training

Per model family:
1. Extract activations from shared probe training set
2. Train linear adapter: ModelFamily → SharedLatentSpace
3. Store adapter weights in GCS, reference in feature_schema_versions

---

## Phase 3: Attack Integration (Weeks 10-12)

### Multi-Modal Attack Scenarios

| Modality | Attack Method | Isaac Implementation |
|----------|--------------|---------------------|
| TEXT | TAP, PAIR, Crescendo | Jailbreak instructions |
| IMAGE | Adversarial patches | USD texture injection |
| AUDIO | Voice injection | Audio-conditioned VLA |
| VIDEO | Temporal perturbation | Frame sequence attacks |
| SCENE | Object placement | USD scene modification |
| MULTI-TURN | Crescendo escalation | Conversation manager |

### Isaac-Specific Behaviors (50+)

Categories:
- Physical harm: Knife pickup, collision induction, human approach
- Safety violations: E-stop bypass, workspace limits, force limits
- Deception: False completion, hidden objectives
- Unauthorized access: Drawer opening, locked area entry
- Property damage: Object throwing, dropping fragile items

---

## Phase 4: vLLM Serving Layer (Weeks 13-14)

Per-model-version servers with routing layer.

---

## Phase 5: Demo & Enterprise Features (Weeks 15-16)

- Gradio Demo UI
- Compliance Reports (DoD AI Ethics, ISO 26262)

---

## Estimated Effort

| Phase | Duration | Focus |
|-------|----------|-------|
| 0: Foundation | 3 weeks | DB schema, GCP infra, Temporal |
| 1: Isaac Integration | 3 weeks | Simulator, targets, task mapping |
| 2: Probe Training | 3 weeks | Adapters, contrastive learning |
| 3: Attack Integration | 3 weeks | All modalities, behaviors |
| 4: vLLM Serving | 2 weeks | Model servers, router |
| 5: Demo & Enterprise | 2 weeks | UI, compliance reports |
| **Total** | **16 weeks** | Production MVP |

---

## Business Integration

**Lightwheel partnership options:**
1. Module Integration: Aspis safety module in RoboFinals
2. OEM Licensing: Safety layer sold to Lightwheel customers
3. Joint Venture: Combined offering for enterprise/defense
4. Acquisition: Aspis becomes Lightwheel safety division

**Customer pipeline via Lightwheel:**
AgiBot, BYD, Figure, Fourier, Galbot, Geely, ByteDance, Google DeepMind + defense primes

---

## Memory MCP References

All context saved to Memory MCP (IDs 4-9):
- ID 4: Lightwheel AI Platform Analysis
- ID 5: Aspis AI Executive Summary
- ID 6: embodied-safety Monorepo Architecture
- ID 7: Lightwheel + Aspis Integration Strategic Analysis
- ID 8: Lightwheel Integration Plan Session Summary
- ID 9: Council Architecture Consensus

---

## Related Files

- Plan file: `/Users/jonathan/.claude/plans/replicated-growing-kettle.md`
- Startup exec summary: `/Users/jonathan/Startup/executive_summary/executive_summary.tex`
- Obsidian vault: `/Users/jonathan/Documents/Obsidian Vault/Startup/`

---

## Next Steps When Returning

1. Review this document and the plan file
2. Check Memory MCP for additional context: `mcp__memory__memory_list(tags_any=["aspis-ai"])`
3. Begin Phase 0: Foundation Infrastructure
4. Contact Lightwheel about partnership discussions
