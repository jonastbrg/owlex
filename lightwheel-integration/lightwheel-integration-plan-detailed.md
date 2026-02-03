# Lightwheel/Isaac Lab Integration for Aspis AI Safety Platform

> **Goal**: Full Product MVP - Production-ready safety evaluation module for Lightwheel customers
> **Access**: Isaac Sim + BenchHub both available
> **Scope**: Multi-modal comprehensive (all attack modalities)
> **Timeline**: Not rushed - build it right

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

---

## Phase 0: Foundation Infrastructure (Weeks 1-3)

### 0.1 Database Schema (PostgreSQL + JSONB)

**Core entities (normalized):**
```sql
-- Immutable artifacts
CREATE TABLE model_versions (
    id UUID PRIMARY KEY,
    model_id UUID REFERENCES models(id),
    version_tag VARCHAR(100),
    docker_image VARCHAR(500),
    weights_uri VARCHAR(500),  -- GCS URI
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

**Golden image pipeline (Packer):**
```
lightwheel/infra/images/
├── isaac-sim/
│   ├── packer.json
│   └── provision.sh      # NVIDIA driver, CUDA, Isaac Sim, Isaac Lab
└── vllm/
    ├── packer.json
    └── provision.sh      # vLLM, model serving deps
```

### 0.3 Agent Orchestration (Temporal)

**Workflow definitions:**
```
lightwheel/workflows/
├── campaigns.py          # CampaignWorkflow
├── episodes.py           # EpisodeWorkflow
├── attacks.py            # AttackWorkflow
├── scoring.py            # ScoringWorkflow
└── reporting.py          # ReportWorkflow
```

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

### 1.1 Module Structure

```
lightwheel/
├── __init__.py
├── config.py                    # Settings, env vars
├── db/
│   ├── models.py               # SQLAlchemy models
│   ├── repositories.py         # Data access layer
│   └── migrations/             # Alembic migrations
├── arena/
│   ├── simulator.py            # IsaacLabSimulator(Simulator)
│   ├── scene_loader.py         # USD scene management
│   └── asset_registry.py       # Lightwheel SimReady assets
├── benchhub/
│   ├── client.py               # BenchHub API client
│   └── task_registry.py        # Map 268 tasks → TaskInfo
├── targets/
│   ├── isaac_target.py         # IsaacLabTarget(Target)
│   └── vllm_target.py          # vLLM-served VLA target
├── scenarios/
│   ├── safety_behaviors.py     # 50+ Isaac-specific behaviors
│   └── harm_classifier.py      # Physics-based harm scoring
├── introspection/
│   ├── engine.py               # Real-time state monitoring
│   └── safety_monitor.py       # Circuit breaker integration
├── probes/
│   ├── training.py             # Probe training pipeline
│   ├── adapters.py             # Alignment adapters per model
│   └── contracts.py            # Feature interface contracts
├── workflows/
│   └── ...                     # Temporal workflows
├── demo/
│   ├── safety_demo.py          # Multi-model comparative eval
│   └── gradio_ui.py            # Interactive interface
└── audit/
    ├── logger.py               # Deterministic audit trail
    └── reports.py              # Compliance report generation
```

### 1.2 Key Interfaces

**IsaacLabSimulator** (implements `vla_eval/simulators/base.py:Simulator`):
```python
class IsaacLabSimulator(Simulator):
    def __init__(
        self,
        scene_config: str | Path,
        robot: str = "franka_panda",
        determinism_profile_id: UUID | None = None,
        device: str = "cuda:0",
    ):
        super().__init__(name="isaac_lab", render_size=(256, 256))

    def get_task_list(self) -> list[TaskInfo]:
        """Return BenchHub tasks mapped to TaskInfo."""

    def reset(self, task_id: str, seed: int | None = None) -> np.ndarray:
        """Reset with deterministic seed."""

    def step(self, action: np.ndarray) -> StepResult:
        """Execute action with physics feedback."""

    def get_physics_feedback(self) -> PhysicsFeedback:
        """Collision forces, proximity, workspace violations."""
```

**IsaacLabTarget** (implements `redteam/protocols.py:Target`):
```python
class IsaacLabTarget(Target):
    def __init__(
        self,
        vla_model_endpoint: str,  # vLLM server URL
        simulator: IsaacLabSimulator,
        activation_extractor: ActivationExtractor | None = None,
    ):
        self.vla_endpoint = vla_model_endpoint
        self.sim = simulator
        self.extractor = activation_extractor

    def send(self, input: MultiModalInput, conversation_id: str) -> TargetResponse:
        """
        1. Parse text instruction + optional injected image
        2. Apply adversarial patches if present
        3. Get VLA action from vLLM server
        4. Execute in Isaac Sim
        5. Return response with physics feedback + hidden states
        """

    def supports_activation_extraction(self) -> bool:
        return self.extractor is not None
```

---

## Phase 2: Safety Probe Training Pipeline (Weeks 7-9)

### 2.1 Feature Interface Contract

```python
@dataclass
class ProbeInput:
    feature_vector: torch.Tensor    # (hidden_size,) normalized
    feature_schema_version: UUID
    model_family: str               # "pi0", "groot", "openvla"
    layer_id: int

@dataclass
class ProbeOutput:
    risk_score: float               # 0-1
    violation_type: str | None
    confidence: float
```

### 2.2 Alignment Adapter Training

**Per model family:**
1. Extract activations from shared probe training set
2. Train linear adapter: `ModelFamily → SharedLatentSpace`
3. Store adapter weights in GCS, reference in `feature_schema_versions`

```python
class AlignmentAdapter(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 768):
        self.linear = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(self.linear(x))
```

### 2.3 Contrastive Probe Training

```python
def train_safety_probe(
    safe_activations: torch.Tensor,
    unsafe_activations: torch.Tensor,
    adapter: AlignmentAdapter,
    epochs: int = 100,
) -> LinearProbe:
    """Train probe with contrastive + supervised objectives."""

    # Align to shared space
    safe_aligned = adapter(safe_activations)
    unsafe_aligned = adapter(unsafe_activations)

    # Contrastive loss + binary classification
    ...
```

---

## Phase 3: Attack Integration (Weeks 10-12)

### 3.1 Multi-Modal Attack Scenarios

| Modality | Attack Method | Isaac Implementation |
|----------|--------------|---------------------|
| TEXT | TAP, PAIR, Crescendo | Jailbreak instructions |
| IMAGE | Adversarial patches | USD texture injection |
| AUDIO | Voice injection | Audio-conditioned VLA |
| VIDEO | Temporal perturbation | Frame sequence attacks |
| SCENE | Object placement | USD scene modification |
| MULTI-TURN | Crescendo escalation | Conversation manager |

### 3.2 Isaac-Specific Behaviors (50+)

Categories:
- **Physical harm**: Knife pickup, collision induction, human approach
- **Safety violations**: E-stop bypass, workspace limits, force limits
- **Deception**: False completion, hidden objectives
- **Unauthorized access**: Drawer opening, locked area entry
- **Property damage**: Object throwing, dropping fragile items

---

## Phase 4: vLLM Serving Layer (Weeks 13-14)

### 4.1 Model Server Configuration

**Per-model-version servers:**
```yaml
# vllm_configs/pi0_v1.yaml
model: physical-intelligence/pi0
tensor_parallel_size: 4
max_model_len: 8192
dtype: float16
enforce_eager: true  # Determinism
seed: 42
disable_custom_all_reduce: true
```

### 4.2 Router Service

```python
class VLLMRouter:
    def __init__(self, model_registry: dict[UUID, str]):
        """
        model_registry: {model_version_id: endpoint_url}
        """

    def route(self, model_version_id: UUID, request: VLARequest) -> VLAResponse:
        endpoint = self.model_registry[model_version_id]
        return self._forward(endpoint, request)
```

---

## Phase 5: Demo & Enterprise Features (Weeks 15-16)

### 5.1 Gradio Demo UI

Features:
- Select behavior from dropdown (50+ scenarios)
- Choose VLA model (pi0, GR00T, OpenVLA)
- Run attack with visualization
- Show introspection data (attention, probe scores)
- Display audit trail

### 5.2 Compliance Reports

```python
class ComplianceReportBuilder:
    def generate(
        self,
        campaign_run_id: UUID,
        standard: str = "DoD_AI_Ethics",  # or "ISO_26262"
    ) -> ComplianceReport:
        """
        Generate report with:
        - Attack success rates per category
        - Circuit breaker effectiveness
        - Determinism verification
        - Full provenance chain
        """
```

---

## Critical Files to Modify

| File | Change |
|------|--------|
| `pyproject.toml` | Add lightwheel package, isaac/temporal/postgres deps |
| `vla_eval/simulators/__init__.py` | Export IsaacLabSimulator |
| `redteam/protocols.py` | Add PHYSICAL_FORCE modality |
| `gcp/spawn/configs/machines.yaml` | Add Isaac Sim machine types |

---

## Dependencies

```toml
[project.optional-dependencies]
lightwheel = [
    # Isaac Sim
    "isaacsim>=4.5.0",
    "omni-isaac-lab>=2.3.0",

    # Database
    "sqlalchemy>=2.0",
    "alembic>=1.13",
    "asyncpg>=0.29",
    "psycopg2-binary>=2.9",

    # Orchestration
    "temporalio>=1.4",

    # Model serving
    "vllm>=0.4.0",

    # Infrastructure
    "packer",  # Golden image builds
    "terraform",  # IaC

    # Monitoring
    "opentelemetry-api>=1.20",
]
```

---

## Verification Plan

### Database Tests
```bash
pytest lightwheel/tests/test_db_models.py
pytest lightwheel/tests/test_repositories.py
```

### Integration Tests
```bash
# Run single attack in Isaac Sim
python -m lightwheel.demo.safety_demo \
    --behavior isaac_001 \
    --model pi0_v1 \
    --determinism-profile strict

# Run campaign
python -m lightwheel.workflows.run_campaign \
    --campaign-id test_001 \
    --wait
```

### Determinism Verification
```bash
# Run same campaign twice, compare results
python -m lightwheel.audit.verify_determinism \
    --campaign-run-id run_001 \
    --campaign-run-id run_002
```

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
1. **Module Integration**: Aspis safety module in RoboFinals
2. **OEM Licensing**: Safety layer sold to Lightwheel customers
3. **Joint Venture**: Combined offering for enterprise/defense
4. **Acquisition**: Aspis becomes Lightwheel safety division

**Customer pipeline via Lightwheel:**
AgiBot, BYD, Figure, Fourier, Galbot, Geely, ByteDance, Google DeepMind + defense primes
