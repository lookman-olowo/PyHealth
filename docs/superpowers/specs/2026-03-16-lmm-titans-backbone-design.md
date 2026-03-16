# Design: LMM (Titans) Backbone for PyHealth

**Author:** Colton Loew
**Date:** 2026-03-16
**Status:** Approved
**Paper:** Behrouz et al. "Titans: Learning to Memorize at Test Time." arXiv 2501.00663, 2025.

---

## Summary

Add the Neural Long-term Memory (LMM) module from Titans as a new
sequence backbone in PyHealth 2.0. LMM uses surprise-based gradient
updates with momentum to preferentially memorize unexpected inputs,
making it suited for EHR data where rare clinical events drive
predictions.

The contribution includes:
- `LMMLayer(nn.Module)` — reusable standalone backbone (like RNNLayer)
- `LMM(BaseModel)` — standalone model (like RNN)
- GRASP integration as `block="LMM"`

---

## Motivation

### Why LMM for EHR?

GRASP currently supports GRU, LSTM, and ConCare backbones. All three
treat every visit in a patient's sequence with the same gating
mechanism. In mortality prediction, the signal often comes from 1-2
critical visits (a sudden lab spike, a new heart failure diagnosis)
buried in a long stable history.

LMM's surprise-based memorization directly addresses this: the memory
network responds most strongly to inputs that produce large gradients
(high surprise), preferentially storing rare but clinically significant
events.

### Why LMM-Only (not MAC/MAG/MAL)?

Titans defines three variants that combine LMM with sliding-window
attention: MAC (Memory as Context), MAG (Memory as Gate), and MAL
(Memory as Layer). All three are designed for ultra-long context
(2M+ tokens). EHR sequences are 10-100 visits — short enough that
the attention components add parameters without meaningful benefit.

| Criterion              | MAC  | MAG  | MAL  | LMM-Only |
|------------------------|------|------|------|----------|
| Architectural fit      | Poor | Moderate | Moderate | Excellent |
| EHR relevance          | Low  | Low-moderate | Low | High |
| GRASP interaction      | Redundant | Gate-on-gate | Clean | Orthogonal |
| Implementation complexity | High | Moderate-high | Moderate | Low-moderate |
| Novelty for paper      | Moderate | Moderate | Low | High |

LMM-Only keeps the core innovation (surprise-based memorization) and
discards the long-context machinery that solves a problem EHR data
doesn't have. See `docs/plans/paper/titans/titans-variants-for-grasp.md`
for the full variant analysis.

### Orthogonality with GRASP

LMM and GRASP address different aspects of clinical prediction:
- **LMM**: "What should I remember from *this* patient's history?"
  (intra-patient temporal memorization)
- **GRASP**: "What can I learn from *similar* patients?"
  (inter-patient knowledge via clustering + GCN + gating)

These are complementary. There is zero architectural redundancy.

---

## Architecture

### LMMLayer Interface

```python
class LMMLayer(nn.Module):
    """Neural Long-term Memory layer for sequence encoding.

    Paper: Ali Behrouz et al. Titans: Learning to Memorize at
        Test Time. arXiv 2501.00663, 2025.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        memory_depth: int = 2,
        use_momentum: bool = True,
        use_weight_decay: bool = True,
        dropout: float = 0.5,
    ) -> None: ...

    def forward(
        self,
        x: torch.Tensor,                      # [B, seq_len, input_size]
        mask: Optional[torch.Tensor] = None,   # [B, seq_len]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (outputs, last_output).

        outputs:     [B, seq_len, hidden_size]
        last_output: [B, hidden_size]
        """
```

This matches `RNNLayer` exactly. GRASP calls
`_, hidden_t = self.backbone(input, mask)` and it works unchanged.

### Internal Components

```
LMMLayer
  proj_k:      Linear(input_size, hidden_size)   — key projection
  proj_v:      Linear(input_size, hidden_size)   — value projection
  proj_q:      Linear(input_size, hidden_size)   — query projection
  memory:      Sequential(                       — the memory MLP
                 Linear(hidden_size, 2 * hidden_size),
                 SiLU(),
                 Linear(2 * hidden_size, hidden_size),
               )
  gate_eta:    Linear(input_size, 1)             — momentum decay
  gate_alpha:  Linear(input_size, 1)             — forgetting rate
  gate_theta:  Linear(input_size, 1)             — learning rate
  dropout_layer: Dropout(dropout)
```

The `memory` module's `nn.Parameter` weights serve as the **initial
memory state** (trained by the outer optimizer). During forward(),
a working copy `mem_weights = {k: v.clone() for k, v in
memory.state_dict().items()}` is created and evolved per-timestep
via the surprise mechanism. See "Autograd Strategy" below.

With hidden_size=32, the memory MLP is 32->64->32 (~4K parameters).

### Forward Pass Logic (per timestep)

```
For each visit t in the sequence:
    1. Project: k_t = proj_k(x_t), v_t = proj_v(x_t), q_t = proj_q(x_t)

    2. Compute surprise: gradient of ||memory(k_t) - v_t||^2
       w.r.t. memory weights

    3. Compute data-dependent gates:
       eta_t   = sigmoid(gate_eta(x_t))     — momentum decay
       alpha_t = sigmoid(gate_alpha(x_t))   — forgetting
       theta_t = softplus(gate_theta(x_t))  — learning rate

    4. Update momentum:
       S_t = eta_t * S_{t-1} - theta_t * gradient

    5. Update memory weights:
       memory.weights = (1 - alpha_t) * memory.weights + S_t

    6. Retrieve:
       output_t = memory(q_t)   (forward pass, no weight update)

Last-output extraction (matches RNNLayer pattern):
    lengths = mask.sum(dim=-1).clamp(min=1)
    last_outputs = outputs[arange(B), lengths - 1, :]
```

### Autograd Strategy (CRITICAL)

The memory weight update involves computing gradients *inside* the
forward pass — the "surprise" signal. This creates two distinct
gradient computations:

1. **Inner loop (per-timestep):** Compute
   `grad(||memory(k_t) - v_t||^2, memory_weights)` to measure
   surprise and update memory weights. This is a functional gradient
   that should NOT be recorded in the outer autograd graph.
2. **Outer loop (training):** Standard `loss.backward()` to train
   all learnable parameters (projections, gates, memory init weights).

**Approach: `torch.func.functional_call` with detached weight tensors.**

Following the Titans paper's stop-gradient convention, memory weight
updates are non-differentiable w.r.t. the outer optimization:

```python
# Memory weights are maintained as plain tensors (not nn.Parameter)
# that evolve during the forward pass. The nn.Parameter versions
# serve only as the initial state.

# At the start of forward():
mem_weights = {k: v.clone() for k, v in self.memory_init.items()}

# Per-timestep surprise computation:
def surprise_loss_fn(weights, k_t, v_t):
    pred = torch.func.functional_call(self.memory, weights, (k_t,))
    return (pred - v_t).pow(2).sum()

grad_fn = torch.func.grad(surprise_loss_fn)
surprise_grad = grad_fn(mem_weights, k_t, v_t)

# Update memory weights (detached from outer graph):
for name in mem_weights:
    S[name] = eta_t * S[name] - theta_t * surprise_grad[name]
    mem_weights[name] = (
        (1 - alpha_t) * mem_weights[name] + S[name]
    ).detach()

# Retrieval IS differentiable — gradients flow back to proj_q
# through the retrieval computation, while mem_weights values
# are treated as fixed (detached) constants at each timestep:
output_t = torch.func.functional_call(
    self.memory, mem_weights, (q_t,)
)
```

**Key points:**
- `self.memory` is an `nn.Module` (Sequential MLP) used only as a
  template for `functional_call`. Its `nn.Parameter` weights serve
  as the initial memory state and are trained by the outer optimizer.
- `mem_weights` is a dict of plain tensors that evolve per-timestep
  via the surprise mechanism. These are `.detach()`ed after each
  update to prevent the outer backward from unrolling through all
  timesteps.
- The retrieval `functional_call` with `mem_weights` creates a fresh
  computation graph from the current memory state, allowing gradients
  to flow back to `proj_q` through the retrieval path.
- This matches the meta-learning "stateless module" pattern (MAML)
  and avoids mutating `nn.Parameter` objects.

**Requires:** PyTorch >= 2.0 for `torch.func.functional_call` and
`torch.func.grad`. PyHealth's `pyproject.toml` already requires
Python >= 3.12, and PyTorch 2.x is standard.

### Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Memory depth | 2 layers (default) | Paper's sweet spot for non-linear memory |
| Memory inner width | 2x hidden_size | Enough capacity without parameter bloat |
| Gates | Data-dependent (per-visit) | Routine visits forget more, surprising visits learn more |
| Activation | SiLU (swish) | Paper convention; smooth gradient helps surprise signal |
| Processing | Sequential (no chunking) | Simplest first; chunking can be added later. Per-step cost: 3 projections + 1 memory forward + 1 gradient + 3 gates + 1 momentum update + 1 weight update + 1 retrieval. Realistic slowdown vs GRU: 5-10x for seq_len=50+. Acceptable for EHR sequences (10-100 visits). |
| Last-output extraction | Index by mask lengths | Matches RNNLayer exactly (rnn.py line 123) |
| Autograd strategy | `torch.func.functional_call` with stop-gradient | Avoids mutating nn.Parameter; clean separation of inner/outer gradients |
| Ablation flags | `use_momentum=True, use_weight_decay=True` constructor args | Enables rows F and G in ablation table without separate model classes |

---

## LMM Standalone Model

Wraps `LMMLayer` in the standard PyHealth `BaseModel` pattern,
identical to how `RNN` wraps `RNNLayer`.

```python
class LMM(BaseModel):
    """Neural Long-term Memory model for clinical prediction.

    Paper: Ali Behrouz et al. Titans: Learning to Memorize at
        Test Time. arXiv 2501.00663, 2025.
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        **kwargs,
    ) -> None: ...

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]: ...
```

Constructor follows the exact PyHealth pattern:
1. `super().__init__(dataset=dataset)`
2. Assert single label key
3. `EmbeddingModel(dataset, embedding_dim)`
4. `nn.ModuleDict` of `LMMLayer` per feature key
5. `nn.Linear(len(feature_keys) * hidden_dim, output_size)`

Forward follows the GRASP pattern (simpler mask derivation):
1. `self.embedding_model(kwargs)` — returns dict of embedded tensors
2. Per-feature: derive mask via `(|x|.sum(dim=-1) != 0).int()`,
   then call `self.lmm[feature_key](x, mask)` — uses `last_output`
3. Concatenate all feature embeddings, FC head, loss, y_prob
4. Return dict with `loss`, `y_prob`, `y_true`, `logit`, optional
   `embed`

Note: The RNN model uses a more robust mask pattern that extracts
explicit masks from processor schemas. The simpler GRASP pattern is
used here for consistency with the primary GRASP integration. If the
LMM standalone model is used with processor types that provide
explicit masks (e.g., TupleTimeTextProcessor), this may need updating.

---

## GRASP Integration

### Changes to `grasp.py`

**Top-level import** (add alongside existing RNNLayer/ConCareLayer
imports at grasp.py lines 25-26, for consistency):

```python
from pyhealth.models.lmm import LMMLayer
```

**In `GRASPLayer.__init__`** (after the LSTM branch):

```python
elif self.block == "LMM":
    self.backbone = LMMLayer(input_dim, hidden_dim, dropout=0)
else:
    raise ValueError(
        f"Unknown block type '{self.block}'. "
        "Supported: 'ConCare', 'GRU', 'LSTM', 'LMM'."
    )
```

**Update `GRASPLayer` docstring** to include LMM:

```python
block: the backbone model used in the GRASP layer
    ('ConCare', 'GRU', 'LSTM', or 'LMM'), default 'ConCare'.
```

**In `grasp_encoder`:** no change needed. LMM returns
`(outputs, last_output)` matching the RNN convention, so it falls
into the existing `else` branch:

```python
else:
    # GRU, LSTM, and now LMM all use this branch
    _, hidden_t = self.backbone(input, mask)
```

**Note:** `memory_depth` is not exposed through GRASP's `**kwargs`
in this initial implementation — the default `memory_depth=2` is
used. Future enhancement: GRASP could accept a `backbone_kwargs`
dict for backbone-specific hyperparameters.

Usage:
```python
model = GRASP(dataset=dataset, block="LMM", ...)
```

---

## File Map

### Files to create

| File | Contents | ~Lines |
|---|---|---|
| `pyhealth/models/lmm.py` | LMMLayer + LMM + smoke test | 400-500 |
| `tests/core/test_lmm.py` | TestLMMLayer + TestLMM + TestGRASPWithLMM | 150 |
| `docs/api/models/pyhealth.models.LMM.rst` | API documentation | 20 |

### Files to modify

| File | Change |
|---|---|
| `pyhealth/models/__init__.py` | Add `from .lmm import LMM, LMMLayer` |
| `pyhealth/models/grasp.py` | Add `elif self.block == "LMM"` (3 lines) |
| `docs/api/models.rst` | Add `models/pyhealth.models.LMM` to toctree |

---

## File Header Convention

```python
# Author: Colton Loew
# Paper title: Titans: Learning to Memorize at Test Time
# Paper link: https://arxiv.org/abs/2501.00663
# Description: Neural Long-term Memory (LMM) module from Titans,
#     adapted as a sequence backbone for EHR clinical prediction.
#     Uses surprise-based memorization with momentum and adaptive
#     forgetting to preferentially encode rare clinical events.
```

---

## Test Plan

### TestLMMLayer
- `test_output_shapes` — verify [B, seq, hidden] and [B, hidden]
- `test_with_mask` — verify mask handling and last-output extraction
- `test_gradient_flows` — verify backward pass produces gradients
- `test_sequence_length_one` — single-visit patient (momentum has
  no history, S_0 = -theta_0 * gradient)
- `test_all_zeros_mask` — empty sequence, verify lengths clamped
  to 1 (matches RNNLayer rnn.py:116)
- `test_determinism` — same seed + input produces identical outputs
  (important because inner gradient could introduce non-determinism)

### TestLMM
- `test_model_initialization` — type, attributes, label_key
- `test_model_forward` — loss/y_prob/y_true/logit keys, shapes, no NaN
- `test_model_backward` — loss.backward() produces gradients
- `test_model_with_embedding` — embed flag returns patient embeddings
- `test_custom_hyperparameters` — non-default memory_depth works

### TestGRASPWithLMM
- `test_grasp_lmm_forward` — GRASP with block="LMM" runs forward
- `test_grasp_lmm_backward` — backward pass produces gradients

### Test conventions
- `torch.manual_seed(42)` for determinism
- Tiny configs: `embedding_dim=8, hidden_dim=4, batch_size=2`
- 2 synthetic patients, in-memory, no network access
- Target: < 1 second total execution

---

## Ablation Table Enabled

| Row | Backbone | Embedding | Tests |
|---|---|---|---|
| A | GRU | Random | Baseline |
| B | GRU | code_mapping | code_mapping value |
| C | GRU | KEEP | KEEP value |
| D | LMM | code_mapping | Titans backbone value |
| E | LMM | KEEP | Full system |
| F | LMM (no momentum) | KEEP | Momentum ablation |
| G | LMM (no weight decay) | KEEP | Forgetting ablation |

---

## Performance Expectations

| Metric | GRU (current) | LMM (expected) |
|---|---|---|
| AUROC | 0.6393 | 0.64-0.68 |
| F1 | 0.0606 | 0.08-0.15 |
| Training speed | ~2-3 min/epoch | ~10-20 min/epoch (5-10x slower, sequential per-step gradient) |
| Parameters | ~2K (GRU) | ~4K (memory MLP) |

Note: The 5-10x slowdown comes from per-timestep gradient computation
(no CuDNN fusion available). For 10-100 visit EHR sequences on an
H200, this translates to hours not days — acceptable for research.
Chunk-wise parallelization can be added later to reduce this to ~2-3x.

---

## Implementation Order

1. `LMMLayer` — the core module with surprise, momentum, decay
2. Smoke test with `torch.randn` — verify shapes
3. Plug into `GRASPLayer` as `block="LMM"`
4. Run one experiment with best config — does it train?
5. `LMM` standalone model (BaseModel wrapper)
6. Full test suite (`tests/core/test_lmm.py`)
7. Documentation files (RST, __init__.py, models.rst)
8. Sweep infrastructure (only after initial results)
