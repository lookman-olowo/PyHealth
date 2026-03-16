# Titans Architecture Variants: Analysis for GRASP EHR Integration

## Building Blocks

Titans has **two core components** that get combined in different ways:

1. **Neural Long-Term Memory (LMM)** — a small neural network (MLP) whose *weights themselves* are the memory. It learns by being "surprised" — when it sees something unexpected, the gradient is large, and it updates its weights more aggressively. Think of it as a brain that pays attention when something unusual happens.

2. **Sliding-Window Attention (Core)** — standard transformer attention, but only over a local window of recent tokens. This is the "short-term memory" — it sees recent context clearly but is blind to the distant past.

The three variants (MAC, MAG, MAL) are different recipes for combining these two ingredients. LMM-Only skips the attention entirely and uses just the memory.

---

## LMM — Neural Long-Term Memory (standalone)

### The Idea

Process a sequence one step at a time. At each step, the memory network tries to predict what it should output. If the prediction is wrong (high surprise), it updates its weights strongly. If the prediction is correct (low surprise), it barely changes.

### The Math, Intuitively

```
Patient visits:  V1 → V2 → V3 → V4 (sudden ICU admission) → V5 → V6

Memory reaction:  low   low   low   HIGH SURPRISE           low   low
                                     (big weight update)
```

The surprise signal has **momentum** — after V4 shocks the memory, the echo persists into V5 and V6. And it has **weight decay** (forgetting) — routine visits gradually fade so the memory doesn't overflow.

### Key Formulas

```
Memory Update:   M_t = (1 - alpha_t) * M_{t-1} + S_t
Surprise:        S_t = eta_t * S_{t-1} - theta_t * nabla_l(M_{t-1}; x_t)
Associative Loss: l(M; x_t) = ||M(k_t) - v_t||^2
Retrieval:       y_t = M*(q_t)   (forward pass without weight update)
```

Where:
- `alpha_t`: data-dependent forgetting rate (weight decay)
- `eta_t`: momentum decay (how past surprise carries forward)
- `theta_t`: learning rate for momentary surprise
- `M*`: forward pass through memory without updating weights

### Paper's Intended Use

The fundamental building block. On its own, it's a recurrent sequence encoder — like a GRU but with learned memorization instead of fixed gates.

### Healthcare Application

A patient has 50 clinic visits over 3 years. 47 are routine checkups. Visit 23 has an unusual creatinine spike. Visit 41 has a new heart failure diagnosis. A GRU treats all 50 visits with the same gating mechanism. LMM's surprise mechanism naturally differentiates:

| Visit | What happens | LMM response |
|---|---|---|
| 1-22 | Routine labs, stable BP | Small updates, gradually forgotten |
| **23** | **Creatinine 4.2 (normally 1.0)** | **Large gradient → strong memory write** |
| 24-40 | Back to normal | Momentum keeps V23's signal alive, slowly decaying |
| **41** | **New HF diagnosis code** | **Another large gradient → strong write** |
| 42-50 | Medication adjustments | Moderate updates |

When you query the final memory state for mortality prediction, V23 and V41 are disproportionately represented — exactly the events a clinician would flag.

### Why LMM Fits GRASP

GRASP already has its own mechanism for "what can I learn from similar patients?" (clustering + GCN + gating). What GRASP lacks is a backbone that distinguishes important visits from routine ones. LMM fills that exact gap. The two are **orthogonal**:

- **LMM**: "What should I remember from *this* patient's history?" (intra-patient)
- **GRASP**: "What can I learn from *similar* patients?" (inter-patient)

---

## MAC — Memory as Context

### The Idea

First, ask the long-term memory "what do you remember that's relevant?" Then combine that retrieved context with the current window of data. Run full attention over the combined set.

### The Flow

```
Step 1: Query memory → "I remember this patient had a kidney issue"
Step 2: Concatenate retrieval + current visits + persistent task knowledge
Step 3: Full attention over all of it → output
Step 4: Update memory with what attention found important
```

### Paper's Intended Use

Ultra-long documents (2M+ tokens) where you can't attend to everything at once. The memory acts as a compressed index of the distant past, and attention decides what's relevant from both the memory and the current chunk.

### Healthcare Application (Theoretical)

Imagine a patient with a **20-year longitudinal record** — thousands of notes, labs, imaging reports. You can't fit it all in one attention window. MAC would:

- Store a compressed representation of years 1-18 in the memory
- Attend over years 19-20 in the current window
- Retrieve relevant historical context (e.g., "patient had thyroid cancer in year 5") when current data triggers that query

### Why MAC Doesn't Fit GRASP

MIMIC-IV data has 10-100 visits per patient. That's short enough to fit entirely in one attention window. MAC's expensive memory-retrieval-then-attend pipeline is solving a problem you don't have. Plus, MAC has its own way of blending historical and current information — which duplicates what GRASP already does with clustering + gating.

### Where MAC Would Shine in Healthcare

If you were building a model to process **full unstructured clinical notes** across a patient's lifetime (hundreds of thousands of tokens), MAC would be the right choice. But that's a different project entirely.

---

## MAG — Memory as Gate

### The Idea

Run attention and memory **in parallel** on the same input. Then combine them multiplicatively — the memory gates the attention output.

### The Flow

```
Path A: Input → Sliding Window Attention → attention_output
Path B: Input → LMM (neural memory)      → memory_output

Final:  output = attention_output × memory_output   (element-wise)
```

The gating means: memory decides *how much* of each attention feature to let through. If memory says "this feature dimension is important based on long history," the gate opens. If not, it closes.

### Paper's Intended Use

Language modeling where you want both local coherence (attention handles nearby word relationships) and long-range memory (LMM handles facts from far back in the document). The gating learns which dimensions need which type of information. This was the **best-performing variant** on language benchmarks (18.61 perplexity vs 19.93 for MAC).

### Healthcare Application (Theoretical)

If you had a model processing both **recent visit sequences** and **long patient history**:

- Attention captures: "in the last 3 visits, blood pressure has been trending up"
- Memory captures: "this patient was diagnosed with diabetes 5 years ago"
- Gate decides: "for the blood pressure features, trust the recent attention; for the diagnosis features, trust the long-term memory"

### Why MAG Doesn't Fit GRASP

Two problems:

1. **Gate-on-gate.** MAG already gates two information streams together (attention x memory). Then GRASP applies a *second* gate: `final = w1 * cluster_info + w2 * backbone_output`. You end up with nested gating, which makes gradient flow harder and makes it impossible to disentangle which gate is contributing what in your ablation study.

2. **The sliding window degenerates.** With 10-100 visits, you'd set the window size to cover the full sequence, which makes sliding-window attention identical to full attention. At that point, the memory and attention see exactly the same input — the architectural motivation disappears. You're just running two parallel encoders and multiplying their outputs.

### Where MAG Would Shine in Healthcare

A system processing **real-time ICU monitoring data** with thousands of timesteps per hour, where you genuinely need both local pattern detection (attention on recent vitals) and long-range memory (what happened 12 hours ago). Continuous waveform data, not discrete visit codes.

---

## MAL — Memory as Layer

### The Idea

The simplest combination — memory processes the input first, then attention processes the output. A sequential pipeline.

### The Flow

```
Input → LMM (memory transforms the representation) → Attention (attends over transformed sequence) → Output
```

### Paper's Intended Use

The "standard" hybrid approach in the literature. The paper explicitly calls it **"the most common hybrid design but less effective than MAC/MAG."** It's included mostly for completeness and comparison.

### Healthcare Application (Theoretical)

The memory layer enriches each visit representation with historical context, then attention captures relationships between the enriched representations. For example:

- LMM transforms V3 from "routine checkup" to "routine checkup *but the memory knows this patient has a history of sudden cardiac events*"
- Attention then sees these enriched representations and can reason about temporal patterns

### Why MAL Doesn't Fit GRASP

1. **The paper says it's the worst variant.** If you're going to publish, using the variant the original authors say underperforms is a hard sell to reviewers.
2. **Attention only sees transformed data.** The attention layer never sees the raw visit embeddings — only the memory's interpretation of them. If the memory module distorts or compresses important local signals, attention can't recover them.
3. **Same overkill problem as MAC/MAG** — the attention component adds parameters for a capability (long-range pairwise attention) that short EHR sequences don't need.

---

## Summary: Why LMM-Only Is the Right Call for GRASP

| Question | MAC | MAG | MAL | **LMM-Only** |
|---|---|---|---|---|
| Does EHR data need the attention component? | No — sequences are 10-100 steps | No — window degenerates to full attention | No — same issue | **N/A — no attention** |
| Does it duplicate GRASP's mechanisms? | Yes — retrieval+blend | Yes — gate-on-gate | No | **No — perfectly orthogonal** |
| Is the core innovation (surprise) preserved? | Yes, but buried under complexity | Yes, but gated with attention | Yes, but filtered through attention | **Yes — front and center** |
| Can you cleanly ablate components? | Hard — too many moving parts | Hard — nested gates | Moderate | **Easy — momentum, decay, depth** |
| Research narrative | "We added a complex architecture" | "We added parallel paths + gating" | "We used the weakest variant" | **"Surprise-based memory captures rare clinical events"** |

### The Research Narrative

LMM-Only keeps the one idea from Titans that actually matters for EHR mortality prediction — **surprise-based memorization of rare but critical events** — and discards the long-context attention machinery that solves a problem EHR data doesn't have.

- **LMM handles temporal memorization** — what to remember from a patient's own history
- **GRASP handles cross-patient knowledge** — learning from similar patients via clustering + GCN + gating

These are orthogonal and complementary. It's the cleanest story, the easiest implementation, and the most honest contribution.

### Implementation Parameters for GRASP

- **Memory depth**: L_M = 2 (2-layer MLP, ~4K parameters with hidden_dim=32)
- **Interface**: Drop-in replacement for GRU — accepts `(input, mask)`, returns `(outputs, last_hidden)`
- **Chunk size**: 16-32 (full sequence fits in one chunk for most patients)
- **Keep momentum and weight decay**: Both are cheap and directly relevant
- **Skip persistent memory initially**: Minor benefit (+0.62 ppl), add later as ablation
- **Skip convolution initially**: Less relevant for aggregated visit embeddings than for language tokens

### Ablation Table This Enables

| Row | Backbone | Embedding | What It Tests |
|---|---|---|---|
| A | GRU | Random | Baseline |
| B | GRU | code_mapping | code_mapping value |
| C | GRU | KEEP | KEEP value |
| D | **LMM** | code_mapping | Titans backbone value |
| E | **LMM** | KEEP | Full system |
| F | **LMM** (no momentum) | KEEP | Momentum ablation |
| G | **LMM** (no weight decay) | KEEP | Forgetting ablation |
