# model - GNN-Based QEC Decoders

Graph neural network decoders for rotated surface code, operating on the detector graph produced by `sampling`.

## Architecture

```
                       ┌─────────────────────────────────────────────┐
                       │              QECDecoder                     │
                       │                                             │
Batch ──► encoder ──►(h, edge_h)                                     │
          (shared)     │                                             │
                       │──► LogicalHead ──► (B, num_obs)             │
                       │    attn_pool ‖ max_pool ‖ edge_pool => MLP  │
                       └─────────────────────────────────────────────┘
```

### Encoder (`encoder.py`)

`DetectorGraphEncoder` runs several rounds of GINEConv message passing on the detector graph with explicit edge co-evolution: each layer updates both node embeddings and edge embeddings.

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Convolution | GINEConv | Edge features (spatial deltas, distances, DEM weight) enter messages natively; 1-WL expressiveness |
| Edge update | `MLP(h_src + h_dst, \|h_src − h_dst\|, e)` | Symmetric, learns edge representations jointly with nodes |
| Normalisation | LayerNorm (nodes and edges) | Stable across variable graph sizes in a batch (d=3 and d=7 mixed) |
| Skip connections | Additive residual per layer (nodes and edges) | Enables deeper networks without degradation |
| Edge projection | Per-layer linear + accumulation | Each layer projects raw edge features and adds previous edge embeddings |

Input/output:

- **In**: `x (N, 6)` node features, `edge_attr (E, 6)` edge features
- **Out**: `h (N, hidden_dim)` node embeddings, `edge_h (E, hidden_dim)` edge embeddings

See [Feature engineering](#feature-engineering) below for the full feature specification.

### Swappable compute operations (`ops.py`)

The encoder and heads forward passes delegate compute-intensive patterns to `model.ops`, which dispatches to one of three backends:

| Backend | Description | Autograd | Use case |
|---------|-------------|----------|----------|
| `pytorch` | Pure PyTorch reference implementations (default) | ✓ | Training and inference |
| `compiled` | `torch.compile`-wrapped PyTorch - same numerics, compiler-driven kernel fusion | ✓ | Training and inference (recommended on GPU) |
| `cuda` | Hand-written CUDA kernels | ✗ | **Inference and benchmarking only** |

> [!WARNING]
> Do not use `backend: "cuda"` for training. The custom CUDA kernels are forward-only - they do not implement autograd backward passes, so gradients will not propagate through the encoder.

Set via `QECDEC_BACKEND` env var or `set_backend()` at runtime.

### Head (`decoder.py`)

**LogicalHead** - graph-level observable prediction:
1. Attention-weighted sum pooling over nodes => `(B, H)`
2. Max pooling over nodes => `(B, H)`
3. Mean pooling over edge embeddings per graph => `(B, H)`
4. Concatenate all three => two-layer MLP => `(B, num_observables)` logits

### Factory

```python
from model import build_model

model = build_model(node_dim=6, edge_dim=6, hidden_dim=128, num_layers=6, num_observables=1)
```

## Decoders (`decoders.py`)

All decoders implement the `Decoder` protocol with `name` property and `decode_batch()`.

A trained checkpoint is restored for inference with `load_gnn_checkpoint`, under
the data contract of the operation being evaluated
(`resolve_profile(operation).data_contract`); a checkpoint trained under any
other contract is rejected at load, naming every axis that differs.  Every
evaluation mode decodes through `GNNDecoder`, which also rejects a model whose
input or output width is not its contract's.

| Decoder | Description |
|---------|-------------|
| `PyMatchingDecoder` | DEM-weighted MWPM via PyMatching |
| `CorrelatedMatchingDecoder` | Two-pass correlated matching via PyMatching |
| `GNNDecoder` | Trained GNN model with graph construction |
| `BeliefMatchingDecoder` | Belief-Matching: BP posteriors re-weight matching, via beliefmatching |
| `TesseractDecoder` | Near-MLE accuracy ceiling: beam search via tesseract-decoder |

A classical baseline takes only its circuit; its package, exact pin and
configuration are listed in `docs/eval_protocol.md` §6.

## Training

### Quick start

```bash
# Train from config
uv run scripts/train_gnn.py -c configs/train.yaml

# Per-distance configs with tuned budgets
uv run scripts/train_gnn.py -c configs/train_memory_d3.yaml

# Use torch.compile backend
uv run scripts/train_gnn.py -c configs/train.yaml --backend compiled

# Continue the killed run of the same config and overrides
uv run scripts/train_gnn.py -c configs/train_memory_d3.yaml --resume
```

One trainer, `Trainer`, serves every operation.  A config listing several `distances` trains one model across them, each shot drawn uniformly from the selected circuits.

### Configuration

Every run names its config with `-c`; CLI arguments override its values, so the YAML file acts as the base and CLI flags provide per-run tweaks.  A key the config does not read (a misspelling, a section of a removed feature) is refused with the file and the key named, as is a config path that does not exist.

```yaml
backend: "pytorch"
compile_mode: "default"
amp_dtype: "bfloat16"
operation: "memory"
output_dir: "./outputs/runs"
model:
  hidden_dim: 128
  num_layers: 6
  dropout: 0.1
optimisation:
  lr: 1.5e-4
  weight_decay: 1.0e-5
  batch_size: 128
sample_budget: 10_000_000
val_interval_samples: 100_000
val_size: 10_000
warmup_fraction: 0.05
patience: 10
seed: 42
```

Programmatic access: `TrainConfig.from_yaml("configs/train.yaml")`.

### Loss function

| Loss | Details |
|------|---------|
| `FocalBCEWithLogitsLoss` | Focal modulation of BCE; `focal_alpha` and `focal_gamma` control class-balance and hard-example emphasis |

### Training details

- **Sample-budget stopping**: training halts after `sample_budget` samples consumed (not epochs)
- **Optimiser**: AdamW (weight_decay configurable, default 1e-4)
- **Scheduler**: Linear warmup (`warmup_fraction` of budget, from 1% of peak lr) followed by cosine annealing to lr/50
- **Mixed precision**: AMP with configurable `amp_dtype` (`bfloat16` or `float16`)
- **Data loading**: persistent workers + prefetch for reduced overhead
- **Checkpointing**: every validation atomically replaces `last.pt` (training state, progress, and the state at the best validation LER so far); `best.pt` is written once, when training ends
- **Reproducibility**: `seed` is the master seed; the training stream and PyTorch's generator are seeded per segment from it, the frozen validation set from its own derived seed

### Outputs

```
outputs/runs/<operation>/d<distances>/<run_id>/
├── config.json    # the resolved spec: config, contract, distances, circuits with sha256
├── last.pt        # resume state, replaced at every validation
├── history.json   # segment starts and per-validation train/val metrics, at the end
└── best.pt        # the best validation state with its calibrated threshold, at the end
```

`<distances>` joins the distances of the circuits trained on (`d3`, `d3_5_7`); `<run_id>` is the first 12 hex digits of the sha256 of `config.json`'s spec.  Any change to a setting, a circuit or the representation names another directory; moving `output_dir` renames none.  A directory is created whole, holding `config.json` and `last.pt`, so it either does not exist or can be resumed.  Training a spec whose directory already holds a run is refused: a run is never overwritten, and a finished run (one with `best.pt`) is never resumed.

`--resume` reopens the run of the same config and overrides from its `last.pt`.  Stim's samplers cannot save their state, so each segment draws a fresh training stream seeded from the master seed and the samples consumed at its start: no shot is reused, and a resumed run is determined by where it was killed (bitwise on CPU; CUDA scatter kernels accumulate with atomics, so no GPU run is bitwise reproducible).  `history.json` lists every segment start.

### Hyperparameter defaults

These are the code defaults in `TrainConfig`.  The shipped `configs/train.yaml` overrides several of them.

| Parameter | Code default | Notes |
|-----------|-------------|-------|
| `hidden_dim` | 64 | Embedding dimensionality |
| `num_layers` | 4 | Message-passing depth |
| `dropout` | 0.1 | Applied in encoder and head |
| `lr` | 1e-3 | Peak learning rate (after warmup) |
| `weight_decay` | 1e-4 | AdamW L2 regularisation |
| `sample_budget` | 1,000,000 | Training halts after this many samples |
| `batch_size` | 64 | Graphs per batch |
| `val_interval_samples` | 50,000 | Validate every N training samples |
| `val_size` | 10,000 | Frozen validation set size |
| `patience` | 10 | Validation checks without improvement |
| `focal_alpha` | 0.75 | Focal loss class-balance weight |
| `focal_gamma` | 1.0 | Focal loss hard-example exponent |
| `warmup_fraction` | 0.05 | Fraction of budget for LR warmup |
| `num_workers` | 4 | DataLoader parallelism |
| `backend` | `pytorch` | Compute backend |

## Evaluation

### Quick start

```bash
# Config-driven evaluation
uv run scripts/eval_gnn.py -c configs/eval_memory_d3.yaml -o outputs/results/eval_memory_d3.json

# Quick sanity check
uv run scripts/eval_gnn.py -c configs/eval_memory_d3.yaml --sanity -o outputs/results/sanity_memory_d3.json

# Dry run (validate config only)
uv run scripts/eval_gnn.py -c configs/eval_memory_d3.yaml --dry-run
```

Each committed eval config names the `best.pt` of its train config's run; a test recomputes that run's directory and fails when the two drift apart.  Every report is `{metadata, points}`, its metadata holding the mode, where the scored shots came from, the resolved config, the checkpoint by path, sha256 of the loaded bytes, samples consumed and threshold, and the device.  Reports are written only to `-o`, never into a run directory.

### Evaluation protocol

For each shot, run the model on (graph, syndrome), threshold the logit at 0 => predicted observable. Compare with ground truth. Report LER per setting with Wilson confidence intervals.

See [`docs/eval_protocol.md`](eval_protocol.md) for the full pre-registered protocol: adaptive stopping, McNemar test, and parity decisions.

### Scaling and baseline diagnostics

```bash
uv run scripts/diagnose_scaling.py
```

Measures every frozen evaluation point without a model: fired detectors and edges per graph over all shots, the flip rate of each observable and of any observable with its Wilson interval, the DEM's probability mass by mechanism class with the share `dem_weight` carries, and MWPM against correlated matching on every shot.  Across distances it applies the sub-threshold rule of `evaluation.distance_scaling` per operation and decoder.  The run samples nothing, so two runs write the same bytes; a committed circuit without a frozen set, or a set whose shots do not hold its manifest's `positive_count`, fails it.

## Dataset interface

The training and evaluation code consumes data from `StreamingSurfaceCodeDataset` (defined in `dataset.py`), which streams samples via the `sampling` module.

Each `__getitem__` returns a PyG `Data` with:

| Field | Shape | Description |
|-------|-------|-------------|
| `x` | `(N, 6)` | Node features (see below) |
| `edge_index` | `(2, E)` | Directed COO (both directions stored) |
| `edge_attr` | `(E, 6)` | Edge features (see below) |
| `y` | `(num_obs,)` | Observable ground truth |
| `logical` | `(num_obs,)` | Always present for evaluation |

The dataset exposes `node_dim` and `edge_dim` attributes so the trainer can construct the model with matching input dimensions.  These dimensions are persisted in the checkpoint for the evaluator.

Important: do not move graph tensors to GPU in the dataset (breaks `num_workers > 0`).  The training loop calls `batch.to(device)`.

## Feature engineering

### Node features (`node_dim = 6`)

The graph is built over **fired detectors only** (complete graph, no boundary node).  Normalisation by `2d` (spatial) and `r` (temporal) makes feature semantics transfer across code distances.

| Column | Name | Formula | Range | Description |
|--------|------|---------|-------|-------------|
| 0 | `x_norm` | `x / (2d)` | [0, 1] | Normalised x-coordinate |
| 1 | `y_norm` | `y / (2d)` | [0, 1] | Normalised y-coordinate |
| 2 | `t_norm` | `t / r` | [0, 1] | Normalised temporal position |
| 3 | `d_x` | `(x - d) / d` | [-1, 1] | Signed boundary distance (x-axis) |
| 4 | `d_y` | `(y - d) / d` | [-1, 1] | Signed boundary distance (y-axis) |
| 5 | `basis` | `int((x+y)/2) % 2` | {0, 1} | Measurement basis (X-check vs Z-check) |

### Edge features (`edge_dim = 6`)

Edges connect every pair of fired detectors (complete graph, both directions).  The first three features are signed deltas in normalised coordinates; the next two are distance metrics; the last is the DEM pairwise error probability.

| Column | Name | Formula | Range | Description |
|--------|------|---------|-------|-------------|
| 0 | `dx` | `x_norm[dst] - x_norm[src]` | [-1, 1] | Signed normalised x-delta |
| 1 | `dy` | `y_norm[dst] - y_norm[src]` | [-1, 1] | Signed normalised y-delta |
| 2 | `dt` | `t_norm[dst] - t_norm[src]` | [-1, 1] | Signed normalised temporal delta |
| 3 | `euclidean` | `√(dx² + dy² + dt²)` | [0, √3] | Euclidean distance in normalised space |
| 4 | `chebyshev` | `max(\|dx\|, \|dy\|, \|dt\|)` | [0, 1] | Chebyshev distance in normalised space |
| 5 | `dem_weight` | from DEM | [0, 1] | Pairwise detector error probability from the detector error model |

The `dem_weight` feature encodes the combined probability that detectors `src` and `dst` are both flipped by the same error mechanism, extracted from `circuit.detector_error_model(decompose_errors=True)`.  Multiple mechanisms affecting the same pair are combined via `p = 1 - ∏(1 - pₖ)`.

A mechanism contributes when it flips exactly two detectors in total (`sampling.graph.dem_weight_mechanisms`): every undecomposed two-detector mechanism, and a decomposed one only when it is listed as two single-detector components.  Single-detector (boundary) mechanisms and every wider decomposition are not represented.  Every classical baseline decoder (MWPM, correlated matching, belief-matching, Tesseract) builds its decoding graph from the whole DEM, so this feature gives the GNN part of that information, not all of it; the share of the DEM's probability mass it carries is recorded per circuit by `scripts/diagnose_scaling.py`.
