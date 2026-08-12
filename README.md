# gnn-surface-code-decoding

End-to-end pipeline for decoding surface codes with graph neural networks: Stim circuit sampling (memory experiments and lattice-surgery operations), detector graph construction, GNN training with on-the-fly data generation (per-distance and curriculum), multi-backend inference (PyTorch, compiled, custom CUDA), statistically rigorous evaluation against classical baselines (MWPM, correlated matching, belief-matching, Tesseract), and performance benchmarking.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest            # verify installation
```

## Dataset generation

Generate Stim circuits and frozen evaluation sets for GNN training and evaluation:

Every artifact tree is partitioned by operation at its root -
`data/circuits/<operation>/`, `data/eval/<operation>/`,
`data/ci_shard/<operation>/` - and static memory is the operation named
`memory`. Each circuit carries a manifest beside it declaring the experiment
it belongs to, and everything sampled from it inherits that identity.

```bash
# Memory circuits for d∈{3,5,7}, p∈{0.003..0.01}
uv run scripts/generate_circuits_memory.py

# Lattice-surgery circuits (ZZ merge/split) via tqec
uv run scripts/generate_circuits_tqec.py

# Frozen evaluation sets
uv run scripts/generate_eval_sets.py --circuit-dir data/circuits/memory

# Small CI shard for test suite
uv run scripts/generate_ci_shard_memory.py
```

See [`docs/sampling.md`](docs/sampling.md) for graph construction, circuit metadata, and the sampling API.

## GNN training

Train a GNN decoder using sample-budget training with on-the-fly Stim sampling:

```bash
uv run scripts/train_gnn.py -c configs/train.yaml

# Per-distance configs with tuned budgets
uv run scripts/train_gnn.py -c configs/train_memory_d3_direct.yaml
uv run scripts/train_gnn.py -c configs/train_memory_d5_direct.yaml
uv run scripts/train_gnn.py -c configs/train_memory_d7_direct.yaml

# Mixed-distance curriculum training (3 => 3+5 => 3+5+7)
uv run scripts/train_gnn.py -c configs/train_memory_mixed_curriculum.yaml
```

See [`docs/model.md`](docs/model.md) for architecture details (GINE encoder, LogicalHead, pooling), hyperparameters, and training configuration.

## Evaluation

Evaluate the GNN against classical baselines on frozen eval sets with adaptive stopping and paired statistical tests:

```bash
# Full eval harness: GNN vs baselines on frozen sets
uv run scripts/eval_gnn.py -c configs/eval_memory_d3_direct.yaml

# Quick sanity check on fresh samples
uv run scripts/eval_gnn.py -c configs/eval_memory_d3_direct.yaml --sanity

# Dry run (validate config without decoding)
uv run scripts/eval_gnn.py -c configs/eval_memory_d3_direct.yaml --dry-run
```

See [`docs/eval_protocol.md`](docs/eval_protocol.md) for the pre-registered stopping rule, McNemar test, and Wilson confidence intervals.

## Deployment and benchmarking

Benchmark inference across backends:

```bash
uv run scripts/benchmark_all.py              # p50/p95/p99 latency, throughput, memory
```

See [`docs/kernels.md`](docs/kernels.md) for the custom CUDA kernels, the backend
selection rules, and the bucketed CUDA-Graphs fast path (inference only).

## Plots

Generate evaluation figures:

```bash
uv run scripts/plot_results.py -v            # LER vs p, and LER vs distance
uv run scripts/plot_calibration.py           # reliability diagrams and ECE
```

## Development

```bash
make fmt                 # ruff format + import sorting
make lint                # ruff check
make test                # pytest
```

## License

MIT License - see [LICENSE](LICENSE).
