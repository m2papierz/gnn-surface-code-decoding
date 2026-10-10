# gnn-surface-code-decoding

End-to-end pipeline for decoding surface codes with graph neural networks: Stim circuit sampling (memory experiments and lattice-surgery operations), detector graph construction, GNN training with on-the-fly data generation, multi-backend inference (PyTorch, compiled, custom CUDA), statistically rigorous evaluation against classical baselines (MWPM, correlated matching, belief-matching, Tesseract), and performance benchmarking.

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
uv run scripts/generate_eval_sets.py --operation memory

# Small CI shard for test suite
uv run scripts/generate_ci_shard_memory.py
```

See [`docs/sampling.md`](docs/sampling.md) for graph construction, circuit metadata, and the sampling API.

## GNN training

Train a GNN decoder using sample-budget training with on-the-fly Stim sampling:

```bash
uv run scripts/train_gnn.py -c configs/train.yaml

# Per-distance configs with tuned budgets
uv run scripts/train_gnn.py -c configs/train_memory_d3.yaml
uv run scripts/train_gnn.py -c configs/train_memory_d5.yaml
uv run scripts/train_gnn.py -c configs/train_memory_d7.yaml

# Continue a killed run of the same config
uv run scripts/train_gnn.py -c configs/train_memory_d3.yaml --resume
```

One trainer serves every operation; a config listing several `distances` trains one model across them.  A run lives in `outputs/runs/<operation>/d<distances>/<run_id>/`, where `run_id` is named by the resolved config and the circuits it trains on, so a run is never overwritten.

See [`docs/model.md`](docs/model.md) for architecture details (GINE encoder, LogicalHead, pooling), hyperparameters, and training configuration.

## Evaluation

Evaluate the GNN against classical baselines on frozen eval sets with adaptive stopping and paired statistical tests:

```bash
# Full eval harness: GNN vs baselines on frozen sets
uv run scripts/eval_gnn.py -c configs/eval_memory_d3.yaml -o outputs/results/eval_memory_d3.json

# Quick sanity check on fresh samples
uv run scripts/eval_gnn.py -c configs/eval_memory_d3.yaml --sanity -o outputs/results/sanity_memory_d3.json

# Dry run (validate config without decoding)
uv run scripts/eval_gnn.py -c configs/eval_memory_d3.yaml --dry-run
```

Each eval config scores the run of its train config.  Every report records the resolved config and the checkpoint it scored, by path and by the sha256 of the bytes that were loaded.

Record the scaling and baseline diagnostics of every frozen evaluation point - graph size, the flip rate of each logical observable, the detector error model's probability mass by mechanism class, MWPM against correlated matching, and which physical error rates are sub-threshold for each decoder:

```bash
uv run scripts/diagnose_scaling.py           # writes outputs/results/scaling_diagnostics.json
```

See [`docs/eval_protocol.md`](docs/eval_protocol.md) for the pre-registered stopping rule, McNemar test, Wilson confidence intervals, and the sub-threshold rule.

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
# LER vs p, and LER vs distance, from one harness report per distance
uv run scripts/plot_results.py --reports outputs/results/eval_memory_d{3,5,7}.json
uv run scripts/plot_calibration.py           # reliability diagrams and ECE
```

## Development

```bash
make setup               # uv sync --locked + pre-commit hook (runs fmt and lint)
make fmt                 # ruff format + import sorting
make lint                # ruff check
make check               # pyright over src/ and scripts/
make test                # pytest
```

Every tool runs at the version `uv.lock` pins.

## License

MIT License - see [LICENSE](LICENSE).
