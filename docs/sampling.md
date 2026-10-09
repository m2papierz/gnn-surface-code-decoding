# sampling - Surface Code Sampling & Graph Construction

Stim circuit sampling and detector graph construction for GNN-based QEC decoders.

The module wraps [Stim](https://github.com/quantumlib/Stim) circuit
simulation and [PyMatching](https://pymatching.readthedocs.io/) decoding
into a reproducible sampling pipeline.

## Concepts

### Surface code memory experiment

`stim.Circuit.generated("surface_code:rotated_memory_x", ...)` creates a *rotated surface code memory-X experiment*: the logical qubit is initialised and measured in the X basis.  During the experiment, stabiliser measurements run for a configurable number of **rounds** (syndrome extraction cycles).

### Lattice-surgery operations

ZZ merge/split circuits are emitted from tqec `BlockGraph` definitions via `sampling.logical_ops`.  Each committed circuit carries a manifest declaring the operation, phase structure, seam, and observable layout.

Training and evaluation read that layout through `sampling.circuit_metadata`, which never imports tqec.  Each operation registers an extractor that takes a committed circuit's path and reads distance, rounds and layout from the manifest beside it.  `build_layout` validates the layout the same way at generation and at load: every detector lies in exactly one phase window, seam detectors lie in a merge phase, and patch ids cover exactly the declared blocks.  A manifest that fails any of these raises, naming the circuit's key.  Every detector declares exactly `(x, y, t)`; one that declares anything else raises, naming the detector and the circuit, at generation and at load alike.

### Detector error model (DEM)

A DEM captures which physical faults trigger which *detectors* (differences between consecutive stabiliser outcomes) and which flip *observables* (the logical measurement).  The module calls `circuit.detector_error_model(decompose_errors=True)` to obtain a graph-like DEM suitable for matching.

### Detector / decoding graph

PyMatching converts the DEM into a weighted graph where nodes are detectors and edges carry error probabilities and MWPM weights.  The module adds a single **virtual boundary node** (index = `num_detectors`) that collapses all boundary edges, making the graph structure stable across settings.

Per-edge **observable flip masks** are extracted from the DEM and stored alongside the graph.  These indicate which logical observables are flipped when a given edge's error occurs - required by belief-matching decoding.

### Labels

| Label | Source | Use case |
|---|---|---|
| `logical` | Stim simulation ground truth | Direct logical-error prediction |

### Rounds

`rounds` controls how many stabiliser measurement cycles run in each memory experiment. More rounds => longer temporal axis => more detectors and edges.

## Quick start

```bash
# Memory circuits for d∈{3,5,7}, p∈{0.003..0.01}
uv run scripts/generate_circuits_memory.py

# Lattice-surgery circuits (ZZ merge/split) via tqec
uv run scripts/generate_circuits_tqec.py

# Frozen evaluation sets from committed circuits
uv run scripts/generate_eval_sets.py --operation memory

# Small CI shard for test suite
uv run scripts/generate_ci_shard_memory.py
```

### Python API

```python
from sampling.profile import resolve_profile
from sampling.sampler import settings_from_circuit_dir

profile = resolve_profile("zz_merge_split")
points = settings_from_circuit_dir(profile.circuit_root, operation=profile.operation)
metadata = profile.metadata_extractor(
    points[0].circuit_path, num_observables=profile.data_contract.num_observables
)
```

## Output structure

All artifacts are partitioned by operation at the root level.

A circuit's identity `(operation, distance, rounds, error_prob)` is read from
the manifest beside it and nowhere else: the filename carries none, and a
circuit without a manifest is rejected by discovery. Discovery is handed the
operation it serves and rejects a root holding a circuit of another; an
extractor is handed the observable count of the contract in use and rejects a
circuit with another count.

### Circuits

```
data/circuits/<operation>/
└── d{distance}_r{rounds}_p{error_prob}.stim
└── d{distance}_r{rounds}_p{error_prob}.manifest.json
```

### Frozen evaluation sets

```
data/eval/<operation>/
└── d{distance}_p{error_prob}/
    ├── data.npz          # syndromes, observables, detector_coords
    └── manifest.json     # generator version, config hash, seed, shot count
```

### CI shards

```
data/ci_shard/<operation>/
├── syndromes.npy
├── observables.npy
├── detector_coords.npy
└── manifest.json
```

## Module layout

```
src/sampling/
├── __init__.py
├── circuit_metadata.py # Committed circuit => CircuitMetadata; phase windows, layout
├── dem.py             # Error mechanisms of a decomposed DEM, read once per circuit
├── experiment.py      # Experiment identity (operation, distance, rounds, p)
├── graph.py           # DEM => DetectorGraph conversion (incl. observable_flips)
├── logical_ops.py     # tqec BlockGraph => committed circuit + manifest
├── profile.py         # Operation profile registry (metric policy, representation)
├── representation.py  # Graph representation versioning and data contracts
├── sampler.py         # Stim circuit building and sampling
└── seeding.py         # Deterministic BLAKE2b seed derivation
```
