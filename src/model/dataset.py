"""
PyG datasets for QEC decoding.

Provides a streaming PyG dataset for QEC decoding via on-the-fly Stim
sampling and fired-detector graph building.

All tensors are returned on CPU.  Move to device in the training loop
via ``batch = batch.to(device)`` - this keeps ``DataLoader(num_workers>0)``
safe (workers cannot share CUDA state).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import IterableDataset
from torch_geometric.data import Data

from sampling.representation import SPATIAL_MEMORY, DataContract, resolve_builder
from sampling.sampler import CircuitSetting, WorkerSampler
from sampling.seeding import stable_seed


if TYPE_CHECKING:
    import stim

    from sampling.graph import CircuitMetadata


logger = logging.getLogger(__name__)


class StreamingSurfaceCodeDataset(IterableDataset):
    """Streaming PyG dataset with on-the-fly Stim sampling.

    Each DataLoader worker owns its own set of Stim
    ``CompiledDetectorSampler`` instances, seeded deterministically from
    ``master_seed`` and ``worker_id`` via BLAKE2b (``stable_seed``).

    Per shot, a setting is sampled uniformly from the configured list,
    giving uniform per-shot ``p`` selection.  The iterator is infinite -
    the training loop controls how many samples to consume.

    Returned ``Data`` fields
    ------------------------
    x : FloatTensor, shape ``(N, 6)``
        Node features: ``[x_norm, y_norm, t_norm, d_x, d_y, basis]``.
    edge_index : LongTensor, shape ``(2, E)``
        Directed COO edges (complete graph on fired detectors).
    edge_attr : FloatTensor, shape ``(E, 6)``
        Edge features: ``[dx, dy, dt, euclidean, chebyshev, dem_weight]``.
    y : FloatTensor, shape ``(num_observables,)``
        Ground-truth observable flip (training target).
    logical : FloatTensor, shape ``(num_observables,)``
        Same as ``y`` (present for eval compatibility).
    num_fired : LongTensor, scalar
        Number of fired detectors (0 for trivially correct shots).
    p : FloatTensor, scalar *(only when* ``include_p_feature=True`` *)*
        Physical error probability for this shot.

    Parameters
    ----------
    settings : sequence of CircuitSetting
        Circuit settings to sample from.
    master_seed : int
        Master seed for reproducibility.
    include_p_feature : bool
        If ``True``, attach ``p`` as a graph-level feature.
    distance_weights : dict mapping int to float, optional
        Per-distance sampling weights.  Within each distance, settings
        are sampled uniformly; across distances, the probability is
        proportional to the weight.  If ``None``, all settings are
        sampled uniformly (default).
    metadata_extractor : callable, optional
        ``(stim.Circuit, distance, rounds) -> CircuitMetadata``.  When
        provided (typically from the operation profile), the workers use
        this instead of the default spatial extractor.
    """

    def __init__(
        self,
        *,
        settings: Sequence[CircuitSetting],
        master_seed: int,
        include_p_feature: bool = False,
        distance_weights: dict[int, float] | None = None,
        contract: DataContract = SPATIAL_MEMORY,
        metadata_extractor: Callable[[stim.Circuit, int, int], CircuitMetadata]
        | None = None,
    ) -> None:
        super().__init__()
        self.settings = list(settings)
        self.master_seed = master_seed
        self.include_p_feature = include_p_feature
        self._contract = contract
        self._builder = resolve_builder(contract.version)
        self._metadata_extractor = metadata_extractor or self._resolve_extractor()

        if not self.settings:
            raise ValueError("At least one CircuitSetting required")

        self._setting_weights = self._compute_setting_weights(distance_weights)

    def _resolve_extractor(
        self,
    ) -> Callable[["stim.Circuit", int, int], "CircuitMetadata"] | None:
        """Resolve metadata extractor from settings' operation profile.

        When the caller omits ``metadata_extractor``, look up the operation
        from the settings.  If all settings share one operation, return the
        profile's extractor; otherwise return ``None`` (the ``WorkerSampler``
        falls back to the spatial extractor).
        """
        from sampling.sampler import ExperimentPoint

        operations: set[str] = set()
        for s in self.settings:
            if isinstance(s, ExperimentPoint):
                operations.add(s.operation)
        if len(operations) != 1:
            return None
        from sampling.profile import resolve_profile

        try:
            profile = resolve_profile(operations.pop())
        except ValueError:
            return None
        return profile.metadata_extractor

    def _compute_setting_weights(
        self, distance_weights: dict[int, float] | None
    ) -> np.ndarray | None:
        """Convert per-distance weights to per-setting weights."""
        if distance_weights is None:
            return None

        dist_counts: dict[int, int] = {}
        for s in self.settings:
            dist_counts[s.distance] = dist_counts.get(s.distance, 0) + 1

        weights = np.array(
            [
                distance_weights.get(s.distance, 0.0) / dist_counts[s.distance]
                for s in self.settings
            ],
            dtype=np.float64,
        )

        total = weights.sum()
        if total <= 0:
            return None
        return weights / total

    @property
    def node_dim(self) -> int:
        """Node feature dimensionality from the data contract."""
        return self._contract.node_dim

    @property
    def edge_dim(self) -> int:
        """Edge feature dimensionality from the data contract."""
        return self._contract.edge_dim

    @property
    def contract(self) -> DataContract:
        """The data contract this dataset builds under."""
        return self._contract

    def __iter__(self) -> Iterator[Data]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id

        worker_seed = stable_seed("worker", f"id={worker_id}", base=self.master_seed)
        assert worker_seed is not None
        sampler = WorkerSampler(
            self.settings,
            worker_seed,
            weights=self._setting_weights,
            metadata_extractor=self._metadata_extractor,
        )

        builder = self._builder
        while True:
            syndrome, obs, meta, error_prob = sampler.sample()
            graph = builder(syndrome, meta)

            data = Data(
                x=torch.from_numpy(graph.node_features),
                edge_index=torch.from_numpy(graph.edge_index),
                edge_attr=torch.from_numpy(graph.edge_features),
                y=torch.from_numpy(obs.astype(np.float32)),
                logical=torch.from_numpy(obs.astype(np.float32)),
                num_fired=torch.tensor(graph.num_fired, dtype=torch.long),
            )

            if self.include_p_feature:
                data.p = torch.tensor(error_prob, dtype=torch.float32)

            yield data
