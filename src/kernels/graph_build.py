"""Batched fired-detector graph construction on the GPU.

Device-side counterpart of :func:`sampling.graph.build_fired_detector_graph`:
a batch of syndrome bit-vectors already resident in device memory becomes a
PyG-batched fired-detector complete graph without a round trip through numpy.

The numpy builder remains the normative definition of the representation -
this module reproduces it bitwise, it does not extend it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from kernels.extension import load
from sampling.graph import EDGE_DIM, NODE_DIM, CircuitMetadata


__all__ = [
    "DeviceGraphBatch",
    "DeviceMetadata",
    "build_fired_detector_graphs",
    "metadata_to_device",
]

_C = load()


@dataclass(frozen=True, slots=True)
class DeviceGraphBatch:
    """A batch of fired-detector graphs held entirely in device memory.

    Carries exactly the fields :meth:`model.decoder.QECDecoder.forward` reads,
    and satisfies the :class:`model.decoder.BatchedGraph` protocol.  Unlike
    ``torch_geometric.data.Batch``, ``num_graphs`` is explicit, so shots that
    fired no detectors keep their row in the output instead of being inferred
    away by ``batch.max() + 1``.

    Parameters
    ----------
    x : Tensor, shape ``(N_total, 6)``, float32
        Node features, shots concatenated in order.
    edge_index : Tensor, shape ``(2, E_total)``, int64
        Directed COO edges with node indices already offset per shot.
    edge_attr : Tensor, shape ``(E_total, 6)``, float32
        Edge features.
    batch : Tensor, shape ``(N_total,)``, int64
        Shot index of each node.
    num_fired : Tensor, shape ``(num_graphs,)``, int64
        Fired-detector count per shot; ``0`` marks a trivially decidable shot.
    num_graphs : int
        Number of shots, including those that fired no detectors.
    """

    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    batch: torch.Tensor
    num_fired: torch.Tensor
    num_graphs: int

    @property
    def total_nodes(self) -> int:
        """Node count across the whole batch.

        Free on the host - a tensor shape, not a device read.  This and
        :attr:`total_edges` are the only shape keys the bucketed runner needs,
        which is what lets it select a rung without any sync.
        """
        return self.x.shape[0]

    @property
    def total_edges(self) -> int:
        """Edge count across the whole batch (free - a tensor shape)."""
        return self.edge_index.shape[1]

    def __post_init__(self) -> None:
        n_total = self.x.shape[0]
        e_total = self.edge_index.shape[1]

        if self.x.shape != (n_total, NODE_DIM):
            raise ValueError(
                f"x shape {tuple(self.x.shape)} != expected ({n_total}, {NODE_DIM})"
            )
        if self.edge_index.shape != (2, e_total):
            raise ValueError(
                f"edge_index shape {tuple(self.edge_index.shape)} "
                f"!= expected (2, {e_total})"
            )
        if self.edge_attr.shape != (e_total, EDGE_DIM):
            raise ValueError(
                f"edge_attr shape {tuple(self.edge_attr.shape)} "
                f"!= expected ({e_total}, {EDGE_DIM})"
            )
        if self.batch.shape != (n_total,):
            raise ValueError(
                f"batch shape {tuple(self.batch.shape)} != expected ({n_total},)"
            )
        if self.num_graphs < 1:
            raise ValueError(f"num_graphs must be >= 1, got {self.num_graphs}")
        if self.num_fired.shape != (self.num_graphs,):
            raise ValueError(
                f"num_fired shape {tuple(self.num_fired.shape)} "
                f"!= expected ({self.num_graphs},)"
            )


@dataclass(frozen=True, slots=True)
class DeviceMetadata:
    """Everything the graph build reads besides the syndromes, on one CUDA device.

    Uploaded once by :func:`metadata_to_device` and reused across batches.
    Every input of the representation travels in this one record, so a build
    cannot run with any of them missing or taken from another circuit.

    Parameters
    ----------
    coords : Tensor, shape ``(D, 3)``, float64
        Detector coordinates, kept float64 so the kernel normalises in double
        precision and rounds once on store, as the numpy builder does.
    dem_weights : Tensor, shape ``(D, D)``, float32
        Pairwise DEM error probabilities, rounded once from float64 as the
        numpy builder rounds its edge features.
    distance : int
        Code distance; spatial normalisation is ``2 * distance``.
    rounds : int
        Syndrome measurement rounds; temporal normalisation.
    """

    coords: torch.Tensor
    dem_weights: torch.Tensor
    distance: int
    rounds: int

    @property
    def num_detectors(self) -> int:
        """Detector count ``D`` of the circuit."""
        return self.coords.shape[0]

    def __post_init__(self) -> None:
        if self.coords.dim() != 2 or self.coords.shape[1] != 3:
            raise ValueError(
                f"coords must have shape (D, 3), got {tuple(self.coords.shape)}"
            )
        if self.coords.dtype != torch.float64:
            raise ValueError(f"coords must be float64, got {self.coords.dtype}")
        d = self.num_detectors
        if self.dem_weights.shape != (d, d):
            raise ValueError(
                f"dem_weights shape {tuple(self.dem_weights.shape)} != "
                f"expected ({d}, {d})"
            )
        if self.dem_weights.dtype != torch.float32:
            raise ValueError(
                f"dem_weights must be float32, got {self.dem_weights.dtype}"
            )
        if not self.coords.is_cuda or self.dem_weights.device != self.coords.device:
            raise ValueError(
                "coords and dem_weights must share one CUDA device, got "
                f"{self.coords.device} and {self.dem_weights.device}"
            )
        if self.distance < 1 or self.rounds < 1:
            raise ValueError(
                f"distance and rounds must be >= 1, got {self.distance} and "
                f"{self.rounds}"
            )


def metadata_to_device(
    metadata: CircuitMetadata,
    device: torch.device | str,
) -> DeviceMetadata:
    """Upload what the graph build reads of *metadata* to a CUDA device.

    Parameters
    ----------
    metadata : CircuitMetadata
        Circuit metadata to upload.
    device : torch.device or str
        Target CUDA device.

    Returns
    -------
    DeviceMetadata
    """
    return DeviceMetadata(
        coords=torch.as_tensor(
            metadata.detector_coords, dtype=torch.float64, device=device
        ).contiguous(),
        dem_weights=torch.as_tensor(
            metadata.dem_edge_weights, dtype=torch.float32, device=device
        ).contiguous(),
        distance=metadata.distance,
        rounds=metadata.rounds,
    )


def build_fired_detector_graphs(
    syndromes: torch.Tensor,
    metadata: DeviceMetadata,
) -> DeviceGraphBatch:
    """Build fired-detector complete graphs for a batch of syndromes.

    Parameters
    ----------
    syndromes : Tensor, shape ``(B, D)``
        Binary syndrome bit-vectors on the metadata's device (``bool``,
        ``uint8`` or any integer dtype; non-zero means fired), ``B >= 1``.
    metadata : DeviceMetadata
        The circuit's metadata, from :func:`metadata_to_device`.

    Returns
    -------
    DeviceGraphBatch

    Raises
    ------
    ValueError
        If *syndromes* is not ``(B, D)`` with ``B >= 1`` and the metadata's
        ``D``, or is not on the metadata's device.
    """
    _validate_syndromes(syndromes, metadata)
    device = syndromes.device

    # Row-major nonzero yields fired detectors already grouped by shot and
    # ascending within a shot - the node ordering the numpy builder produces.
    fired = torch.nonzero(syndromes, as_tuple=False)
    shot = fired[:, 0].contiguous()
    detector = fired[:, 1].contiguous()
    num_nodes = shot.shape[0]

    node_count = syndromes.count_nonzero(dim=1).to(torch.int64)
    node_prefix = _exclusive_prefix(node_count)
    edge_prefix = _exclusive_prefix(node_count * (node_count - 1))

    # The one unavoidable device-to-host sync: the edge buffer cannot be
    # allocated without its size.  Everything downstream reads shapes off the
    # resulting tensors instead of the device, so this is the only one.
    num_edges = int(edge_prefix[-1])

    x = torch.empty((num_nodes, NODE_DIM), dtype=torch.float32, device=device)
    _C.fired_detector_node_features(
        coords=metadata.coords,
        detector=detector,
        slot=torch.arange(num_nodes, dtype=torch.int64, device=device),
        out=x,
        distance=metadata.distance,
        rounds=metadata.rounds,
    )

    edge_index = torch.empty((2, num_edges), dtype=torch.int64, device=device)
    edge_attr = torch.empty((num_edges, EDGE_DIM), dtype=torch.float32, device=device)
    _C.fired_detector_edges(
        node_features=x,
        edge_prefix=edge_prefix,
        node_base=node_prefix[:-1],
        node_count=node_count,
        edge_base=edge_prefix[:-1],
        dem_weights=metadata.dem_weights,
        detector_ids=detector,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_edges=num_edges,
    )

    return DeviceGraphBatch(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=shot,
        num_fired=node_count,
        num_graphs=syndromes.shape[0],
    )


def _validate_syndromes(syndromes: torch.Tensor, metadata: DeviceMetadata) -> None:
    """Raise unless *syndromes* is a non-empty ``(B, D)`` batch for *metadata*."""
    if syndromes.dim() != 2 or syndromes.shape[0] < 1:
        raise ValueError(
            "syndromes must have shape (B, D) with B >= 1, "
            f"got {tuple(syndromes.shape)}"
        )
    if syndromes.shape[1] != metadata.num_detectors:
        raise ValueError(
            f"syndrome detectors {syndromes.shape[1]} != metadata detectors "
            f"{metadata.num_detectors}"
        )
    if syndromes.device != metadata.coords.device:
        raise ValueError(
            f"syndromes on {syndromes.device} but the metadata on "
            f"{metadata.coords.device}; both must be on one CUDA device"
        )


def _exclusive_prefix(counts: torch.Tensor) -> torch.Tensor:
    """Return the ``(B + 1,)`` exclusive prefix sum of a ``(B,)`` count vector."""
    prefix = torch.zeros(counts.shape[0] + 1, dtype=torch.int64, device=counts.device)
    prefix[1:] = counts.cumsum(dim=0)
    return prefix
