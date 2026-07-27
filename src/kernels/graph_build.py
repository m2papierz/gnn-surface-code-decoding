"""Batched fired-detector graph construction on the GPU.

Device-side counterpart of :func:`sampling.graph.build_fired_detector_graph`:
a batch of syndrome bit-vectors already resident in device memory becomes a
PyG-batched fired-detector complete graph without a round trip through numpy.

The numpy builder remains the normative definition of the representation —
this module reproduces it bitwise, it does not extend it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from kernels._C import fired_detector_edges as _cuda_edges
from kernels._C import fired_detector_node_features as _cuda_node_features
from sampling.graph import EDGE_DIM, NODE_DIM, CircuitMetadata


__all__ = [
    "DeviceGraphBatch",
    "build_fired_detector_graphs",
    "detector_coords_to_device",
]


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
    edge_attr : Tensor, shape ``(E_total, 5)``, float32
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

        Free on the host — a tensor shape, not a device read.  This and
        :attr:`total_edges` are the only shape keys the bucketed runner needs,
        which is what lets it select a rung without any sync.
        """
        return self.x.shape[0]

    @property
    def total_edges(self) -> int:
        """Edge count across the whole batch (free — a tensor shape)."""
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


def detector_coords_to_device(
    metadata: CircuitMetadata,
    device: torch.device | str,
) -> torch.Tensor:
    """Upload detector coordinates once for repeated graph builds.

    Kept float64 so the kernel can normalise in double precision and round
    once on store, exactly as the numpy builder does.

    Parameters
    ----------
    metadata : CircuitMetadata
        Circuit metadata whose ``detector_coords`` are uploaded.
    device : torch.device or str
        Target CUDA device.

    Returns
    -------
    Tensor, shape ``(D, 3)``, float64
    """
    coords = metadata.detector_coords[:, :3]
    return torch.as_tensor(coords, dtype=torch.float64, device=device).contiguous()


def build_fired_detector_graphs(
    syndromes: torch.Tensor,
    coords: torch.Tensor,
    *,
    distance: int,
    rounds: int,
) -> DeviceGraphBatch:
    """Build fired-detector complete graphs for a batch of syndromes.

    Parameters
    ----------
    syndromes : Tensor, shape ``(B, D)``
        Binary syndrome bit-vectors on a CUDA device (``bool``, ``uint8`` or
        any integer dtype; non-zero means fired).
    coords : Tensor, shape ``(D, 3)``, float64
        Detector coordinates on the same device, from
        :func:`detector_coords_to_device`.
    distance : int
        Code distance, used for spatial normalisation as ``2 * distance``.
    rounds : int
        Syndrome measurement rounds, used for temporal normalisation.

    Returns
    -------
    DeviceGraphBatch

    Raises
    ------
    ValueError
        If the inputs are not CUDA tensors, are the wrong rank, or disagree
        on the detector count.
    """
    if syndromes.dim() != 2:
        raise ValueError(
            f"syndromes must have shape (B, D), got {tuple(syndromes.shape)}"
        )
    if not syndromes.is_cuda or not coords.is_cuda:
        raise ValueError(
            "syndromes and coords must be CUDA tensors "
            f"(got {syndromes.device} and {coords.device})"
        )
    if coords.dim() != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must have shape (D, 3), got {tuple(coords.shape)}")
    if coords.shape[0] != syndromes.shape[1]:
        raise ValueError(
            f"coords rows {coords.shape[0]} != syndrome detectors {syndromes.shape[1]}"
        )

    num_shots = syndromes.shape[0]
    if num_shots < 1:
        raise ValueError("syndromes must contain at least one shot, got 0")
    device = syndromes.device

    # Row-major nonzero yields fired detectors already grouped by shot and
    # ascending within a shot — the node ordering the numpy builder produces.
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
    slot = torch.arange(num_nodes, dtype=torch.int64, device=device)
    _cuda_node_features(coords, detector, slot, x, distance, rounds)

    edge_index = torch.empty((2, num_edges), dtype=torch.int64, device=device)
    edge_attr = torch.empty((num_edges, EDGE_DIM), dtype=torch.float32, device=device)
    _cuda_edges(
        x,
        edge_prefix,
        node_prefix[:-1],
        node_count,
        edge_prefix[:-1],
        edge_index,
        edge_attr,
        num_edges,
    )

    return DeviceGraphBatch(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=shot,
        num_fired=node_count,
        num_graphs=num_shots,
    )


def _exclusive_prefix(counts: torch.Tensor) -> torch.Tensor:
    """Return the ``(B + 1,)`` exclusive prefix sum of a ``(B,)`` count vector."""
    prefix = torch.zeros(counts.shape[0] + 1, dtype=torch.int64, device=counts.device)
    prefix[1:] = counts.cumsum(dim=0)
    return prefix
