"""Bucketed CUDA-Graphs forward for small fired-detector graphs.

Fired-detector graphs are tiny and variable-shaped; kernel launch overhead
dominates the latency path.  This module captures the model forward pass as a
CUDA Graph per bucket and replays it, removing per-invocation launch overhead.

Layout
------
Real nodes are copied to ``[0, N_real)`` of a fixed-size buffer, keeping the
indices they already had, so ``edge_index`` needs no remapping — the copy is
four ``copy_`` calls and nothing else.  Node and edge capacity are drawn from
two *independent* ladders over the batch totals.  Coupling them is what made
the original design pathological: complete-graph edge count is quadratic in
the per-graph node rung, so padding every graph to the batch maximum inflated
edge work by a measured 1.9x-26.5x and turned this path into a regression
against eager everywhere except tiny batches.

Padding is exact, not approximate.  Unused node slots and the sink node all
carry batch index ``B`` — one graph beyond the real batch — and padded edges
are self-loops on the sink, so messages can only flow *into* padding, never
out of it.  Pooling runs at size ``B + 1`` and the sink row is discarded.  The
model is untouched and no mask threads through the head.

The runner is single-consumer and not thread-safe: each bucket owns
pre-allocated tensors that replay overwrites in place.  ``forward`` returns a
view into the bucket's static output buffer, valid until the next call.

Capture is lazy, and a first-time bucket captures inside the calling stream.
Because rungs are selected from *batch totals*, which vary shot to shot, a real
syndrome stream keeps reaching new buckets well past startup — 24 distinct rung
pairs over 20k shots at d=7/B=1.  Anything serving live traffic should call
:meth:`BucketedGraphRunner.prewarm` with a representative sample of totals
first, so those captures are paid at setup rather than as stalls on the latency
path.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Final

import torch

from model.decoder import QECDecoder


logger = logging.getLogger(__name__)


def _geometric_ladder(
    start: int, stop: int, ratio: float, align: int
) -> tuple[int, ...]:
    """Build an ascending ladder of capacities, each a multiple of *align*.

    A geometric ladder bounds padding waste at roughly ``ratio`` regardless of
    scale, while keeping the number of distinct CUDA-Graph captures small —
    each rung costs a capture and its own static buffers.

    Parameters
    ----------
    start, stop : int
        First and last capacity to cover (both inclusive of rounding).
    ratio : float
        Growth factor between rungs; the padding-waste bound.
    align : int
        Each rung is rounded up to a multiple of this.

    Returns
    -------
    tuple of int
    """
    rungs: list[int] = []
    value = float(start)
    while value <= stop:
        rung = math.ceil(value / align) * align
        if not rungs or rung > rungs[-1]:
            rungs.append(rung)
        value *= ratio
    return tuple(rungs)


# Ratio 1.25 bounds padding waste at ~25%, against the 1.9x-26.5x the
# node-coupled scheme produced.  Ranges cover a single d=3 shot (2 nodes,
# 2 edges) through a 512-shot d=7 batch (~23k nodes, ~509k edges).
DEFAULT_NODE_BUCKETS: Final[tuple[int, ...]] = _geometric_ladder(
    start=8, stop=1 << 16, ratio=1.25, align=8
)
DEFAULT_EDGE_BUCKETS: Final[tuple[int, ...]] = _geometric_ladder(
    start=32, stop=1 << 21, ratio=1.25, align=32
)

_WARMUP_ITERS: Final[int] = 3


@dataclass(slots=True)
class _BucketBuffers:
    """Pre-allocated static tensors for one (node rung, edge rung) pair.

    ``filled_nodes`` and ``filled_edges`` are the high-water marks of the last
    call.  Only the slack between the previous and current extents is reset,
    so the per-call clear costs O(delta) instead of O(capacity).
    """

    node_rung: int
    edge_rung: int
    sink_node: int

    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    batch: torch.Tensor
    output: torch.Tensor

    graph: torch.cuda.CUDAGraph | None = field(default=None, init=False)
    filled_nodes: int = field(default=0, init=False)
    filled_edges: int = field(default=0, init=False)


def _rung_for(needed: int, ladder: tuple[int, ...], axis: str) -> int:
    """Return the smallest ladder rung >= *needed*.

    Raises
    ------
    ValueError
        If *needed* exceeds the top rung.  Failing loudly is deliberate: an
        unplanned recapture inside a timed run would silently contaminate the
        latency it is meant to measure.
    """
    for rung in ladder:
        if needed <= rung:
            return rung
    raise ValueError(
        f"batch needs {needed} {axis} but the largest {axis} rung is "
        f"{ladder[-1]}; widen the ladder explicitly and pay the recapture "
        f"cost at setup"
    )


class BucketedGraphRunner:
    """CUDA-Graphs-captured forward pass, bucketed by batch node and edge totals.

    Parameters
    ----------
    model : QECDecoder
        Trained decoder in eval mode on a CUDA device.
    batch_size : int
        Fixed shot count per call.
    node_buckets, edge_buckets : tuple of int
        Ascending capacity ladders.  A batch exceeding either top rung raises
        rather than triggering a fresh capture.
    node_dim, edge_dim : int
        Feature dimensions (v2 defaults: 6, 5).
    """

    def __init__(
        self,
        model: QECDecoder,
        *,
        batch_size: int,
        node_buckets: tuple[int, ...] = DEFAULT_NODE_BUCKETS,
        edge_buckets: tuple[int, ...] = DEFAULT_EDGE_BUCKETS,
        node_dim: int = 6,
        edge_dim: int = 5,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if not next(model.parameters()).is_cuda:
            raise ValueError("model must be on a CUDA device")

        self.model = model
        self.batch_size = batch_size
        self.node_buckets = tuple(sorted(node_buckets))
        self.edge_buckets = tuple(sorted(edge_buckets))
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.device = next(model.parameters()).device

        # Logits per graph, read from the head rather than assumed, so the
        # static buffers track num_observables instead of a hardcoded 1.
        self.out_dim = model.head.mlp[-1].out_features

        self._buckets: dict[tuple[int, int], _BucketBuffers] = {}
        self._empty_output = torch.zeros(
            batch_size, self.out_dim, dtype=torch.float32, device=self.device
        )

    def _get_or_create(self, node_rung: int, edge_rung: int) -> _BucketBuffers:
        key = (node_rung, edge_rung)
        cached = self._buckets.get(key)
        if cached is not None:
            return cached

        buf = self._allocate(node_rung, edge_rung)
        self._capture(buf)
        self._buckets[key] = buf
        return buf

    def prewarm(self, totals: Iterable[tuple[int, int]]) -> int:
        """Capture every bucket the given ``(node total, edge total)`` pairs select.

        Capture is otherwise lazy, and lazy is the wrong default for a served
        decoder: rung selection depends on the *batch totals*, which vary shot
        to shot, so a real syndrome stream keeps hitting first-time buckets long
        after startup.  Measured on 20k Stim shots at p=0.01, the number of
        distinct rung pairs actually touched is:

            d=7  B=1   -> 24 pairs (top two cover only 40% of batches)
            d=7  B=128 ->  2 pairs
            d=5  B=1   -> 18 pairs
            d=3  B=1   ->  5 pairs

        Every one of those is a CUDA-Graph capture, and a capture that lands in
        the serving stream stalls it for far longer than the 0.817 ms p50 the
        path exists to deliver.  Passing a representative sample of totals here
        moves that cost to setup, which is where this component already puts
        the equivalent cost of widening a ladder.

        Idempotent: pairs mapping to an already-captured bucket are skipped.

        Parameters
        ----------
        totals : iterable of (int, int)
            Observed ``(total_nodes, total_edges)`` per batch — e.g. from
            :class:`~kernels.graph_build.DeviceGraphBatch` over a calibration
            sample.  Pairs with zero nodes are ignored: they short-circuit
            before any bucket is selected.

        Returns
        -------
        int
            Number of buckets newly captured.

        Raises
        ------
        ValueError
            If any pair exceeds the top rung of either ladder — the same
            fail-loud rule :meth:`forward` applies, surfaced at setup instead
            of mid-stream.
        """
        before = len(self._buckets)
        for n_nodes, n_edges in totals:
            if n_nodes == 0:
                continue
            self._get_or_create(
                _rung_for(n_nodes, self.node_buckets, "nodes"),
                _rung_for(n_edges, self.edge_buckets, "edges"),
            )

        captured = len(self._buckets) - before
        logger.info(
            "Prewarmed %d new bucket(s); %d captured in total",
            captured,
            len(self._buckets),
        )
        return captured

    def _allocate(self, node_rung: int, edge_rung: int) -> _BucketBuffers:
        # One extra row holds the sink node; every padded edge self-loops on
        # it.  The old design gave the sink a whole graph's worth of nodes and
        # edges, which at B=1 doubled the work for nothing.
        sink = node_rung
        total_nodes = node_rung + 1

        return _BucketBuffers(
            node_rung=node_rung,
            edge_rung=edge_rung,
            sink_node=sink,
            x=torch.zeros(total_nodes, self.node_dim, device=self.device),
            edge_index=torch.full(
                (2, edge_rung), sink, dtype=torch.int64, device=self.device
            ),
            edge_attr=torch.zeros(edge_rung, self.edge_dim, device=self.device),
            batch=torch.full(
                (total_nodes,), self.batch_size, dtype=torch.int64, device=self.device
            ),
            output=torch.zeros(self.batch_size + 1, self.out_dim, device=self.device),
        )

    def _capture(self, buf: _BucketBuffers) -> None:
        """Warm up on a side stream, then capture the forward pass."""

        @dataclass(slots=True)
        class _StaticBatch:
            x: torch.Tensor
            edge_index: torch.Tensor
            edge_attr: torch.Tensor
            batch: torch.Tensor
            num_graphs: int

        static = _StaticBatch(
            x=buf.x,
            edge_index=buf.edge_index,
            edge_attr=buf.edge_attr,
            batch=buf.batch,
            num_graphs=self.batch_size + 1,
        )

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(_WARMUP_ITERS):
                buf.output.copy_(self.model(static))
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            buf.output.copy_(self.model(static))
        buf.graph = graph

        logger.info(
            "Captured CUDA Graph: nodes<=%d edges<=%d B=%d",
            buf.node_rung,
            buf.edge_rung,
            self.batch_size,
        )

    def _fill(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch_vec: torch.Tensor,
        buf: _BucketBuffers,
    ) -> None:
        """Copy real data in and reset only the slack the last call dirtied.

        Real node indices survive unchanged because real nodes land at
        ``[0, N_real)``, so ``edge_index`` is copied verbatim.
        """
        n_nodes = x.shape[0]
        n_edges = edge_index.shape[1]

        # Reset only what the previous call wrote past the new extent; the
        # rest of the buffer is still in its allocated (padded) state.
        if buf.filled_nodes > n_nodes:
            buf.x[n_nodes : buf.filled_nodes].zero_()
            buf.batch[n_nodes : buf.filled_nodes].fill_(self.batch_size)
        if buf.filled_edges > n_edges:
            buf.edge_index[:, n_edges : buf.filled_edges].fill_(buf.sink_node)
            buf.edge_attr[n_edges : buf.filled_edges].zero_()

        buf.x[:n_nodes].copy_(x)
        buf.batch[:n_nodes].copy_(batch_vec)
        buf.edge_index[:, :n_edges].copy_(edge_index)
        buf.edge_attr[:n_edges].copy_(edge_attr)

        buf.filled_nodes = n_nodes
        buf.filled_edges = n_edges

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch_vec: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """Run the bucketed CUDA-Graphs forward pass.

        Both shape keys are tensor shapes, so bucket selection costs no device
        traffic.

        Parameters
        ----------
        x : Tensor, shape ``(N_total, node_dim)``
        edge_index : Tensor, shape ``(2, E_total)``
        edge_attr : Tensor, shape ``(E_total, edge_dim)``
        batch_vec : Tensor, shape ``(N_total,)``
        num_graphs : int
            Shot count; must equal the configured ``batch_size``.

        Returns
        -------
        Tensor, shape ``(num_graphs, 1)``
            A view into the bucket's static output buffer, valid until the
            next :meth:`forward` call on this runner.
        """
        if num_graphs != self.batch_size:
            raise ValueError(
                f"num_graphs {num_graphs} != configured batch_size {self.batch_size}"
            )

        n_nodes = x.shape[0]
        if n_nodes == 0:
            return self._empty_output

        buf = self._get_or_create(
            _rung_for(n_nodes, self.node_buckets, "nodes"),
            _rung_for(edge_index.shape[1], self.edge_buckets, "edges"),
        )
        self._fill(x, edge_index, edge_attr, batch_vec, buf)

        assert buf.graph is not None  # set by _capture before first use
        buf.graph.replay()

        return buf.output[:num_graphs]

    @torch.no_grad()
    def forward_from_batch(self, batch: object) -> torch.Tensor:
        """Forward from a :class:`kernels.graph_build.DeviceGraphBatch`.

        Parameters
        ----------
        batch
            Object exposing ``x``, ``edge_index``, ``edge_attr``, ``batch``
            and ``num_graphs``.

        Returns
        -------
        Tensor, shape ``(num_graphs, 1)``
        """
        return self.forward(
            x=batch.x,  # type: ignore[attr-defined]
            edge_index=batch.edge_index,  # type: ignore[attr-defined]
            edge_attr=batch.edge_attr,  # type: ignore[attr-defined]
            batch_vec=batch.batch,  # type: ignore[attr-defined]
            num_graphs=batch.num_graphs,  # type: ignore[attr-defined]
        )
