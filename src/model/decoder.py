"""
Decoding head and full decoder for QEC graph decoding.

:class:`LogicalHead` performs graph-level classification of observable flips
using attention-weighted, max, and sum node pooling.
"""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_max_pool
from torch_geometric.utils import softmax

from model.encoder import DetectorGraphEncoder
from sampling.representation import SPATIAL


class BatchedGraph(Protocol):
    """Structural interface of a batched graph consumed by :class:`QECDecoder`.

    Satisfied by ``torch_geometric.data.Batch`` and by
    ``kernels.graph_build.DeviceGraphBatch``, so the device-side fast path
    needs no adapter at the model boundary.

    Attributes
    ----------
    x : Tensor, shape ``(N_total, node_dim)``
    edge_index : Tensor, shape ``(2, E_total)``
    edge_attr : Tensor, shape ``(E_total, edge_dim)``
    batch : Tensor, shape ``(N_total,)``
        Graph index of each node.
    num_graphs : int
        Graph count including graphs with zero nodes.
    """

    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    batch: torch.Tensor
    num_graphs: int


class LogicalHead(nn.Module):
    """
    Graph-level head for predicting logical observable flips.

    Concatenates three node-level pooling channels - attention-weighted
    sum, max, and (unweighted) sum - feeding a 3H-dimensional vector
    into a two-layer MLP.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of encoder output embeddings.
    num_observables : int
        Number of logical observables to predict (default: 1).
    dropout : float
        Dropout probability in the MLP (default: 0.1).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_observables: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        # Input: attn_pool (H) | max_pool (H) | sum_pool (H)
        self.mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_observables),
        )

    def forward(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Predict logical observable logits.

        Parameters
        ----------
        h : Tensor, shape (N_total, hidden_dim)
            Node embeddings from the encoder (batched).
        batch : Tensor, shape (N_total,)
            PyG batch vector mapping each node to its graph index.
        num_graphs : int, optional
            Total number of graphs in the batch (including empty ones
            with zero nodes).  When provided, pooling output covers all
            graph indices; otherwise inferred from ``batch.max() + 1``,
            which silently drops empty graphs.

        Returns
        -------
        Tensor, shape (B, num_observables)
        """
        attn = softmax(self.gate(h), batch, num_nodes=num_graphs)
        g_attn = global_add_pool(h * attn, batch, size=num_graphs)
        g_max = global_max_pool(h, batch, size=num_graphs)
        g_max = torch.nan_to_num(g_max, neginf=0.0)
        g_sum = global_add_pool(h, batch, size=num_graphs)

        return self.mlp(torch.cat([g_attn, g_max, g_sum], dim=-1))


class QECDecoder(nn.Module):
    """
    Full decoder: encoder backbone + LogicalHead.

    Parameters
    ----------
    encoder : DetectorGraphEncoder
        Shared message-passing backbone.
    head : LogicalHead
        Graph-level prediction head.
    """

    def __init__(
        self,
        encoder: DetectorGraphEncoder,
        head: LogicalHead,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, batch: BatchedGraph) -> torch.Tensor:
        """Run full decode pipeline on a batched graph.

        Returns
        -------
        Tensor, shape ``(B, num_observables)``
        """
        h, _ = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        return self.head(
            h,
            batch=batch.batch,
            num_graphs=batch.num_graphs,
        )


def build_model(
    *,
    node_dim: int = SPATIAL.node_dim,
    edge_dim: int = SPATIAL.edge_dim,
    hidden_dim: int = 128,
    num_layers: int = 6,
    num_observables: int = 1,
    dropout: float = 0.1,
) -> QECDecoder:
    """
    Construct a full :class:`QECDecoder`.

    Parameters
    ----------
    node_dim : int
        Input node feature dimensionality (default: 6).
    edge_dim : int
        Input edge feature dimensionality (default: 6).
    hidden_dim, num_layers, dropout
        Encoder architecture parameters.
    num_observables : int
        Number of logical observables.

    Returns
    -------
    QECDecoder
    """
    encoder = DetectorGraphEncoder(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )

    head = LogicalHead(
        hidden_dim=hidden_dim,
        num_observables=num_observables,
        dropout=dropout,
    )

    return QECDecoder(encoder, head)
