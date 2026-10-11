"""
Message-passing encoder for detector-graph decoding.

The GNN backbone of :class:`model.decoder.QECDecoder`.  Transforms raw node
features and edge attributes into learned node embeddings via neighbourhood
message passing with explicit edge co-evolution: between consecutive node
updates, an edge update refines the edge embeddings the next node update reads.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, LayerNorm

from model.ops import (
    Backend,
    EdgeUpdateWeights,
    edge_update,
    fused_norm_residual_dropout,
    get_backend,
)


class _NodeUpdate(nn.Module):
    """
    Node update of one message-passing layer: GINEConv over the layer's edge
    embeddings, then norm, dropout and residual.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of node and edge embeddings.
    edge_dim : int
        Dimensionality of *raw* edge features (before projection).
    dropout : float
        Dropout probability applied after the norm.
    """

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float) -> None:
        super().__init__()

        mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv = GINEConv(nn=mlp, edge_dim=hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        # mode="node" normalises each row over its own hidden channels.  PyG's
        # default mode="graph", called without a batch vector, takes statistics
        # over the whole batched tensor, which makes one shot's logits depend
        # on the other shots sharing its batch.
        self.norm = LayerNorm(hidden_dim, mode="node")
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_h: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        One round of node message passing.

        Parameters
        ----------
        x : Tensor, shape (N, hidden_dim)
            Node embeddings.
        edge_index : Tensor, shape (2, E)
            Edge indices in COO format.
        edge_attr : Tensor, shape (E, edge_dim)
            Raw edge features, projected afresh by every layer.
        edge_h : Tensor or None, shape (E, hidden_dim)
            Edge embeddings from the preceding edge update; None in the first
            layer, which has none.

        Returns
        -------
        x : Tensor, shape (N, hidden_dim)
            Updated node embeddings.
        e : Tensor, shape (E, hidden_dim)
            The edge embeddings the messages were built from, which the
            following edge update refines.
        """
        e = self.edge_proj(edge_attr)
        if edge_h is not None:
            e = e + edge_h

        h = self.conv(x, edge_index, e)
        h = fused_norm_residual_dropout(h, x, self.norm, self.dropout, self.training)
        return h, e


class _EdgeUpdate(nn.Module):
    """
    Edge update between two node updates: ``MLP([h_src + h_dst | |h_src -
    h_dst| | e])``, then norm, dropout and residual on ``e``.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of node and edge embeddings.
    dropout : float
        Dropout probability applied after the norm.
    """

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()

        self.in_proj = nn.Linear(3 * hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = LayerNorm(hidden_dim, mode="node")
        self.dropout = nn.Dropout(dropout)

        # Column blocks of in_proj, derived lazily and re-derived whenever the
        # parameter changes identity or is mutated in place.  The nn.Linear
        # remains the source of truth, so the split never reaches a checkpoint.
        self._fused_weights: EdgeUpdateWeights | None = None
        self._fused_weights_key: tuple[int, int] | None = None

    def _cuda_weights(self) -> EdgeUpdateWeights | None:
        """Return the cached weight split, or None off the fast path.

        The blocks are detached, so building them during training would both
        strand the parameter's gradient and rebuild on every optimiser step
        (the version counter bumps each time).  Returning None keeps the
        ``pytorch`` and ``compiled`` backends on ``nn.Linear`` untouched.
        """
        if get_backend() is not Backend.CUDA:
            return None

        weight = self.in_proj.weight
        key = (weight.data_ptr(), weight._version)
        if self._fused_weights_key != key or self._fused_weights is None:
            self._fused_weights = EdgeUpdateWeights.from_linear(self.in_proj)
            self._fused_weights_key = key
        return self._fused_weights

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, e: torch.Tensor
    ) -> torch.Tensor:
        """
        Refine edge embeddings from the node embeddings at their endpoints.

        Parameters
        ----------
        x : Tensor, shape (N, hidden_dim)
            Node embeddings after the preceding node update.
        edge_index : Tensor, shape (2, E)
            Edge indices in COO format.
        e : Tensor, shape (E, hidden_dim)
            Edge embeddings the preceding node update read.

        Returns
        -------
        Tensor, shape (E, hidden_dim)
        """
        z = edge_update(x, edge_index, e, self.in_proj, self._cuda_weights())
        z = self.out_proj(F.relu(z, inplace=True))
        return fused_norm_residual_dropout(z, e, self.norm, self.dropout, self.training)


class DetectorGraphEncoder(nn.Module):
    """
    GNN encoder producing node embeddings from detector graphs.

    ``num_layers`` node updates alternate with ``num_layers - 1`` edge updates.
    The last node update is followed by none: the head pools node embeddings
    only, so an edge embedding computed after it would reach no output.

    Parameters
    ----------
    node_dim : int
        Input node feature dimensionality, the contract's.
    edge_dim : int
        Input edge feature dimensionality, the contract's.
    hidden_dim : int
        Hidden embedding dimensionality.
    num_layers : int
        Number of message-passing layers.
    dropout : float
        Dropout probability (default: 0.1).
    """

    def __init__(
        self,
        *,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.node_updates = nn.ModuleList(
            [_NodeUpdate(hidden_dim, edge_dim, dropout) for _ in range(num_layers)]
        )
        self.edge_updates = nn.ModuleList(
            [_EdgeUpdate(hidden_dim, dropout) for _ in range(num_layers - 1)]
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode a detector graph into learned node embeddings.

        Parameters
        ----------
        x : Tensor, shape (N, node_dim)
            Raw node features.
        edge_index : Tensor, shape (2, E)
            Directed edge indices in COO format (bidirectional).
        edge_attr : Tensor, shape (E, edge_dim)
            Raw edge attributes.

        Returns
        -------
        Tensor, shape (N, hidden_dim)
            Node embeddings.
        """
        h, e = self.node_updates[0](self.node_proj(x), edge_index, edge_attr, None)
        for edge_step, node_step in zip(
            self.edge_updates, self.node_updates[1:], strict=True
        ):
            h, e = node_step(h, edge_index, edge_attr, edge_step(h, edge_index, e))
        return h
