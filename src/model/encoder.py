"""
Message-passing encoder for detector-graph decoding.

Provides the shared GNN backbone used by all three decoding heads.
Transforms raw node features and edge attributes into learned node
*and edge* embeddings via neighbourhood message passing with explicit
edge co-evolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, LayerNorm

from model.ops import (
    Backend,
    EdgeUpdateWeights,
    edge_update,
    fused_norm_residual_dropout,
    get_backend,
)
from sampling.representation import SPATIAL


class _GINEBlock(nn.Module):
    """
    Single message-passing layer: node update via GINEConv followed by
    explicit edge update, with residual connections, norms, and dropout.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of node and edge embeddings.
    edge_dim : int
        Dimensionality of *raw* edge features (before projection).
    dropout : float
        Dropout probability applied after norms.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        dropout: float = 0.1,
    ) -> None:
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

        self.edge_update = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_norm = LayerNorm(hidden_dim, mode="node")

        # Column blocks of edge_update[0], derived lazily and re-derived
        # whenever the parameter changes identity or is mutated in place.
        # The nn.Linear remains the source of truth, so checkpoints written
        # before the split existed still load unchanged.
        self._edge_weights: EdgeUpdateWeights | None = None
        self._edge_weights_key: tuple[int, int] | None = None

    def _fused_edge_weights(self) -> EdgeUpdateWeights | None:
        """Return the cached weight split, or None off the fast path.

        The blocks are detached, so building them during training would both
        strand the parameter's gradient and rebuild on every optimiser step
        (the version counter bumps each time).  Returning None keeps the
        ``pytorch`` and ``compiled`` backends on ``nn.Linear`` untouched.
        """
        if get_backend() is not Backend.CUDA:
            return None

        linear = self.edge_update[0]
        assert isinstance(linear, nn.Linear)  # constructed above
        key = (linear.weight.data_ptr(), linear.weight._version)
        if self._edge_weights_key != key or self._edge_weights is None:
            self._edge_weights = EdgeUpdateWeights.from_linear(linear)
            self._edge_weights_key = key
        return self._edge_weights

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_h: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        One round of node + edge message passing.

        Parameters
        ----------
        x : Tensor, shape (N, hidden_dim)
            Node embeddings.
        edge_index : Tensor, shape (2, E)
            Edge indices in COO format.
        edge_attr : Tensor, shape (E, edge_dim)
            Raw edge features (projected internally per layer).
        edge_h : Tensor or None, shape (E, hidden_dim)
            Learned edge embeddings from previous layer.

        Returns
        -------
        x : Tensor, shape (N, hidden_dim)
        edge_h : Tensor, shape (E, hidden_dim)
        """
        e = self.edge_proj(edge_attr)
        if edge_h is not None:
            e = e + edge_h

        # Node update: GINEConv + norm/residual
        residual = x
        x = self.conv(x, edge_index, e)
        x = fused_norm_residual_dropout(
            x,
            residual,
            self.norm,
            self.dropout,
            self.training,
        )

        # Edge update: symmetric features fused into their projection, then
        # the rest of the MLP, then norm/residual.
        e_new = edge_update(
            x, edge_index, e, self.edge_update[0], self._fused_edge_weights()
        )
        for layer in self.edge_update[1:]:
            e_new = layer(e_new)
        e_new = fused_norm_residual_dropout(
            e_new,
            e,
            self.edge_norm,
            self.dropout,
            self.training,
        )

        return x, e_new


class DetectorGraphEncoder(nn.Module):
    """
    GNN encoder producing node and edge embeddings from detector graphs.

    Parameters
    ----------
    node_dim : int
        Input node feature dimensionality (default: 6).
    edge_dim : int
        Input edge feature dimensionality (default: 6).
    hidden_dim : int
        Hidden embedding dimensionality (default: 128).
    num_layers : int
        Number of message-passing layers (default: 6).
    dropout : float
        Dropout probability (default: 0.1).
    """

    def __init__(
        self,
        node_dim: int = SPATIAL.node_dim,
        edge_dim: int = SPATIAL.edge_dim,
        hidden_dim: int = 128,
        num_layers: int = 6,
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
        self.layers = nn.ModuleList(
            [
                _GINEBlock(hidden_dim, edge_dim=edge_dim, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode detector graph into learned node and edge embeddings.

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
        h : Tensor, shape (N, hidden_dim)
            Node embeddings.
        edge_h : Tensor, shape (E, hidden_dim)
            Edge embeddings.
        """
        h = self.node_proj(x)
        edge_h = None

        for layer in self.layers:
            h, edge_h = layer(h, edge_index, edge_attr, edge_h)

        return h, edge_h
