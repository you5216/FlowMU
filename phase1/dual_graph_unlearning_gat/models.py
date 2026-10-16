                      
                                                                    

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class QueryAwareGATEncoder(nn.Module):
                                                                            

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 128,
        node_type_count: int = 4,
        edge_type_count: int = 5,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads.")
        self.hidden_dim = int(hidden_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.node_type_emb = nn.Embedding(node_type_count, hidden_dim)
        self.edge_type_emb = nn.Embedding(edge_type_count, hidden_dim)
        self.edge_weight_proj = nn.Linear(1, hidden_dim)
        self.query_proj = nn.Linear(input_dim, hidden_dim)
        self.query_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.gat1 = GATConv(
            hidden_dim,
            hidden_dim // heads,
            heads=heads,
            concat=True,
            dropout=dropout,
            edge_dim=hidden_dim,
            add_self_loops=True,
        )
        self.gat2 = GATConv(
            hidden_dim,
            hidden_dim // heads,
            heads=heads,
            concat=True,
            dropout=dropout,
            edge_dim=hidden_dim,
            add_self_loops=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()

    def _edge_attr(self, edge_type: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        return self.edge_type_emb(edge_type) + self.edge_weight_proj(edge_weight.float())

    def _project_query_for_nodes(self, query: torch.Tensor, node_count: int, dtype: torch.dtype) -> torch.Tensor:
        if query.dim() == 1:
            query = query.unsqueeze(0)
        query = query.to(dtype)
        if query.size(0) == 1:
            return self.query_proj(query).expand(node_count, -1)
        if query.size(0) == node_count:
            return self.query_proj(query)
        raise ValueError(
            f"Query tensor must have shape [dim], [1, dim], or [num_nodes, dim]; got {tuple(query.shape)}."
        )

    def forward(
        self,
        x: torch.Tensor,
        node_type: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_weight: torch.Tensor,
        query: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h0 = self.input_proj(x) + self.node_type_emb(node_type)
        if query is not None:
            q = self._project_query_for_nodes(query, h0.size(0), x.dtype)
            gate = self.query_gate(torch.cat([h0, q, self.node_type_emb(node_type)], dim=-1))
            h0 = h0 * (1.0 + gate)

        edge_attr = self._edge_attr(edge_type, edge_weight)
        h = self.gat1(h0, edge_index, edge_attr=edge_attr)
        h = self.norm1(self.act(h) + h0)
        h1 = h
        h = self.gat2(h, edge_index, edge_attr=edge_attr)
        h = self.norm2(self.act(h) + h1)
        return h


class RAGGAT(nn.Module):
                                                                                     

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        community_count: int,
        aux_feature_dim: int = 0,
        aux_logit_weight: float = 1.0,
        gat_logit_weight: float = 1.0,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.aux_feature_dim = int(aux_feature_dim)
        self.aux_logit_weight = float(aux_logit_weight)
        self.gat_logit_weight = float(gat_logit_weight)
        self.encoder = QueryAwareGATEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            node_type_count=1,
            edge_type_count=1,
            heads=heads,
            dropout=dropout,
        )
        self.community_head = nn.Linear(hidden_dim, community_count)
        self.edge_weight_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.contrast_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.drop_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.aux_drop_head = (
            nn.Sequential(
                nn.Linear(self.aux_feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
            )
            if self.aux_feature_dim > 0
            else None
        )

    @staticmethod
    def pair_features(z: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        zi = z[left]
        zj = z[right]
        return torch.cat([zi, zj, torch.abs(zi - zj), zi * zj], dim=-1)

    def encode(self, batch, query: torch.Tensor | None = None) -> torch.Tensor:
        return self.encoder(batch.x, batch.node_type, batch.edge_index, batch.edge_type, batch.edge_weight, query)

    def community_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.community_head(z)

    def edge_weight_logits(self, z: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.edge_weight_head(self.pair_features(z, left, right)).squeeze(-1)

    def contrast_logits(self, z: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.contrast_head(self.pair_features(z, left, right)).squeeze(-1)

    def drop_logits(
        self,
        z: torch.Tensor,
        query: torch.Tensor,
        aux_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.encoder._project_query_for_nodes(query, z.size(0), z.dtype)
        logits = self.drop_head(torch.cat([z, q], dim=-1)) * self.gat_logit_weight
        if self.aux_drop_head is not None and aux_features is not None:
            logits = logits + self.aux_drop_head(aux_features.to(dtype=z.dtype)) * self.aux_logit_weight
        return logits


class MemoryGAT(nn.Module):
                                                                                     

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        community_count: int,
        node_type_count: int,
        edge_type_count: int,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = QueryAwareGATEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            node_type_count=node_type_count,
            edge_type_count=edge_type_count,
            heads=heads,
            dropout=dropout,
        )
        self.node_type_head = nn.Linear(hidden_dim, node_type_count)
        self.community_head = nn.Linear(hidden_dim, community_count)
        self.edge_class_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, edge_type_count),
        )
        self.edge_score_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.temporal_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.contrast_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.punish_head = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def pair_features(z: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        zi = z[left]
        zj = z[right]
        return torch.cat([zi, zj, torch.abs(zi - zj), zi * zj], dim=-1)

    def encode(self, batch, query: torch.Tensor | None = None) -> torch.Tensor:
        return self.encoder(batch.x, batch.node_type, batch.edge_index, batch.edge_type, batch.edge_weight, query)

    def node_type_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.node_type_head(z)

    def community_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.community_head(z)

    def edge_class_logits(self, z: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.edge_class_head(self.pair_features(z, left, right))

    def edge_score_logits(self, z: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.edge_score_head(self.pair_features(z, left, right)).squeeze(-1)

    def temporal_logits(self, z: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.temporal_head(self.pair_features(z, left, right)).squeeze(-1)

    def contrast_logits(self, z: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.contrast_head(self.pair_features(z, left, right)).squeeze(-1)

    def punish_logits(
        self,
        z: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        pair = self.pair_features(z, left, right)
        edge_emb = self.encoder.edge_type_emb(edge_type)
        return self.punish_head(torch.cat([pair, edge_emb], dim=-1)).squeeze(-1)
