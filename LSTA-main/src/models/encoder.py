import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional encoding along tour positions (Vaswani et al., 2017)."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: Tensor of shape (N,) with integer tour positions.
        Returns:
            Tensor of shape (N, d_model)
        """
        # Clamp positions to buffer size
        pos_clamped = torch.clamp(positions, 0, self.pe.size(0) - 1)
        return self.pe[pos_clamped]


class RouteMaskedTransformerLayer(nn.Module):
    """
    Transformer encoder layer with route-specific attention mask.
    Prevents computation/attention between nodes belonging to different routes.
    """
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(
        self,
        src: torch.Tensor,
        route_membership: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            src: (1, N, d_model)
            route_membership: (N,) tensor where 0 = depot, 1 = route 1, 2 = route 2.
        """
        # Build attention mask: (N, N)
        # Allowed if same route, or if one of the nodes is the depot (route 0)
        route_i = route_membership.unsqueeze(1)  # (N, 1)
        route_j = route_membership.unsqueeze(0)  # (1, N)

        # True where attention is DISALLOWED (attn_mask in PyTorch MultiheadAttention)
        is_same_route = (route_i == route_j)
        is_depot = (route_i == 0) | (route_j == 0)
        allowed = is_same_route | is_depot
        disallowed_mask = ~allowed  # BoolTensor (N, N)

        # MultiheadAttention
        src2, _ = self.self_attn(
            query=src,
            key=src,
            value=src,
            attn_mask=disallowed_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class L2SegEncoder(nn.Module):
    """
    Two-stage Hybrid Encoder of L2Seg (Section 4.1 & Appendix D.3):
    1. Input Feature Projection + Positional Encoding:
       h_init = Concat(MLP(x_i), PosEnc(pos_i)) in R^{2 * d_h} (d_h = 128)
    2. Route-level Transformer (L_TFM = 2 layers, 2 heads, FFN=512, Dropout=0.1, ReLU)
       with route-specific attention mask.
    3. Global Graph Attention Network (L_GNN = 2 layers of TransformerConv, 1 head, edge_dim=3)
       operating on the combined solution + K-NN graph.

    Output:
       H_GNN in R^{N x d_h}
    """
    def __init__(
        self,
        node_in_dim: int = 25,
        edge_in_dim: int = 3,
        hidden_dim: int = 128,
        num_tfm_layers: int = 2,
        num_gnn_layers: int = 2,
        tfm_heads: int = 2,
        gnn_heads: int = 1,
        ffn_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 1. Feature projections
        self.node_mlp = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pos_encoder = SinusoidalPositionalEmbedding(d_model=hidden_dim)

        # 2. Route-level Transformer (operates on 2 * hidden_dim = 256)
        tfm_dim = 2 * hidden_dim
        self.tfm_layers = nn.ModuleList([
            RouteMaskedTransformerLayer(
                d_model=tfm_dim,
                nhead=tfm_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
            )
            for _ in range(num_tfm_layers)
        ])
        # Project down from 2 * hidden_dim to hidden_dim
        self.tfm_proj = nn.Linear(tfm_dim, hidden_dim)

        # 3. Global Graph Transformer Convolution (TransformerConv in PyG)
        self.gnn_layers = nn.ModuleList([
            TransformerConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=gnn_heads,
                concat=False,
                edge_dim=edge_in_dim,
                dropout=dropout,
            )
            for _ in range(num_gnn_layers)
        ])
        self.gnn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_gnn_layers)
        ])
        self.gnn_activation = nn.ReLU()

    def forward(
        self,
        node_feats: torch.Tensor,
        tour_positions: torch.Tensor,
        route_membership: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of L2Seg Shared Encoder.

        Args:
            node_feats: (N, 25) float tensor
            tour_positions: (N,) int/long tensor of customer positions in tour
            route_membership: (N,) int/long tensor of route IDs (0=depot, 1=R1, 2=R2)
            edge_index: (2, E) long tensor
            edge_attr: (E, 3) float tensor

        Returns:
            H_GNN: (N, hidden_dim) node embeddings
        """
        # Step 1: Initial Embedding h_init = Concat(h_MLP, h_POS) in R^{2 * d_h}
        h_mlp = self.node_mlp(node_feats)                     # (N, d_h)
        h_pos = self.pos_encoder(tour_positions)               # (N, d_h)
        h_init = torch.cat([h_mlp, h_pos], dim=-1)             # (N, 2 * d_h)

        # Step 2: Route-level Transformer with Route Attention Masks
        # Shape for PyTorch MultiheadAttention is (batch_size=1, N, 2 * d_h)
        h_tfm = h_init.unsqueeze(0)
        for layer in self.tfm_layers:
            h_tfm = layer(h_tfm, route_membership=route_membership)
        h_tfm = h_tfm.squeeze(0)                               # (N, 2 * d_h)
        h_tfm = self.tfm_proj(h_tfm)                           # (N, d_h)

        # Step 3: Global Graph Encoding via TransformerConv
        h_gnn = h_tfm
        for conv, norm in zip(self.gnn_layers, self.gnn_norms):
            h_res = h_gnn
            h_out = conv(h_gnn, edge_index=edge_index, edge_attr=edge_attr)
            h_gnn = norm(self.gnn_activation(h_out) + h_res)

        return h_gnn
