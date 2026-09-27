import torch
import torch.nn as nn
from typing import Optional


class L2SegNARDecoder(nn.Module):
    r"""
    Non-Autoregressive (NAR) Decoder for L2Seg (Section 4.1 & Appendix D.3).
    
    Predicts the node instability probability p^{NAR} in one shot:
        p^{NAR} = Sigmoid(MLP(H^{GNN})) \in [0, 1]^N

    Architecture:
    - Input: Node embeddings H^{GNN} of shape (N, hidden_dim).
    - MLP: Linear(hidden_dim, hidden_dim) -> ReLU -> Dropout(0.1) -> Linear(hidden_dim, 1) -> Sigmoid.
    - Output: Probabilities of shape (N,) indicating whether each node is unstable.
    """
    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, h_gnn: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_gnn: (N, hidden_dim) or (B, N, hidden_dim) node embeddings from Encoder.
        Returns:
            p_nar: (N,) or (B, N) probabilities in range [0, 1].
        """
        out = self.mlp(h_gnn)  # (..., 1)
        return out.squeeze(-1)

    def compute_loss(
        self,
        p_nar: torch.Tensor,
        labels: torch.Tensor,
        w_pos: float = 9.0,
    ) -> torch.Tensor:
        r"""
        Weighted Binary Cross-Entropy Loss (Section 4.2):
            L_NAR = - \sum [ w_pos * y * log(p) + (1 - y) * log(1 - p) ]

        Args:
            p_nar: (N,) predicted probabilities.
            labels: (N,) binary ground truth (1 if node in V_unstable, else 0).
            w_pos: Weight for positive/unstable nodes (default 9.0 from Table 10).
        """
        eps = 1e-7
        p_clamped = torch.clamp(p_nar, eps, 1.0 - eps)
        pos_loss = w_pos * labels * torch.log(p_clamped)
        neg_loss = (1.0 - labels) * torch.log(1.0 - p_clamped)
        return -torch.mean(pos_loss + neg_loss)
