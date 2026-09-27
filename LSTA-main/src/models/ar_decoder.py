import math
from typing import List, Tuple, Optional, Dict, Set
import torch
import torch.nn as nn
import torch.nn.functional as F


class ARTransformerBlock(nn.Module):
    """Transformer block with MultiheadAttention and FeedForward network for AR decoding."""
    def __init__(self, d_model: int = 128, nhead: int = 1, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (1, seq_len, d_model)
        x2, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout1(x2))
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout2(x2))
        return x


class L2SegARDecoder(nn.Module):
    """
    Autoregressive (AR) Decoder for L2Seg (Section 4.1 & Appendix D.3).
    
    Models unstable edge interdependencies by sequentially alternating between:
    - Deletion stage: identifies unstable edges (L_delete = 1 MHA layer, 1 head)
    - Insertion stage: introduces bridging pseudo-edges or end token (L_insert = 4 MHA layers, 1 head)
    """
    def __init__(
        self,
        hidden_dim: int = 128,
        num_delete_layers: int = 1,
        num_insert_layers: int = 4,
        nhead: int = 1,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 1. Sequence GRU (1 layer)
        self.gru = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)

        # 2. Learnable parameter alpha for end token h_end
        self.raw_alpha = nn.Parameter(torch.tensor(0.5))

        # 3. Shallow decoder for Deletion stage (L_delete = 1)
        self.delete_layers = nn.ModuleList([
            ARTransformerBlock(hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(num_delete_layers)
        ])

        # 4. Deep decoder for Insertion stage (L_insert = 4)
        self.insert_layers = nn.ModuleList([
            ARTransformerBlock(hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(num_insert_layers)
        ])

        # 5. Pointer scoring matrices: W_q maps h_c (6 * hidden_dim) to hidden_dim
        # h_c concatenates first 3 tokens of H^(0) and H^(L) along feature dimension -> 3 * 2 * hidden_dim = 6 * hidden_dim
        self.W_q = nn.Linear(6 * hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)

    @property
    def alpha(self) -> torch.Tensor:
        """Bounded alpha in [0, 1] for end token."""
        return torch.sigmoid(self.raw_alpha)

    def init_hidden(self, h_gnn: torch.Tensor) -> torch.Tensor:
        """Initial GRU hidden state is the graph average embedding h_0^{hidden} = mean(H^{GNN})."""
        return torch.mean(h_gnn, dim=0, keepdim=True)  # (1, hidden_dim)

    def compute_h_end(self, h_gnn: torch.Tensor, h_pi_0: torch.Tensor) -> torch.Tensor:
        r"""
        Termination token:
            h_end = \alpha * h_{\pi_0}^{GNN} + (1 - \alpha) * mean(H^{GNN})
        """
        mean_h = torch.mean(h_gnn, dim=0, keepdim=True)  # (1, hidden_dim)
        a = self.alpha
        return a * h_pi_0.unsqueeze(0) + (1.0 - a) * mean_h  # (1, hidden_dim)

    def decode_step(
        self,
        h_gnn: torch.Tensor,
        h_pi_0: torch.Tensor,
        h_pi_prev: torch.Tensor,
        h_hidden_prev: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        stage: str,  # "delete" or "insert"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one decoding step (Equation 2 in Section 4.1):
        
        Args:
            h_gnn: (N, hidden_dim) all node embeddings
            h_pi_0: (hidden_dim,) initial node embedding
            h_pi_prev: (hidden_dim,) previous node embedding
            h_hidden_prev: (1, hidden_dim) previous GRU hidden state
            candidate_embeddings: (M, hidden_dim) embeddings of available candidate nodes
            stage: "delete" or "insert"

        Returns:
            probs: (M,) softmax probability distribution over candidate nodes
            h_hidden_new: (1, hidden_dim) updated GRU hidden state
        """
        # Step 1: Update sequence GRU state
        h_hidden_new = self.gru(h_pi_prev.unsqueeze(0), h_hidden_prev)  # (1, hidden_dim)
        h_seq = h_hidden_new  # (1, hidden_dim)

        # Step 2: Form context embedding H_context = Concat(h_pi_0, h_pi_prev, h_seq)
        # Shape: (3, hidden_dim)
        h_context = torch.stack([h_pi_0, h_pi_prev, h_seq.squeeze(0)], dim=0)

        # Step 3: H^(0) = Concat(H_context, H_a) -> shape (3 + M, hidden_dim)
        H_0 = torch.cat([h_context, candidate_embeddings], dim=0)  # (3 + M, hidden_dim)

        # Step 4: Pass through MHA layers
        H_l = H_0.unsqueeze(0)  # (1, 3 + M, hidden_dim)
        layers = self.delete_layers if stage == "delete" else self.insert_layers
        for block in layers:
            H_l = block(H_l)
        H_LMHA = H_l.squeeze(0)  # (3 + M, hidden_dim)

        # Step 5: Construct context vector h_c by concatenating first 3 rows of H^(0) and H^(LMHA)
        # H_0[:3] is (3, d_h), H_LMHA[:3] is (3, d_h) -> flatten to (6 * d_h,)
        h_c = torch.cat([H_0[:3].flatten(), H_LMHA[:3].flatten()], dim=0)  # (6 * hidden_dim,)

        # Step 6: Attention scoring
        q = self.W_q(h_c)  # (hidden_dim,)
        # Candidates are from index 3 onwards
        cand_features = H_LMHA[3:]  # (M, hidden_dim)
        k = self.W_k(cand_features)  # (M, hidden_dim)

        scale = math.sqrt(self.hidden_dim)
        logits = torch.matmul(k, q) / scale  # (M,)
        probs = F.softmax(logits, dim=-1)

        return probs, h_hidden_new

    def compute_sequence_loss(
        self,
        h_gnn: torch.Tensor,
        node_sequence: List[int],
        local_routes: List[List[int]],
        w_insert: float = 0.8,
        w_delete: float = 0.2,
    ) -> torch.Tensor:
        r"""
        Compute weighted cross-entropy loss for an AR sequence (Section 4.2):
            L_AR = - \sum_{k} w_delete * log p_{delete} - \sum_{k} w_insert * log p_{insert}

        Args:
            h_gnn: (N, hidden_dim)
            node_sequence: [pi_0, pi_1, ..., pi_m, END_TOKEN_ID]
            local_routes: Local routes of the subproblem to determine valid solution edges
            w_insert: weight for insertion steps (default 0.8)
            w_delete: weight for deletion steps (default 0.2)
        """
        if len(node_sequence) < 2:
            return torch.tensor(0.0, device=h_gnn.device, requires_grad=True)

        END_TOKEN_ID = -1
        N = h_gnn.size(0)

        # Precompute solution adjacency for deletion candidates
        adj: Dict[int, Set[int]] = {i: set() for i in range(N)}
        for r in local_routes:
            for idx in range(len(r) - 1):
                u, v = r[idx], r[idx + 1]
                adj[u].add(v)
                adj[v].add(u)

        h_pi_0 = h_gnn[node_sequence[0]]
        h_hidden = self.init_hidden(h_gnn)
        loss = torch.tensor(0.0, device=h_gnn.device)

        deleted_edges: Set[Tuple[int, int]] = set()

        for step in range(len(node_sequence) - 1):
            curr_node = node_sequence[step]
            target_node = node_sequence[step + 1]
            is_delete = (step % 2 == 0)

            h_pi_prev = h_gnn[curr_node]

            if is_delete:
                # DELETION STAGE: candidates are solution neighbors of curr_node not yet deleted
                cand_ids = [
                    v for v in adj[curr_node]
                    if (min(curr_node, v), max(curr_node, v)) not in deleted_edges
                ]
                if not cand_ids or target_node not in cand_ids:
                    # In case of edge-case labels, fallback to all solution neighbors
                    cand_ids = list(adj[curr_node])
                if target_node not in cand_ids:
                    cand_ids.append(target_node)

                cand_embeddings = h_gnn[cand_ids]
                probs, h_hidden = self.decode_step(
                    h_gnn=h_gnn,
                    h_pi_0=h_pi_0,
                    h_pi_prev=h_pi_prev,
                    h_hidden_prev=h_hidden,
                    candidate_embeddings=cand_embeddings,
                    stage="delete",
                )
                target_idx = cand_ids.index(target_node)
                loss = loss - w_delete * torch.log(probs[target_idx] + 1e-8)
                deleted_edges.add((min(curr_node, target_node), max(curr_node, target_node)))

            else:
                # INSERTION STAGE: candidates are nodes not connected to curr_node, PLUS h_end
                h_end = self.compute_h_end(h_gnn, h_pi_0)
                # Candidates: all nodes + end token
                all_customer_nodes = [i for i in range(N) if i != curr_node]
                cand_ids = all_customer_nodes + [END_TOKEN_ID]

                cust_embeddings = h_gnn[all_customer_nodes]
                cand_embeddings = torch.cat([cust_embeddings, h_end], dim=0)

                probs, h_hidden = self.decode_step(
                    h_gnn=h_gnn,
                    h_pi_0=h_pi_0,
                    h_pi_prev=h_pi_prev,
                    h_hidden_prev=h_hidden,
                    candidate_embeddings=cand_embeddings,
                    stage="insert",
                )
                target_idx = cand_ids.index(target_node)
                loss = loss - w_insert * torch.log(probs[target_idx] + 1e-8)

        return loss

    @torch.no_grad()
    def predict_unstable_edges(
        self,
        h_gnn: torch.Tensor,
        pi_0: int,
        local_routes: List[List[int]],
        max_steps: int = 10,
    ) -> List[Tuple[int, int]]:
        """
        Greedy autoregressive decoding starting from initial node pi_0.
        Returns list of deleted (unstable) edges in local node indices.
        """
        N = h_gnn.size(0)
        adj: Dict[int, Set[int]] = {i: set() for i in range(N)}
        for r in local_routes:
            for idx in range(len(r) - 1):
                u, v = r[idx], r[idx + 1]
                adj[u].add(v)
                adj[v].add(u)

        h_pi_0 = h_gnn[pi_0]
        h_hidden = self.init_hidden(h_gnn)
        curr_node = pi_0

        deleted_edges: List[Tuple[int, int]] = []
        deleted_set: Set[Tuple[int, int]] = set()

        for step in range(max_steps):
            is_delete = (step % 2 == 0)
            h_pi_prev = h_gnn[curr_node]

            if is_delete:
                # Deletion candidates
                cand_ids = [
                    v for v in adj[curr_node]
                    if (min(curr_node, v), max(curr_node, v)) not in deleted_set
                ]
                if not cand_ids:
                    break

                cand_embeddings = h_gnn[cand_ids]
                probs, h_hidden = self.decode_step(
                    h_gnn=h_gnn,
                    h_pi_0=h_pi_0,
                    h_pi_prev=h_pi_prev,
                    h_hidden_prev=h_hidden,
                    candidate_embeddings=cand_embeddings,
                    stage="delete",
                )
                chosen_idx = int(torch.argmax(probs).item())
                chosen_node = cand_ids[chosen_idx]

                edge = (min(curr_node, chosen_node), max(curr_node, chosen_node))
                deleted_edges.append(edge)
                deleted_set.add(edge)
                curr_node = chosen_node

            else:
                # Insertion candidates
                h_end = self.compute_h_end(h_gnn, h_pi_0)
                all_nodes = [i for i in range(N) if i != curr_node]
                cand_ids = all_nodes + [-1]  # -1 represents end token

                cust_embeddings = h_gnn[all_nodes]
                cand_embeddings = torch.cat([cust_embeddings, h_end], dim=0)

                probs, h_hidden = self.decode_step(
                    h_gnn=h_gnn,
                    h_pi_0=h_pi_0,
                    h_pi_prev=h_pi_prev,
                    h_hidden_prev=h_hidden,
                    candidate_embeddings=cand_embeddings,
                    stage="insert",
                )
                chosen_idx = int(torch.argmax(probs).item())
                chosen_node = cand_ids[chosen_idx]

                if chosen_node == -1:
                    # End token chosen -> terminate AR decoding
                    break
                curr_node = chosen_node

        return deleted_edges
