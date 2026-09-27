from typing import List, Tuple, Set, Optional
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

from .encoder import L2SegEncoder
from .nar_decoder import L2SegNARDecoder
from .ar_decoder import L2SegARDecoder
from features.subproblem import SubProblem
from features.node_features import extract_subproblem_node_features
from features.edge_features import extract_subproblem_edges


class L2SegModel(nn.Module):
    """
    Complete L2Seg Framework (Learning to Segment for Vehicle Routing Problems).
    
    Contains:
    - Shared Encoder (Route-Masked Transformer + TransformerConv GNN)
    - NAR Decoder (One-shot node instability prediction)
    - AR Decoder (Sequential alternating Deletion/Insertion local edge detection)
    
    Supports:
    - L2Seg-NAR: Algorithm 4
    - L2Seg-AR:  Algorithm 5
    - L2Seg-SYN: Algorithm 3 (Synergized prediction - Best performance)
    """
    def __init__(
        self,
        node_in_dim: int = 25,
        edge_in_dim: int = 3,
        hidden_dim: int = 128,
        num_tfm_layers: int = 2,
        num_gnn_layers: int = 2,
        num_ar_delete_layers: int = 1,
        num_ar_insert_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = L2SegEncoder(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            num_tfm_layers=num_tfm_layers,
            num_gnn_layers=num_gnn_layers,
            dropout=dropout,
        )
        self.nar_decoder = L2SegNARDecoder(
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.ar_decoder = L2SegARDecoder(
            hidden_dim=hidden_dim,
            num_delete_layers=num_ar_delete_layers,
            num_insert_layers=num_ar_insert_layers,
            dropout=dropout,
        )

    @classmethod
    def load_pretrained(cls, nar_path: str = "checkpoints/nar_model.pt", ar_path: str = "checkpoints/ar_model.pt", device: str = "cpu"):
        return load_trained_l2seg_model(nar_path=nar_path, ar_path=ar_path, device=device)

    def forward_nar(
        self,
        node_feats: torch.Tensor,
        tour_positions: torch.Tensor,
        route_membership: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for NAR training / evaluation."""
        h_gnn = self.encoder(
            node_feats=node_feats,
            tour_positions=tour_positions,
            route_membership=route_membership,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        p_nar = self.nar_decoder(h_gnn)
        return p_nar

    def encode_subproblem(self, subproblem: SubProblem) -> torch.Tensor:
        """Helper to extract features and compute GNN node embeddings for a SubProblem."""
        node_feats = extract_subproblem_node_features(subproblem).to(self.device)
        tour_positions = torch.from_numpy(subproblem.tour_positions).long().to(self.device)
        route_membership = torch.from_numpy(subproblem.route_membership).long().to(self.device)
        edge_index, edge_attr = extract_subproblem_edges(subproblem)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)

        return self.encoder(
            node_feats=node_feats,
            tour_positions=tour_positions,
            route_membership=route_membership,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def predict_subproblem_syn(
        self,
        subproblem: SubProblem,
        threshold: float = 0.6,
        n_clusters: int = 3,
    ) -> Set[Tuple[int, int]]:
        r"""
        L2Seg-SYN Synergized Prediction on a SubProblem (Algorithm 3, Page 29):
        1. Global unstable node detection via NAR model: \hat{y}^{NAR} = {x_i | p_i^{NAR} >= \eta}.
        2. K-means clustering on predicted unstable nodes to group into regional clusters.
        3. Identify starting initial node x_{\pi_0} with highest probability in each cluster.
        4. Local unstable edge detection using AR model starting from each x_{\pi_0}.
        
        Returns:
            Set of original (u_orig, v_orig) unstable edges for this subproblem.
        """
        h_gnn = self.encode_subproblem(subproblem)
        p_nar = self.nar_decoder(h_gnn).cpu().numpy()  # (N_sub,)

        # Identify unstable nodes above threshold (excluding depot at index 0)
        unstable_mask = (p_nar >= threshold)
        unstable_mask[0] = False  # Depot handled separately
        candidate_nodes = np.where(unstable_mask)[0]

        if len(candidate_nodes) == 0:
            # Fallback: select top-k customer nodes with highest probabilities
            cust_probs = p_nar[1:]
            if len(cust_probs) > 0:
                top_idx = np.argsort(cust_probs)[-min(n_clusters, len(cust_probs)):]
                candidate_nodes = top_idx + 1  # Offset by 1 for depot
            else:
                return set()

        # Step 2: Cluster candidate nodes using K-Means on coords
        k = min(n_clusters, len(candidate_nodes))
        cand_coords = subproblem.coords[candidate_nodes]

        if k == 1 or len(candidate_nodes) <= k:
            clusters = {i: [node] for i, node in enumerate(candidate_nodes)}
        else:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(cand_coords)
            clusters = {c: [] for c in range(k)}
            for node, cluster_id in zip(candidate_nodes, kmeans.labels_):
                clusters[cluster_id].append(node)

        # Step 3: Select initial node with highest p_nar within each cluster
        initial_nodes = []
        for cluster_id, nodes in clusters.items():
            if not nodes:
                continue
            best_node = max(nodes, key=lambda u: p_nar[u])
            initial_nodes.append(best_node)

        # Step 4: Run AR decoder from each initial node
        unstable_local_edges: Set[Tuple[int, int]] = set()
        for pi_0 in initial_nodes:
            ar_edges = self.ar_decoder.predict_unstable_edges(
                h_gnn=h_gnn,
                pi_0=pi_0,
                local_routes=subproblem.local_routes,
                max_steps=10,
            )
            for u_loc, v_loc in ar_edges:
                unstable_local_edges.add((min(u_loc, v_loc), max(u_loc, v_loc)))

        # Map local edge indices back to original graph node indices
        unstable_orig_edges: Set[Tuple[int, int]] = set()
        for u_loc, v_loc in unstable_local_edges:
            u_orig = subproblem.local_to_orig[u_loc]
            v_orig = subproblem.local_to_orig[v_loc]
            unstable_orig_edges.add((min(u_orig, v_orig), max(u_orig, v_orig)))

        return unstable_orig_edges

    @torch.no_grad()
    def predict_subproblem_nar(
        self,
        subproblem: SubProblem,
        threshold: float = 0.6,
    ) -> Set[Tuple[int, int]]:
        """
        L2Seg-NAR Non-Autoregressive Prediction (Algorithm 4, Page 30):
        Marks all solution edges connected to predicted unstable nodes as unstable.
        """
        h_gnn = self.encode_subproblem(subproblem)
        p_nar = self.nar_decoder(h_gnn).cpu().numpy()

        unstable_nodes = {i for i in range(1, subproblem.num_nodes) if p_nar[i] >= threshold}

        unstable_orig_edges: Set[Tuple[int, int]] = set()
        for r in subproblem.local_routes:
            for idx in range(len(r) - 1):
                u, v = r[idx], r[idx + 1]
                if u in unstable_nodes or v in unstable_nodes:
                    u_orig = subproblem.local_to_orig[u]
        return unstable_orig_edges


def load_trained_l2seg_model(
    nar_path: str = "checkpoints/nar_model.pt",
    ar_path: str = "checkpoints/ar_model.pt",
    device: str = "cpu",
) -> L2SegModel:
    """Load trained weights into unified L2SegModel."""
    import os
    model = L2SegModel(
        node_in_dim=25,
        edge_in_dim=3,
        hidden_dim=128,
        num_tfm_layers=2,
        num_gnn_layers=2,
        num_ar_delete_layers=1,
        num_ar_insert_layers=4,
        dropout=0.0,
    ).to(device)

    # Load NAR weights
    if os.path.exists(nar_path):
        ckpt_nar = torch.load(nar_path, map_location=device, weights_only=False)
        if "encoder_state_dict" in ckpt_nar:
            model.encoder.load_state_dict(ckpt_nar["encoder_state_dict"])
        if "decoder_state_dict" in ckpt_nar:
            model.nar_decoder.load_state_dict(ckpt_nar["decoder_state_dict"])

    # Load AR weights
    if os.path.exists(ar_path):
        ckpt_ar = torch.load(ar_path, map_location=device, weights_only=False)
        if "decoder_state_dict" in ckpt_ar:
            model.ar_decoder.load_state_dict(ckpt_ar["decoder_state_dict"])

    model.eval()
    return model


def predict_unstable_edges_l2seg_syn(
    model: L2SegModel,
    instance: Any,
    routes: List[List[int]],
    threshold: float = 0.6,
    n_clusters: int = 3,
) -> Set[Tuple[int, int]]:
    """
    Run full L2Seg-SYN Synergized Prediction across all adjacent route pairs (Algorithm 3):
    1. Partition P into ~|R| subproblems P_TR.
    2. In each subproblem:
       - NAR detects candidate unstable nodes.
       - K-Means clusters them to find focal regions.
       - AR decodes alternating deletion/insertion moves from focal nodes.
    3. Aggregate all predicted unstable edges across subproblems.
    """
    from features.subproblem import decompose_into_adjacent_subproblems
    subproblems = decompose_into_adjacent_subproblems(instance, routes)
    global_unstable_edges: Set[Tuple[int, int]] = set()

    for sub in subproblems:
        sub_edges = model.predict_subproblem_syn(
            subproblem=sub,
            threshold=threshold,
            n_clusters=n_clusters,
        )
        global_unstable_edges.update(sub_edges)

    return global_unstable_edges
