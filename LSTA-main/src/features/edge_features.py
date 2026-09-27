from typing import Tuple, Set
import numpy as np
import torch
from .subproblem import SubProblem


def extract_subproblem_edges(
    subproblem: SubProblem,
    k_nearest: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construct graph edge_index and edge_attr according to Table 8 and Section 4.1:
    - Edge set includes:
        1. Solution edges: bidirectional edges from current local routes E_R.
        2. K-NN edges: edges connecting each node to its k nearest neighbors.
    - 3-dimensional Edge Features:
        1. Travel cost: Euclidean distance dist(u, v)
        2. In-solution indicator: 1.0 if edge in solution, 0.0 otherwise
        3. Travel cost rank: rank of distance w.r.t. the source node, normalized in [0, 1]

    Returns:
        edge_index: LongTensor of shape (2, num_edges)
        edge_attr: FloatTensor of shape (num_edges, 3)
    """
    coords = subproblem.coords.copy()
    if np.max(coords) > 1.5 or np.min(coords) < -0.5:
        c_min = coords.min(axis=0)
        c_max = coords.max(axis=0)
        span = np.maximum(c_max - c_min, 1e-6)
        coords = (coords - c_min) / np.max(span)

    num_nodes = subproblem.num_nodes

    # Compute Euclidean distance matrix
    diff = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

    # Identify solution edges (bidirectional)
    solution_edges_set: Set[Tuple[int, int]] = set()
    for r in subproblem.local_routes:
        for idx in range(len(r) - 1):
            u, v = r[idx], r[idx + 1]
            solution_edges_set.add((u, v))
            solution_edges_set.add((v, u))

    # Collect directed edges
    edges_dict = {}  # (u, v) -> [dist, in_sol, rank]

    # For each node u, find k nearest neighbors
    sorted_neighbors = np.argsort(dist_matrix, axis=1)  # row u sorted by dist

    effective_k = min(k_nearest, num_nodes - 1)

    for u in range(num_nodes):
        # 1. Add solution edges incident to u
        for v in range(num_nodes):
            if (u, v) in solution_edges_set:
                edges_dict[(u, v)] = [dist_matrix[u, v], 1.0, 0.0]

        # 2. Add K-NN edges from u
        for rank, v in enumerate(sorted_neighbors[u, 1 : effective_k + 1], start=1):
            if (u, v) not in edges_dict:
                edges_dict[(u, v)] = [dist_matrix[u, v], 0.0, 0.0]

    # Assign normalized distance rank for each edge outgoing from u
    # Group edges by source node u
    out_edges_per_node = {u: [] for u in range(num_nodes)}
    for (u, v), feat in edges_dict.items():
        out_edges_per_node[u].append((feat[0], (u, v)))

    for u in range(num_nodes):
        edge_list = out_edges_per_node[u]
        if not edge_list:
            continue
        # Sort by distance
        edge_list.sort(key=lambda x: x[0])
        max_rank = max(len(edge_list) - 1, 1)
        for rank_idx, (_, edge) in enumerate(edge_list):
            edges_dict[edge][2] = float(rank_idx) / max_rank

    # Build tensors
    edge_list = list(edges_dict.keys())
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()  # (2, E)
    edge_attrs = np.array([edges_dict[e] for e in edge_list], dtype=np.float32)
    edge_attr = torch.from_numpy(edge_attrs)  # (E, 3)

    return edge_index, edge_attr
