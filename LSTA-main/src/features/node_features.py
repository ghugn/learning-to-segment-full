import math
from typing import List
import numpy as np
import torch
from .subproblem import SubProblem


def extract_subproblem_node_features(subproblem: SubProblem) -> torch.Tensor:
    """
    Extract enhanced 25-dimensional node features according to Table 8 (Page 28)
    and Appendix D.3 (Page 32) of the L2Seg paper.

    Dimensions:
    1.  [0:2]   The xy coordinates (2)
    2.  [2:3]   The normalized demand d_i / C (1)
    3.  [3:5]   The centroid of the subtour for each node (2)
    4.  [5:9]   The coordinates of the two nodes connecting to each node (4)
    5.  [9:11]  The travel cost of the two edges connecting to each node (2)
    6.  [11:13] The relative xy coordinates w.r.t. the depot (2)
    7.  [13:14] The angle w.r.t. the depot (1)
    8.  [14:15] The weighted angle w.r.t. the depot by distance (1)
    9.  [15:18] The distances of the closest 3 neighbors for each node (3)
    10. [18:21] Percentage of K nearest nodes in same subtour (K = 5, 15, 40) (3)
    11. [21:24] Percentage of K% nearest nodes in same subtour (K = 5%, 15%, 40%) (3)
    12. [24:25] The distance to the depot (1)

    Total dimension: 25
    """
    coords = subproblem.coords.copy()
    # Normalize coordinates to [0, 1] if input scale is large (e.g. CVRPLib Set-X instances)
    if np.max(coords) > 1.5 or np.min(coords) < -0.5:
        c_min = coords.min(axis=0)
        c_max = coords.max(axis=0)
        span = np.maximum(c_max - c_min, 1e-6)
        coords = (coords - c_min) / np.max(span)

    demands = subproblem.demands  # (N_sub,)
    capacity = subproblem.capacity
    num_nodes = subproblem.num_nodes
    depot_coord = coords[0]

    # Precompute pairwise distances
    diff = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

    # Feature 1: xy coordinates (2)
    feat_coords = coords.copy()

    # Feature 2: normalized demand (1)
    feat_demand = (demands / max(capacity, 1.0))[:, None]

    # Feature 3: Subtour centroid (2)
    # Subproblem has 2 routes (local routes)
    feat_centroids = np.zeros((num_nodes, 2), dtype=np.float32)
    route_centroids = {}
    for r_id, r in enumerate(subproblem.local_routes, start=1):
        custs = r[1:-1]
        if custs:
            route_centroids[r_id] = np.mean(coords[custs], axis=0)
        else:
            route_centroids[r_id] = depot_coord

    for i in range(num_nodes):
        r_id = subproblem.route_membership[i]
        if r_id == 0:
            feat_centroids[i] = depot_coord
        else:
            feat_centroids[i] = route_centroids.get(r_id, depot_coord)

    # Feature 4 & 5: Predecessor and Successor coordinates (4) & Travel costs (2)
    feat_conn_coords = np.zeros((num_nodes, 4), dtype=np.float32)
    feat_conn_costs = np.zeros((num_nodes, 2), dtype=np.float32)

    # Map each local node to its prev and next nodes in its route
    for r in subproblem.local_routes:
        for idx in range(1, len(r) - 1):
            curr_node = r[idx]
            prev_node = r[idx - 1]
            next_node = r[idx + 1]

            feat_conn_coords[curr_node, 0:2] = coords[prev_node]
            feat_conn_coords[curr_node, 2:4] = coords[next_node]
            feat_conn_costs[curr_node, 0] = dist_matrix[prev_node, curr_node]
            feat_conn_costs[curr_node, 1] = dist_matrix[curr_node, next_node]

    # For depot, set connecting nodes as starts/ends of routes
    feat_conn_coords[0, 0:2] = coords[subproblem.local_routes[0][1]] if len(subproblem.local_routes[0]) > 2 else depot_coord
    feat_conn_coords[0, 2:4] = coords[subproblem.local_routes[1][1]] if len(subproblem.local_routes[1]) > 2 else depot_coord

    # Feature 6: Relative xy coordinates w.r.t depot (2)
    feat_rel_coords = coords - depot_coord[None, :]

    # Feature 7: Angle w.r.t depot (1)
    feat_angles = np.zeros((num_nodes, 1), dtype=np.float32)
    for i in range(num_nodes):
        if i == 0:
            feat_angles[i, 0] = 0.0
        else:
            dx = coords[i, 0] - depot_coord[0]
            dy = coords[i, 1] - depot_coord[1]
            feat_angles[i, 0] = math.atan2(dy, dx) / math.pi

    # Feature 12: Distance to depot (1)
    feat_dist_depot = dist_matrix[:, 0:1]

    # Feature 8: Distance-weighted angle w.r.t depot (1)
    feat_weighted_angles = feat_angles * feat_dist_depot

    # Feature 9: Distances of closest 3 neighbors (3)
    feat_dist_3nn = np.zeros((num_nodes, 3), dtype=np.float32)
    sorted_dists = np.sort(dist_matrix, axis=1)  # col 0 is self-distance 0.0
    for i in range(num_nodes):
        avail = sorted_dists[i, 1:4]
        feat_dist_3nn[i, :len(avail)] = avail

    # Feature 10 & 11: % of K nearest nodes within same subtour (K = 5, 15, 40)
    # and % of K% nearest nodes within same subtour (K = 5%, 15%, 40%)
    feat_pct_k = np.zeros((num_nodes, 3), dtype=np.float32)
    feat_pct_k_pct = np.zeros((num_nodes, 3), dtype=np.float32)

    k_values = [5, 15, 40]
    k_pct_values = [0.05, 0.15, 0.40]

    # Sort neighbor indices by distance (excluding self at index 0)
    sorted_neighbor_indices = np.argsort(dist_matrix, axis=1)[:, 1:]

    for i in range(num_nodes):
        my_route = subproblem.route_membership[i]
        neighbors = sorted_neighbor_indices[i]
        num_avail = len(neighbors)
        if num_avail == 0 or my_route == 0:
            continue

        # K fixed values
        for idx, k in enumerate(k_values):
            k_eff = min(k, num_avail)
            top_k_routes = subproblem.route_membership[neighbors[:k_eff]]
            same_count = np.sum(top_k_routes == my_route)
            feat_pct_k[i, idx] = same_count / max(k_eff, 1)

        # K percent values
        for idx, pct in enumerate(k_pct_values):
            k_eff = max(1, min(int(round(pct * num_avail)), num_avail))
            top_k_routes = subproblem.route_membership[neighbors[:k_eff]]
            same_count = np.sum(top_k_routes == my_route)
            feat_pct_k_pct[i, idx] = same_count / max(k_eff, 1)

    # Concatenate all 25 features
    node_features = np.hstack([
        feat_coords,           # 2
        feat_demand,           # 1
        feat_centroids,        # 2
        feat_conn_coords,      # 4
        feat_conn_costs,       # 2
        feat_rel_coords,       # 2
        feat_angles,           # 1
        feat_weighted_angles,  # 1
        feat_dist_3nn,         # 3
        feat_pct_k,            # 3
        feat_pct_k_pct,        # 3
        feat_dist_depot,       # 1
    ]).astype(np.float32)

    assert node_features.shape[1] == 25, f"Expected 25 node features, got {node_features.shape[1]}"
    return torch.from_numpy(node_features)
