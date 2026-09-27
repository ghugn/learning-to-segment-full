import math
from typing import List, Tuple, Dict
import numpy as np
from fsta.types import CVRPInstance


class SubProblem:
    """
    A subproblem formed by a pair of adjacent routes P_TR = {R_i, R_j}.
    According to the paper (Section 4.2 & Section 4.3):
    - Decomposing the full CVRP graph into adjacent route pairs avoids GPU memory bottlenecks.
    - Each subproblem contains the depot (node 0) and the customers in R_i and R_j.
    - Node indices are remapped to local range [0, N_sub - 1] with depot staying as 0.
    """
    def __init__(
        self,
        instance: CVRPInstance,
        route_i_idx: int,
        route_j_idx: int,
        route_i: List[int],
        route_j: List[int],
    ):
        self.route_i_idx = route_i_idx
        self.route_j_idx = route_j_idx
        self.orig_routes = [route_i, route_j]
        self.capacity = instance.capacity

        # Collect unique original nodes (depot is 0)
        custs_i = route_i[1:-1]
        custs_j = route_j[1:-1]
        all_orig_nodes = [0] + custs_i + custs_j

        self.local_to_orig: List[int] = all_orig_nodes
        self.orig_to_local: Dict[int, int] = {orig: loc for loc, orig in enumerate(all_orig_nodes)}

        # Local routes
        self.local_routes = [
            [self.orig_to_local[u] for u in route_i],
            [self.orig_to_local[u] for u in route_j],
        ]

        # Extract coords and demands for subproblem
        self.coords = instance.coords[all_orig_nodes]
        self.demands = instance.demands[all_orig_nodes]
        self.num_nodes = len(all_orig_nodes)

        # Route ID for each local node (0 for depot, 1 for route_i, 2 for route_j)
        self.route_membership = np.zeros(self.num_nodes, dtype=np.int64)
        for u in custs_i:
            self.route_membership[self.orig_to_local[u]] = 1
        for u in custs_j:
            self.route_membership[self.orig_to_local[u]] = 2

        # Position along the tour (0 for depot, 1..k for customer order in tour)
        self.tour_positions = np.zeros(self.num_nodes, dtype=np.int64)
        for pos, u in enumerate(custs_i, start=1):
            self.tour_positions[self.orig_to_local[u]] = pos
        for pos, u in enumerate(custs_j, start=1):
            self.tour_positions[self.orig_to_local[u]] = pos


def get_route_centroids(instance: CVRPInstance, routes: List[List[int]]) -> np.ndarray:
    """Compute 2D centroid of customers for each route."""
    centroids = []
    for r in routes:
        custs = r[1:-1]
        if custs:
            centroids.append(np.mean(instance.coords[custs], axis=0))
        else:
            centroids.append(instance.coords[0])
    return np.array(centroids)


def decompose_into_adjacent_subproblems(
    instance: CVRPInstance,
    routes: List[List[int]],
    strategy: str = "polar",
    max_spatial_neighbors: int = 1,
) -> List[SubProblem]:
    """
    Partition the full CVRP problem P with solution R into subproblems,
    each formed by grouping nodes from two adjacent routes (Section 4.3).
    
    Strategies:
    - "polar" (default): Circular angular order of route centroids w.r.t. the depot.
    - "spatial": Euclidean distance between route centroids in 2D space.
    - "hybrid": Combines polar circular neighbors with spatial nearest neighbors.
    """
    num_routes = len(routes)
    if num_routes < 2:
        return [SubProblem(instance, 0, 0, routes[0], routes[0])]

    depot = instance.coords[0]
    route_angles: List[Tuple[float, int]] = []

    for r_idx, r in enumerate(routes):
        custs = r[1:-1]
        if not custs:
            continue
        pts = instance.coords[custs]
        centroid = np.mean(pts, axis=0)
        angle = math.atan2(centroid[1] - depot[1], centroid[0] - depot[0])
        route_angles.append((angle, r_idx))

    route_angles.sort(key=lambda x: x[0])
    ordered_route_indices = [idx for _, idx in route_angles]

    subproblems: List[SubProblem] = []
    added_pairs = set()

    # 1. Polar circular adjacent pairs
    if strategy in ("polar", "hybrid"):
        m = len(ordered_route_indices)
        for i in range(m):
            idx_curr = ordered_route_indices[i]
            idx_next = ordered_route_indices[(i + 1) % m]
            pair_key = (min(idx_curr, idx_next), max(idx_curr, idx_next))
            if pair_key not in added_pairs:
                added_pairs.add(pair_key)
                subproblems.append(
                    SubProblem(
                        instance=instance,
                        route_i_idx=idx_curr,
                        route_j_idx=idx_next,
                        route_i=routes[idx_curr],
                        route_j=routes[idx_next],
                    )
                )

    # 2. Spatial Euclidean nearest neighbors
    if strategy in ("spatial", "hybrid") and len(routes) >= 3:
        centroids = get_route_centroids(instance, routes)
        for i in range(len(routes)):
            if len(routes[i]) <= 2:
                continue
            dists = np.linalg.norm(centroids - centroids[i], axis=1)
            dists[i] = 1e9
            nearest_indices = np.argsort(dists)
            added_for_i = 0
            for j in nearest_indices:
                if len(routes[j]) <= 2:
                    continue
                pair_key = (min(i, int(j)), max(i, int(j)))
                if pair_key not in added_pairs:
                    added_pairs.add(pair_key)
                    subproblems.append(
                        SubProblem(
                            instance=instance,
                            route_i_idx=i,
                            route_j_idx=int(j),
                            route_i=routes[i],
                            route_j=routes[j],
                        )
                    )
                    added_for_i += 1
                    if added_for_i >= max_spatial_neighbors:
                        break

    return subproblems
