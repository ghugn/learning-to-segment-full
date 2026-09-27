from typing import List, Set, Tuple, Optional
import numpy as np
from .types import CVRPInstance
from .partition import normalize_edge


def random_edge_selector(
    routes: List[List[int]],
    ratio: float = 0.3,
    seed: Optional[int] = None,
) -> Set[Tuple[int, int]]:
    """
    Randomly select a fraction of internal customer edges to be unstable.
    (Baseline 'Random FSTA' from Section 5.4 of the paper).

    Args:
        routes: Current CVRP routes.
        ratio: Fraction of internal edges to mark as unstable (e.g. 0.3 for 30%).
        seed: Random seed for reproducibility.

    Returns:
        Set of undirected (min, max) edges.
    """
    rng = np.random.RandomState(seed)
    internal_edges: List[Tuple[int, int]] = []

    for route in routes:
        custs = route[1:-1]
        for i in range(len(custs) - 1):
            internal_edges.append(normalize_edge(custs[i], custs[i + 1]))

    if not internal_edges:
        return set()

    num_select = max(1, int(round(len(internal_edges) * ratio)))
    num_select = min(num_select, len(internal_edges))
    chosen_indices = rng.choice(len(internal_edges), size=num_select, replace=False)
    
    return {internal_edges[idx] for idx in chosen_indices}


def longest_edge_selector(
    instance: CVRPInstance,
    routes: List[List[int]],
    ratio: float = 0.2,
) -> Set[Tuple[int, int]]:
    """
    Heuristic edge selector: Select the longest internal edges in the current solution.
    Long edges are geometrically prime candidates for 2-opt/3-opt destruction.

    Args:
        instance: CVRPInstance with distance matrix.
        routes: Current CVRP routes.
        ratio: Top fraction of longest internal edges to select.

    Returns:
        Set of undirected (min, max) edges.
    """
    edge_costs: List[Tuple[float, Tuple[int, int]]] = []

    for route in routes:
        custs = route[1:-1]
        for i in range(len(custs) - 1):
            u, v = custs[i], custs[i + 1]
            dist = instance.get_distance(u, v)
            edge_costs.append((dist, normalize_edge(u, v)))

    if not edge_costs:
        return set()

    # Sort descending by distance
    edge_costs.sort(key=lambda x: x[0], reverse=True)
    num_select = max(1, int(round(len(edge_costs) * ratio)))
    num_select = min(num_select, len(edge_costs))

    return {edge for _, edge in edge_costs[:num_select]}


def oracle_diff_selector(
    routes_before: List[List[int]],
    routes_after: List[List[int]],
) -> Set[Tuple[int, int]]:
    r"""
    The 'Look-Ahead Oracle' selector (Appendix B.1.1 & Section 4.2 of the paper).
    Identifies differing edges between current solution R and an improved solution R+:
        E_diff = (E_R \ E_{R+}) U (E_{R+} \ E_R)

    Returns the deleted/unstable edges from E_R that were broken in R+.
    """
    def extract_edges(sol: List[List[int]]) -> Set[Tuple[int, int]]:
        edges: Set[Tuple[int, int]] = set()
        for r in sol:
            for i in range(len(r) - 1):
                edges.add(normalize_edge(r[i], r[i + 1]))
        return edges

    edges_before = extract_edges(routes_before)
    edges_after = extract_edges(routes_after)

    # Edges present in R that were removed in R+
    deleted_edges = edges_before - edges_after
    
    # Filter out edges connected to depot (as depot edges are already cut by default in FSTA)
    customer_unstable_edges = {
        (u, v) for u, v in deleted_edges if u != 0 and v != 0
    }
    return customer_unstable_edges
