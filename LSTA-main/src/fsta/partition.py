from typing import List, Set, Tuple, Union
import numpy as np
from .types import CVRPInstance, Segment


def normalize_edge(u: int, v: int) -> Tuple[int, int]:
    """Convert an undirected edge (u, v) into a canonical sorted tuple (min, max)."""
    return (min(u, v), max(u, v))


def partition_solution(
    instance: CVRPInstance,
    routes: List[List[int]],
    unstable_edges: Union[Set[Tuple[int, int]], List[Tuple[int, int]]],
) -> List[List[Segment]]:
    """
    Partition each route into disjoint segments based on unstable edges.
    
    According to the FSTA paper (Section 3.2 and Appendix B.1.4):
    - Edges connected to the depot (0, v) are always treated as unstable (cut points).
    - Unstable customer edges (u, v) are removed.
    - Consecutive customer nodes connected by stable edges form a Segment.
    - Segments can consist of a single node or multiple nodes.

    Args:
        instance: CVRPInstance with demands and coordinates.
        routes: List of routes, each formatted as [0, v_1, v_2, ..., v_k, 0].
        unstable_edges: Collection of (u, v) tuples indicating unstable edges.

    Returns:
        List of lists of Segments, one list per route: [[Seg1, Seg2, ...], [Seg3, ...]].
    """
    # Convert unstable edges into canonical undirected set
    unstable_set: Set[Tuple[int, int]] = {normalize_edge(u, v) for u, v in unstable_edges}

    partitioned_routes: List[List[Segment]] = []
    global_segment_id = 0

    for route_idx, route in enumerate(routes):
        # Validate route format: must start and end at depot 0
        if len(route) < 2 or route[0] != 0 or route[-1] != 0:
            raise ValueError(f"Route {route_idx} must start and end at depot 0. Got: {route}")

        customers = route[1:-1]
        if not customers:
            # Empty route [0, 0]
            partitioned_routes.append([])
            continue

        route_segments: List[Segment] = []
        current_segment_nodes: List[int] = [customers[0]]

        for t in range(len(customers) - 1):
            u = customers[t]
            v = customers[t + 1]
            edge = normalize_edge(u, v)

            if edge in unstable_set:
                # The edge between u and v is unstable -> cut and finalize current segment
                tot_demand = float(np.sum(instance.demands[current_segment_nodes]))
                internal_cost = 0.0
                for idx in range(len(current_segment_nodes) - 1):
                    internal_cost += instance.get_distance(
                        current_segment_nodes[idx], current_segment_nodes[idx + 1]
                    )

                route_segments.append(
                    Segment(
                        segment_id=global_segment_id,
                        route_id=route_idx,
                        nodes=list(current_segment_nodes),
                        total_demand=tot_demand,
                        internal_cost=internal_cost,
                    )
                )
                global_segment_id += 1
                # Start new segment with node v
                current_segment_nodes = [v]
            else:
                # Edge is stable -> append v to current segment
                current_segment_nodes.append(v)

        # Finalize the last segment of the route
        tot_demand = float(np.sum(instance.demands[current_segment_nodes]))
        internal_cost = 0.0
        for idx in range(len(current_segment_nodes) - 1):
            internal_cost += instance.get_distance(
                current_segment_nodes[idx], current_segment_nodes[idx + 1]
            )

        route_segments.append(
            Segment(
                segment_id=global_segment_id,
                route_id=route_idx,
                nodes=list(current_segment_nodes),
                total_demand=tot_demand,
                internal_cost=internal_cost,
            )
        )
        global_segment_id += 1

        partitioned_routes.append(route_segments)

    return partitioned_routes
