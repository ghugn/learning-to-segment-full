from typing import List, Tuple, Dict
import numpy as np
from .types import CVRPInstance, AggregatedProblem


def recover_solution(
    agg_problem: AggregatedProblem,
    agg_routes: List[List[int]],
) -> List[List[int]]:
    """
    Recover the original CVRP solution R from the aggregated solution \tilde{R}.
    
    According to the FSTA Monotonicity Theorem (Section 3.2 and Appendix B.2):
    - When a dual hypernode pair (\tilde{x}_j, \tilde{x}_k) is traversed from head to tail,
      it expands to the forward original segment (x_j -> ... -> x_k).
    - When traversed from tail to head, it expands to the reversed segment (x_k -> ... -> x_j).
    - Single hypernodes expand to their single customer node.

    Args:
        agg_problem: AggregatedProblem holding segment mappings.
        agg_routes: Re-optimized routes on the aggregated graph.

    Returns:
        Recovered routes on original graph, e.g. [[0, v1, v2, ..., 0], ...].
    """
    recovered_routes: List[List[int]] = []

    for route_idx, agg_route in enumerate(agg_routes):
        if len(agg_route) < 2 or agg_route[0] != 0 or agg_route[-1] != 0:
            raise ValueError(
                f"Aggregated route {route_idx} must start and end at depot 0. Got: {agg_route}"
            )

        recovered_route: List[int] = [0]
        i = 1
        n = len(agg_route) - 1  # Excluding the final depot 0

        while i < n:
            curr_node = agg_route[i]

            if curr_node not in agg_problem.node_to_segment:
                raise ValueError(
                    f"Unknown aggregated node {curr_node} in route {route_idx}."
                )

            seg_id, role = agg_problem.node_to_segment[curr_node]
            segment = agg_problem.segments[seg_id]

            if role == "single":
                # Single customer node
                recovered_route.append(segment.start_node)
                i += 1
            elif role == "head":
                # Must be followed immediately by its tail hypernode
                if i + 1 >= n:
                    raise ValueError(
                        f"Head hypernode {curr_node} for segment {seg_id} is at route end without its tail."
                    )
                next_node = agg_route[i + 1]
                next_info = agg_problem.node_to_segment.get(next_node)
                if next_info != (seg_id, "tail"):
                    raise ValueError(
                        f"Fixed edge broken! Expected tail hypernode for segment {seg_id} after {curr_node}, got {next_node} ({next_info})."
                    )
                # Forward traversal
                recovered_route.extend(segment.nodes)
                i += 2  # Consume both head and tail
            elif role == "tail":
                # Must be followed immediately by its head hypernode (reversed traversal)
                if i + 1 >= n:
                    raise ValueError(
                        f"Tail hypernode {curr_node} for segment {seg_id} is at route end without its head."
                    )
                next_node = agg_route[i + 1]
                next_info = agg_problem.node_to_segment.get(next_node)
                if next_info != (seg_id, "head"):
                    raise ValueError(
                        f"Fixed edge broken! Expected head hypernode for segment {seg_id} after {curr_node}, got {next_node} ({next_info})."
                    )
                # Reverse traversal
                recovered_route.extend(list(reversed(segment.nodes)))
                i += 2  # Consume both tail and head
            else:
                raise ValueError(f"Unrecognized hypernode role: {role}")

        recovered_route.append(0)
        recovered_routes.append(recovered_route)

    return recovered_routes


def compute_route_cost(instance: CVRPInstance, route: List[int]) -> float:
    """Compute Euclidean travel cost of a single route starting and ending at 0."""
    cost = 0.0
    for i in range(len(route) - 1):
        cost += instance.get_distance(route[i], route[i + 1])
    return cost


def compute_solution_cost(instance: CVRPInstance, routes: List[List[int]]) -> float:
    """Compute total travel cost of a full solution."""
    return sum(compute_route_cost(instance, r) for r in routes)


def validate_cvrp_solution(
    instance: CVRPInstance,
    routes: List[List[int]],
) -> Tuple[bool, str]:
    """
    Validate that a CVRP solution is feasible:
    1. Every route starts and ends at depot 0.
    2. Every customer 1..N is visited exactly once.
    3. Vehicle capacity C is not exceeded on any route.

    Returns:
        (is_valid, error_message_or_ok)
    """
    num_customers = instance.num_customers
    visited = [0] * (num_customers + 1)

    for r_idx, route in enumerate(routes):
        if len(route) < 2:
            return False, f"Route {r_idx} has fewer than 2 nodes: {route}"
        if route[0] != 0 or route[-1] != 0:
            return False, f"Route {r_idx} does not start and end at depot 0: {route}"

        route_demand = 0.0
        for node in route[1:-1]:
            if node <= 0 or node > num_customers:
                return False, f"Invalid customer index {node} in route {r_idx}"
            visited[node] += 1
            route_demand += instance.demands[node]

        if route_demand > instance.capacity + 1e-6:
            return False, (
                f"Capacity violated on route {r_idx}: "
                f"demand {route_demand:.2f} > capacity {instance.capacity:.2f}"
            )

    unvisited = [i for i in range(1, num_customers + 1) if visited[i] == 0]
    multi_visited = [i for i in range(1, num_customers + 1) if visited[i] > 1]

    if unvisited:
        return False, f"Customers not visited: {unvisited[:10]} (total {len(unvisited)})"
    if multi_visited:
        return False, f"Customers visited more than once: {multi_visited[:10]}"

    return True, "Solution is valid and feasible."
