import math
from typing import List, Optional
import numpy as np
from .types import CVRPInstance
from .recovery import validate_cvrp_solution


def build_greedy_sweep_solution(instance: CVRPInstance) -> List[List[int]]:
    """
    Construct a fast initial feasible CVRP solution using the simple Angular Sweep heuristic
    with intra-route 2-opt post-processing.
    """
    depot_coord = instance.coords[0]
    num_cust = instance.num_customers
    customer_indices = list(range(1, num_cust + 1))

    # Calculate angles w.r.t. depot
    angles = []
    for c in customer_indices:
        dx = instance.coords[c, 0] - depot_coord[0]
        dy = instance.coords[c, 1] - depot_coord[1]
        angles.append(math.atan2(dy, dx))

    sorted_customers = [c for _, c in sorted(zip(angles, customer_indices))]

    routes: List[List[int]] = []
    curr_route = [0]
    curr_load = 0.0

    for c in sorted_customers:
        d = instance.demands[c]
        if curr_load + d <= instance.capacity:
            curr_route.append(c)
            curr_load += d
        else:
            curr_route.append(0)
            routes.append(curr_route)
            curr_route = [0, c]
            curr_load = d

    if len(curr_route) > 1:
        curr_route.append(0)
        routes.append(curr_route)

    dist = instance.dist_matrix
    optimized_routes: List[List[int]] = []
    for r in routes:
        best_r = r[:]
        improved = True
        while improved:
            improved = False
            for i in range(1, len(best_r) - 2):
                for j in range(i + 1, len(best_r) - 1):
                    delta = (
                        dist[best_r[i - 1], best_r[j]]
                        + dist[best_r[i], best_r[j + 1]]
                        - dist[best_r[i - 1], best_r[i]]
                        - dist[best_r[j], best_r[j + 1]]
                    )
                    if delta < -1e-6:
                        best_r[i : j + 1] = best_r[i : j + 1][::-1]
                        improved = True
        optimized_routes.append(best_r)

    is_valid, msg = validate_cvrp_solution(instance, optimized_routes)
    if not is_valid:
        raise RuntimeError(f"Generated initial solution is invalid: {msg}")

    return optimized_routes


def build_sector_clustered_solution(
    instance: CVRPInstance,
    k_veh: int = 6,
    alpha_init: float = 0.95,
    time_limit_per_sector: float = 1.2,
    seed: int = 42,
) -> List[List[int]]:
    """
    Official Initial Solution Heuristic from Appendix D.1 (page 31) of ICLR 2026 paper:
    (Inspired by Li et al. 2021):
    - Partitions customers by polar angle w.r.t. depot into groups approaching
      alpha_init * K_veh * Capacity.
    - Solves each subproblem independently using a fast local/metaheuristic solver (PyVRP).
    - Assembles the sector solutions into a high-quality global feasible CVRP solution.
    """
    try:
        from solvers.pyvrp_solver import PyVRPSolver
        solver = PyVRPSolver(seed=seed)
    except Exception:
        return build_greedy_sweep_solution(instance)

    depot_coord = instance.coords[0]
    num_cust = instance.num_customers
    customer_indices = list(range(1, num_cust + 1))
    angles = [
        math.atan2(instance.coords[c, 1] - depot_coord[1], instance.coords[c, 0] - depot_coord[0])
        for c in customer_indices
    ]
    sorted_custs = [c for _, c in sorted(zip(angles, customer_indices))]

    target_demand = alpha_init * k_veh * instance.capacity
    sectors = []
    curr_sec = []
    curr_load = 0.0

    for c in sorted_custs:
        d = instance.demands[c]
        if curr_load + d <= target_demand or not curr_sec:
            curr_sec.append(c)
            curr_load += d
        else:
            sectors.append(curr_sec)
            curr_sec = [c]
            curr_load = d

    if curr_sec:
        sectors.append(curr_sec)

    all_routes: List[List[int]] = []
    for sec in sectors:
        if not sec:
            continue
        sec_nodes = [0] + sec
        sec_coords = instance.coords[sec_nodes]
        sec_demands = instance.demands[sec_nodes]
        sec_inst = CVRPInstance(coords=sec_coords, demands=sec_demands, capacity=instance.capacity)

        try:
            local_routes, _, _ = solver.solve(sec_inst, time_limit=time_limit_per_sector)
            for r in local_routes:
                orig_r = [sec_nodes[u] for u in r]
                all_routes.append(orig_r)
        except Exception:
            # Fallback for this sector: greedy sweep
            sec_greedy_routes = build_greedy_sweep_solution(sec_inst)
            for r in sec_greedy_routes:
                orig_r = [sec_nodes[u] for u in r]
                all_routes.append(orig_r)

    is_valid, msg = validate_cvrp_solution(instance, all_routes)
    if not is_valid:
        # Fallback to standard greedy sweep
        return build_greedy_sweep_solution(instance)

    return all_routes


def build_angular_sweep_solution(
    instance: CVRPInstance,
    method: str = "auto",
    k_veh: int = 6,
) -> List[List[int]]:
    """
    Main entry point for generating initial feasible CVRP solutions.
    
    If method == "auto":
        - Uses fast greedy sweep for small instances (< 60 customers, e.g. tests)
        - Uses paper Appendix D.1 Sector Clustering (K_veh=6) for large instances (>= 60 customers)
    If method == "sector":
        - Always uses paper Appendix D.1 Sector Clustering
    If method == "greedy":
        - Uses simple Angular Sweep + 2-opt
    """
    if method == "greedy" or (method == "auto" and instance.num_customers < 60):
        return build_greedy_sweep_solution(instance)
    return build_sector_clustered_solution(instance, k_veh=k_veh)
