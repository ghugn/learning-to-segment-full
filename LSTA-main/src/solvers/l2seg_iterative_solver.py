import time
import copy
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from fsta.types import CVRPInstance, Segment, AggregatedProblem
from fsta.init_solution import build_angular_sweep_solution
from fsta.partition import partition_solution, normalize_edge
from fsta.aggregation import aggregate_segments
from fsta.recovery import recover_solution, validate_cvrp_solution
from fsta.local_search import (
    fsta_macro_local_search,
    parse_aggregated_route_into_blocks,
    blocks_to_aggregated_route,
    Block,
)
from features.subproblem import SubProblem, decompose_into_adjacent_subproblems, get_route_centroids
from .lns import LNSSolver


class L2SegIterativeSolver:
    """
    Algorithm 1: Iteratively Re-optimize Routing Problems with FSTA + L2Seg
    (Published as conference paper at ICLR 2026).
    
    Coordinates the full cycle:
    1. Unstable Edges Detection (via L2SegModel neural inference).
    2. Segment Partitioning (identifying stable segments).
    3. Hypernode Aggregation (reducing problem graph size by 70%-85%).
    4. Re-optimization with Backbone Solver (LNS).
    5. Solution Recovery (expanding hypernodes back with 100% feasibility guarantee).
    6. Monotonic solution update until time limit TTL is reached.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        backbone: str = "lns",  # "lns" is the paper backbone solver
        device: str = "cpu",
        seed: int = 42,
    ):
        self.model = model
        self.backbone_name = backbone.lower()
        self.device = device
        self.seed = seed

        self.lns = LNSSolver(seed=seed)

    def compute_cost(self, instance: CVRPInstance, routes: List[List[int]]) -> float:
        dist = instance.dist_matrix
        return sum(
            sum(dist[r[i], r[i + 1]] for i in range(len(r) - 1))
            for r in routes
        )

    def detect_unstable_edges(
        self,
        instance: CVRPInstance,
        routes: List[List[int]],
        threshold: float = 0.6,
        strategy: str = "polar",
    ) -> List[Tuple[int, int]]:
        """Run L2Seg neural model or heuristic fallback to predict unstable edges."""
        if self.model is None:
            # Fallback to longest edges heuristic if model not loaded
            all_edges = []
            for r in routes:
                for i in range(len(r) - 1):
                    all_edges.append((r[i], r[i + 1], instance.dist_matrix[r[i], r[i + 1]]))
            all_edges.sort(key=lambda x: x[2], reverse=True)
            k = max(2, int(len(all_edges) * 0.25))
            return [normalize_edge(u, v) for u, v, _ in all_edges[:k]]

        subproblems = decompose_into_adjacent_subproblems(instance, routes, strategy=strategy)
        predicted_unstable: Set[Tuple[int, int]] = set()

        for sub in subproblems:
            if sub.num_nodes <= 3:
                continue
            pred_edges = self.model.predict_subproblem_syn(
                subproblem=sub,
                threshold=threshold,
                n_clusters=3,
            )
            for u_orig, v_orig in pred_edges:
                predicted_unstable.add(normalize_edge(u_orig, v_orig))

        return list(predicted_unstable)

    def solve(
        self,
        instance: CVRPInstance,
        initial_routes: Optional[List[List[int]]] = None,
        time_limit: float = 30.0,
        max_iterations: Optional[int] = None,
        reopt_time_per_iter: float = 3.0,
    ) -> Dict[str, Any]:
        """
        Run Algorithm 1 until time limit is reached.
        
        Returns:
            Dict containing:
            - best_routes
            - best_cost
            - initial_cost
            - gap_improved_pct
            - total_time
            - iterations_run
            - average_compression_pct
        """
        start_time = time.perf_counter()

        if initial_routes is None:
            current_routes = build_angular_sweep_solution(instance)
        else:
            current_routes = [r[:] for r in initial_routes]

        current_cost = self.compute_cost(instance, current_routes)
        best_routes = [r[:] for r in current_routes]
        best_cost = current_cost
        initial_cost = current_cost

        iteration = 0
        compressions = []
        stagnation = 0
        max_iter = max_iterations if max_iterations is not None else 1000000

        while (time.perf_counter() - start_time) < time_limit and iteration < max_iter:
            iteration += 1
            iter_start = time.perf_counter()

            # Dynamic threshold & strategy adaptation when stagnating
            curr_threshold = max(0.30, 0.60 - 0.05 * min(6, stagnation))
            decomp_strategy = "hybrid" if stagnation >= 1 else "polar"

            # 1. Unstable Edges Detection via L2Seg
            unstable_edges = self.detect_unstable_edges(
                instance, current_routes, threshold=curr_threshold, strategy=decomp_strategy
            )

            # 2. Segment Partitioning
            partitioned = partition_solution(instance, current_routes, unstable_edges)

            # 3. Hypernode Aggregation
            agg_problem = aggregate_segments(instance, partitioned, embed_internal_cost=True)
            comp_pct = (1.0 - agg_problem.num_nodes / instance.num_nodes) * 100.0
            compressions.append(comp_pct)

            # 4. Re-optimization with Backbone Solver on Reduced Problem
            time_left = max(0.5, time_limit - (time.perf_counter() - start_time))
            step_time = min(reopt_time_per_iter, time_left)

            # 4. Re-optimization with Backbone Solver on Reduced Problem (FSTA Macro-Block Search)
            improved_agg_routes = fsta_macro_local_search(agg_problem, max_passes=5)

            # 5. Solution Recovery (Feasibility & Monotonicity Guaranteed)
            recovered_routes = recover_solution(agg_problem, improved_agg_routes)
            rec_cost = self.compute_cost(instance, recovered_routes)

            # 5b. Backbone LNS Re-optimization on Unstable Neighborhoods (Appendix C.1)
            unstable_nodes = {u for u, v in unstable_edges if u != 0} | {v for u, v in unstable_edges if v != 0}
            if unstable_nodes:
                lns_routes, lns_cost, _ = self.lns.solve(
                    instance, recovered_routes, time_limit=step_time, allowed_removal_nodes=unstable_nodes
                )
                if lns_cost < rec_cost:
                    recovered_routes = lns_routes
                    rec_cost = lns_cost

            # 6. Monotonic update
            if rec_cost < current_cost - 1e-6:
                current_routes = recovered_routes
                current_cost = rec_cost
                stagnation = 0
                if current_cost < best_cost:
                    best_routes = [r[:] for r in current_routes]
                    best_cost = current_cost
                    elapsed_now = time.perf_counter() - start_time
                    print(f"      -> [L2Seg-Iter {iteration:03d} | t={elapsed_now:6.1f}s] New Best: {best_cost:.3f} (Compressed: {comp_pct:.1f}%)")
            else:
                stagnation += 1
                # If stagnation >= 6, gentle boundary shake perturbation to escape local basin
                if stagnation >= 6 and len(current_routes) >= 2:
                    centroids = get_route_centroids(instance, current_routes)
                    i = int(iteration % len(current_routes))
                    dists = np.linalg.norm(centroids - centroids[i], axis=1)
                    dists[i] = 1e9
                    j = int(np.argmin(dists))
                    r_i = current_routes[i]
                    r_j = current_routes[j]
                    if len(r_i) > 3 and len(r_j) > 3:
                        node_to_move = r_i[-2]
                        dem = instance.demands[node_to_move]
                        r_j_dem = sum(instance.demands[u] for u in r_j[1:-1])
                        if r_j_dem + dem <= instance.capacity:
                            new_r_i = r_i[:-2] + [0]
                            new_r_j = r_j[:-1] + [node_to_move, 0]
                            current_routes[i] = new_r_i
                            current_routes[j] = new_r_j
                            current_cost = self.compute_cost(instance, current_routes)
                            stagnation = 2  # reset partial stagnation to explore new basin

            if iteration % 15 == 0:
                elapsed_now = time.perf_counter() - start_time
                print(f"      .. [L2Seg-Iter {iteration:03d} | t={elapsed_now:6.1f}s] Current Best: {best_cost:.3f}")

            # If iteration took too long, check loop exit
            if (time.perf_counter() - start_time) >= time_limit:
                break

        total_time = time.perf_counter() - start_time
        gain_pct = (initial_cost - best_cost) / initial_cost * 100.0 if initial_cost > 0 else 0.0

        return {
            "best_routes": best_routes,
            "best_cost": best_cost,
            "initial_cost": initial_cost,
            "gain_pct": gain_pct,
            "total_time": total_time,
            "iterations": iteration,
            "avg_compression_pct": float(np.mean(compressions)) if compressions else 0.0,
            "is_valid": validate_cvrp_solution(instance, best_routes)[0],
        }
