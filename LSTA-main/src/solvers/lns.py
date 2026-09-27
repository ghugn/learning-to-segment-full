import copy
import time
import math
import numpy as np
from typing import List, Tuple, Set, Optional

from fsta.types import CVRPInstance


class LNSSolver:
    """
    Large Neighborhood Search (Shaw, 1998; Ropke & Pisinger, 2006).
    Standard baseline and backbone solver for CVRP.
    
    Features:
    - Destroy: Shaw (relatedness) removal, Random removal, Worst removal.
    - Repair: Greedy best insertion, Regret-2 insertion.
    - Acceptance: Simulated Annealing with geometric cooling.
    """

    def __init__(
        self,
        destroy_fraction_range: Tuple[float, float] = (0.1, 0.3),
        shaw_weights: Tuple[float, float] = (0.8, 0.2), # dist, demand
        cooling_rate: float = 0.995,
        seed: int = 42,
    ):
        self.min_q_frac, self.max_q_frac = destroy_fraction_range
        self.w_dist, self.w_dem = shaw_weights
        self.cooling_rate = cooling_rate
        self.rng = np.random.RandomState(seed)

    def route_cost(self, r: List[int], dist: np.ndarray) -> float:
        return sum(dist[r[i], r[i + 1]] for i in range(len(r) - 1))

    def solution_cost(self, routes: List[List[int]], dist: np.ndarray) -> float:
        return sum(self.route_cost(r, dist) for r in routes)

    def solve(
        self,
        instance: CVRPInstance,
        initial_routes: List[List[int]],
        time_limit: float = 10.0,
        max_iterations: int = 1000,
        allowed_removal_nodes: Optional[Set[int]] = None,
    ) -> Tuple[List[List[int]], float, int]:
        """
        Run LNS on instance starting from initial_routes.
        
        Returns:
            (best_routes, best_cost, num_iterations)
        """
        start_time = time.perf_counter()
        dist = instance.dist_matrix
        demands = instance.demands
        capacity = instance.capacity
        n_clients = instance.num_customers

        current_routes = [r[:] for r in initial_routes if len(r) > 2]
        current_cost = self.solution_cost(current_routes, dist)

        best_routes = [r[:] for r in current_routes]
        best_cost = current_cost

        # Initial temperature for SA: accept a 5% worsening move with p=0.5
        T = max(1.0, current_cost * 0.05 / math.log(2.0))
        T_min = 1e-3

        iteration = 0
        while (time.perf_counter() - start_time) < time_limit and iteration < max_iterations:
            iteration += 1

            # Determine number of customers to remove: q in [q_min, q_max]
            # Standard Shaw (1998) & Ropke & Pisinger (2006) for large CVRP: cap at 10-25 nodes
            n_target = len(allowed_removal_nodes) if allowed_removal_nodes is not None else n_clients
            q_min = max(2, min(10, int(self.min_q_frac * n_target)))
            q_max = max(q_min + 1, min(25, int(self.max_q_frac * n_target)))
            q = self.rng.randint(q_min, q_max + 1)

            # Choose destroy operator: 0 = Shaw, 1 = Worst, 2 = Random
            dest_choice = self.rng.choice([0, 1, 2], p=[0.5, 0.3, 0.2])
            if dest_choice == 0:
                removed_nodes, remaining_routes = self._shaw_removal(current_routes, instance, q, allowed_removal_nodes)
            elif dest_choice == 1:
                removed_nodes, remaining_routes = self._worst_removal(current_routes, instance, q, allowed_removal_nodes)
            else:
                removed_nodes, remaining_routes = self._random_removal(current_routes, q, allowed_removal_nodes)

            # Choose repair operator: 70% Regret-2, 30% Greedy
            if self.rng.rand() < 0.7:
                cand_routes = self._regret2_insertion(remaining_routes, removed_nodes, instance)
            else:
                cand_routes = self._greedy_insertion(remaining_routes, removed_nodes, instance)

            cand_cost = self.solution_cost(cand_routes, dist)

            # Acceptance criterion: SA
            delta = cand_cost - current_cost
            if delta < 0 or (T > T_min and self.rng.rand() < math.exp(-delta / T)):
                current_routes = cand_routes
                current_cost = cand_cost

                if current_cost < best_cost - 1e-6:
                    best_routes = [r[:] for r in current_routes]
                    best_cost = current_cost

            # Cool down
            T = max(T_min, T * self.cooling_rate)

        return best_routes, best_cost, iteration

    def _random_removal(
        self,
        routes: List[List[int]],
        q: int,
        allowed_removal_nodes: Optional[Set[int]] = None,
    ) -> Tuple[List[int], List[List[int]]]:
        all_nodes = [u for r in routes for u in r[1:-1] if allowed_removal_nodes is None or u in allowed_removal_nodes]
        if not all_nodes:
            return [], routes
        q = min(q, len(all_nodes))
        removed = list(self.rng.choice(all_nodes, size=q, replace=False))
        rem_set = set(removed)

        remaining_routes: List[List[int]] = []
        for r in routes:
            filt = [0] + [u for u in r[1:-1] if u not in rem_set] + [0]
            if len(filt) > 2:
                remaining_routes.append(filt)

        return removed, remaining_routes

    def _worst_removal(
        self,
        routes: List[List[int]],
        instance: CVRPInstance,
        q: int,
        allowed_removal_nodes: Optional[Set[int]] = None,
    ) -> Tuple[List[int], List[List[int]]]:
        dist = instance.dist_matrix
        savings: List[Tuple[float, int]] = []

        for r in routes:
            for i in range(1, len(r) - 1):
                u = r[i]
                if allowed_removal_nodes is not None and u not in allowed_removal_nodes:
                    continue
                prev_n, next_n = r[i - 1], r[i + 1]
                cost_gain = dist[prev_n, u] + dist[u, next_n] - dist[prev_n, next_n]
                savings.append((cost_gain, u))

        if not savings:
            return [], routes

        q = min(q, len(savings))
        savings.sort(key=lambda x: x[0], reverse=True)
        p = 3.0
        removed: List[int] = []
        while len(removed) < q and savings:
            idx = int(math.floor(len(savings) * (self.rng.rand() ** p)))
            removed.append(savings.pop(idx)[1])

        rem_set = set(removed)
        remaining_routes = [
            [0] + [u for u in r[1:-1] if u not in rem_set] + [0]
            for r in routes
        ]
        remaining_routes = [r for r in remaining_routes if len(r) > 2]
        return removed, remaining_routes

    def _shaw_removal(
        self,
        routes: List[List[int]],
        instance: CVRPInstance,
        q: int,
        allowed_removal_nodes: Optional[Set[int]] = None,
    ) -> Tuple[List[int], List[List[int]]]:
        dist = instance.dist_matrix
        demands = instance.demands
        max_dist = max(1.0, float(dist.max()))
        max_dem = max(1.0, float(demands.max()))

        all_nodes = [u for r in routes for u in r[1:-1] if allowed_removal_nodes is None or u in allowed_removal_nodes]
        if not all_nodes:
            return [], routes

        q = min(q, len(all_nodes))
        seed_node = self.rng.choice(all_nodes)
        removed = [seed_node]
        rem_set = {seed_node}

        p = 4.0
        while len(removed) < q and len(rem_set) < len(all_nodes):
            last = removed[-1]
            candidates = [u for u in all_nodes if u not in rem_set]
            # Compute relatedness: lower R means more related
            rel = [
                self.w_dist * (dist[last, u] / max_dist)
                + self.w_dem * (abs(demands[last] - demands[u]) / max_dem)
                for u in candidates
            ]
            sorted_cand = [c for _, c in sorted(zip(rel, candidates))]
            idx = int(math.floor(len(sorted_cand) * (self.rng.rand() ** p)))
            chosen = sorted_cand[idx]
            removed.append(chosen)
            rem_set.add(chosen)

        remaining_routes = [
            [0] + [u for u in r[1:-1] if u not in rem_set] + [0]
            for r in routes
        ]
        remaining_routes = [r for r in remaining_routes if len(r) > 2]
        return removed, remaining_routes

    def _greedy_insertion(
        self,
        routes: List[List[int]],
        removed_nodes: List[int],
        instance: CVRPInstance,
    ) -> List[List[int]]:
        routes = [r[:] for r in routes]
        dist = instance.dist_matrix
        demands = instance.demands
        capacity = instance.capacity
        route_loads = [sum(demands[node] for node in r[1:-1]) for r in routes]

        self.rng.shuffle(removed_nodes)

        for u in removed_nodes:
            d_u = demands[u]
            best_cost_inc = float("inf")
            best_r_idx = -1
            best_pos = -1

            # Try inserting into existing routes
            for r_idx, r in enumerate(routes):
                if route_loads[r_idx] + d_u > capacity:
                    continue

                for pos in range(1, len(r)):
                    p, n = r[pos - 1], r[pos]
                    inc = dist[p, u] + dist[u, n] - dist[p, n]
                    if inc < best_cost_inc:
                        best_cost_inc = inc
                        best_r_idx = r_idx
                        best_pos = pos

            if best_r_idx != -1:
                routes[best_r_idx].insert(best_pos, u)
                route_loads[best_r_idx] += d_u
            else:
                # Open a new single-customer route: [0, u, 0]
                routes.append([0, u, 0])
                route_loads.append(d_u)

        return routes

    def _regret2_insertion(
        self,
        routes: List[List[int]],
        removed_nodes: List[int],
        instance: CVRPInstance,
    ) -> List[List[int]]:
        routes = [r[:] for r in routes]
        dist = instance.dist_matrix
        demands = instance.demands
        capacity = instance.capacity
        route_loads = [sum(demands[node] for node in r[1:-1]) for r in routes]

        unassigned = set(removed_nodes)

        while unassigned:
            max_regret = -float("inf")
            best_node = -1
            best_insert_pos: Optional[Tuple[int, int]] = None

            for u in unassigned:
                d_u = demands[u]
                best_inc = float("inf")
                second_best_inc = float("inf")
                node_best_pos: Optional[Tuple[int, int]] = None

                for r_idx, r in enumerate(routes):
                    if route_loads[r_idx] + d_u > capacity:
                        continue

                    for pos in range(1, len(r)):
                        p, n = r[pos - 1], r[pos]
                        inc = dist[p, u] + dist[u, n] - dist[p, n]
                        if inc < best_inc:
                            second_best_inc = best_inc
                            best_inc = inc
                            node_best_pos = (r_idx, pos)
                        elif inc < second_best_inc:
                            second_best_inc = inc

                # Cost to open a new route [0, u, 0]
                new_route_inc = 2.0 * dist[0, u]
                if new_route_inc < best_inc:
                    second_best_inc = best_inc
                    best_inc = new_route_inc
                    node_best_pos = (len(routes), 1)
                elif new_route_inc < second_best_inc:
                    second_best_inc = new_route_inc

                regret = second_best_inc - best_inc
                if regret > max_regret:
                    max_regret = regret
                    best_node = u
                    best_insert_pos = node_best_pos

            # Perform the best regret insertion
            unassigned.remove(best_node)
            if best_insert_pos is not None:
                r_idx, pos = best_insert_pos
                if r_idx < len(routes):
                    routes[r_idx].insert(pos, best_node)
                    route_loads[r_idx] += demands[best_node]
                else:
                    routes.append([0, best_node, 0])
                    route_loads.append(demands[best_node])
            else:
                routes.append([0, best_node, 0])
                route_loads.append(demands[best_node])

        return routes
