import os
import time
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from fsta.types import CVRPInstance

try:
    import pyvrp
    from pyvrp import Model
    from pyvrp.stop import MaxRuntime, MaxIterations
    HAS_PYVRP = True
except ImportError:
    HAS_PYVRP = False


class PyVRPSolver:
    """
    Hybrid Genetic Search (HGS - Vidal, 2022) via official PyVRP package.
    Used both as the state-of-the-art classical baseline and as a backbone solver.
    """

    def __init__(self, scale_factor: float = 1000.0, seed: int = 42):
        self.scale_factor = scale_factor
        self.seed = seed

    def is_available(self) -> bool:
        return HAS_PYVRP

    def solve(
        self,
        instance: CVRPInstance,
        time_limit: float = 5.0,
        max_iterations: Optional[int] = None,
        vrp_file_path: Optional[str] = None,
    ) -> Tuple[List[List[int]], float, float]:
        """
        Solve CVRP instance using PyVRP (HGS).
        
        Args:
            instance: CVRPInstance
            time_limit: Time limit in seconds
            max_iterations: Optional iteration limit
            vrp_file_path: If provided and exists, reads native TSPLIB .vrp file directly
            
        Returns:
            (best_routes, best_cost, solve_time_seconds)
        """
        if not HAS_PYVRP:
            raise RuntimeError("PyVRP is not installed. Run `pip install pyvrp`.")

        t0 = time.perf_counter()

        # 1. Native TSPLIB file solving (fastest C++ parser)
        if vrp_file_path and os.path.exists(vrp_file_path):
            pyvrp_data = pyvrp.read(vrp_file_path, round_func="round")
            stop_crit = MaxRuntime(time_limit) if max_iterations is None else MaxIterations(max_iterations)
            res = pyvrp.solve(pyvrp_data, stop=stop_crit, display=False)
            t_solve = time.perf_counter() - t0
            
            routes: List[List[int]] = []
            for r in res.best.routes():
                route_nodes = [act.idx for act in r if act.is_client()]
                if route_nodes:
                    routes.append([0] + route_nodes + [0])

            cost = float(res.cost())
            return routes, cost, t_solve

        # 2. Fast ProblemData formulation (instant C++ matrix pass-through)
        coords = instance.coords
        demands = instance.demands
        num_clients = instance.num_customers
        capacity = int(round(instance.capacity))

        locs = [
            pyvrp.Location(
                x=int(round(coords[i, 0] * self.scale_factor)),
                y=int(round(coords[i, 1] * self.scale_factor)),
            )
            for i in range(len(coords))
        ]
        depots = [pyvrp.Depot(location=0)]
        clients = [
            pyvrp.Client(location=i, delivery=[max(1, int(round(demands[i])))])
            for i in range(1, len(coords))
        ]
        vehicle_types = [
            pyvrp.VehicleType(
                num_available=max(num_clients, 50),
                capacity=[capacity],
            )
        ]

        dist_mat_scaled = (instance.dist_matrix * self.scale_factor).round().astype(int)
        # Ensure 0 diagonal
        np.fill_diagonal(dist_mat_scaled, 0)

        data = pyvrp.ProblemData(
            locations=locs,
            clients=clients,
            depots=depots,
            vehicle_types=vehicle_types,
            distance_matrices=[dist_mat_scaled],
            duration_matrices=[dist_mat_scaled],
        )

        stop_crit = MaxRuntime(time_limit) if max_iterations is None else MaxIterations(max_iterations)
        res = pyvrp.solve(data, stop=stop_crit, display=False)
        t_solve = time.perf_counter() - t0

        routes = []
        for r in res.best.routes():
            client_nodes = [act.idx + 1 for act in r if act.is_client()]
            if client_nodes:
                routes.append([0] + client_nodes + [0])

        real_cost = sum(
            sum(instance.dist_matrix[r[k], r[k + 1]] for k in range(len(r) - 1))
            for r in routes
        )

        return routes, real_cost, t_solve
