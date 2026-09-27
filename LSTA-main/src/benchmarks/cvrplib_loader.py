import os
import urllib.request
from typing import Tuple, Dict, Any, Optional
import numpy as np
import vrplib
from fsta.types import CVRPInstance


INSTANCES_DIR = os.path.join(os.path.dirname(__file__), "instances")
BASE_URL = "https://raw.githubusercontent.com/PyVRP/VRPLIB/main/tests/data/cvrplib/Vrp-Set-X/X/"


def ensure_cvrplib_instance(name: str) -> Tuple[str, Optional[str]]:
    """
    Ensure the .vrp instance file and .sol solution file exist locally in benchmarks/instances.
    If not, download them from the official VRPLIB GitHub repository.
    """
    os.makedirs(INSTANCES_DIR, exist_ok=True)
    vrp_path = os.path.join(INSTANCES_DIR, f"{name}.vrp")
    sol_path = os.path.join(INSTANCES_DIR, f"{name}.sol")

    # Download .vrp if missing
    if not os.path.exists(vrp_path):
        url = BASE_URL + f"{name}.vrp"
        try:
            print(f"[*] Downloading {name}.vrp from GitHub...")
            content = urllib.request.urlopen(url, timeout=10).read()
            with open(vrp_path, "wb") as f:
                f.write(content)
        except Exception as e:
            raise FileNotFoundError(f"Could not download {name}.vrp: {e}")

    # Download .sol if missing (optional)
    if not os.path.exists(sol_path):
        url = BASE_URL + f"{name}.sol"
        try:
            content = urllib.request.urlopen(url, timeout=10).read()
            with open(sol_path, "wb") as f:
                f.write(content)
        except Exception:
            sol_path = None

    return vrp_path, sol_path if os.path.exists(sol_path) else None


def load_cvrplib_instance(name: str) -> Tuple[CVRPInstance, Optional[float], Optional[dict]]:
    """
    Load a CVRPLib instance by name (e.g. 'X-n101-k25', 'X-n1001-k43').
    
    Returns:
        instance: CVRPInstance with coordinates, demands, capacity, dist_matrix.
        bks_cost: Best Known Solution cost (from .sol if available).
        raw_data: Raw parsed dictionary from vrplib.
    """
    vrp_path, sol_path = ensure_cvrplib_instance(name)
    raw_data = vrplib.read_instance(vrp_path)

    raw_coords = raw_data["node_coord"].astype(np.float64)
    demands = raw_data["demand"].astype(np.float64)
    capacity = float(raw_data["capacity"])

    # Build CVRPInstance (original units)
    instance = CVRPInstance(
        coords=raw_coords,
        demands=demands,
        capacity=capacity,
    )

    bks_cost = None
    if sol_path and os.path.exists(sol_path):
        try:
            sol_data = vrplib.read_solution(sol_path)
            bks_cost = float(sol_data.get("cost", 0.0))
        except Exception:
            pass

    return instance, bks_cost, raw_data


def generate_synthetic_cvrp_instance(
    num_customers: int = 1000,
    capacity: float = 200.0,
    seed: int = 42,
) -> CVRPInstance:
    """
    Generate synthetic CVRP instance according to the exact paper specifications:
    - Uniform coordinates in [0, 1]^2, depot at [0.5, 0.5].
    - Uniform integer demands in [1, 9].
    - Specified vehicle capacity (e.g. C=200 for CVRP1k, C=300 for CVRP2k).
    """
    rng = np.random.RandomState(seed)
    depot_coord = np.array([[0.5, 0.5]], dtype=np.float64)
    cust_coords = rng.uniform(0.0, 1.0, size=(num_customers, 2)).astype(np.float64)
    coords = np.vstack([depot_coord, cust_coords])

    demands = np.zeros(num_customers + 1, dtype=np.float64)
    demands[1:] = rng.randint(1, 10, size=num_customers).astype(np.float64)

    return CVRPInstance(
        coords=coords,
        demands=demands,
        capacity=capacity,
    )
