import pytest
import numpy as np
import os
import torch

from fsta.types import CVRPInstance
from fsta.init_solution import build_angular_sweep_solution
from fsta.recovery import validate_cvrp_solution
from solvers.lns import LNSSolver
from solvers.pyvrp_solver import PyVRPSolver
from solvers.l2seg_iterative_solver import L2SegIterativeSolver
from benchmarks.benchmark_suite import load_trained_l2seg_model
from benchmarks.benchmark_table2 import (
    TABLE2_OFFICIAL_LITERATURE,
    generate_benchmark_instance,
    format_table2_markdown,
)


def create_toy_instance(num_customers: int = 15, seed: int = 42) -> CVRPInstance:
    rng = np.random.RandomState(seed)
    coords = np.vstack([[0.5, 0.5], rng.uniform(0.0, 1.0, size=(num_customers, 2))])
    demands = np.concatenate([[0.0], rng.randint(1, 6, size=num_customers).astype(np.float64)])
    return CVRPInstance(coords=coords, demands=demands, capacity=20.0)


def test_lns_solver_feasibility_and_monotonicity():
    inst = create_toy_instance(num_customers=20)
    init_routes = build_angular_sweep_solution(inst)
    init_cost = sum(
        sum(inst.dist_matrix[r[i], r[i + 1]] for i in range(len(r) - 1))
        for r in init_routes
    )

    lns = LNSSolver(seed=42)
    best_routes, best_cost, iters = lns.solve(inst, init_routes, time_limit=1.0, max_iterations=50)

    assert best_cost <= init_cost + 1e-6
    valid, msg = validate_cvrp_solution(inst, best_routes)
    assert valid, f"LNS solution invalid: {msg}"


def test_pyvrp_solver_execution():
    solver = PyVRPSolver(seed=42)
    if not solver.is_available():
        pytest.skip("PyVRP is not available")

    inst = create_toy_instance(num_customers=15)
    routes, cost, t = solver.solve(inst, time_limit=1.0)

    assert len(routes) > 0
    assert cost > 0.0
    valid, msg = validate_cvrp_solution(inst, routes)
    assert valid, f"PyVRP solution invalid: {msg}"


def test_l2seg_iterative_solver_with_trained_model():
    nar_path = "checkpoints/nar_model.pt"
    ar_path = "checkpoints/ar_model.pt"
    if not (os.path.exists(nar_path) and os.path.exists(ar_path)):
        pytest.skip("Checkpoints not present")

    model = load_trained_l2seg_model(nar_path, ar_path, device="cpu")
    inst = create_toy_instance(num_customers=20)
    init_routes = build_angular_sweep_solution(inst)

    solver = L2SegIterativeSolver(model=model, backbone="lns", device="cpu")
    res = solver.solve(inst, init_routes, time_limit=2.0, max_iterations=2)

    assert res["is_valid"]
    assert res["best_cost"] <= res["initial_cost"] + 1e-6
    assert res["avg_compression_pct"] >= 0.0


def test_table2_structure_and_export():
    assert "cvrp" in TABLE2_OFFICIAL_LITERATURE
    assert len(TABLE2_OFFICIAL_LITERATURE["cvrp"]) >= 15

    # Check key methods exist
    methods = [r["method"] for r in TABLE2_OFFICIAL_LITERATURE["cvrp"]]
    assert any("HGS" in m for m in methods)
    assert any("NDS" in m for m in methods)
    assert any("L2D" in m for m in methods)
    assert any("L2Seg-SYN-L2D" in m for m in methods)

    md = format_table2_markdown()
    assert "# Table 2" in md
    assert "NDS (Hottung et al., 2025)" in md
