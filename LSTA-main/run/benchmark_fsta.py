import time
from typing import List, Dict, Any
import numpy as np
from fsta import (
    CVRPInstance,
    partition_solution,
    aggregate_segments,
    recover_solution,
    compute_solution_cost,
    validate_cvrp_solution,
    longest_edge_selector,
    random_edge_selector,
    build_angular_sweep_solution,
    fsta_macro_local_search,
    original_cvrp_local_search,
)


def generate_benchmark_instance(
    num_customers: int,
    capacity: float,
    seed: int = 42,
) -> CVRPInstance:
    """Generate a standard CVRP benchmark instance according to Solomon / Zheng et al."""
    rng = np.random.RandomState(seed)
    # Depot at center [0.5, 0.5]
    depot = np.array([[0.5, 0.5]])
    # Customers uniformly distributed in [0, 1]^2
    cust_coords = rng.uniform(0.0, 1.0, size=(num_customers, 2))
    coords = np.vstack([depot, cust_coords])
    # Demands integer between 1 and 9
    demands = np.concatenate([[0.0], rng.randint(1, 10, size=num_customers).astype(np.float64)])
    return CVRPInstance(coords=coords, demands=demands, capacity=capacity)


def run_benchmark_for_scale(num_customers: int, capacity: float, seed: int = 42) -> Dict[str, Any]:
    print(f"\n{'='*75}")
    print(f"--- BENCHMARK SCALE: N = {num_customers} Customers | Capacity = {capacity} ---")
    print(f"{'='*75}")

    instance = generate_benchmark_instance(num_customers, capacity, seed=seed)

    # 1. Generate initial solution via Angular Sweep
    t0 = time.perf_counter()
    init_routes = build_angular_sweep_solution(instance)
    t_init = time.perf_counter() - t0
    c_init = compute_solution_cost(instance, init_routes)
    is_valid, msg = validate_cvrp_solution(instance, init_routes)
    assert is_valid, msg
    print(f"[Init Solution] Num Routes: {len(init_routes)} | Cost: {c_init:.3f} | Gen Time: {t_init*1000:.1f}ms")

    # 2. Baseline: Local Search directly on full original graph P
    t0 = time.perf_counter()
    base_routes = original_cvrp_local_search(instance, init_routes, max_passes=5)
    t_baseline = time.perf_counter() - t0
    c_baseline = compute_solution_cost(instance, base_routes)
    base_valid, _ = validate_cvrp_solution(instance, base_routes)
    base_gain = (c_init - c_baseline) / c_init * 100.0

    print(f"[Baseline Full Search] Cost: {c_baseline:.3f} (-{base_gain:.2f}%) | Time: {t_baseline*1000:.1f}ms | Feasible: {base_valid}")

    results = {
        "N": num_customers,
        "Capacity": capacity,
        "InitCost": c_init,
        "Baseline": {
            "Cost": c_baseline,
            "Gain%": base_gain,
            "Time_ms": t_baseline * 1000,
            "Feasible": base_valid,
        },
        "FSTA_Methods": {},
    }

    # 3. Test FSTA with different Edge Selectors
    selectors = [
        ("FSTA (Longest 25%)", lambda inst, r: longest_edge_selector(inst, r, ratio=0.25)),
        ("FSTA (Random 25%)", lambda inst, r: random_edge_selector(r, ratio=0.25, seed=seed)),
    ]

    for name, selector_fn in selectors:
        # Step A: Identify unstable edges
        t_sel_0 = time.perf_counter()
        unstable_edges = selector_fn(instance, init_routes)
        t_sel = time.perf_counter() - t_sel_0

        # Step B: Segment Partitioning
        t_part_0 = time.perf_counter()
        partitioned = partition_solution(instance, init_routes, unstable_edges)
        t_part = time.perf_counter() - t_part_0

        # Step C: Dual Hypernode Aggregation
        t_agg_0 = time.perf_counter()
        agg_prob = aggregate_segments(instance, partitioned, embed_internal_cost=True)
        t_agg = time.perf_counter() - t_agg_0

        num_agg_nodes = agg_prob.num_nodes
        compression_ratio = (1.0 - num_agg_nodes / num_customers) * 100.0

        # Step D: Re-optimization on Aggregated Graph \tilde{P}
        t_search_0 = time.perf_counter()
        agg_opt_routes = fsta_macro_local_search(agg_prob, max_passes=5)
        t_search = time.perf_counter() - t_search_0

        # Step E: Recovery on Original Graph P
        t_rec_0 = time.perf_counter()
        recovered_routes = recover_solution(agg_prob, agg_opt_routes)
        t_rec = time.perf_counter() - t_rec_0

        c_fsta = compute_solution_cost(instance, recovered_routes)
        fsta_valid, val_msg = validate_cvrp_solution(instance, recovered_routes)
        fsta_gain = (c_init - c_fsta) / c_init * 100.0

        total_fsta_time = (t_sel + t_part + t_agg + t_search + t_rec)
        speedup = t_baseline / total_fsta_time if total_fsta_time > 0 else 1.0

        print(
            f"[{name}] N_tilde: {num_agg_nodes} (Reduced {compression_ratio:.1f}%) | "
            f"Cost: {c_fsta:.3f} (-{fsta_gain:.2f}%) | "
            f"Search Time: {t_search*1000:.1f}ms | Total Time: {total_fsta_time*1000:.1f}ms | "
            f"Speedup: {speedup:.2f}x | Feasible: {fsta_valid}"
        )

        results["FSTA_Methods"][name] = {
            "N_tilde": num_agg_nodes,
            "Compression%": compression_ratio,
            "Cost": c_fsta,
            "Gain%": fsta_gain,
            "SearchTime_ms": t_search * 1000,
            "TotalTime_ms": total_fsta_time * 1000,
            "Speedup": speedup,
            "Feasible": fsta_valid,
        }

    return results


def main():
    print("=" * 75)
    print("      BENCHMARKING FSTA DECOMPOSITION & RECOVERY (WITHOUT L2SEG)      ")
    print("  Testing Compression, Monotonicity, Runtime Speedup & 100% Feasibility")
    print("=" * 75)

    scales = [
        (100, 50.0),
        (250, 80.0),
        (500, 150.0),
    ]

    all_results = []
    for n, cap in scales:
        res = run_benchmark_for_scale(num_customers=n, capacity=cap, seed=42)
        all_results.append(res)

    # Print Summary Table
    print("\n" + "=" * 95)
    print("                             SUMMARY BENCHMARK RESULTS")
    print("=" * 95)
    print(
        f"{'Scale (N)':<10} | {'Method':<20} | {'Nodes':<12} | {'Cost':<10} | {'Improv %':<10} | {'Time (ms)':<10} | {'Speedup':<8} | {'Feasible'}"
    )
    print("-" * 95)

    for r in all_results:
        n = r["N"]
        # Baseline
        base = r["Baseline"]
        print(
            f"N={n:<8} | {'Baseline Full':<20} | {n:<12} | {base['Cost']:<10.2f} | {base['Gain%']:<10.2f} | {base['Time_ms']:<10.1f} | {'1.00x':<8} | {base['Feasible']}"
        )

        # FSTA methods
        for m_name, m_res in r["FSTA_Methods"].items():
            print(
                f"N={n:<8} | {m_name:<20} | {m_res['N_tilde']} (-{m_res['Compression%']:.0f}%)    | {m_res['Cost']:<10.2f} | {m_res['Gain%']:<10.2f} | {m_res['TotalTime_ms']:<10.1f} | {m_res['Speedup']:<7.2f}x | {m_res['Feasible']}"
            )
        print("-" * 95)


if __name__ == "__main__":
    main()
