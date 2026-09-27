import os
import sys
import time
import argparse
import json
from typing import List, Dict, Any, Optional
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fsta.types import CVRPInstance
from fsta.init_solution import build_angular_sweep_solution
from fsta.partition import partition_solution
from fsta.aggregation import aggregate_segments
from fsta.recovery import recover_solution, compute_solution_cost, validate_cvrp_solution
from fsta.local_search import fsta_macro_local_search, original_cvrp_local_search
from fsta.edge_selectors import longest_edge_selector
from models.l2seg_model import load_trained_l2seg_model, predict_unstable_edges_l2seg_syn
from benchmarks.cvrplib_loader import load_cvrplib_instance, generate_synthetic_cvrp_instance

try:
    import pyvrp
    from pyvrp.stop import MaxRuntime
    HAS_PYVRP = True
except ImportError:
    HAS_PYVRP = False


def run_single_benchmark(
    name: str,
    instance: CVRPInstance,
    bks_cost: Optional[float] = None,
    hgs_ref_cost: Optional[float] = None,
    model: Optional[Any] = None,
    max_passes: int = 5,
    run_pyvrp_sec: float = 3.0,
    cvrp_file_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the complete benchmark protocol on a single instance:
    1. Initial Solution (Angular Sweep)
    2. Baseline Full Search (uncompressed graph)
    3. Heuristic FSTA (Longest 25% edges)
    4. L2Seg-SYN + FSTA (AI-guided)
    5. PyVRP (HGS reference) if available
    """
    N = instance.num_customers
    res: Dict[str, Any] = {
        "name": name,
        "N": N,
        "capacity": instance.capacity,
        "bks": bks_cost,
        "hgs_paper": hgs_ref_cost,
    }

    # 1. Initial Solution
    t0 = time.perf_counter()
    init_routes = build_angular_sweep_solution(instance)
    t_init = time.perf_counter() - t0
    c_init = compute_solution_cost(instance, init_routes)
    is_valid, _ = validate_cvrp_solution(instance, init_routes)

    res["initial"] = {
        "cost": c_init,
        "time": t_init,
        "routes": len(init_routes),
        "valid": is_valid,
    }

    # 2. Baseline Full Search (Direct local search on full graph)
    t0 = time.perf_counter()
    base_routes = original_cvrp_local_search(instance, init_routes, max_passes=max_passes)
    t_base = time.perf_counter() - t0
    c_base = compute_solution_cost(instance, base_routes)
    b_valid, _ = validate_cvrp_solution(instance, base_routes)

    res["baseline_full"] = {
        "cost": c_base,
        "gain": (c_init - c_base) / c_init * 100.0,
        "time": t_base,
        "valid": b_valid,
    }

    # 3. Heuristic FSTA (Longest 25% edges)
    t0 = time.perf_counter()
    unstable_longest = longest_edge_selector(instance, init_routes, ratio=0.25)
    part_longest = partition_solution(instance, init_routes, unstable_longest)
    agg_longest = aggregate_segments(instance, part_longest, embed_internal_cost=True)
    opt_longest = fsta_macro_local_search(agg_longest, max_passes=max_passes)
    rec_longest = recover_solution(agg_longest, opt_longest)
    t_longest = time.perf_counter() - t0
    c_longest = compute_solution_cost(instance, rec_longest)
    l_valid, _ = validate_cvrp_solution(instance, rec_longest)

    res["fsta_longest"] = {
        "nodes": agg_longest.num_nodes,
        "reduction": (1.0 - agg_longest.num_nodes / N) * 100.0,
        "cost": c_longest,
        "gain": (c_init - c_longest) / c_init * 100.0,
        "time": t_longest,
        "speedup": t_base / t_longest if t_longest > 0 else 1.0,
        "valid": l_valid,
    }

    # 4. AI L2Seg-SYN + FSTA
    if model is not None:
        t0 = time.perf_counter()
        t_ai_0 = time.perf_counter()
        unstable_ai = predict_unstable_edges_l2seg_syn(model, instance, init_routes, threshold=0.55, n_clusters=3)
        t_ai_inf = time.perf_counter() - t_ai_0

        part_ai = partition_solution(instance, init_routes, unstable_ai)
        agg_ai = aggregate_segments(instance, part_ai, embed_internal_cost=True)
        opt_ai = fsta_macro_local_search(agg_ai, max_passes=max_passes)
        rec_ai = recover_solution(agg_ai, opt_ai)
        t_ai_total = time.perf_counter() - t0
        c_ai = compute_solution_cost(instance, rec_ai)
        ai_valid, _ = validate_cvrp_solution(instance, rec_ai)

        res["l2seg_syn"] = {
            "unstable_count": len(unstable_ai),
            "nodes": agg_ai.num_nodes,
            "reduction": (1.0 - agg_ai.num_nodes / N) * 100.0,
            "cost": c_ai,
            "gain": (c_init - c_ai) / c_init * 100.0,
            "inf_time": t_ai_inf,
            "time": t_ai_total,
            "speedup": t_base / t_ai_total if t_ai_total > 0 else 1.0,
            "valid": ai_valid,
        }

    # 5. PyVRP (HGS Vidal 2022) reference
    if HAS_PYVRP and cvrp_file_path and os.path.exists(cvrp_file_path):
        try:
            t0 = time.perf_counter()
            pyvrp_data = pyvrp.read(cvrp_file_path, round_func="round")
            pyvrp_res = pyvrp.solve(pyvrp_data, stop=MaxRuntime(run_pyvrp_sec), display=False)
            t_pyvrp = time.perf_counter() - t0
            c_pyvrp = float(pyvrp_res.cost())

            res["pyvrp_hgs"] = {
                "cost": c_pyvrp,
                "time": t_pyvrp,
                "feasible": pyvrp_res.is_feasible(),
            }
        except Exception as e:
            res["pyvrp_hgs"] = {"error": str(e)}

    # Compute Gaps to Golden Standard (BKS or Paper HGS)
    ref_target = bks_cost or hgs_ref_cost
    if ref_target:
        res["initial"]["gap_bks"] = (res["initial"]["cost"] - ref_target) / ref_target * 100.0
        res["baseline_full"]["gap_bks"] = (res["baseline_full"]["cost"] - ref_target) / ref_target * 100.0
        res["fsta_longest"]["gap_bks"] = (res["fsta_longest"]["cost"] - ref_target) / ref_target * 100.0
        if "l2seg_syn" in res:
            res["l2seg_syn"]["gap_bks"] = (res["l2seg_syn"]["cost"] - ref_target) / ref_target * 100.0
        if "pyvrp_hgs" in res and "cost" in res["pyvrp_hgs"]:
            res["pyvrp_hgs"]["gap_bks"] = (res["pyvrp_hgs"]["cost"] - ref_target) / ref_target * 100.0

    return res


def print_benchmark_table(results: List[Dict[str, Any]]):
    """Print a clean Markdown-compatible benchmark comparison table."""
    print("\n" + "=" * 115)
    print("                      BENCHMARK COMPARISON TABLE: L2Seg + FSTA vs BASELINES & BKS")
    print("=" * 115)
    header = (
        f"{'Instance':<14} | {'N':<5} | {'BKS / HGS':<10} | {'Initial':<12} | "
        f"{'Full Search':<12} | {'FSTA (Longest)':<15} | {'L2Seg-SYN (AI)':<15} | {'PyVRP (HGS)':<12}"
    )
    print(header)
    print("-" * 115)

    for r in results:
        name = r["name"]
        n = r["N"]
        bks_str = f"{r['bks']:.0f}" if r["bks"] is not None else (f"{r['hgs_paper']:.2f}" if r.get("hgs_paper") else "N/A")
        
        c_init = f"{r['initial']['cost']:.1f}"
        c_base = f"{r['baseline_full']['cost']:.1f}"
        
        # Longest
        c_long = f"{r['fsta_longest']['cost']:.1f}"
        if "gap_bks" in r["fsta_longest"]:
            c_long += f" (+{r['fsta_longest']['gap_bks']:.1f}%)"
            
        # AI
        c_ai = "N/A"
        if "l2seg_syn" in r:
            c_ai = f"{r['l2seg_syn']['cost']:.1f}"
            if "gap_bks" in r["l2seg_syn"]:
                c_ai += f" (+{r['l2seg_syn']['gap_bks']:.1f}%)"

        # PyVRP
        c_pyvrp = "N/A"
        if "pyvrp_hgs" in r and "cost" in r["pyvrp_hgs"]:
            c_pyvrp = f"{r['pyvrp_hgs']['cost']:.1f}"
            if "gap_bks" in r["pyvrp_hgs"]:
                c_pyvrp += f" (+{r['pyvrp_hgs']['gap_bks']:.1f}%)"

        print(f"{name:<14} | {n:<5} | {bks_str:<10} | {c_init:<12} | {c_base:<12} | {c_long:<15} | {c_ai:<15} | {c_pyvrp:<12}")

    print("=" * 115)


def print_compression_and_speedup_table(results: List[Dict[str, Any]]):
    """Print graph compression ratio and speedup details."""
    print("\n" + "=" * 105)
    print("                          GRAPH COMPRESSION & RUNTIME ANALYSIS")
    print("=" * 105)
    header = (
        f"{'Instance':<14} | {'Orig N':<8} | {'Compressed N':<14} | "
        f"{'Compression %':<15} | {'AI Time':<10} | {'Base Time':<10} | {'Speedup':<8}"
    )
    print(header)
    print("-" * 105)

    for r in results:
        if "l2seg_syn" not in r:
            continue
        name = r["name"]
        orig_n = r["N"]
        comp_n = r["l2seg_syn"]["nodes"]
        comp_pct = r["l2seg_syn"]["reduction"]
        t_ai = f"{r['l2seg_syn']['time']*1000:.1f}ms"
        t_base = f"{r['baseline_full']['time']*1000:.1f}ms"
        speedup = f"{r['l2seg_syn']['speedup']:.2f}x"

        print(f"{name:<14} | {orig_n:<8} | {comp_n:<14} | {comp_pct:<15.1f}% | {t_ai:<10} | {t_base:<10} | {speedup:<8}")

    print("=" * 105)


def main():
    parser = argparse.ArgumentParser(description="Official Benchmark Suite for L2Seg + FSTA")
    parser.add_argument("--nar_path", type=str, default="checkpoints/nar_model.pt")
    parser.add_argument("--ar_path", type=str, default="checkpoints/ar_model.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_json", type=str, default="benchmarks/benchmark_results.json")
    args = parser.parse_args()

    print("\n[*] Initializing L2Seg Benchmark Suite...")
    model = load_trained_l2seg_model(nar_path=args.nar_path, ar_path=args.ar_path, device=args.device)

    results: List[Dict[str, Any]] = []

    # 1. CVRPLib Set-X instances (Table 11 in paper)
    cvrplib_instances = ["X-n101-k25", "X-n153-k22", "X-n280-k17", "X-n502-k39", "X-n1001-k43"]
    print("\n" + "#" * 60)
    print("  PHASE 1: CVRPLib Benchmark (Set-X, matching Table 11)")
    print("#" * 60)

    for inst_name in cvrplib_instances:
        print(f"\n--> Running on CVRPLib instance: {inst_name}...")
        inst, bks, _ = load_cvrplib_instance(inst_name)
        vrp_file = os.path.join(os.path.dirname(__file__), "instances", f"{inst_name}.vrp")

        res = run_single_benchmark(
            name=inst_name,
            instance=inst,
            bks_cost=bks,
            model=model,
            max_passes=5,
            run_pyvrp_sec=3.0,
            cvrp_file_path=vrp_file,
        )
        results.append(res)
        print(f"    [+] Initial Cost: {res['initial']['cost']:.1f}")
        print(f"    [+] Baseline Full Cost: {res['baseline_full']['cost']:.1f}")
        print(f"    [+] FSTA Longest Cost: {res['fsta_longest']['cost']:.1f} (Nodes: {res['fsta_longest']['nodes']})")
        if "l2seg_syn" in res:
            print(f"    [+] L2Seg-SYN AI Cost: {res['l2seg_syn']['cost']:.1f} (Nodes: {res['l2seg_syn']['nodes']})")
        if "pyvrp_hgs" in res and "cost" in res["pyvrp_hgs"]:
            print(f"    [+] PyVRP (HGS) Cost: {res['pyvrp_hgs']['cost']:.1f} (Gap: {res['pyvrp_hgs'].get('gap_bks', 0):.2f}%)")
        if bks:
            print(f"    [*] BKS Target: {bks:.1f}")

    # 2. Synthetic CVRP instances (Table 2 in paper)
    print("\n" + "#" * 60)
    print("  PHASE 2: Synthetic CVRP (CVRP1k & CVRP2k, matching Table 2)")
    print("#" * 60)

    synthetic_configs = [
        {"name": "CVRP1k-Syn", "N": 1000, "C": 200.0, "hgs_paper": 41.20, "l2seg_lns_paper": 41.36},
        {"name": "CVRP2k-Syn", "N": 2000, "C": 300.0, "hgs_paper": 57.20, "l2seg_lns_paper": 56.08},
    ]

    for cfg in synthetic_configs:
        print(f"\n--> Generating and running on Synthetic instance: {cfg['name']} (N={cfg['N']}, C={cfg['C']})...")
        syn_inst = generate_synthetic_cvrp_instance(num_customers=cfg["N"], capacity=cfg["C"], seed=42)
        res = run_single_benchmark(
            name=cfg["name"],
            instance=syn_inst,
            hgs_ref_cost=cfg["hgs_paper"],
            model=model,
            max_passes=5,
            run_pyvrp_sec=0.0,
            cvrp_file_path=None,
        )
        res["l2seg_lns_paper"] = cfg["l2seg_lns_paper"]
        results.append(res)
        print(f"    [+] Initial Cost: {res['initial']['cost']:.2f}")
        print(f"    [+] FSTA Longest Cost: {res['fsta_longest']['cost']:.2f} (Nodes: {res['fsta_longest']['nodes']})")
        if "l2seg_syn" in res:
            print(f"    [+] L2Seg-SYN AI Cost: {res['l2seg_syn']['cost']:.2f} (Nodes: {res['l2seg_syn']['nodes']})")
        print(f"    [*] Paper HGS Reference: {cfg['hgs_paper']:.2f} | Paper L2Seg-SYN-LNS: {cfg['l2seg_lns_paper']:.2f}")

    # Print summary tables
    print_benchmark_table(results)
    print_compression_and_speedup_table(results)

    # Save results to JSON
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Full benchmark results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
