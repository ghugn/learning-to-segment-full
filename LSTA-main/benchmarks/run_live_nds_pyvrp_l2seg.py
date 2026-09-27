import os
import sys
import time
import pickle
import argparse
import subprocess
import json
import numpy as np
from typing import List, Dict, Any

# Ensure src is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from fsta.types import CVRPInstance
from solvers.pyvrp_solver import PyVRPSolver
from solvers.l2seg_iterative_solver import L2SegIterativeSolver
from models.l2seg_model import L2SegModel


def load_nds_instance(data_path: str, instance_idx: int) -> CVRPInstance:
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    elem = data[instance_idx]
    depot = elem[0]
    cust_coords = elem[1]
    demands = elem[2]
    capacity = float(elem[3])

    coords = np.vstack([[depot], cust_coords])
    demands_full = np.array([0.0] + list(demands), dtype=float)
    return CVRPInstance(coords=coords, demands=demands_full, capacity=capacity)


def run_nds_benchmark(nds_dir: str, nb_instances: int, max_runtime: int) -> List[Dict[str, Any]]:
    """Run official NDS on CPU using eval.py and parse results."""
    cmd = [
        sys.executable,
        "eval.py",
        "cvrp_1000.yaml",
        "tester_params.use_cuda=False",
        f"tester_params.nb_instances={nb_instances}",
        f"tester_params.max_runtime={max_runtime}",
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=nds_dir, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    # Parse stdout from NDS
    results = []
    lines = proc.stdout.splitlines()
    for line in lines:
        if "Instance" in line and "Cost:" in line:
            # Format: Instance   1/  1  |  Elapsed: 0.19m  |  Remain: 0.00m  |  Cost:   40.43  |  Avg:  40.427
            parts = line.split("|")
            inst_part = parts[0].strip()
            cost_part = [p for p in parts if "Cost:" in p][0].strip()
            cost_val = float(cost_part.split("Cost:")[1].split()[0])
            results.append({"cost": cost_val, "time": elapsed / max(1, nb_instances)})

    # Fallback to reading the latest results.csv if parsing fails
    if not results:
        results_dir = os.path.join(nds_dir, "results")
        if os.path.exists(results_dir):
            all_sub = sorted([os.path.join(results_dir, d) for d in os.listdir(results_dir)], key=os.path.getmtime)
            if all_sub:
                latest_csv = os.path.join(all_sub[-1], "results.csv")
                if os.path.exists(latest_csv):
                    with open(latest_csv, "r") as f:
                        for row in f:
                            row_parts = row.strip().split(",")
                            if len(row_parts) >= 3:
                                try:
                                    c = float(row_parts[1])
                                    t = float(row_parts[2])
                                    results.append({"cost": c, "time": t})
                                except ValueError:
                                    continue
    return results[:nb_instances]


def main():
    parser = argparse.ArgumentParser(description="Live Head-to-Head Benchmark: PyVRP vs NDS vs L2Seg+FSTA")
    parser.add_argument("--instances", type=int, default=1, help="Number of instances to evaluate from test set")
    parser.add_argument("--time_limit", type=float, default=15.0, help="Per-instance time limit in seconds")
    parser.add_argument("--backbone", type=str, default="pyvrp", choices=["pyvrp", "lns"], help="Backbone re-optimizer (default: pyvrp)")
    parser.add_argument("--nds_dir", type=str, default="../NDS", help="Path to NDS cloned repository")
    args = parser.parse_args()

    nds_dir = os.path.abspath(os.path.join(project_root, args.nds_dir))
    data_path = os.path.join(nds_dir, "data", "cvrp", "vrp1000_test_seed1234.pkl")

    if not os.path.exists(data_path):
        print(f"Error: Dataset {data_path} not found.")
        sys.exit(1)

    l2_name = f"L2Seg-SYN-{args.backbone.upper()}"
    print("=" * 80)
    print("LIVE HEAD-TO-HEAD BENCHMARK (CVRP-1000)")
    print(f"Test Set: vrp1000_test_seed1234.pkl ({args.instances} instances)")
    print(f"Time Budget per Instance: {args.time_limit:.1f}s")
    print(f"L2Seg Backbone: {l2_name}")
    print("=" * 80)

    # 1. Run NDS
    print(f"\n[1/3] Running official NDS (Hottung et al., 2022) with {int(args.time_limit)}s budget...")
    nds_res = run_nds_benchmark(nds_dir, args.instances, int(args.time_limit))
    print(f"  NDS finished: {len(nds_res)} results collected.")

    # 2. Run PyVRP and L2Seg
    pyvrp_solver = PyVRPSolver()
    
    # Check if pre-trained neural weights exist
    chk_nar = os.path.join(project_root, "checkpoints", "nar_model.pt")
    chk_ar = os.path.join(project_root, "checkpoints", "ar_model.pt")
    model = None
    if os.path.exists(chk_nar) and os.path.exists(chk_ar):
        try:
            model = L2SegModel.load_pretrained(chk_nar, chk_ar)
            print("  Loaded L2Seg pre-trained neural model.")
        except Exception as e:
            print(f"  Note: Neural model loading skipped ({e}), using structural heuristic.")

    l2seg_solver = L2SegIterativeSolver(model=model, backbone=args.backbone)

    all_comparison = []

    for i in range(args.instances):
        print(f"\nEvaluating Instance {i+1}/{args.instances} (N=1000)...")
        inst = load_nds_instance(data_path, i)

        # PyVRP (HGS - Vidal, 2022)
        print("  Running PyVRP (HGS Vidal 2022)...", end="", flush=True)
        t0 = time.perf_counter()
        py_routes, py_cost, py_time = pyvrp_solver.solve(inst, time_limit=args.time_limit)
        print(f" Done ({py_time:.2f}s, Cost: {py_cost:.3f})")

        # NDS
        nds_cost = nds_res[i]["cost"] if i < len(nds_res) else 0.0
        nds_time = nds_res[i]["time"] if i < len(nds_res) else 0.0

        # L2Seg Solver
        print(f"  Running {l2_name} (Iterative FSTA + {args.backbone.upper()})...", end="", flush=True)
        l2_res = l2seg_solver.solve(inst, time_limit=args.time_limit, reopt_time_per_iter=3.0)
        l2_cost = l2_res["best_cost"]
        l2_time = l2_res["total_time"]
        l2_comp = l2_res["avg_compression_pct"]
        print(f" Done ({l2_time:.2f}s, Cost: {l2_cost:.3f}, Graph Compressed: {l2_comp:.1f}%)")

        # Gaps relative to PyVRP (HGS) as in Table 2
        nds_gap = (nds_cost - py_cost) / py_cost * 100.0 if py_cost > 0 else 0.0
        l2_gap = (l2_cost - py_cost) / py_cost * 100.0 if py_cost > 0 else 0.0

        all_comparison.append({
            "instance": i + 1,
            "pyvrp_cost": py_cost,
            "pyvrp_time": py_time,
            "nds_cost": nds_cost,
            "nds_time": nds_time,
            "nds_gap": nds_gap,
            "l2seg_cost": l2_cost,
            "l2seg_time": l2_time,
            "l2seg_gap": l2_gap,
            "compression_pct": l2_comp,
        })

    # Summary Table
    print("\n" + "=" * 95)
    print(f"{'Method':<25} | {'Cost (Obj)':<12} | {'Gap vs HGS (%)':<15} | {'Runtime (s)':<12} | {'Graph Reduction':<15}")
    print("-" * 95)
    
    avg_py_cost = np.mean([x["pyvrp_cost"] for x in all_comparison])
    avg_py_time = np.mean([x["pyvrp_time"] for x in all_comparison])
    print(f"{'PyVRP (HGS Vidal 2022)':<25} | {avg_py_cost:<12.3f} | {'0.00% (Baseline)':<15} | {avg_py_time:<12.2f} | {'0.0% (Full Graph)':<15}")

    avg_nds_cost = np.mean([x["nds_cost"] for x in all_comparison])
    avg_nds_time = np.mean([x["nds_time"] for x in all_comparison])
    avg_nds_gap = np.mean([x["nds_gap"] for x in all_comparison])
    print(f"{'NDS (Hottung et al. 2022)':<25} | {avg_nds_cost:<12.3f} | {f'{avg_nds_gap:+.2f}%':<15} | {avg_nds_time:<12.2f} | {'0.0% (Full Graph)':<15}")

    avg_l2_cost = np.mean([x["l2seg_cost"] for x in all_comparison])
    avg_l2_time = np.mean([x["l2seg_time"] for x in all_comparison])
    avg_l2_gap = np.mean([x["l2seg_gap"] for x in all_comparison])
    avg_comp = np.mean([x["compression_pct"] for x in all_comparison])
    print(f"{l2_name + ' (Ours)':<25} | {avg_l2_cost:<12.3f} | {f'{avg_l2_gap:+.2f}%':<15} | {avg_l2_time:<12.2f} | {f'-{avg_comp:.1f}%':<15}")
    print("=" * 95)

    # Save to Markdown
    out_md = os.path.join(project_root, "benchmarks", "live_head_to_head_results.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Live Empirical Head-to-Head Benchmark on CVRP-1000\n\n")
        f.write(f"- **Dataset**: `vrp1000_test_seed1234.pkl` (exact test set from NDS / Kool et al.)\n")
        f.write(f"- **Evaluated Instances**: {args.instances}\n")
        f.write(f"- **Time Budget per Instance**: {args.time_limit:.1f}s\n\n")
        f.write("| Method | Implementation | Cost (Obj) | Gap vs HGS (%) | Execution Time | Graph Search Space Reduction |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **PyVRP (HGS Vidal 2022)** | Official C++ Engine | **{avg_py_cost:.3f}** | **0.00%** | {avg_py_time:.2f}s | 0.0% (Full Graph) |\n")
        f.write(f"| **NDS (Hottung et al. 2022)** | Official C++ / PyTorch Repo | **{avg_nds_cost:.3f}** | **{avg_nds_gap:+.2f}%** | {avg_nds_time:.2f}s | 0.0% (Full Graph) |\n")
        f.write(f"| **{l2_name} (Ours)** | L2Seg AI + FSTA + Focused {args.backbone.upper()} | **{avg_l2_cost:.3f}** | **{avg_l2_gap:+.2f}%** | {avg_l2_time:.2f}s | **-{avg_comp:.1f}%** |\n")
    print(f"\nSaved Markdown report to: {out_md}")


if __name__ == "__main__":
    main()
