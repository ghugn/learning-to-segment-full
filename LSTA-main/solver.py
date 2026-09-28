#!/usr/bin/env python
"""
Main Entry Point for L2Seg + FSTA (Level 2: Topological Coarse-Graining).
Matches the laboratory structure of COPT-MT-main.

Usage:
    python solver.py                    # Run default end-to-end inference
    python solver.py --mode infer       # Run end-to-end inference
    python solver.py --mode benchmark   # Run official CVRPLib & Synthetic benchmark
    python solver.py --mode visualize   # Generate 6-stage Matplotlib process plot
    python solver.py --mode test        # Run unit tests
    python solver.py --mode train-nar   # Train NAR model
    python solver.py --mode train-ar    # Train AR model
"""

import os
import sys
import argparse

# Ensure root and src are on sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(root_dir, "src")
for p in [root_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    parser = argparse.ArgumentParser(
        description="L2Seg & FSTA: Learning to Segment for Vehicle Routing Problems (ICLR 2026)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="infer",
        choices=["infer", "benchmark", "table2", "multiscale", "visualize", "test", "train-nar", "train-ar"],
        help="Execution mode (default: infer)",
    )
    parser.add_argument("--customers", type=int, default=150, help="Number of customers for infer/visualize")
    parser.add_argument("--capacity", type=float, default=50.0, help="Vehicle capacity")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="assets/l2seg_fsta_process.png", help="Path for visualization output")
    parser.add_argument("--backbone", type=str, default="lns", choices=["lns"], help="Backbone solver for L2Seg (ICLR 2026: LNS)")
    parser.add_argument("--dataset", type=str, default=None, help="Path to .pkl dataset file (e.g. ../NDS/data/cvrp/vrp1000_test_seed1234.pkl)")
    parser.add_argument("--instance_idx", type=int, default=0, help="Instance index in dataset")
    parser.add_argument("--time_limit", type=float, default=15.0, help="Time limit in seconds for solving")
    parser.add_argument("--time_1k", type=int, default=150, help="Time limit for CVRP-1000")
    parser.add_argument("--time_2k", type=int, default=240, help="Time limit for CVRP-2000")
    parser.add_argument("--time_5k", type=int, default=300, help="Time limit for CVRP-5000")

    args, unknown = parser.parse_known_args()

    if args.mode == "infer":
        if args.dataset:
            import pickle
            import numpy as np
            from fsta.types import CVRPInstance
            from solvers.l2seg_iterative_solver import L2SegIterativeSolver
            from models.l2seg_model import L2SegModel

            print(f"\n[*] Loading CVRP instance {args.instance_idx} from dataset: {args.dataset}")
            with open(args.dataset, "rb") as f:
                data = pickle.load(f)
            elem = data[args.instance_idx]
            coords = np.vstack([[elem[0]], elem[1]])
            demands_full = np.array([0.0] + list(elem[2]), dtype=float)
            instance = CVRPInstance(coords=coords, demands=demands_full, capacity=float(elem[3]))

            print(f"[*] Problem Scale: N = {len(instance.coords) - 1} Customers | Capacity = {instance.capacity}")
            print(f"[*] Solver Backbone: L2Seg-SYN-{args.backbone.upper()} | Time Budget: {args.time_limit}s")

            model = None
            if os.path.exists("checkpoints/nar_model.pt") and os.path.exists("checkpoints/ar_model.pt"):
                try:
                    model = L2SegModel.load_pretrained("checkpoints/nar_model.pt", "checkpoints/ar_model.pt")
                    print("[+] Loaded pre-trained L2Seg neural model weights.")
                except Exception as e:
                    print(f"[-] Neural weight loading notice: {e}, using heuristic segmenter.")

            solver = L2SegIterativeSolver(model=model, backbone=args.backbone)
            res = solver.solve(instance, time_limit=args.time_limit)
            print("\n" + "=" * 80)
            print(f"SOLUTION SUMMARY: Cost = {res['best_cost']:.3f} | Total Time = {res['total_time']:.2f}s | Graph Compressed = {res['avg_compression_pct']:.1f}%")
            print("=" * 80)
        else:
            from run.infer import run_pipeline
            print("\n[*] Running L2Seg-SYN + FSTA End-to-End Pipeline...")
            run_pipeline(
                num_customers=args.customers,
                capacity=args.capacity,
                seed=args.seed,
                nar_path="checkpoints/nar_model.pt",
                ar_path="checkpoints/ar_model.pt",
            )

    elif args.mode == "benchmark":
        from benchmarks.benchmark_suite import main as benchmark_main
        print("\n[*] Launching Official Benchmark Suite...")
        sys.argv = [sys.argv[0]] + unknown
        benchmark_main()

    elif args.mode == "table2":
        from benchmarks.benchmark_table2 import main as table2_main
        print("\n[*] Launching Official Table 2 SOTA Benchmark Suite...")
        sys.argv = [sys.argv[0]] + unknown
        table2_main()

    elif args.mode == "multiscale":
        from benchmarks.run_multiscale_benchmark import main as multiscale_main
        print("\n[*] Launching Multi-Scale Benchmark Suite (1k, 2k, 3k across PyVRP, NDS, L2Seg)...")
        sys.argv = [sys.argv[0], f"--backbone={args.backbone}", f"--time_1k={args.time_1k}", f"--time_2k={args.time_2k}", f"--time_3k={args.time_3k}"] + unknown
        multiscale_main()

    elif args.mode == "visualize":
        from run.visualize import plot_l2seg_fsta_process
        from fsta.types import CVRPInstance
        from fsta.init_solution import build_angular_sweep_solution
        from fsta.partition import partition_solution
        from fsta.aggregation import aggregate_segments
        from fsta.recovery import recover_solution
        from fsta.local_search import fsta_macro_local_search
        from models.l2seg_model import load_trained_l2seg_model, predict_unstable_edges_l2seg_syn
        import numpy as np

        print(f"\n[*] Generating 6-Stage Process Plot (N={args.customers}, C={args.capacity})...")
        rng = np.random.RandomState(args.seed)
        depot_coord = np.array([[0.5, 0.5]])
        cust_coords = rng.uniform(0.0, 1.0, size=(args.customers, 2))
        coords = np.vstack([depot_coord, cust_coords])
        demands = np.concatenate([[0.0], rng.randint(1, 10, size=args.customers).astype(np.float64)])
        instance = CVRPInstance(coords=coords, demands=demands, capacity=args.capacity)

        init_routes = build_angular_sweep_solution(instance)
        model = load_trained_l2seg_model("checkpoints/nar_model.pt", "checkpoints/ar_model.pt", device="cpu")
        unstable = predict_unstable_edges_l2seg_syn(model, instance, init_routes, threshold=0.55, n_clusters=3)
        part = partition_solution(instance, init_routes, unstable)
        agg = aggregate_segments(instance, part, embed_internal_cost=True)
        opt = fsta_macro_local_search(agg, max_passes=5)
        rec = recover_solution(agg, opt)

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        plot_l2seg_fsta_process(
            instance=instance,
            init_routes=init_routes,
            unstable_edges=unstable,
            partitioned_routes=part,
            agg_problem=agg,
            opt_agg_routes=opt,
            recovered_routes=rec,
            output_path=args.output,
            dpi=300,
        )
        print(f"[+] Plot saved to: {args.output}")

    elif args.mode == "test":
        import pytest
        print("\n[*] Running Unit Test Suite...")
        exit_code = pytest.main(["tests/"])
        sys.exit(exit_code)

    elif args.mode == "train-nar":
        from run.train_nar import main as train_nar_main
        sys.argv = [sys.argv[0]] + unknown
        train_nar_main()

    elif args.mode == "train-ar":
        from run.train_ar import main as train_ar_main
        sys.argv = [sys.argv[0]] + unknown
        train_ar_main()


if __name__ == "__main__":
    main()
