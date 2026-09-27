import os
import sys
import argparse
import time
from typing import Set, Tuple, List, Dict, Any
import numpy as np
import torch

# Ensure project root and src/ are in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(root_dir, "src")
for p in [root_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fsta.types import CVRPInstance
from fsta.init_solution import build_angular_sweep_solution
from fsta.partition import partition_solution
from fsta.aggregation import aggregate_segments
from fsta.recovery import (
    recover_solution,
    compute_solution_cost,
    validate_cvrp_solution,
)
from fsta.local_search import (
    fsta_macro_local_search,
    original_cvrp_local_search,
)
from fsta.edge_selectors import longest_edge_selector
from features.subproblem import SubProblem, decompose_into_adjacent_subproblems
from models.l2seg_model import L2SegModel


def load_trained_l2seg_model(
    nar_path: str = "checkpoints/nar_model.pt",
    ar_path: str = "checkpoints/ar_model.pt",
    device: str = "cpu",
) -> L2SegModel:
    """Load trained weights into unified L2SegModel."""
    model = L2SegModel(
        node_in_dim=25,
        edge_in_dim=3,
        hidden_dim=128,
        num_tfm_layers=2,
        num_gnn_layers=2,
        num_ar_delete_layers=1,
        num_ar_insert_layers=4,
        dropout=0.0,
    ).to(device)

    # Load NAR weights
    if os.path.exists(nar_path):
        ckpt_nar = torch.load(nar_path, map_location=device, weights_only=False)
        if "encoder_state_dict" in ckpt_nar:
            model.encoder.load_state_dict(ckpt_nar["encoder_state_dict"])
        if "decoder_state_dict" in ckpt_nar:
            model.nar_decoder.load_state_dict(ckpt_nar["decoder_state_dict"])
        print(f"[+] Loaded NAR weights from: {nar_path}")

    # Load AR weights
    if os.path.exists(ar_path):
        ckpt_ar = torch.load(ar_path, map_location=device, weights_only=False)
        if "decoder_state_dict" in ckpt_ar:
            model.ar_decoder.load_state_dict(ckpt_ar["decoder_state_dict"])
        print(f"[+] Loaded AR weights from: {ar_path}")

    model.eval()
    return model


def predict_unstable_edges_l2seg_syn(
    model: L2SegModel,
    instance: CVRPInstance,
    routes: List[List[int]],
    threshold: float = 0.6,
    n_clusters: int = 3,
) -> Set[Tuple[int, int]]:
    """
    Run full L2Seg-SYN Synergized Prediction across all adjacent route pairs (Algorithm 3):
    1. Partition P into ~|R| subproblems P_TR.
    2. In each subproblem:
       - NAR detects candidate unstable nodes.
       - K-Means clusters them to find focal regions.
       - AR decodes alternating deletion/insertion moves from focal nodes.
    3. Aggregate all predicted unstable edges across subproblems.
    """
    subproblems = decompose_into_adjacent_subproblems(instance, routes)
    global_unstable_edges: Set[Tuple[int, int]] = set()

    for sub in subproblems:
        sub_edges = model.predict_subproblem_syn(
            subproblem=sub,
            threshold=threshold,
            n_clusters=n_clusters,
        )
        global_unstable_edges.update(sub_edges)

    return global_unstable_edges


def run_pipeline(
    num_customers: int = 150,
    capacity: float = 50.0,
    seed: int = 42,
    nar_path: str = "checkpoints/nar_model.pt",
    ar_path: str = "checkpoints/ar_model.pt",
    device: str = "cpu",
):
    print(f"\n{'='*80}")
    print(f"      L2Seg-SYN + FSTA END-TO-END INFERENCE & BENCHMARK PIPELINE      ")
    print(f"{'='*80}")
    print(f"Instance scale: N = {num_customers} Customers | Vehicle Capacity C = {capacity}")

    # 1. Create Synthetic CVRP instance
    rng = np.random.RandomState(seed)
    coords = np.vstack([[0.5, 0.5], rng.uniform(0.0, 1.0, size=(num_customers, 2))])
    demands = np.concatenate([[0.0], rng.randint(1, 9, size=num_customers).astype(np.float64)])
    instance = CVRPInstance(coords=coords, demands=demands, capacity=capacity)

    # 2. Initial Solution (Angular Sweep)
    t0 = time.perf_counter()
    init_routes = build_angular_sweep_solution(instance)
    t_init = time.perf_counter() - t0
    c_init = compute_solution_cost(instance, init_routes)
    is_valid, msg = validate_cvrp_solution(instance, init_routes)
    assert is_valid, msg
    print(f"[1. Initial Solution] Routes: {len(init_routes)} | Cost: {c_init:.3f} | Gen Time: {t_init*1000:.1f}ms")

    # 3. Baseline: Local Search directly on full uncompressed graph P
    t0 = time.perf_counter()
    base_routes = original_cvrp_local_search(instance, init_routes, max_passes=5)
    t_baseline = time.perf_counter() - t0
    c_baseline = compute_solution_cost(instance, base_routes)
    base_valid, _ = validate_cvrp_solution(instance, base_routes)
    base_gain = (c_init - c_baseline) / c_init * 100.0
    print(f"[2. Baseline Full Search] Cost: {c_baseline:.3f} (-{base_gain:.2f}%) | Time: {t_baseline*1000:.1f}ms | Feasible: {base_valid}")

    # 4. Non-learning Heuristic: FSTA with Longest 25% edges
    t0 = time.perf_counter()
    unstable_longest = longest_edge_selector(instance, init_routes, ratio=0.25)
    part_longest = partition_solution(instance, init_routes, unstable_longest)
    agg_longest = aggregate_segments(instance, part_longest, embed_internal_cost=True)
    opt_longest = fsta_macro_local_search(agg_longest, max_passes=5)
    rec_longest = recover_solution(agg_longest, opt_longest)
    t_longest = time.perf_counter() - t0
    c_longest = compute_solution_cost(instance, rec_longest)
    longest_valid, _ = validate_cvrp_solution(instance, rec_longest)
    longest_gain = (c_init - c_longest) / c_init * 100.0
    longest_speedup = t_baseline / t_longest if t_longest > 0 else 1.0
    print(
        f"[3. FSTA (Longest 25%)] Nodes: {agg_longest.num_nodes} (-{(1-agg_longest.num_nodes/num_customers)*100:.1f}%) | "
        f"Cost: {c_longest:.3f} (-{longest_gain:.2f}%) | Time: {t_longest*1000:.1f}ms | "
        f"Speedup: {longest_speedup:.2f}x | Feasible: {longest_valid}"
    )

    # 5. AI-Guided: FSTA with L2Seg-SYN (Learned Neural Model)
    print("\n[*] Loading AI models for L2Seg-SYN...")
    model = load_trained_l2seg_model(nar_path=nar_path, ar_path=ar_path, device=device)

    t0 = time.perf_counter()
    # Step A: Neural inference (NAR + K-Means + AR)
    t_ai_0 = time.perf_counter()
    unstable_ai = predict_unstable_edges_l2seg_syn(model, instance, init_routes, threshold=0.55, n_clusters=3)
    t_ai = time.perf_counter() - t_ai_0

    # Step B: FSTA Compression
    part_ai = partition_solution(instance, init_routes, unstable_ai)
    agg_ai = aggregate_segments(instance, part_ai, embed_internal_cost=True)

    # Step C: Re-optimization on Aggregated Graph
    t_search_0 = time.perf_counter()
    opt_ai = fsta_macro_local_search(agg_ai, max_passes=5)
    t_search = time.perf_counter() - t_search_0

    # Step D: Recovery on Original Graph
    rec_ai = recover_solution(agg_ai, opt_ai)
    t_total_ai = time.perf_counter() - t0

    c_ai = compute_solution_cost(instance, rec_ai)
    ai_valid, ai_msg = validate_cvrp_solution(instance, rec_ai)
    ai_gain = (c_init - c_ai) / c_init * 100.0
    ai_speedup = t_baseline / t_total_ai if t_total_ai > 0 else 1.0

    print(
        f"[4. L2Seg-SYN + FSTA (AI)] Nodes: {agg_ai.num_nodes} (-{(1-agg_ai.num_nodes/num_customers)*100:.1f}%) | "
        f"Cost: {c_ai:.3f} (-{ai_gain:.2f}%) | "
        f"AI Inference: {t_ai*1000:.1f}ms | Search Time: {t_search*1000:.1f}ms | Total: {t_total_ai*1000:.1f}ms | "
        f"Speedup: {ai_speedup:.2f}x | Feasible: {ai_valid}"
    )

    # Summary Table
    print("\n" + "=" * 95)
    print("                                   FINAL COMPARISON")
    print("=" * 95)
    print(f"{'Method':<25} | {'Graph Nodes':<14} | {'Travel Cost':<12} | {'Improvement %':<14} | {'Total Time':<12} | {'Speedup'}")
    print("-" * 95)
    print(f"{'1. Baseline Full Search':<25} | {num_customers:<14} | {c_baseline:<12.3f} | {base_gain:<14.2f} | {t_baseline*1000:<10.1f}ms | 1.00x")
    print(f"{'2. FSTA (Longest 25%)':<25} | {agg_longest.num_nodes:<14} | {c_longest:<12.3f} | {longest_gain:<14.2f} | {t_longest*1000:<10.1f}ms | {longest_speedup:.2f}x")
    print(f"{'3. L2Seg-SYN + FSTA (AI)':<25} | {agg_ai.num_nodes:<14} | {c_ai:<12.3f} | {ai_gain:<14.2f} | {t_total_ai*1000:<10.1f}ms | {ai_speedup:.2f}x")
    print("=" * 95)
    print(f"[Feasibility Validation] All constraints satisfied: {ai_valid} ({ai_msg})")


def main():
    parser = argparse.ArgumentParser(description="Run L2Seg-SYN End-to-End Pipeline")
    parser.add_argument("--customers", type=int, default=150, help="Number of customers")
    parser.add_argument("--capacity", type=float, default=50.0, help="Vehicle capacity")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--nar_path", type=str, default="checkpoints/nar_model.pt", help="Path to NAR checkpoint")
    parser.add_argument("--ar_path", type=str, default="checkpoints/ar_model.pt", help="Path to AR checkpoint")
    args = parser.parse_args()

    run_pipeline(
        num_customers=args.customers,
        capacity=args.capacity,
        seed=args.seed,
        nar_path=args.nar_path,
        ar_path=args.ar_path,
    )


if __name__ == "__main__":
    main()
