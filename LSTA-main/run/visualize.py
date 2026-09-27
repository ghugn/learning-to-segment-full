import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Set, Tuple, List

# Add project root and src to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(root_dir, "src")
for p in [root_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fsta.types import CVRPInstance, Segment, AggregatedProblem
from fsta.init_solution import build_angular_sweep_solution
from fsta.partition import partition_solution
from fsta.aggregation import aggregate_segments
from fsta.recovery import recover_solution, compute_solution_cost, validate_cvrp_solution
from fsta.local_search import fsta_macro_local_search
from models.l2seg_model import load_trained_l2seg_model, predict_unstable_edges_l2seg_syn


def plot_l2seg_fsta_process(
    instance: CVRPInstance,
    init_routes: List[List[int]],
    unstable_edges: Set[Tuple[int, int]],
    partitioned_routes: List[List[Segment]],
    agg_problem: AggregatedProblem,
    opt_agg_routes: List[List[int]],
    recovered_routes: List[List[int]],
    output_path: str = "l2seg_fsta_process.png",
    dpi: int = 300,
):
    """
    Export the 6-stage visualization of L2Seg + FSTA matching Figure 1 from the ICLR 2026 paper:
    (a) The original CVRP instance
    (b) Unstable edges detection (highlighting unstable edges in red)
    (c) Segment partitioning (unstable edges removed, showing intact segments)
    (d) Hypernode aggregation (contracted graph P_tilde with dual hypernodes)
    (e) Re-opt. w/ backbone solvers (reconnecting routes on contracted graph)
    (f) Solution recovery (full expanded solution on original graph)
    """
    coords = instance.coords
    N = instance.num_customers
    depot = coords[0]

    # Normalize unstable edges set for bidirectional lookup
    norm_unstable = set()
    for u, v in unstable_edges:
        norm_unstable.add((min(u, v), max(u, v)))

    # Set up matplotlib style
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.5), constrained_layout=True)
    fig.patch.set_facecolor("white")

    def format_ax(ax, title: str):
        ax.set_aspect("equal")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(labelsize=8, direction="in")
        ax.set_title(title, y=-0.16, fontsize=11, fontweight="medium")

    # -------------------------------------------------------------
    # (a) The original CVRP instance
    # -------------------------------------------------------------
    ax_a = axes[0, 0]
    # Draw customer points
    ax_a.scatter(coords[1:, 0], coords[1:, 1], c="black", s=5, zorder=3)
    # Draw initial routes
    for r in init_routes:
        r_coords = coords[r]
        ax_a.plot(r_coords[:, 0], r_coords[:, 1], c="black", lw=0.5, alpha=0.85, zorder=2)
    # Draw depot
    ax_a.scatter(depot[0], depot[1], c="#ff9800", marker="s", s=36, edgecolor="black", lw=0.6, zorder=5)
    format_ax(ax_a, f"(a) The original CVRP instance (N={N})")

    # -------------------------------------------------------------
    # (b) Unstable edges detection
    # -------------------------------------------------------------
    ax_b = axes[0, 1]
    ax_b.scatter(coords[1:, 0], coords[1:, 1], c="black", s=5, zorder=3)
    # Plot stable vs unstable edges
    for r in init_routes:
        for i in range(len(r) - 1):
            u, v = r[i], r[i + 1]
            pair = (min(u, v), max(u, v))
            is_unstable = pair in norm_unstable
            if is_unstable:
                ax_b.plot(
                    [coords[u, 0], coords[v, 0]],
                    [coords[u, 1], coords[v, 1]],
                    c="#e53935",
                    lw=1.1,
                    alpha=0.9,
                    zorder=3,
                )
            else:
                ax_b.plot(
                    [coords[u, 0], coords[v, 0]],
                    [coords[u, 1], coords[v, 1]],
                    c="black",
                    lw=0.4,
                    alpha=0.6,
                    zorder=2,
                )
    ax_b.scatter(depot[0], depot[1], c="#ff9800", marker="s", s=36, edgecolor="black", lw=0.6, zorder=5)
    format_ax(ax_b, "(b) Unstable edges detection")

    # -------------------------------------------------------------
    # (c) Segment partitioning
    # -------------------------------------------------------------
    ax_c = axes[0, 2]
    ax_c.scatter(coords[1:, 0], coords[1:, 1], c="black", s=5, zorder=3)
    # Draw only intact segments (unstable edges cut!)
    for route_segs in partitioned_routes:
        for seg in route_segs:
            if len(seg.nodes) > 1:
                seg_coords = coords[seg.nodes]
                ax_c.plot(seg_coords[:, 0], seg_coords[:, 1], c="black", lw=0.7, alpha=0.9, zorder=2)
    ax_c.scatter(depot[0], depot[1], c="#ff9800", marker="s", s=36, edgecolor="black", lw=0.6, zorder=5)
    format_ax(ax_c, "(c) Segment partitioning")

    # -------------------------------------------------------------
    # (d) Hypernode aggregation
    # -------------------------------------------------------------
    ax_d = axes[1, 0]
    agg_coords = agg_problem.coords
    # Draw hypernodes
    ax_d.scatter(agg_coords[1:, 0], agg_coords[1:, 1], c="black", s=6, zorder=3)
    # Draw internal fixed edges of dual hypernodes
    for u, v in agg_problem.fixed_edges:
        ax_d.plot(
            [agg_coords[u, 0], agg_coords[v, 0]],
            [agg_coords[u, 1], agg_coords[v, 1]],
            c="black",
            lw=0.8,
            alpha=0.9,
            zorder=2,
        )
    ax_d.scatter(agg_coords[0, 0], agg_coords[0, 1], c="#ff9800", marker="s", s=36, edgecolor="black", lw=0.6, zorder=5)
    comp_pct = (1.0 - agg_problem.num_nodes / N) * 100.0
    format_ax(ax_d, f"(d) Hypernode aggregation (N={agg_problem.num_nodes}, -{comp_pct:.0f}%)")

    # -------------------------------------------------------------
    # (e) Re-opt. w/ backbone solvers
    # -------------------------------------------------------------
    ax_e = axes[1, 1]
    ax_e.scatter(agg_coords[1:, 0], agg_coords[1:, 1], c="black", s=6, zorder=3)
    # Draw re-optimized routes on aggregated graph
    fixed_set = set(agg_problem.fixed_edges)
    for r in opt_agg_routes:
        for i in range(len(r) - 1):
            u, v = r[i], r[i + 1]
            pair = (min(u, v), max(u, v))
            if pair in fixed_set:
                # Internal segment edge
                ax_e.plot(
                    [agg_coords[u, 0], agg_coords[v, 0]],
                    [agg_coords[u, 1], agg_coords[v, 1]],
                    c="black",
                    lw=0.8,
                    alpha=0.9,
                    zorder=2,
                )
            elif u == 0 or v == 0:
                # Depot ray on aggregated graph (thin grey)
                ax_e.plot(
                    [agg_coords[u, 0], agg_coords[v, 0]],
                    [agg_coords[u, 1], agg_coords[v, 1]],
                    c="#90a4ae",
                    lw=0.5,
                    alpha=0.6,
                    zorder=1,
                )
            else:
                # Re-optimized connecting edge between hypernodes (blue)
                ax_e.plot(
                    [agg_coords[u, 0], agg_coords[v, 0]],
                    [agg_coords[u, 1], agg_coords[v, 1]],
                    c="#1e88e5",
                    lw=1.0,
                    alpha=0.9,
                    zorder=3,
                )
    ax_e.scatter(agg_coords[0, 0], agg_coords[0, 1], c="#ff9800", marker="s", s=36, edgecolor="black", lw=0.6, zorder=5)
    format_ax(ax_e, "(e) Re-opt. w/ backbone solvers")

    # -------------------------------------------------------------
    # (f) Solution recovery
    # -------------------------------------------------------------
    ax_f = axes[1, 2]
    ax_f.scatter(coords[1:, 0], coords[1:, 1], c="black", s=5, zorder=3)
    # Initial edges set for comparison
    init_edges_set = set()
    for r in init_routes:
        for i in range(len(r) - 1):
            init_edges_set.add((min(r[i], r[i + 1]), max(r[i], r[i + 1])))

    # Draw recovered full solution
    for r in recovered_routes:
        for i in range(len(r) - 1):
            u, v = r[i], r[i + 1]
            pair = (min(u, v), max(u, v))
            if pair in init_edges_set:
                ax_f.plot(
                    [coords[u, 0], coords[v, 0]],
                    [coords[u, 1], coords[v, 1]],
                    c="black",
                    lw=0.5,
                    alpha=0.8,
                    zorder=2,
                )
            else:
                # New / improved edge in recovered solution (blue)
                ax_f.plot(
                    [coords[u, 0], coords[v, 0]],
                    [coords[u, 1], coords[v, 1]],
                    c="#1e88e5",
                    lw=0.8,
                    alpha=0.9,
                    zorder=3,
                )
    ax_f.scatter(depot[0], depot[1], c="#ff9800", marker="s", s=36, edgecolor="black", lw=0.6, zorder=5)
    c_final = compute_solution_cost(instance, recovered_routes)
    format_ax(ax_f, f"(f) Solution recovery (Cost={c_final:.1f})")

    # Save image
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[+] Successfully exported visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize L2Seg + FSTA Compression Process")
    parser.add_argument("--customers", type=int, default=500, help="Number of customers (e.g. 500 or 1000)")
    parser.add_argument("--capacity", type=float, default=150.0, help="Vehicle capacity")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--nar_path", type=str, default="checkpoints/nar_model.pt")
    parser.add_argument("--ar_path", type=str, default="checkpoints/ar_model.pt")
    parser.add_argument("--output", type=str, default="l2seg_fsta_process.png")
    args = parser.parse_args()

    print(f"\n[*] Generating Synthetic CVRP Instance: N = {args.customers}, C = {args.capacity}...")
    rng = np.random.RandomState(args.seed)
    depot_coord = np.array([[0.5, 0.5]])
    cust_coords = rng.uniform(0.0, 1.0, size=(args.customers, 2))
    coords = np.vstack([depot_coord, cust_coords])
    demands = np.concatenate([[0.0], rng.randint(1, 10, size=args.customers).astype(np.float64)])
    instance = CVRPInstance(coords=coords, demands=demands, capacity=args.capacity)

    # 1. Initial Solution
    print("[*] Generating Initial Solution (Angular Sweep)...")
    init_routes = build_angular_sweep_solution(instance)
    c_init = compute_solution_cost(instance, init_routes)
    print(f"    Initial Cost: {c_init:.2f} ({len(init_routes)} routes)")

    # 2. AI Model Prediction (L2Seg-SYN)
    print("[*] Loading L2Seg-SYN AI Model...")
    model = load_trained_l2seg_model(nar_path=args.nar_path, ar_path=args.ar_path, device="cpu")
    print("[*] Predicting Unstable Edges (L2Seg-SYN)...")
    unstable_edges = predict_unstable_edges_l2seg_syn(
        model=model,
        instance=instance,
        routes=init_routes,
        threshold=0.55,
        n_clusters=3,
    )
    print(f"    Predicted Unstable Edges: {len(unstable_edges)}")

    # 3. FSTA Partitioning
    print("[*] Performing FSTA Segment Partitioning...")
    partitioned_routes = partition_solution(instance, init_routes, unstable_edges)

    # 4. FSTA Aggregation
    print("[*] Performing Hypernode Aggregation...")
    agg_problem = aggregate_segments(instance, partitioned_routes, embed_internal_cost=True)
    comp_pct = (1.0 - agg_problem.num_nodes / args.customers) * 100.0
    print(f"    Original Nodes: {args.customers} -> Aggregated Nodes: {agg_problem.num_nodes} (Compressed {comp_pct:.1f}%)")

    # 5. Macro Local Search (Re-optimization)
    print("[*] Re-optimizing Aggregated Problem with Macro Local Search...")
    opt_agg_routes = fsta_macro_local_search(agg_problem, max_passes=5)

    # 6. Solution Recovery
    print("[*] Recovering Full CVRP Solution...")
    recovered_routes = recover_solution(agg_problem, opt_agg_routes)
    c_rec = compute_solution_cost(instance, recovered_routes)
    is_valid, msg = validate_cvrp_solution(instance, recovered_routes)
    print(f"    Recovered Cost: {c_rec:.2f} (Gain: {(c_init - c_rec)/c_init*100:.2f}%) | Feasible: {is_valid}")

    # Export Visualization
    plot_l2seg_fsta_process(
        instance=instance,
        init_routes=init_routes,
        unstable_edges=unstable_edges,
        partitioned_routes=partitioned_routes,
        agg_problem=agg_problem,
        opt_agg_routes=opt_agg_routes,
        recovered_routes=recovered_routes,
        output_path=args.output,
        dpi=300,
    )

    # Also copy to brain artifact directory if exists
    artifact_dir = r"C:\Users\ADMIN\.gemini\antigravity\brain\93706ddd-9291-44d0-9387-bdc69162df51"
    if os.path.exists(artifact_dir):
        dest = os.path.join(artifact_dir, "l2seg_fsta_process.png")
        import shutil
        shutil.copyfile(args.output, dest)
        print(f"[+] Also copied to artifact directory: {dest}")


if __name__ == "__main__":
    main()
