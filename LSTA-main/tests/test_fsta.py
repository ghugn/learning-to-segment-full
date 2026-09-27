import pytest
import numpy as np
from fsta.types import CVRPInstance
from fsta.partition import partition_solution, normalize_edge
from fsta.aggregation import aggregate_segments
from fsta.recovery import (
    recover_solution,
    compute_solution_cost,
    validate_cvrp_solution,
)


def create_sample_cvrp(num_customers: int = 8, capacity: float = 30.0, seed: int = 42) -> CVRPInstance:
    """Create a deterministic synthetic CVRP instance."""
    rng = np.random.RandomState(seed)
    # Depot at [0.5, 0.5]
    coords = np.vstack([[0.5, 0.5], rng.uniform(0.0, 1.0, size=(num_customers, 2))])
    # Demands between 2 and 6
    demands = np.concatenate([[0.0], rng.randint(2, 7, size=num_customers).astype(np.float64)])
    return CVRPInstance(coords=coords, demands=demands, capacity=capacity)


def test_partition_solution():
    instance = create_sample_cvrp(num_customers=6)
    # Two routes: 0 -> 1 -> 2 -> 3 -> 0, and 0 -> 4 -> 5 -> 6 -> 0
    routes = [
        [0, 1, 2, 3, 0],
        [0, 4, 5, 6, 0],
    ]
    # Mark edge (2, 3) and edge (4, 5) as unstable
    unstable_edges = [(2, 3), (5, 4)]
    
    partitioned = partition_solution(instance, routes, unstable_edges)
    assert len(partitioned) == 2
    
    # Route 0 should be split into: [1, 2] and [3]
    r0_segs = partitioned[0]
    assert len(r0_segs) == 2
    assert r0_segs[0].nodes == [1, 2]
    assert r0_segs[1].nodes == [3]
    assert r0_segs[1].is_single_node is True

    # Route 1 should be split into: [4] and [5, 6]
    r1_segs = partitioned[1]
    assert len(r1_segs) == 2
    assert r1_segs[0].nodes == [4]
    assert r1_segs[1].nodes == [5, 6]


def test_fsta_aggregation_and_exact_cost_match():
    instance = create_sample_cvrp(num_customers=6)
    routes = [
        [0, 1, 2, 3, 0],
        [0, 4, 5, 6, 0],
    ]
    # Ensure initial solution is feasible
    is_valid, msg = validate_cvrp_solution(instance, routes)
    assert is_valid, msg

    orig_cost = compute_solution_cost(instance, routes)

    # Cut edge (1, 2)
    unstable_edges = [(1, 2)]
    partitioned = partition_solution(instance, routes, unstable_edges)
    
    # Aggregate with embed_internal_cost=True
    agg_problem = aggregate_segments(instance, partitioned, embed_internal_cost=True)
    
    # Verify demands sum is preserved
    assert np.isclose(np.sum(agg_problem.demands), np.sum(instance.demands))

    # Initial aggregated routes should evaluate to the EXACT same cost as original
    agg_cost = 0.0
    for agg_route in agg_problem.initial_aggregated_routes:
        for i in range(len(agg_route) - 1):
            agg_cost += agg_problem.dist_matrix[agg_route[i], agg_route[i + 1]]

    assert np.isclose(agg_cost, orig_cost, atol=1e-8), f"Agg cost {agg_cost} != orig cost {orig_cost}"

    # Recover solution directly from initial aggregated routes
    recovered = recover_solution(agg_problem, agg_problem.initial_aggregated_routes)
    assert recovered == routes
    
    rec_cost = compute_solution_cost(instance, recovered)
    assert np.isclose(rec_cost, orig_cost, atol=1e-8)


def test_fsta_bidirectional_segment_traversal():
    """Test that reversing a dual hypernode pair in the aggregated route correctly reverses customer order."""
    instance = create_sample_cvrp(num_customers=5)
    # Route: 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 0
    routes = [[0, 1, 2, 3, 4, 5, 0]]
    # Cut (1, 2) and (4, 5) -> Segments: [1], [2, 3, 4], [5]
    unstable_edges = [(1, 2), (4, 5)]
    
    partitioned = partition_solution(instance, routes, unstable_edges)
    agg_problem = aggregate_segments(instance, partitioned, embed_internal_cost=True)
    
    # seg 0: [1] (single)
    # seg 1: [2, 3, 4] (head, tail)
    # seg 2: [5] (single)
    agg_route = agg_problem.initial_aggregated_routes[0]
    # Format: [0, h_1, h_head, h_tail, h_5, 0]
    head_idx = agg_route[2]
    tail_idx = agg_route[3]
    
    # Reverse traversal of the segment: swap head and tail
    reversed_agg_route = [agg_route[0], agg_route[1], tail_idx, head_idx, agg_route[4], agg_route[5]]
    
    recovered = recover_solution(agg_problem, [reversed_agg_route])
    # The recovered route should have [2, 3, 4] reversed to [4, 3, 2]
    assert recovered == [[0, 1, 4, 3, 2, 5, 0]]
    
    # Check validity
    is_valid, msg = validate_cvrp_solution(instance, recovered)
    assert is_valid, msg


def test_random_cvrp_compression_and_recovery():
    """Stress test with 50 customers and random cuts."""
    rng = np.random.RandomState(123)
    num_cust = 50
    instance = create_sample_cvrp(num_customers=num_cust, capacity=50.0, seed=123)

    # Build simple sequential routes satisfying capacity
    routes = []
    curr_route = [0]
    curr_load = 0.0
    for c in range(1, num_cust + 1):
        if curr_load + instance.demands[c] <= instance.capacity:
            curr_route.append(c)
            curr_load += instance.demands[c]
        else:
            curr_route.append(0)
            routes.append(curr_route)
            curr_route = [0, c]
            curr_load = instance.demands[c]
    curr_route.append(0)
    routes.append(curr_route)

    is_valid, msg = validate_cvrp_solution(instance, routes)
    assert is_valid, msg

    # Randomly select ~25% of internal customer edges to be unstable
    candidate_edges = []
    for r in routes:
        custs = r[1:-1]
        for t in range(len(custs) - 1):
            candidate_edges.append((custs[t], custs[t + 1]))

    num_unstable = max(1, len(candidate_edges) // 4)
    chosen_indices = rng.choice(len(candidate_edges), size=num_unstable, replace=False)
    unstable_edges = [candidate_edges[i] for i in chosen_indices]

    # FSTA Partition & Aggregate
    partitioned = partition_solution(instance, routes, unstable_edges)
    agg_problem = aggregate_segments(instance, partitioned, embed_internal_cost=True)

    # Verify problem size reduction
    assert agg_problem.num_nodes <= num_cust

    # Recover
    recovered = recover_solution(agg_problem, agg_problem.initial_aggregated_routes)
    is_valid, msg = validate_cvrp_solution(instance, recovered)
    assert is_valid, msg
    assert recovered == routes

    # Cost invariance
    orig_cost = compute_solution_cost(instance, routes)
    rec_cost = compute_solution_cost(instance, recovered)
    assert np.isclose(orig_cost, rec_cost, atol=1e-8)


def test_monotonicity_improvement():
    """Verify Monotonicity: an improvement on \tilde{P} strictly yields an improvement on P."""
    coords = np.array([
        [0.0, 0.0],  # 0: Depot
        [0.0, 1.0],  # 1: A
        [0.0, 2.0],  # 2: B
        [2.0, 1.0],  # 3: D (crossed)
        [2.0, 2.0],  # 4: C
    ])
    demands = np.array([0.0, 5.0, 5.0, 5.0, 5.0])
    inst = CVRPInstance(coords=coords, demands=demands, capacity=30.0)

    # Initial crossed route: 0 -> 1 -> 2 -> 3 -> 4 -> 0
    routes = [[0, 1, 2, 3, 4, 0]]
    unstable_edges = [(2, 3)]

    partitioned = partition_solution(inst, routes, unstable_edges)
    agg = aggregate_segments(inst, partitioned, embed_internal_cost=True)

    cost_init = compute_solution_cost(inst, routes)

    # Re-optimize on aggregated graph by reversing segment 2 (swap head 3 and tail 4 to 4 -> 3)
    agg_route_opt = [[0, 1, 2, 4, 3, 0]]
    rec = recover_solution(agg, agg_route_opt)
    cost_opt = compute_solution_cost(inst, rec)

    assert cost_opt < cost_init, f"Expected cost reduction, got {cost_opt} >= {cost_init}"
    is_valid, msg = validate_cvrp_solution(inst, rec)
    assert is_valid, msg
    assert rec == [[0, 1, 2, 4, 3, 0]]


def test_edge_selectors():
    """Verify non-learning edge selectors: random, longest, and oracle diff."""
    from fsta.edge_selectors import (
        random_edge_selector,
        longest_edge_selector,
        oracle_diff_selector,
    )

    routes = [[0, 1, 2, 3, 4, 5, 0], [0, 6, 7, 8, 0]]
    rand_edges = random_edge_selector(routes, ratio=0.4, seed=42)
    assert len(rand_edges) > 0

    coords = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.2],
        [0.3, 0.3],
        [5.0, 5.0],  # (3, 4) is long
        [5.1, 5.1],
    ])
    demands = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    inst = CVRPInstance(coords=coords, demands=demands, capacity=20.0)
    route2 = [[0, 1, 2, 3, 4, 5, 0]]
    long_edges = longest_edge_selector(inst, route2, ratio=0.25)
    assert (3, 4) in long_edges

    # Oracle selector: deleted edges between R and R+
    r_before = [[0, 1, 2, 3, 4, 0]]
    r_after = [[0, 1, 3, 2, 4, 0]]
    oracle_edges = oracle_diff_selector(r_before, r_after)
    assert (1, 2) in oracle_edges
    assert (3, 4) in oracle_edges


