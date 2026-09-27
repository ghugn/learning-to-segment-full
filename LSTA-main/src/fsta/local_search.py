import copy
from typing import List, Tuple, Dict
import numpy as np
from .types import CVRPInstance, AggregatedProblem


class Block:
    """
    Represents an atomic macro-block in the aggregated problem.
    Can be either a single hypernode [u] or a dual hypernode [head, tail].
    """
    def __init__(self, seg_id: int, is_single: bool, head: int, tail: int, is_reversed: bool = False):
        self.seg_id = seg_id
        self.is_single = is_single
        self.head = head
        self.tail = tail
        self.is_reversed = is_reversed

    @property
    def entry_node(self) -> int:
        if self.is_single:
            return self.head
        return self.tail if self.is_reversed else self.head

    @property
    def exit_node(self) -> int:
        if self.is_single:
            return self.head
        return self.head if self.is_reversed else self.tail

    def to_node_list(self) -> List[int]:
        if self.is_single:
            return [self.head]
        return [self.tail, self.head] if self.is_reversed else [self.head, self.tail]

    def flipped(self) -> "Block":
        if self.is_single:
            return self
        return Block(self.seg_id, False, self.head, self.tail, not self.is_reversed)


def parse_aggregated_route_into_blocks(
    agg_problem: AggregatedProblem,
    agg_route: List[int],
) -> List[Block]:
    """Parse a raw aggregated route into a list of atomic Blocks."""
    blocks: List[Block] = []
    i = 1
    n = len(agg_route) - 1

    while i < n:
        u = agg_route[i]
        seg_id, role = agg_problem.node_to_segment[u]
        seg = agg_problem.segments[seg_id]

        if role == "single":
            blocks.append(Block(seg_id, True, u, u, False))
            i += 1
        elif role == "head":
            tail_idx = agg_route[i + 1]
            blocks.append(Block(seg_id, False, u, tail_idx, is_reversed=False))
            i += 2
        elif role == "tail":
            head_idx = agg_route[i + 1]
            blocks.append(Block(seg_id, False, head_idx, u, is_reversed=True))
            i += 2
        else:
            raise ValueError(f"Unknown role {role}")

    return blocks


def blocks_to_aggregated_route(blocks: List[Block]) -> List[int]:
    """Convert a list of Blocks back to a raw aggregated route [0, ..., 0]."""
    route = [0]
    for b in blocks:
        route.extend(b.to_node_list())
    route.append(0)
    return route


def compute_blocks_route_cost(agg_problem: AggregatedProblem, blocks: List[Block]) -> float:
    """Compute travel cost of a route represented by blocks."""
    if not blocks:
        return 0.0
    dist = agg_problem.dist_matrix
    cost = 0.0

    # Depot to first block entry
    cost += dist[0, blocks[0].entry_node]

    # Between blocks
    for i in range(len(blocks) - 1):
        # Internal block cost is included in dist_matrix between entry and exit if embedded
        b_curr = blocks[i]
        b_next = blocks[i + 1]
        if not b_curr.is_single:
            cost += dist[b_curr.entry_node, b_curr.exit_node]
        cost += dist[b_curr.exit_node, b_next.entry_node]

    # Last block internal cost & exit to depot
    b_last = blocks[-1]
    if not b_last.is_single:
        cost += dist[b_last.entry_node, b_last.exit_node]
    cost += dist[b_last.exit_node, 0]

    return cost


def compute_blocks_route_demand(agg_problem: AggregatedProblem, blocks: List[Block]) -> float:
    """Compute total demand of a route represented by blocks."""
    total_d = 0.0
    for b in blocks:
        total_d += agg_problem.segments[b.seg_id].total_demand
    return total_d


def fsta_macro_local_search(
    agg_problem: AggregatedProblem,
    max_passes: int = 10,
) -> List[List[int]]:
    """
    Perform macro-block Local Search on the aggregated problem \tilde{P}:
    - Intra-route Block 2-opt (inverting sequences of blocks)
    - Intra-route Block Flip (reversing single dual hypernode orientation)
    - Inter-route Block Relocate (moving a block to another route within capacity)

    Guarantees:
    - Never breaks internal dual hypernode fixed edges.
    - Operates on a much smaller search space.
    - Yields strictly monotone improvement.
    """
    # Parse initial routes into blocks
    routes_blocks: List[List[Block]] = [
        parse_aggregated_route_into_blocks(agg_problem, r)
        for r in agg_problem.initial_aggregated_routes
    ]

    improved = True
    passes = 0

    while improved and passes < max_passes:
        improved = False
        passes += 1

        # 1. Intra-route moves
        for r_idx in range(len(routes_blocks)):
            blocks = routes_blocks[r_idx]
            m = len(blocks)
            if m < 1:
                continue

            current_cost = compute_blocks_route_cost(agg_problem, blocks)

            # Move A: Block Flip (try reversing individual dual hypernodes)
            for i in range(m):
                if blocks[i].is_single:
                    continue
                # Flip block i
                blocks[i] = blocks[i].flipped()
                new_cost = compute_blocks_route_cost(agg_problem, blocks)
                if new_cost < current_cost - 1e-6:
                    current_cost = new_cost
                    improved = True
                else:
                    # Revert
                    blocks[i] = blocks[i].flipped()

            # Move B: Intra-route 2-opt on blocks
            for i in range(m - 1):
                for j in range(i + 1, m):
                    # Invert blocks[i:j+1] and flip each inverted block
                    sub = [b.flipped() for b in reversed(blocks[i : j + 1])]
                    candidate_blocks = blocks[:i] + sub + blocks[j + 1 :]
                    new_cost = compute_blocks_route_cost(agg_problem, candidate_blocks)
                    if new_cost < current_cost - 1e-6:
                        blocks = candidate_blocks
                        routes_blocks[r_idx] = blocks
                        current_cost = new_cost
                        improved = True
                        break
                if improved:
                    break

        # 2. Inter-route Relocate
        num_routes = len(routes_blocks)
        for r1 in range(num_routes):
            if not routes_blocks[r1]:
                continue
            for r2 in range(num_routes):
                if r1 == r2:
                    continue

                r1_blocks = routes_blocks[r1]
                r2_blocks = routes_blocks[r2]
                r2_demand = compute_blocks_route_demand(agg_problem, r2_blocks)

                for i in range(len(r1_blocks)):
                    target_block = r1_blocks[i]
                    block_demand = agg_problem.segments[target_block.seg_id].total_demand

                    if r2_demand + block_demand > agg_problem.capacity:
                        continue

                    # Current combined cost
                    c_old = compute_blocks_route_cost(agg_problem, r1_blocks) + compute_blocks_route_cost(agg_problem, r2_blocks)

                    # Try inserting into all positions in r2
                    r1_cand = r1_blocks[:i] + r1_blocks[i + 1 :]
                    for pos in range(len(r2_blocks) + 1):
                        for flipped in [False, True]:
                            blk = target_block.flipped() if flipped else target_block
                            r2_cand = r2_blocks[:pos] + [blk] + r2_blocks[pos:]
                            c_new = compute_blocks_route_cost(agg_problem, r1_cand) + compute_blocks_route_cost(agg_problem, r2_cand)

                            if c_new < c_old - 1e-6:
                                routes_blocks[r1] = r1_cand
                                routes_blocks[r2] = r2_cand
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    # Reconstruct raw aggregated routes
    return [blocks_to_aggregated_route(b) for b in routes_blocks]


def original_cvrp_local_search(
    instance: CVRPInstance,
    routes: List[List[int]],
    max_passes: int = 10,
) -> List[List[int]]:
    """
    Standard node-level local search (2-opt & relocate) on the full, uncompressed graph P.
    Used as direct baseline for runtime and solution quality comparison.
    """
    routes = copy.deepcopy(routes)
    dist = instance.dist_matrix
    capacity = instance.capacity
    demands = instance.demands

    def route_cost(r: List[int]) -> float:
        return sum(dist[r[i], r[i + 1]] for i in range(len(r) - 1))

    improved = True
    passes = 0

    while improved and passes < max_passes:
        improved = False
        passes += 1

        # Intra-route 2-opt
        for r_idx in range(len(routes)):
            r = routes[r_idx]
            n = len(r)
            if n <= 3:
                continue

            c_best = route_cost(r)
            for i in range(1, n - 2):
                for j in range(i + 1, n - 1):
                    # 2-opt swap
                    new_r = r[:i] + list(reversed(r[i : j + 1])) + r[j + 1 :]
                    new_c = route_cost(new_r)
                    if new_c < c_best - 1e-6:
                        routes[r_idx] = new_r
                        c_best = new_c
                        improved = True
                        break
                if improved:
                    break

        # Inter-route Relocate
        num_routes = len(routes)
        for r1 in range(num_routes):
            if len(routes[r1]) <= 2:
                continue
            for r2 in range(num_routes):
                if r1 == r2:
                    continue

                r1_nodes = routes[r1]
                r2_nodes = routes[r2]
                r2_dem = sum(demands[u] for u in r2_nodes[1:-1])

                for i in range(1, len(r1_nodes) - 1):
                    u = r1_nodes[i]
                    if r2_dem + demands[u] > capacity:
                        continue

                    old_cost = route_cost(r1_nodes) + route_cost(r2_nodes)
                    cand_r1 = r1_nodes[:i] + r1_nodes[i + 1 :]

                    for pos in range(1, len(r2_nodes)):
                        cand_r2 = r2_nodes[:pos] + [u] + r2_nodes[pos:]
                        new_cost = route_cost(cand_r1) + route_cost(cand_r2)

                        if new_cost < old_cost - 1e-6:
                            routes[r1] = cand_r1
                            routes[r2] = cand_r2
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    return routes
