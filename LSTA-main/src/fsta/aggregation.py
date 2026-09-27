from typing import List, Tuple, Dict
import numpy as np
from .types import CVRPInstance, Segment, AggregatedProblem


def aggregate_segments(
    instance: CVRPInstance,
    partitioned_routes: List[List[Segment]],
    embed_internal_cost: bool = True,
) -> AggregatedProblem:
    """
    Construct the reduced problem \tilde{P} using Dual Hypernode Aggregation for CVRP.
    
    According to the FSTA paper (Section 3.2, Table 7, and Appendix B.1.5):
    - Multi-node segments S_{j, k} (length >= 2) are represented by dual hypernodes:
        * \tilde{x}_j with coords of start customer x_j
        * \tilde{x}_k with coords of end customer x_k
        * Demands: \tilde{d}_j = \tilde{d}_k = 0.5 * sum(d_m)
        * Fixed edge: (\tilde{x}_j, \tilde{x}_k) must be traversed together.
    - Single-node segments S_{j, j} (length == 1) are represented by 1 hypernode.
    - Depot remains node 0.

    Args:
        instance: Original CVRPInstance.
        partitioned_routes: Segments per route from partition_solution.
        embed_internal_cost: If True, set dist(\tilde{x}_j, \tilde{x}_k) to the segment's
                             internal travel cost, making f(\tilde{R}) == f(R) directly.
                             If False, set dist(\tilde{x}_j, \tilde{x}_k) = 0 and track constant_offset.

    Returns:
        AggregatedProblem instance.
    """
    # Depot is always aggregated node 0
    agg_coords: List[np.ndarray] = [instance.coords[0].copy()]
    agg_demands: List[float] = [0.0]
    
    # Mapping: agg_node_idx -> (segment_id, 'head' | 'tail' | 'single')
    node_to_segment: Dict[int, Tuple[int, str]] = {}
    
    # Mapping: segment_id -> Segment
    segments_dict: Dict[int, Segment] = {}
    
    # Store hypernode index assignments for each segment:
    # segment_id -> (head_idx, tail_idx) or (single_idx, single_idx)
    segment_to_nodes: Dict[int, Tuple[int, int]] = {}
    
    fixed_edges: List[Tuple[int, int]] = []
    constant_offset = 0.0

    current_agg_idx = 1
    for route_segs in partitioned_routes:
        for seg in route_segs:
            segments_dict[seg.segment_id] = seg
            constant_offset += seg.internal_cost

            if seg.is_single_node:
                # Single customer node -> 1 hypernode
                node_orig = seg.start_node
                agg_coords.append(instance.coords[node_orig].copy())
                agg_demands.append(instance.demands[node_orig])
                
                node_to_segment[current_agg_idx] = (seg.segment_id, "single")
                segment_to_nodes[seg.segment_id] = (current_agg_idx, current_agg_idx)
                current_agg_idx += 1
            else:
                # Multi-node segment -> Dual hypernodes (head and tail)
                head_orig = seg.start_node
                tail_orig = seg.end_node
                half_demand = seg.total_demand / 2.0

                head_idx = current_agg_idx
                tail_idx = current_agg_idx + 1

                # Head hypernode
                agg_coords.append(instance.coords[head_orig].copy())
                agg_demands.append(half_demand)
                node_to_segment[head_idx] = (seg.segment_id, "head")

                # Tail hypernode
                agg_coords.append(instance.coords[tail_orig].copy())
                agg_demands.append(half_demand)
                node_to_segment[tail_idx] = (seg.segment_id, "tail")

                segment_to_nodes[seg.segment_id] = (head_idx, tail_idx)
                fixed_edges.append((min(head_idx, tail_idx), max(head_idx, tail_idx)))

                current_agg_idx += 2

    num_agg_nodes = len(agg_coords)
    agg_coords_arr = np.array(agg_coords, dtype=np.float64)
    agg_demands_arr = np.array(agg_demands, dtype=np.float64)

    # Compute Euclidean distance matrix between all pairs of coordinates
    diff = agg_coords_arr[:, None, :] - agg_coords_arr[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

    # Adjust distance for dual hypernode internal edges
    for seg_id, seg in segments_dict.items():
        if not seg.is_single_node:
            head_idx, tail_idx = segment_to_nodes[seg_id]
            if embed_internal_cost:
                dist_matrix[head_idx, tail_idx] = seg.internal_cost
                dist_matrix[tail_idx, head_idx] = seg.internal_cost
            else:
                dist_matrix[head_idx, tail_idx] = 0.0
                dist_matrix[tail_idx, head_idx] = 0.0

    # Build the initial aggregated solution \tilde{R}
    initial_aggregated_routes: List[List[int]] = []
    for route_segs in partitioned_routes:
        if not route_segs:
            initial_aggregated_routes.append([0, 0])
            continue

        agg_route = [0]
        for seg in route_segs:
            if seg.is_single_node:
                node_idx = segment_to_nodes[seg.segment_id][0]
                agg_route.append(node_idx)
            else:
                head_idx, tail_idx = segment_to_nodes[seg.segment_id]
                # In initial route, segment was traversed from head to tail
                agg_route.extend([head_idx, tail_idx])
        agg_route.append(0)
        initial_aggregated_routes.append(agg_route)

    return AggregatedProblem(
        coords=agg_coords_arr,
        demands=agg_demands_arr,
        capacity=instance.capacity,
        dist_matrix=dist_matrix,
        fixed_edges=fixed_edges,
        segments=segments_dict,
        node_to_segment=node_to_segment,
        initial_aggregated_routes=initial_aggregated_routes,
        constant_offset=0.0 if embed_internal_cost else constant_offset,
    )
