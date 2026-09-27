from typing import List, Tuple, Set, Dict, Optional
import numpy as np
import torch

from fsta.types import CVRPInstance
from fsta.partition import normalize_edge
from features.subproblem import SubProblem, decompose_into_adjacent_subproblems


def extract_solution_edges(routes: List[List[int]]) -> Set[Tuple[int, int]]:
    """Extract undirected edges (min, max) from a CVRP solution."""
    edges: Set[Tuple[int, int]] = set()
    for r in routes:
        for i in range(len(r) - 1):
            edges.add(normalize_edge(r[i], r[i + 1]))
    return edges


def extract_differing_edges(
    routes_before: List[List[int]],
    routes_after: List[List[int]],
) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """
    Compute deleted and inserted edges between R and R+ (Section 4.2):
        E_deleted = E_R \\ E_{R+}
        E_inserted = E_{R+} \\ E_R
    """
    edges_before = extract_solution_edges(routes_before)
    edges_after = extract_solution_edges(routes_after)

    deleted = edges_before - edges_after
    inserted = edges_after - edges_before

    return deleted, inserted


def extract_nar_labels(
    subproblem: SubProblem,
    deleted_edges: Set[Tuple[int, int]],
    inserted_edges: Set[Tuple[int, int]],
) -> torch.Tensor:
    """
    Construct binary labels for NAR model (Section 4.2):
    A node x is labeled 1 if it is an endpoint of any differing edge in E_diff, else 0.
    Depot (node 0) is labeled 0 as it is always fixed.
    
    Returns:
        Tensor of shape (N_sub,) with values in {0.0, 1.0}.
    """
    diff_edges = deleted_edges | inserted_edges
    unstable_orig_nodes: Set[int] = set()
    for u, v in diff_edges:
        if u != 0:
            unstable_orig_nodes.add(u)
        if v != 0:
            unstable_orig_nodes.add(v)

    labels = torch.zeros(subproblem.num_nodes, dtype=torch.float32)
    for loc_idx, orig_idx in enumerate(subproblem.local_to_orig):
        if loc_idx == 0:
            labels[loc_idx] = 0.0
        elif orig_idx in unstable_orig_nodes:
            labels[loc_idx] = 1.0

    return labels


def extract_ar_sequences(
    subproblem: SubProblem,
    deleted_edges: Set[Tuple[int, int]],
    inserted_edges: Set[Tuple[int, int]],
    max_seq_len: int = 12,
) -> List[List[int]]:
    """
    Construct alternating node sequences for AR model using DFS (Algorithm 2 & Figure 4):
    - Alternates between DELETION (traversing an edge in E_deleted)
      and INSERTION (traversing an edge in E_inserted).
    - Sequences terminate with END_TOKEN_ID = -1.
    
    Returns:
        List of sequences, e.g. [[pi_0, pi_1, pi_2, ..., -1], ...].
    """
    END_TOKEN_ID = -1
    local_adj_del: Dict[int, Set[int]] = {i: set() for i in range(subproblem.num_nodes)}
    local_adj_ins: Dict[int, Set[int]] = {i: set() for i in range(subproblem.num_nodes)}

    # Map original differing edges to local indices within this subproblem
    for u_orig, v_orig in deleted_edges:
        if u_orig in subproblem.orig_to_local and v_orig in subproblem.orig_to_local:
            u_loc = subproblem.orig_to_local[u_orig]
            v_loc = subproblem.orig_to_local[v_orig]
            local_adj_del[u_loc].add(v_loc)
            local_adj_del[v_loc].add(u_loc)

    for u_orig, v_orig in inserted_edges:
        if u_orig in subproblem.orig_to_local and v_orig in subproblem.orig_to_local:
            u_loc = subproblem.orig_to_local[u_orig]
            v_loc = subproblem.orig_to_local[v_orig]
            local_adj_ins[u_loc].add(v_loc)
            local_adj_ins[v_loc].add(u_loc)

    # Find starting nodes: any customer node that has an incident deleted edge
    start_candidates = [
        u for u in range(1, subproblem.num_nodes)
        if len(local_adj_del[u]) > 0
    ]

    sequences: List[List[int]] = []
    visited_del_edges: Set[Tuple[int, int]] = set()

    for start_node in start_candidates:
        curr_node = start_node
        path = [curr_node]
        step = 0

        # Alternate: step even -> delete, step odd -> insert
        while len(path) < max_seq_len:
            is_delete = (step % 2 == 0)

            if is_delete:
                # Need an unvisited deleted edge from curr_node
                avail = [
                    v for v in local_adj_del[curr_node]
                    if normalize_edge(curr_node, v) not in visited_del_edges
                ]
                if not avail:
                    break
                next_node = avail[0]
                visited_del_edges.add(normalize_edge(curr_node, next_node))
                path.append(next_node)
                curr_node = next_node
            else:
                # Need an inserted edge from curr_node
                avail = list(local_adj_ins[curr_node])
                if not avail:
                    break
                next_node = avail[0]
                path.append(next_node)
                curr_node = next_node

            step += 1

        # A valid sequence must contain at least 1 delete and 1 insert, ending with END_TOKEN_ID
        if len(path) >= 3:
            path.append(END_TOKEN_ID)
            sequences.append(path)

    return sequences
