import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import Dataset

from fsta.types import CVRPInstance
from fsta.init_solution import build_angular_sweep_solution
from fsta.local_search import original_cvrp_local_search
from features.subproblem import SubProblem, decompose_into_adjacent_subproblems
from features.node_features import extract_subproblem_node_features
from features.edge_features import extract_subproblem_edges
from .label_extractor import (
    extract_differing_edges,
    extract_nar_labels,
    extract_ar_sequences,
)


@dataclass
class L2SegSample:
    """A single subproblem training sample with node/edge features and supervisory labels."""
    node_feats: torch.Tensor          # (N_sub, 25)
    tour_positions: torch.Tensor      # (N_sub,)
    route_membership: torch.Tensor    # (N_sub,)
    edge_index: torch.Tensor          # (2, E)
    edge_attr: torch.Tensor           # (E, 3)
    nar_labels: torch.Tensor          # (N_sub,)
    ar_sequences: List[List[int]]     # List of alternating DFS sequences
    local_routes: List[List[int]]     # [[0, ...], [0, ...]]


class L2SegDataset(Dataset):
    """PyTorch Dataset containing L2Seg training samples."""
    def __init__(self, samples: Optional[List[L2SegSample]] = None):
        self.samples = samples if samples is not None else []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> L2SegSample:
        return self.samples[idx]

    def save(self, filepath: str) -> None:
        """Save dataset to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        torch.save(self.samples, filepath)

    @classmethod
    def load(cls, filepath: str) -> "L2SegDataset":
        """Load dataset from disk."""
        samples = torch.load(filepath, weights_only=False)
        return cls(samples)


def create_training_samples_from_instance(
    instance: CVRPInstance,
    initial_routes: Optional[List[List[int]]] = None,
    solver_passes: int = 5,
    oracle: str = "pyvrp",
) -> List[L2SegSample]:
    """
    Generate training samples from a single CVRP instance (Algorithm 2 in paper):
    1. Obtain initial solution R (via Angular Sweep if not provided).
    2. Obtain improved solution R+ via local solver lookahead (PyVRP expert or local search).
    3. Extract differing edges E_diff = (E_deleted, E_inserted).
    4. Decompose P into adjacent route subproblems P_TR.
    5. For each subproblem, extract node/edge features and supervisory labels.
    """
    if initial_routes is None:
        initial_routes = build_angular_sweep_solution(instance)

    # 1. Run look-ahead solver to obtain improved solution R+
    improved_routes = None
    if oracle.lower() == "pyvrp":
        try:
            from solvers.pyvrp_solver import PyVRPSolver
            pyvrp = PyVRPSolver()
            routes_cand, _, _ = pyvrp.solve(instance, time_limit=1.0)
            if routes_cand and len(routes_cand) > 0:
                improved_routes = routes_cand
        except Exception:
            improved_routes = None

    if improved_routes is None:
        improved_routes = original_cvrp_local_search(instance, initial_routes, max_passes=solver_passes)

    # 2. Extract differing edges
    deleted_edges, inserted_edges = extract_differing_edges(initial_routes, improved_routes)

    # 3. Decompose into adjacent route subproblems
    subproblems = decompose_into_adjacent_subproblems(instance, initial_routes)

    samples: List[L2SegSample] = []
    for sub in subproblems:
        # Features
        node_feats = extract_subproblem_node_features(sub)
        tour_pos = torch.from_numpy(sub.tour_positions).long()
        route_mem = torch.from_numpy(sub.route_membership).long()
        edge_index, edge_attr = extract_subproblem_edges(sub)

        # Supervisory Labels
        nar_labels = extract_nar_labels(sub, deleted_edges, inserted_edges)
        ar_seqs = extract_ar_sequences(sub, deleted_edges, inserted_edges)

        samples.append(
            L2SegSample(
                node_feats=node_feats,
                tour_positions=tour_pos,
                route_membership=route_mem,
                edge_index=edge_index,
                edge_attr=edge_attr,
                nar_labels=nar_labels,
                ar_sequences=ar_seqs,
                local_routes=sub.local_routes,
            )
        )

    return samples
