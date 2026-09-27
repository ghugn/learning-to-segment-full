import os
import tempfile
import pytest
import numpy as np
import torch

from fsta.types import CVRPInstance
from fsta.init_solution import build_angular_sweep_solution
from data.label_extractor import (
    extract_differing_edges,
    extract_nar_labels,
    extract_ar_sequences,
)
from data.dataset import (
    L2SegDataset,
    create_training_samples_from_instance,
)
from features.subproblem import decompose_into_adjacent_subproblems


def create_mock_instance(num_customers: int = 25, capacity: float = 25.0) -> CVRPInstance:
    rng = np.random.RandomState(42)
    coords = np.vstack([[0.5, 0.5], rng.uniform(0.0, 1.0, size=(num_customers, 2))])
    demands = np.concatenate([[0.0], rng.randint(1, 6, size=num_customers).astype(np.float64)])
    return CVRPInstance(coords=coords, demands=demands, capacity=capacity)


def test_differing_edges_extraction():
    # R: 0 -> 1 -> 2 -> 3 -> 0
    # R+: 0 -> 1 -> 3 -> 2 -> 0 (swapped 2 and 3)
    # Deleted: (1, 2), (3, 0)
    # Inserted: (1, 3), (2, 0)
    r_before = [[0, 1, 2, 3, 0]]
    r_after = [[0, 1, 3, 2, 0]]

    deleted, inserted = extract_differing_edges(r_before, r_after)
    assert (1, 2) in deleted
    assert (0, 3) in deleted
    assert (1, 3) in inserted
    assert (0, 2) in inserted


def test_nar_and_ar_label_extraction():
    instance = create_mock_instance(num_customers=20)
    routes = build_angular_sweep_solution(instance)
    sub = decompose_into_adjacent_subproblems(instance, routes)[0]

    # Mock differing edges: delete (u, v) and insert (v, w)
    loc_nodes = sub.local_to_orig
    u_orig = loc_nodes[1]
    v_orig = loc_nodes[2]
    w_orig = loc_nodes[3]

    del_edges = {(min(u_orig, v_orig), max(u_orig, v_orig))}
    ins_edges = {(min(v_orig, w_orig), max(v_orig, w_orig))}

    nar_labels = extract_nar_labels(sub, del_edges, ins_edges)
    assert nar_labels[0] == 0.0  # Depot always 0
    assert nar_labels[1] == 1.0  # Node u is unstable
    assert nar_labels[2] == 1.0  # Node v is unstable
    assert nar_labels[3] == 1.0  # Node w is unstable

    # Test AR sequence DFS
    ar_seqs = extract_ar_sequences(sub, del_edges, ins_edges)
    assert len(ar_seqs) > 0
    seq = ar_seqs[0]
    # Format: [pi_0, pi_1, pi_2, ..., -1]
    assert seq[-1] == -1
    assert len(seq) >= 4


def test_full_dataset_generation_and_persistence():
    instance = create_mock_instance(num_customers=25)
    samples = create_training_samples_from_instance(instance, solver_passes=3)

    assert len(samples) > 0
    s0 = samples[0]
    assert s0.node_feats.shape[1] == 25
    assert s0.edge_attr.shape[1] == 3
    assert s0.nar_labels.shape[0] == s0.node_feats.shape[0]

    # Test Dataset save and load
    dataset = L2SegDataset(samples)
    assert len(dataset) == len(samples)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        dataset.save(tmp_path)
        loaded = L2SegDataset.load(tmp_path)
        assert len(loaded) == len(dataset)
        assert torch.equal(loaded[0].node_feats, s0.node_feats)
        assert torch.equal(loaded[0].nar_labels, s0.nar_labels)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
