import pytest
import numpy as np
import torch
from fsta.types import CVRPInstance
from fsta.init_solution import build_angular_sweep_solution
from features.subproblem import decompose_into_adjacent_subproblems
from features.node_features import extract_subproblem_node_features
from features.edge_features import extract_subproblem_edges
from models.encoder import L2SegEncoder
from models.nar_decoder import L2SegNARDecoder
from models.ar_decoder import L2SegARDecoder
from models.l2seg_model import L2SegModel


def create_mock_instance(num_customers: int = 20, capacity: float = 20.0) -> CVRPInstance:
    rng = np.random.RandomState(42)
    coords = np.vstack([[0.5, 0.5], rng.uniform(0.0, 1.0, size=(num_customers, 2))])
    demands = np.concatenate([[0.0], rng.randint(1, 6, size=num_customers).astype(np.float64)])
    return CVRPInstance(coords=coords, demands=demands, capacity=capacity)


def test_nar_decoder_forward_and_loss():
    instance = create_mock_instance(num_customers=15)
    routes = build_angular_sweep_solution(instance)
    sub = decompose_into_adjacent_subproblems(instance, routes)[0]

    node_feats = extract_subproblem_node_features(sub)
    tour_positions = torch.from_numpy(sub.tour_positions).long()
    route_membership = torch.from_numpy(sub.route_membership).long()
    edge_index, edge_attr = extract_subproblem_edges(sub, k_nearest=10)

    encoder = L2SegEncoder(hidden_dim=64, num_tfm_layers=1, num_gnn_layers=1)
    nar_decoder = L2SegNARDecoder(hidden_dim=64)

    h_gnn = encoder(node_feats, tour_positions, route_membership, edge_index, edge_attr)
    p_nar = nar_decoder(h_gnn)

    # Check shapes and bounds
    assert p_nar.shape == (sub.num_nodes,)
    assert (p_nar >= 0.0).all() and (p_nar <= 1.0).all()

    # Create dummy binary labels
    labels = torch.zeros(sub.num_nodes, dtype=torch.float32)
    labels[1] = 1.0
    labels[3] = 1.0

    loss = nar_decoder.compute_loss(p_nar, labels, w_pos=9.0)
    assert loss.item() > 0.0

    # Backpropagation
    loss.backward()
    for param in nar_decoder.parameters():
        if param.requires_grad:
            assert param.grad is not None


def test_ar_decoder_forward_and_loss():
    instance = create_mock_instance(num_customers=15)
    routes = build_angular_sweep_solution(instance)
    sub = decompose_into_adjacent_subproblems(instance, routes)[0]

    h_gnn = torch.randn(sub.num_nodes, 64, requires_grad=True)
    ar_decoder = L2SegARDecoder(hidden_dim=64, num_delete_layers=1, num_insert_layers=1)

    # Test dummy sequence: [node_1 -> delete node_2 -> insert node_4 -> end(-1)]
    seq = [1, 2, 4, -1]
    loss = ar_decoder.compute_sequence_loss(
        h_gnn=h_gnn,
        node_sequence=seq,
        local_routes=sub.local_routes,
        w_insert=0.8,
        w_delete=0.2,
    )

    assert loss.item() > 0.0
    loss.backward()

    # Gradients should reach h_gnn and ar_decoder
    assert h_gnn.grad is not None
    assert ar_decoder.W_q.weight.grad is not None
    assert ar_decoder.raw_alpha.grad is not None

    # Test greedy decoding inference
    deleted_edges = ar_decoder.predict_unstable_edges(
        h_gnn=h_gnn.detach(),
        pi_0=1,
        local_routes=sub.local_routes,
        max_steps=4,
    )
    assert isinstance(deleted_edges, list)


def test_l2seg_unified_model_syn_inference():
    instance = create_mock_instance(num_customers=25)
    routes = build_angular_sweep_solution(instance)
    sub = decompose_into_adjacent_subproblems(instance, routes)[0]

    model = L2SegModel(
        node_in_dim=25,
        edge_in_dim=3,
        hidden_dim=64,
        num_tfm_layers=1,
        num_gnn_layers=1,
        num_ar_delete_layers=1,
        num_ar_insert_layers=1,
    )
    model.eval()

    # Test L2Seg-SYN prediction
    syn_edges = model.predict_subproblem_syn(sub, threshold=0.5, n_clusters=2)
    assert isinstance(syn_edges, set)

    # Test L2Seg-NAR prediction
    nar_edges = model.predict_subproblem_nar(sub, threshold=0.5)
    assert isinstance(nar_edges, set)

    # Verify that edges contain valid original node IDs
    for u, v in syn_edges:
        assert 0 <= u <= instance.num_customers
        assert 0 <= v <= instance.num_customers
