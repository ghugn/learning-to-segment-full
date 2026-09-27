import pytest
import numpy as np
import torch
from fsta.types import CVRPInstance
from fsta.init_solution import build_angular_sweep_solution
from features.subproblem import decompose_into_adjacent_subproblems
from features.node_features import extract_subproblem_node_features
from features.edge_features import extract_subproblem_edges
from models.encoder import L2SegEncoder


def create_mock_instance(num_customers: int = 30, capacity: float = 25.0) -> CVRPInstance:
    """Create a deterministic CVRP instance for feature and model testing."""
    rng = np.random.RandomState(42)
    coords = np.vstack([[0.5, 0.5], rng.uniform(0.0, 1.0, size=(num_customers, 2))])
    demands = np.concatenate([[0.0], rng.randint(1, 6, size=num_customers).astype(np.float64)])
    return CVRPInstance(coords=coords, demands=demands, capacity=capacity)


def test_subproblem_decomposition():
    instance = create_mock_instance(num_customers=25)
    routes = build_angular_sweep_solution(instance)
    assert len(routes) >= 2

    subproblems = decompose_into_adjacent_subproblems(instance, routes)
    # Number of subproblems should equal number of routes
    assert len(subproblems) == len(routes)

    sub0 = subproblems[0]
    # Local node 0 must be depot
    assert sub0.local_to_orig[0] == 0
    assert sub0.route_membership[0] == 0
    assert sub0.num_nodes > 2
    assert len(sub0.local_routes) == 2


def test_node_features_dimension_and_validity():
    instance = create_mock_instance(num_customers=20)
    routes = build_angular_sweep_solution(instance)
    subproblems = decompose_into_adjacent_subproblems(instance, routes)
    sub = subproblems[0]

    node_feats = extract_subproblem_node_features(sub)

    # Check shape: exactly (N_sub, 25)
    assert isinstance(node_feats, torch.Tensor)
    assert node_feats.shape == (sub.num_nodes, 25)

    # Check for NaNs or Infs
    assert not torch.isnan(node_feats).any()
    assert not torch.isinf(node_feats).any()

    # Check demand normalization: demand / C in [0, 1]
    assert (node_feats[:, 2] >= 0.0).all() and (node_feats[:, 2] <= 1.0).all()


def test_edge_features_dimension_and_validity():
    instance = create_mock_instance(num_customers=20)
    routes = build_angular_sweep_solution(instance)
    subproblems = decompose_into_adjacent_subproblems(instance, routes)
    sub = subproblems[0]

    edge_index, edge_attr = extract_subproblem_edges(sub, k_nearest=15)

    # Check shapes
    assert edge_index.dim() == 2 and edge_index.shape[0] == 2
    assert edge_attr.dim() == 2 and edge_attr.shape[1] == 3
    assert edge_index.shape[1] == edge_attr.shape[0]

    # No NaNs or Infs
    assert not torch.isnan(edge_attr).any()
    assert not torch.isinf(edge_attr).any()

    # In-solution flag should be 0.0 or 1.0
    in_sol = edge_attr[:, 1]
    assert ((in_sol == 0.0) | (in_sol == 1.0)).all()


def test_l2seg_encoder_forward_and_backward():
    """Test full forward pass and gradient backpropagation through L2SegEncoder."""
    instance = create_mock_instance(num_customers=20)
    routes = build_angular_sweep_solution(instance)
    sub = decompose_into_adjacent_subproblems(instance, routes)[0]

    node_feats = extract_subproblem_node_features(sub)
    tour_positions = torch.from_numpy(sub.tour_positions).long()
    route_membership = torch.from_numpy(sub.route_membership).long()
    edge_index, edge_attr = extract_subproblem_edges(sub, k_nearest=15)

    encoder = L2SegEncoder(
        node_in_dim=25,
        edge_in_dim=3,
        hidden_dim=128,
        num_tfm_layers=2,
        num_gnn_layers=2,
        tfm_heads=2,
        gnn_heads=1,
        ffn_dim=512,
        dropout=0.1,
    )

    encoder.train()
    h_gnn = encoder(
        node_feats=node_feats,
        tour_positions=tour_positions,
        route_membership=route_membership,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )

    # Output should have shape (N_sub, hidden_dim)
    assert h_gnn.shape == (sub.num_nodes, 128)
    assert not torch.isnan(h_gnn).any()

    # Verify backpropagation / gradient flow
    dummy_loss = h_gnn.sum()
    dummy_loss.backward()

    # Ensure gradients exist on model parameters
    for name, param in encoder.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"
