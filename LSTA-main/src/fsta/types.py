from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
import numpy as np


@dataclass
class CVRPInstance:
    """
    Representation of a Capacitated Vehicle Routing Problem (CVRP) instance.
    
    Attributes:
        coords: np.ndarray of shape (N + 1, 2) where coords[0] is the depot.
        demands: np.ndarray of shape (N + 1,) where demands[0] == 0.
        capacity: Maximum vehicle load C.
        dist_matrix: Optional precomputed pairwise distance matrix of shape (N + 1, N + 1).
    """
    coords: np.ndarray
    demands: np.ndarray
    capacity: float
    dist_matrix: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.dist_matrix is None:
            diff = self.coords[:, None, :] - self.coords[None, :, :]
            self.dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

    @property
    def num_customers(self) -> int:
        return len(self.coords) - 1

    @property
    def num_nodes(self) -> int:
        return len(self.coords)

    def get_distance(self, u: int, v: int) -> float:
        return float(self.dist_matrix[u, v])


@dataclass
class Segment:
    """
    A stable segment S_{j, k} = (x_j -> ... -> x_k) consisting of consecutive nodes.
    
    Attributes:
        segment_id: Unique identifier for the segment.
        route_id: Original route index.
        nodes: Ordered list of original customer indices [x_j, ..., x_k].
        total_demand: Sum of demands of all customers in this segment.
        internal_cost: Sum of edge distances within the segment (0 if len == 1).
    """
    segment_id: int
    route_id: int
    nodes: List[int]
    total_demand: float
    internal_cost: float

    @property
    def start_node(self) -> int:
        return self.nodes[0]

    @property
    def end_node(self) -> int:
        return self.nodes[-1]

    @property
    def length(self) -> int:
        return len(self.nodes)

    @property
    def is_single_node(self) -> bool:
        return len(self.nodes) == 1


@dataclass
class AggregatedProblem:
    """
    The reduced problem \tilde{P} produced by FSTA hypernode aggregation.
    
    Attributes:
        coords: Coordinates of aggregated nodes of shape (\tilde{N} + 1, 2).
        demands: Demands of aggregated nodes of shape (\tilde{N} + 1,).
        capacity: Original vehicle capacity C.
        dist_matrix: Distance matrix for the aggregated graph (\tilde{N} + 1, \tilde{N} + 1).
        fixed_edges: Pairs of aggregated node indices (\tilde{u}, \tilde{v}) that MUST
                     be traversed consecutively (e.g. dual hypernode edges).
        segments: Mapping from segment_id to Segment object.
        node_to_segment: Mapping from aggregated node index -> (segment_id, 'single' | 'head' | 'tail').
        initial_aggregated_routes: The initial solution \tilde{R} mapped to aggregated indices.
        constant_offset: Fixed cost of internal edges in segments, ensuring:
                         f(R) = f(\tilde{R}) if internal cost is excluded from dist_matrix,
                         or f(R) == f(\tilde{R}) exactly if internal cost is embedded.
    """
    coords: np.ndarray
    demands: np.ndarray
    capacity: float
    dist_matrix: np.ndarray
    fixed_edges: List[Tuple[int, int]]
    segments: Dict[int, Segment]
    node_to_segment: Dict[int, Tuple[int, str]]
    initial_aggregated_routes: List[List[int]]
    constant_offset: float = 0.0

    @property
    def num_nodes(self) -> int:
        return len(self.coords) - 1
