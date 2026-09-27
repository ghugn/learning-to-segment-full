"""
Solvers package for L2Seg + FSTA.
Contains:
- LNSSolver: Large Neighborhood Search (Shaw, 1998)
- PyVRPSolver: Hybrid Genetic Search (Vidal, 2022 via PyVRP)
- L2SegIterativeSolver: Algorithm 1 from ICLR 2026 paper
"""

from .lns import LNSSolver
from .pyvrp_solver import PyVRPSolver
from .l2seg_iterative_solver import L2SegIterativeSolver

__all__ = ["LNSSolver", "PyVRPSolver", "L2SegIterativeSolver"]
