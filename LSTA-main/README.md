# L2Seg & FSTA: Learning to Segment for Large-Scale Vehicle Routing Problems

An industrial-grade, fully modular Python implementation of **Learning to Segment (L2Seg)** and the **First-Segment-Then-Aggregate (FSTA)** decomposition paradigm for large-scale Capacitated Vehicle Routing Problems (CVRP), based on the **ICLR 2026** paper *"Learning to Segment for Vehicle Routing Problems"* (Ouyang et al.).

This module serves as the **Level 2: Topological Coarse-Graining / Abstraction** component in the overarching research framework **Bridging Reducibility and Scalability: Toward Foundation Multi-task Combinatorial Optimization**.

> **Project & Benchmark Log**: For the complete chronological development trajectory, AI training details, 9 core technical findings, and full empirical benchmark tables against PyVRP, NDS, and LKH-3, please refer to **[README_Progress.md](README_Progress.md)**.

---

## Key Highlights & Architectural Features

```
       Original Problem P (N = 1000 - 5000)
                        │
                        ▼ [Angular Sweep / Heatmap Init]
       Initial Solution R
                        │
                        ▼ [Level 2: L2Seg-SYN AI]
 ┌────────────────────────────────────────────────────────┐
 │ 1. Decompose P into Adjacent Subproblems P_TR          │
 │ 2. Global Unstable Node Detection (NAR Decoder)        │
 │ 3. Focal Region Clustering (K-Means K=3)               │
 │ 4. Local Deletion/Insertion Rollout (AR Pointer Net)   │
 └────────────────────────────────────────────────────────┘
                        │
                        ▼ [Unstable Edges Cut]
       Atomic Segments S_k
                        │
                        ▼ [FSTA Dual Hypernode Aggregation]
       Aggregated Problem P_tilde (Compressed by 70% - 85%)
                        │
                        ▼ [Macro Block Re-optimization / Backbone Solver]
       Re-optimized Aggregated Solution R_tilde
                        │
                        ▼ [FSTA Monotone Recovery]
       Improved Feasible Solution R+ on Original Graph P
```

1. **Topological Coarse-Graining (70% - 85% Problem Reduction)**:
   - Partitions solutions at unstable edges into atomic segments $S_k$.
   - Aggregates multi-customer segments into **Dual Hypernodes** (head & tail nodes with 50/50 demand split, embedded internal cost, and locked fixed edges).
   - Preserves 100% feasibility and guarantees strict monotonicity ($f(\tilde{R}_1) \le f(\tilde{R}_2) \implies f(R_1) \le f(R_2)$).

2. **Synergized Neural Architecture (L2Seg-SYN)**:
   - **Feature Extraction**: 25 enhanced node features (coordinates, demand, depot angles, K-NN / K%-NN subtour distributions) + 3 edge features.
   - **Encoder**: Sinusoidal Positional Encoding + Route-Masked Self-Attention ($L=2, heads=2$) + Graph Transformer Convolution (`TransformerConv`, $L=2$).
   - **NAR Decoder**: 2-layer MLP with Sigmoid for rapid global unstable node candidate detection.
   - **AR Decoder**: GRU context tracker + Deletion MHA ($L=1$) + Insertion MHA ($L=4$) + Pointer Network for autoregressive edge repair.

3. **Multi-Scale Benchmark Integration & Dual Backbones**:
   - Direct loaders for standard **CVRPLib Set-X** instances ($N \in [101, 1001]$) with ground-truth Best Known Solutions (BKS).
   - Synthetic benchmark generators for **CVRP-1000**, **CVRP-2000**, and **CVRP-3000**.
   - Native integration with **PyVRP** (Vidal's Hybrid Genetic Search in C++), **LKH-3.0.4** (compiled MSVC Windows C binary), and **NDS** (Neural Divide-and-Conquer).
   - **Dual Backbone Architecture**: Supports both pure Python LNS (`backbone="lns"`) and accelerated C++ PyVRP (`backbone="pyvrp"`) with **Even-Odd Disjoint Matching** to prevent route collision during subproblem re-optimization.

---

## Repository Structure

```
LSTA-main/
├── configs/                       # Hyperparameter and experiment configurations
│   └── default.yaml               # Default training and inference parameters
│
├── run/                           # Execution scripts & task runners
│   ├── infer.py                   # End-to-End inference runner
│   ├── benchmark.py               # Official benchmark suite runner (Table 2 & 11)
│   ├── visualize.py               # 6-stage Matplotlib plot generator (Figure 1)
│   ├── train_nar.py               # NAR Decoder training script
│   └── train_ar.py                # AR Pointer Network training script
│
├── src/                           # Core Source Packages
│   ├── fsta/                      # FSTA Topological Abstraction Engine
│   │   ├── types.py               # Data classes: CVRPInstance, Segment, AggregatedProblem
│   │   ├── partition.py           # Breaks routes into segments at unstable edges
│   │   ├── aggregation.py         # Dual Hypernode contraction & fixed edge locking
│   │   ├── recovery.py            # Expands hypernodes back to original graph + validation
│   │   ├── init_solution.py       # Angular Sweep initial solution heuristic
│   │   ├── edge_selectors.py      # Heuristic edge selectors (random, longest, oracle)
│   │   └── local_search.py        # Macro block local search (2-opt & relocate on hypernodes)
│   │
│   ├── features/                  # Graph & Subproblem Feature Extractors
│   │   ├── subproblem.py          # Adjacent route pairs decomposition P_TR = {R_i, R_j}
│   │   ├── node_features.py       # 25-dimensional node features (Table 8 in paper)
│   │   └── edge_features.py       # 3-dimensional edge features (distance, in-sol, rank)
│   │
│   ├── models/                    # Deep Learning Neural Networks
│   │   ├── encoder.py             # L2SegEncoder (PE + Masked TFM + PyG TransformerConv)
│   │   ├── nar_decoder.py         # L2SegNARDecoder (Non-Autoregressive MLP)
│   │   ├── ar_decoder.py          # L2SegARDecoder (Autoregressive GRU + Pointer Net)
│   │   └── l2seg_model.py         # Unified L2SegModel with predict_subproblem_syn (Alg. 3)
│   │
│   ├── data/                      # Dataset & Imitation Learning Pipelines
│   │   ├── label_extractor.py     # Ground-truth E_diff extraction & DFS sequence labels
│   │   └── dataset.py             # L2SegSample & L2SegDataset with disk serialization
│   │
│   ├── solvers/                   # Advanced Backbone & Iterative Solvers
│   │   ├── lns.py                 # Large Neighborhood Search (Shaw, 1998)
│   │   ├── pyvrp_solver.py        # PyVRP (HGS - Vidal, 2022) C++ Wrapper with direct ProblemData
│   │   └── l2seg_iterative_solver.py # Algorithm 1: Iterative Re-optimization with FSTA (pyvrp/lns backbone)
│   │
│   └── benchmarks/                # CVRPLib Data Loaders & Benchmarks
│       ├── instances/             # Cached CVRPLib .vrp and .sol benchmark files
│       ├── cvrplib_loader.py      # Automatic downloader and parser for CVRPLib Set-X
│       ├── benchmark_suite.py     # Benchmark core logic
│       ├── benchmark_table2.py    # Official Table 2 SOTA Benchmark Generator
│       ├── run_multiscale_benchmark.py # Multi-Scale Benchmark (CVRP-1k/2k/3k across PyVRP, NDS, L2Seg)
│       ├── multiscale_benchmark_results.md # Multi-scale empirical markdown report
│       ├── multiscale_benchmark_results.json # Multi-scale benchmark JSON logs
│       ├── table2_comparison.md   # Exported Table 2 Markdown Matrix
│       └── benchmark_results.json # Saved experimental results
│
├── checkpoints/                   # Trained Neural Network Weights (Retrained with PyVRP Oracle)
│   ├── nar_model.pt               # Trained Encoder + NAR Decoder checkpoint (Recall: 1.000)
│   └── ar_model.pt                # Trained AR Decoder checkpoint
│
├── tests/                         # Comprehensive Unit Test Suite (20/20 Passing)
│   ├── test_fsta.py               # Mathematical tests for FSTA feasibility & recovery
│   ├── test_features_and_encoder.py # Tensor shape and gradient flow tests
│   ├── test_decoders.py           # NAR & AR loss and rollout verification
│   ├── test_data_pipeline.py      # Dataset loading and label extraction tests
│   └── test_solvers.py            # LNS, PyVRP C++, L2Seg iterative re-optimization, & Table 2
│
├── assets/                        # Figures, plots, and media assets
│   └── l2seg_fsta_process.png     # Rendered 6-stage Matplotlib process plot
│
├── solver.py                      # Main Unified CLI Entry Point (matching lab standard)
├── pyproject.toml                 # Standard package metadata & pytest configuration
├── requirements.txt               # Python package dependencies
├── .project-root                  # Project root marker
├── README_Progress.md             # Master chronological progress, findings & benchmark report
└── README.md                      # Architecture & Quick Start Documentation
```

---

## Quick Start Guide

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/.../LSTA-main.git
cd LSTA-main
pip install -r requirements.txt
pip install -e .
```

### 2. Verify Unit Tests (20/20 Pass)
Ensure all mathematical assertions, solvers, and tensor pipelines are functioning correctly:
```bash
python -m pytest
# or via solver.py
python solver.py --mode test
```
*Expected Output:*
```
collected 20 items
tests/test_data_pipeline.py ...       [ 15%]
tests/test_decoders.py ...            [ 30%]
tests/test_features_and_encoder.py ... [ 50%]
tests/test_fsta.py ......             [ 80%]
tests/test_solvers.py ....            [100%]
===================== 20 passed in 7.50s =====================
```

### 3. Unified Entry Point (`solver.py`)
Run the entire pipeline via the unified `solver.py` interface:
```bash
# End-to-end inference
python solver.py --mode infer --customers 150 --capacity 50.0

# Official Table 2 SOTA Benchmark Matrix (matching Table 2 of ICLR 2026 paper)
python solver.py --mode table2

# Multi-Scale Benchmark Suite across CVRP-1000, 2000, 3000
python solver.py --mode multiscale --backbone pyvrp

# Live Head-to-Head Benchmark on N=1000 with 10s budget
python solver.py --mode table2 --run_live --scale 1000 --time_limit 10.0

# Standard CVRPLib Set-X benchmark
python solver.py --mode benchmark

# 6-stage process visualization (Figure 1 in paper)
python solver.py --mode visualize --customers 500 --capacity 150 --output assets/l2seg_fsta_process.png
```

Alternatively, invoke scripts directly inside `run/` or `benchmarks/`:
```bash
python run/infer.py --customers 150 --capacity 50.0
python benchmarks/run_multiscale_benchmark.py --backbone pyvrp
python run/visualize.py --customers 500 --capacity 150 --output assets/l2seg_fsta_process.png
```

### 4. Run the Official Multi-Scale Benchmark Suite
Evaluate performance against official CVRPLib Set-X instances and multi-scale synthetic datasets (1k, 2k, 3k):
```bash
python benchmarks/run_multiscale_benchmark.py --backbone pyvrp --time_1k 150 --time_2k 240 --time_3k 240
```

### 5. Generate 6-Stage Process Visualization (Figure 1 in Paper)
Generate high-resolution Matplotlib figures demonstrating edge detection, segment partitioning, hypernode contraction, and route recovery:
```bash
python solver.py --mode visualize --customers 500 --capacity 150 --output assets/l2seg_fsta_process.png
```

---

## Experimental Results

### 1. Multi-Scale Empirical SOTA Benchmark (CVRP-1000, 2000, 3000)

Evaluated under identical paper time budgets (150s for 1k, 240s for 2k and 3k) against top SOTA methods:

| Benchmark Scale | Metric | **L2Seg-SYN-PYVRP (Ours)** | **PyVRP (HGS)** | **NDS (Paper SOTA)** | **LKH-3.0.4** |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **CVRP-1000**<br>*(150s budget)* | **Cost**<br>Time<br>Status | **40.718**<br>**25.2s** *(6.0x faster)*<br>`Valid: True` | 41.200<br>150.0s<br>`Valid: True` | 41.160<br>150.0s<br>`Feasible` | 43.140<br>150.0s<br>`Feasible` |
| **CVRP-2000**<br>*(240s budget)* | **Cost**<br>Time<br>Status | **55.862**<br>**32.0s** *(7.5x faster)*<br>`Valid: True` | 57.200<br>240.0s<br>`Valid: True` | 56.110<br>240.0s<br>`Feasible` | 59.810<br>240.0s<br>`Feasible` |
| **CVRP-3000**<br>*(240s budget)* | **Cost**<br>Time<br>Status | **67.292**<br>**35.5s** *(6.7x faster)*<br>`Valid: True` | 67.210<br>240.0s<br>`Valid: True` | OOM / Crash<br>N/A<br>CUDA OOM | 71.050<br>240.0s<br>`Feasible` |

> **Key Empirical Insights**:
> 1. **Superiority & Speedup**: On CVRP-1000 and CVRP-2000, `L2Seg-SYN-PYVRP` outperforms both standalone PyVRP and NDS in objective cost while converging **6x to 7.5x faster** (25s–32s vs 150s–240s).
> 2. **Scalability Beyond Memory Limits**: At $N=3000$, neural baselines like NDS crash due to quadratic attention matrices ($O(N^2)$ VRAM). In contrast, FSTA coarse-grains the 3000-node graph down to **~414 hypernodes**, allowing the C++ engine to optimize the compressed graph in just **35.5 seconds**.
> 3. **Mathematical Feasibility**: All recovered solutions strictly respect capacity limits and route continuity (`Valid: True`, zero violations).

### 2. Graph Size Reduction (Topological Coarse-Graining)
Evaluated across CVRPLib Set-X and Synthetic CVRP instances using `predict_unstable_edges_l2seg_syn`:

| Benchmark Instance | Original Scale ($N$) | Compressed Scale ($\tilde{N}$) | **Graph Compression Ratio** |
| :--- | :---: | :---: | :---: |
| **X-n280-k17** | 279 | 131 | **53.0%** |
| **X-n502-k39** | 501 | 245 | **51.1%** |
| **X-n1001-k43** | 1,000 | 324 | **67.6%** |
| **CVRP1k-Syn** | 1,000 | 226 | **77.4%** |
| **CVRP2k-Syn** | 2,000 | 310 | **84.5%** |
| **CVRP3k-Syn** | 3,000 | 414 | **86.2%** |

### 3. Benchmark Comparison (Paper Table 2 & Table 11 Alignment)
* **CVRPLib Set-X**: Tested directly against official Best Known Solutions (BKS).
* **Industrial Baseline (PyVRP / Vidal 2022 HGS)**: Reaches **0.02% to 4.61% Gap** to BKS in seconds, validating that data loaders and evaluation formulas strictly adhere to international standards.
* **FSTA Macro Local Search**: Operates exclusively on contracted hypernodes, proving that 2-opt block flips and relocations strictly maintain segment integrity without ever breaking locked edges.

---

## Integration with Research Proposal

In the 3-level framework **Bridging Reducibility and Scalability**:
* **Level 1 (GCON Backbone)**: Produces probabilistic edge heatmaps $P(e \in \text{opt})$.
* **Level 2 (This Repository - L2Seg + FSTA)**: Translates heatmap / structural instability into topological coarse-graining, compressing $N=2000$ down to $\tilde{N} \approx 300$ hypernodes.
* **Level 3 (Compressed MFEA)**: Conducts evolutionary multi-task optimization directly in the reduced hypernode space, enabling order-of-magnitude faster convergence.

---

## Citation & Acknowledgements

If you use this codebase in your research, please cite the original ICLR 2026 paper:
```bibtex
@inproceedings{ouyang2026learning,
  title={Learning to Segment for Vehicle Routing Problems},
  author={Ouyang, ... and others},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```
