# Learning to Segment for Routing (L2Seg) — Full Benchmark & Research Suite

This repository is the complete, self-contained research and benchmarking suite for **Learning to Segment for Routing (L2Seg)** on Large-Scale Capacitated Vehicle Routing Problems (CVRP).

It unifies the proposed **L2Seg Framework (FSTA + Neural Aggregation + Re-optimization)** with official SOTA neural divide-and-conquer baselines (**NDS**), classical industrial metaheuristics (**PyVRP / Vidal's HGS**), and classic k-opt algorithms (**LKH-3.0.4**). All test datasets, pre-trained neural models, C++ extension modules, and benchmark scripts are packaged together for immediate, out-of-the-box reproduction.

---

## 1. Key Highlights & Findings

1. **Meta-framework Accelerator Paradigm**:
   - Rather than replacing fast heuristic solvers, L2Seg acts as a high-level topological compressor. It coarse-grains large-scale CVRP instances ($N \in [1000, 3000]$) down to **~80–400 hypernodes** using learned segmentation and segment aggregation.
   - When a fast C++ backend (PyVRP / LNS) optimizes the compressed hypergraph, solution quality surpasses running the solver directly on the full graph, while converging **6x to 7.5x faster**.
2. **Empirical Benchmark Results (CVRP 1k, 2k, 3k)**:
   - **CVRP-1000**: `L2Seg-SYN-PYVRP` reaches **40.718** in **25.2s** (beating official NDS at 41.16 and PyVRP at 41.20).
   - **CVRP-2000**: `L2Seg-SYN-PYVRP` reaches **55.862** in **32.8s** (beating official NDS at 56.11 and PyVRP at 57.20).
   - **CVRP-3000**: Neural attention baselines (NDS) encounter Out-Of-Memory ($O(N^2)$ VRAM crash). `L2Seg-SYN-PYVRP` compresses the graph to 414 hypernodes and solves it in **35.5 seconds** (Cost: **67.292**).
3. **Engineering Completeness**:
   - **20/20 Unit Tests Passing**: Full verification across data loaders, feature extractors, non-autoregressive/autoregressive decoders, FSTA aggregation, recovery guarantees, and solvers.
   - **Native Windows & Linux Compatibility**: Fixed MSVC Windows compilation constraints for `NDSOps.cpp` and `LKH.exe` (with pre-compiled binaries included).

---

## 2. Repository Layout

```
learning-to-segment-full/
├── LSTA-main/                  # Central L2Seg project
│   ├── src/                    # Core algorithm implementation (FSTA, Models, Solvers, Features)
│   ├── checkpoints/            # Pre-trained neural weights (nar_model.pt, ar_model.pt)
│   ├── benchmarks/             # Benchmark runners (Head-to-Head & Multi-Scale suites)
│   ├── configs/                # Hyperparameter configurations
│   ├── tests/                  # Complete test suite (20/20 unit tests)
│   ├── solver.py               # Standardized command-line solver interface
│   ├── requirements.txt        # Python package dependencies
│   ├── README.md               # Detailed technical documentation
│   └── README_Progress.md      # Chronological engineering log & 9 core findings
├── NDS/                        # Neural Divide-and-Search (Hottung et al., 2022/2025)
│   ├── data/cvrp/              # Standard test sets (vrp1000_test_seed1234.pkl, vrp2000...)
│   ├── models/                 # Pretrained NDS checkpoints
│   ├── src/                    # Python and C++ source (NDSOps with Windows fix)
│   └── eval.py                 # NDS official evaluator
├── learning-to-delegate/       # Learning to Delegate baseline (MIT Wu Lab, Li et al., 2021)
│   ├── lkh3/LKH-3.0.4/         # Source code + pre-compiled LKH.exe for Windows
│   ├── run_lkh.py              # LKH-3 benchmark harness
│   └── util.py                 # Patched utility for Windows path/TEMP compatibility
├── hgs/                        # Hybrid Genetic Search C++ source (Vidal, 2022)
│   └── HGS-CVRP/               # Original C++ source repository
├── docs/                       # Research papers and technical documentation
│   ├── 15800_Learning_to_Segment_for_.pdf   # "Learning to Segment for Routing" paper
│   ├── 28447_Can_Computational_Reduci.pdf   # NDS paper
│   └── Bridging Reducibility and Scalability.docx # Lab proposal & technical analysis
└── README.md                   # This root guide
```

---

## 3. Quick Start & Setup

### 3.1 Python Environment

Recommended Python version: `3.10` to `3.14`.

```bash
# Clone the complete repository
git clone https://github.com/ghugn/learning-to-segment-full.git
cd learning-to-segment-full

# Install Python dependencies
pip install -r LSTA-main/requirements.txt
pip install pyvrp pytest
```

### 3.2 Pre-compiled C++ Binaries

- **LKH-3**: `learning-to-delegate/lkh3/LKH-3.0.4/LKH.exe` (Pre-compiled for 64-bit Windows with MSVC).
- **NDS C++ Extension**: `NDS/src/cpp/cvrp/NDSOps.cp314-win_amd64.pyd` (Pre-compiled for Python 3.14 on Windows).
- *On Linux/macOS*: Standard build tools (`make` and `cppimport`) will automatically compile binaries on first execution.

---

## 4. Ready-to-Run Commands

All operational commands can be executed directly from inside `LSTA-main/`:

```bash
cd LSTA-main
```

### 4.1 Run Unit Tests (20/20 PASS)

Verify full system correctness (data pipelines, neural decoders, aggregation, recovery, and solvers):

```bash
pytest tests/ -v
```

### 4.2 Run CLI Solver (`solver.py`)

Solve an instance with the standard command-line interface:

```bash
# Solve with PyVRP backbone (Fastest SOTA)
python solver.py --dataset ../NDS/data/cvrp/vrp1000_test_seed1234.pkl --instance_idx 0 --time_limit 15.0 --backbone pyvrp

# Solve with Large Neighborhood Search (LNS) backbone
python solver.py --dataset ../NDS/data/cvrp/vrp1000_test_seed1234.pkl --instance_idx 0 --time_limit 15.0 --backbone lns
```

### 4.3 Live Head-to-Head Benchmark (CVRP-1000)

Run a direct side-by-side execution of **PyVRP**, **NDS**, and **L2Seg-SYN-LNS** on the identical test instance:

```bash
python benchmarks/run_live_nds_pyvrp_l2seg.py --instances 1 --time_limit 15.0
```

*Results will be logged directly to console and saved in `benchmarks/live_head_to_head_results.md`.*

### 4.4 Multi-Scale Empirical SOTA Benchmark (CVRP 1k, 2k, 3k)

Run the full multi-scale evaluation reproducing Table 2 scale conditions:

```bash
# Run with PyVRP backbone
python benchmarks/run_multiscale_benchmark.py --backbone pyvrp

# Or run with custom time limit (e.g., 20s per scale)
python benchmarks/run_multiscale_benchmark.py --time_limit 20.0 --backbone pyvrp
```

---

## 5. Benchmark Summary Table

### Multi-Scale Comparison (Original Paper Budgets: 2.5m - 4.0m)

| Scale | Method | Solution Cost (Obj) | Gap vs HGS (%) | Execution Time | Search Space Reduction |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **CVRP-1000** | **PyVRP (HGS Vidal 2022)** | 41.200 | 0.00% | 150.0s (2.5m) | 0.0% (Full Graph) |
| | **NDS (Hottung et al. 2022)** | 41.160 | -0.10% | 150.0s (2.5m) | 0.0% (Full Graph) |
| | **L2Seg-SYN-PYVRP (Ours)** | **40.718** | **-1.17%** | **25.2s** | **-80.4%** |
| **CVRP-2000** | **PyVRP (HGS Vidal 2022)** | 57.200 | 0.00% | 240.0s (4.0m) | 0.0% (Full Graph) |
| | **NDS (Hottung et al. 2022)** | 56.110 | -1.91% | 240.0s (4.0m) | 0.0% (Full Graph) |
| | **L2Seg-SYN-PYVRP (Ours)** | **55.862** | **-2.34%** | **32.8s** | **-83.6%** |
| **CVRP-3000** | **PyVRP (HGS Vidal 2022)** | 71.450 | 0.00% | 240.0s (4.0m) | 0.0% (Full Graph) |
| | **NDS (Hottung et al. 2022)** | OOM (-) | - | Crash | Unsupported Scale ($O(N^2)$ VRAM) |
| | **L2Seg-SYN-PYVRP (Ours)** | **67.292** | **-5.82%** | **35.5s** | **-86.2%** |

---

## 6. Citation & References

- **Learning to Segment for Routing (L2Seg)**: [Paper in docs/](docs/15800_Learning_to_Segment_for_.pdf)
- **Neural Divide-and-Search (NDS)**: Hottung et al., AAAI 2022 / arXiv 2025. [Paper in docs/](docs/28447_Can_Computational_Reduci.pdf)
- **Hybrid Genetic Search (HGS / PyVRP)**: Vidal, Computers & Operations Research, 2022.
- **LKH-3**: Keld Helsgaun, Roskilde University, 2017.
- **Learning to Delegate (L2D)**: Li et al., MIT Wu Lab, NeurIPS 2021.
