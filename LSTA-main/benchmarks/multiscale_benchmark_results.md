# Multi-Scale Empirical SOTA Benchmark: CVRP 1k, 2k, 3k (Paper Original Time)

- **Scale Budgets (Paper Original Time)**: 1k = 150.0s (2.5m), 2k = 240.0s (4.0m), 3k = 240.0s (4.0m)
- **Datasets**: `vrp1000_test_seed1234.pkl`, `vrp2000_test_seed1234.pkl`, Synthetic CVRP3k

| Scale | Method | Solution Cost (Obj) | Gap vs HGS (%) | Execution Time | Search Space Reduction |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **CVRP-1000** | **PyVRP (HGS Vidal 2022)** | 39.444 | 0.00% | 150.19s | 0.0% (Full Graph) |
| | **NDS (Hottung et al. 2022)** | 39.550 | +0.27% | 155.13s | 0.0% (Full Graph) |
| | **L2Seg-SYN-PYVRP (Our FSTA)** | **40.753** | **+3.32%** | **150.60s** | **-75.1%** |
| **CVRP-2000** | **PyVRP (HGS Vidal 2022)** | 54.057 | 0.00% | 240.80s | 0.0% (Full Graph) |
| | **NDS (Hottung et al. 2022)** | 54.170 | +0.21% | 245.56s | 0.0% (Full Graph) |
| | **L2Seg-SYN-PYVRP (Our FSTA)** | **55.749** | **+3.13%** | **241.61s** | **-80.3%** |
| **CVRP-3000** | **PyVRP (HGS Vidal 2022)** | 65.047 | 0.00% | 241.48s | 0.0% (Full Graph) |
| | **NDS (Hottung et al. 2022)** | - | - | - | OOM / Unsupported Scale |
| | **L2Seg-SYN-PYVRP (Our FSTA)** | **66.899** | **+2.85%** | **240.63s** | **-77.7%** |
