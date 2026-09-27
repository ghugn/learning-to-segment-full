# Multi-Scale Empirical SOTA Benchmark: CVRP 1k, 2k, 3k (Paper Original Time)

- **Scale Budgets (Paper Original Time)**: 1k = 150.0s (2.5m), 2k = 240.0s (4.0m), 3k = 240.0s (4.0m)
- **Datasets**: `vrp1000_test_seed1234.pkl`, `vrp2000_test_seed1234.pkl`, Synthetic CVRP3k

| Scale | Method | Solution Cost (Obj) | Gap vs HGS (%) | Execution Time | Search Space Reduction |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **CVRP-1000** | **PyVRP (HGS Vidal 2022)** | 39.444 | 0.00% | 150.24s | 0.0% (Full Graph) |
| | **NDS (Hottung et al. 2022)** | 39.390 | -0.14% | 154.97s | 0.0% (Full Graph) |
| | **L2Seg-SYN-LNS (Our FSTA)** | **42.355** | **+7.38%** | **151.85s** | **-79.2%** |
| **CVRP-2000** | **PyVRP (HGS Vidal 2022)** | 54.055 | 0.00% | 240.85s | 0.0% (Full Graph) |
| | **NDS (Hottung et al. 2022)** | 54.150 | +0.18% | 245.22s | 0.0% (Full Graph) |
| | **L2Seg-SYN-LNS (Our FSTA)** | **58.509** | **+8.24%** | **240.10s** | **-86.1%** |
| **CVRP-3000** | **PyVRP (HGS Vidal 2022)** | 65.040 | 0.00% | 241.53s | 0.0% (Full Graph) |
| | **NDS (Hottung et al. 2022)** | - | - | - | OOM / Unsupported Scale |
| | **L2Seg-SYN-LNS (Our FSTA)** | **69.206** | **+6.41%** | **243.12s** | **-86.2%** |
