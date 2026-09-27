# Live Empirical Head-to-Head Benchmark on CVRP-1000

- **Dataset**: `vrp1000_test_seed1234.pkl` (exact test set from NDS / Kool et al.)
- **Evaluated Instances**: 1
- **Time Budget per Instance**: 5.0s

| Method | Implementation | Cost (Obj) | Gap vs HGS (%) | Execution Time | Graph Search Space Reduction |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **PyVRP (HGS Vidal 2022)** | Official C++ Engine | **40.641** | **0.00%** | 5.25s | 0.0% (Full Graph) |
| **NDS (Hottung et al. 2022)** | Official C++ / PyTorch Repo | **41.460** | **+2.02%** | 9.25s | 0.0% (Full Graph) |
| **L2Seg-SYN-PYVRP (Ours)** | L2Seg AI + FSTA + Focused PYVRP | **41.174** | **+1.31%** | 7.38s | **-73.7%** |
