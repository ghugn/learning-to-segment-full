# Live Empirical Head-to-Head Benchmark on CVRP-1000

- **Dataset**: `vrp1000_test_seed1234.pkl` (exact test set from NDS / Kool et al.)
- **Evaluated Instances**: 1
- **Time Budget per Instance**: 20.0s

| Method | Implementation | Cost (Obj) | Gap vs HGS (%) | Execution Time | Graph Search Space Reduction |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **PyVRP (HGS Vidal 2022)** | Official C++ Engine | **39.886** | **0.00%** | 23.35s | 0.0% (Full Graph) |
| **NDS (Hottung et al. 2022)** | Official C++ / PyTorch Repo | **40.150** | **+0.66%** | 24.30s | 0.0% (Full Graph) |
| **L2Seg-SYN-LNS** | L2Seg AI + FSTA + Focused LNS | **42.480** | **+6.50%** | 20.04s | **-78.5%** |
