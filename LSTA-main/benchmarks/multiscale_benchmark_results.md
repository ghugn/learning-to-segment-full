# Bảng Kết Quả Đánh Giá Đa Quy Mô: CVRP 1k, 2k, 3k (Accelerated Time-to-Quality Benchmark)

- **Ngân sách PyVRP/NDS Baseline**: 1k = 150.0s (2.5m), 2k = 240.0s (4.0m), 3k = 240.0s (4.0m)
- **Ngân sách L2Seg Tăng Tốc (Ours)**: 1k = 30.0s (5x Speedup), 2k = 60.0s (4x Speedup), 3k = 60.0s (4x Speedup)
- **Tập dữ liệu**: `vrp1000_test_seed1234.pkl`, `vrp2000_test_seed1234.pkl`, Synthetic CVRP3k

| Quy mô đề bài | Thuật toán / Mô hình | Chi phí đạt được (Cost ↓) | Chênh lệch Gap vs HGS | Thời gian chạy (Time) | Độ nén không gian (Search Space Reduction) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **CVRP-1000** | **PyVRP (HGS Vidal 2022)** | **39.444** | 0.00% (Baseline) | 150.19s | 0.0% (Đồ thị đầy đủ) |
| | **NDS (Hottung et al. 2022)** | 39.550 | +0.27% | 155.13s | 0.0% (Đồ thị đầy đủ) |
| | **L2Seg-SYN-PYVRP (Ours)** | **40.644** | **+3.04%** | **30.99s (Nhanh gấp 4.8x!)** | **-72.5% (Nén đồ thị!)** |
| **CVRP-2000** | **PyVRP (HGS Vidal 2022)** | **54.057** | 0.00% (Baseline) | 240.80s | 0.0% (Đồ thị đầy đủ) |
| | **NDS (Hottung et al. 2022)** | 54.170 | +0.21% | 245.56s | 0.0% (Đồ thị đầy đủ) |
| | **L2Seg-SYN-PYVRP (Ours)** | **55.886** | **+3.38%** | **60.92s (Nhanh gấp 4.0x!)** | **-81.1% (Nén đồ thị!)** |
| **CVRP-3000** | **PyVRP (HGS Vidal 2022)** | **65.047** | 0.00% (Baseline) | 241.48s | 0.0% (Đồ thị đầy đủ) |
| | **NDS (Hottung et al. 2022)** | Bị sập OOM (-) | — | — | OOM / Tràn VRAM ($O(N^2)$) |
| | **L2Seg-SYN-PYVRP (Ours)** | **67.547** | **+3.84%** | **63.96s (Nhanh gấp 3.8x!)** | **-77.8% (Nén đồ thị!)** |
