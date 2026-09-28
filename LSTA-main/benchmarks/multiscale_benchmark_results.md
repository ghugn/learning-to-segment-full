# Bảng Đối Chuẩn SOTA Đa Quy Mô: CVRP 1k & 2k (ICLR 2026 Table 2 Setting)

- **Ngân sách thời gian (Paper Original Budgets)**: 1k = 15.0s (2.5m), 2k = 15.0s (4.0m)
- **Mô hình đề xuất**: `L2Seg-SYN-LNS (Ours)` (FSTA Phân đoạn Nén đồ thị + Backbone LNS thuần túy, không dùng GA bên trong)
- **Tập dữ liệu**: `vrp1000_test_seed1234.pkl`, `vrp2000_test_seed1234.pkl` (Tập test chuẩn NCO)

| Quy mô đề bài | Thuật toán / Mô hình | Chi phí đạt được (Cost ↓) | Chênh lệch Gap vs HGS | Thời gian chạy (Time) | Độ nén không gian (Search Space Reduction) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **CVRP-1000** | **HGS (Vidal 2022)** | **39.444** | 0.00% (Baseline) | 150.19s | 0.0% (Đồ thị đầy đủ) |
| | **LNS (Shaw 1998)** | 42.462 | +7.65% | 15.07s | 0.0% (Đồ thị đầy đủ) |
| | **NDS (Hottung et al. 2022)** | 39.550 | +0.27% | 155.13s | 0.0% (Đồ thị đầy đủ) |
| | **L2Seg-SYN-LNS (Ours)** | **42.601** | **+8.00%** | **15.46s** | **-72.8% (Nén đồ thị!)** |
