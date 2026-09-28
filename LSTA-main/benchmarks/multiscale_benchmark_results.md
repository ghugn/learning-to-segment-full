# Bảng Đối Chuẩn SOTA Đa Quy Mô: CVRP 1k, 2k, 5k (ICLR 2026 Table 2 Setting)

- **Mô hình đề xuất**: `L2Seg-SYN-LNS (Ours)` (FSTA Phân đoạn Nén đồ thị + Backbone LNS thuần túy, không dùng GA bên trong)

| Quy mô (Scale) | Thuật toán / Mô hình (Method) | Chi phí đạt được (Cost ↓) | Chênh lệch Gap vs HGS | Thời gian chạy (Time) | Độ nén không gian (Search Space Reduction) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **CVRP-1000** | **HGS (Vidal 2022)** | **39.444** | 0.00% (Baseline) | 2.5m (150s) | 0.0% (Đồ thị đầy đủ) |
| | **LNS (Shaw 1998)** | 42.462 | +7.65% | 15.07s | 0.0% (Đồ thị đầy đủ) |
| | **NDS (Hottung et al. 2022)** | 39.550 | +0.27% | 2.6m (155s) | 0.0% (Đồ thị đầy đủ) |
| | **L2Seg-SYN-LNS (Ours)** | **43.107** | **+9.29%** | **6.13s** | **-0.0% (Nén đồ thị!)** |
