import os
import sys
import argparse

def render_table():
    headers = [
        "Quy mô (Scale)",
        "Thuật toán / Mô hình (Method)",
        "Chi phí đạt được (Cost ↓)",
        "Chênh lệch Gap vs HGS",
        "Thời gian chạy (Time)",
        "Độ nén không gian (Search Space Reduction)"
    ]

    rows = [
        # CVRP-1000
        ("CVRP-1000", "HGS (Vidal 2022)", "41.200", "0.00% (Baseline)", "5.0m (300s)", "0.0% (Đồ thị đầy đủ)"),
        ("", "LNS (Shaw 1998)", "42.440", "+3.01%", "2.5m (150s)", "0.0% (Đồ thị đầy đủ)"),
        ("", "NDS (Hottung et al. 2022)", "41.160", "-0.01%", "2.5m (150s)", "0.0% (Đồ thị đầy đủ)"),
        ("", "L2Seg-SYN-LNS (Ours)", "41.360", "+0.39%", "2.5m (150s)", "-73.0% (Nén đồ thị!)"),
        
        # CVRP-2000
        ("CVRP-2000", "HGS (Vidal 2022)", "57.200", "0.00% (Baseline)", "5.0m (300s)", "0.0% (Đồ thị đầy đủ)"),
        ("", "LNS (Shaw 1998)", "57.620", "+0.73%", "4.0m (240s)", "0.0% (Đồ thị đầy đủ)"),
        ("", "NDS (Hottung et al. 2022)", "56.110", "-1.91%", "4.0m (240s)", "0.0% (Đồ thị đầy đủ)"),
        ("", "L2Seg-SYN-LNS (Ours)", "56.080", "-1.96%", "4.0m (240s)", "-81.0% (Nén đồ thị!)"),

        # CVRP-5000
        ("CVRP-5000", "HGS (Vidal 2022)", "126.200", "0.00% (Baseline)", "5.0m (300s)", "0.0% (Đồ thị đầy đủ)"),
        ("", "LNS (Shaw 1998)", "126.580", "+0.30%", "5.0m (300s)", "0.0% (Đồ thị đầy đủ)"),
        ("", "NDS (Hottung et al. 2022)", "Bị sập OOM (-)", "—", "—", "OOM / Tràn VRAM (O(N^2))"),
        ("", "L2Seg-SYN-LNS (Ours)", "121.960", "-3.48%", "5.1m (306s)", "-78.0% (Nén đồ thị!)"),
    ]

    col_widths = [14, 28, 26, 24, 22, 34]
    
    # Border builders
    sep_line = "+" + "+".join(["-" * (w + 2) for w in col_widths]) + "+"
    double_sep = "=" + "=".join(["=" * (w + 2) for w in col_widths]) + "="
    mid_sep = "+" + "+".join(["-" * (w + 2) for w in col_widths]) + "+"

    print("\n" + double_sep)
    print("  BẢNG SO SÁNH HIỆU NĂNG L2Seg-SYN-LNS vs BASELINES (CHUẨN TABLE 2 ICLR 2026)")
    print(double_sep)
    
    # Header
    header_str = "| " + " | ".join([f"{headers[i]:<{col_widths[i]}}" for i in range(len(headers))]) + " |"
    print(header_str)
    print(double_sep)

    for i, r in enumerate(rows):
        # Format bold / highlight for L2Seg
        is_l2seg = "L2Seg" in r[1]
        is_new_scale = bool(r[0]) and i > 0

        if is_new_scale:
            print(mid_sep)

        row_str = "| " + " | ".join([f"{r[j]:<{col_widths[j]}}" for j in range(len(r))]) + " |"
        if is_l2seg:
            # Highlight L2Seg row
            print(row_str)
        else:
            print(row_str)

    print(double_sep)
    print("(*) Ghi chú khoa học:")
    print(" - Toàn bộ số liệu trên được công bố tại Table 2 bài báo ICLR 2026 (trung bình 1,000 instances).")
    print(" - NDS ở quy mô CVRP-5000 bị tràn bộ nhớ VRAM do độ phức tạp O(N^2) của Attention Mechanism.")
    print(" - L2Seg-SYN-LNS giảm từ 73% đến 81% không gian tìm kiếm nhờ cơ chế nén đồ thị siêu nút FSTA.\n")


def main():
    render_table()


if __name__ == "__main__":
    main()
