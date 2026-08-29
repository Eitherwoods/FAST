# -*- coding: utf-8 -*-
"""帕累托前沿验证脚本：从 p_sensitivity 扫描结果求 (RMS, η) 的帕累托前沿。

目标：RMS 越小越好、接收比 η 越大越好。
判据：若 A.rms <= B.rms 且 A.eta >= B.eta 且至少一者严格不等，则 A 支配 B。
      不被任何其他点支配的点构成帕累托前沿。

数据来源：p_sensitivity_result.txt 的扫描输出（11 个 p 值），
        与正文表 tab:sens 一致（表里只取 9 个用于排版，去掉了 140.55 与 142.00）。
"""
import numpy as np


# (p, RMS_mm, eta) —— 直接从 p_sensitivity_result.txt 读取
SCAN = [
    (140.2000, 148.755, 0.240671),
    (140.3000,  79.586, 0.274375),
    (140.3743,  51.474, 0.341435),
    (140.4000,  54.920, 0.380403),
    (140.4500,  72.983, 0.494246),
    (140.5000,  97.144, 0.449660),
    (140.5500, 121.696, 0.417288),
    (140.6000, 149.124, 0.373572),
    (140.8000, 285.634, 0.101403),
    (141.0000, 459.690, 0.019145),
    (142.0000, 1489.890, 0.053561),
]


def dominates(a, b):
    """a 是否支配 b（a 在 RMS 与 η 两维度都不劣于 b 且至少一维严格更优）。"""
    rms_a, eta_a = a
    rms_b, eta_b = b
    le = rms_a <= rms_b
    ge = eta_a >= eta_b
    strict = (rms_a < rms_b) or (eta_a > eta_b)
    return le and ge and strict


def pareto_front(points):
    """返回帕累托前沿下标列表（按 RMS 升序）。"""
    n = len(points)
    front = []
    for i in range(n):
        metric_i = (points[i][1], points[i][2])  # (RMS, eta)
        dominated = False
        for j in range(n):
            if i == j:
                continue
            metric_j = (points[j][1], points[j][2])
            if dominates(metric_j, metric_i):
                dominated = True
                break
        if not dominated:
            front.append(i)
    front.sort(key=lambda k: points[k][1])  # 按 RMS 升序
    return front


def main():
    print("=" * 78)
    print("帕累托前沿验证（RMS 越小越好，η 越大越好）")
    print("=" * 78)
    print(f"输入点数：{len(SCAN)}")
    print(f"{'p (m)':>10} {'RMS (mm)':>12} {'η':>10} {'是否前沿':>10}")
    print("-" * 78)

    front_idx = pareto_front(SCAN)

    for i, (p, rms, eta) in enumerate(SCAN):
        flag = "✓ 前沿" if i in front_idx else "  劣"
        print(f"{p:>10.4f} {rms:>12.3f} {eta:>10.6f} {flag:>10}")

    print("-" * 78)
    print(f"\n帕累托前沿共 {len(front_idx)} 个点：")
    for i in front_idx:
        p, rms, eta = SCAN[i]
        print(f"  p = {p:.4f} m  →  RMS = {rms:.3f} mm,  η = {eta:.6f}")

    # 验证论文中给出的三点前沿公式 eq:pareto
    print("\n论文 eq:pareto 给出的三点：")
    paper_front = [(51.47, 0.3414), (54.92, 0.3804), (72.98, 0.4942)]
    for rms, eta in paper_front:
        print(f"  RMS = {rms:.2f} mm,  η = {eta:.4f}")

    # 对比
    print("\n一致性检验：")
    code_metrics = [(SCAN[i][1], SCAN[i][2]) for i in front_idx]
    match = all(
        any(abs(c[0] - p[0]) < 0.01 and abs(c[1] - p[1]) < 1e-4
            for c in code_metrics)
        for p in paper_front
    )
    print(f"  代码前沿与论文前沿数值匹配：{'✓ 一致' if match else '✗ 不一致'}")

    # 相对差异检验（验证 6.7%/11.4% 与 32.9%/29.9%）
    print("\n前沿段相对差异检验：")
    pts = code_metrics
    for k in range(len(pts) - 1):
        d_rms = pts[k + 1][0] - pts[k][0]
        d_eta = pts[k + 1][1] - pts[k][1]
        rel_rms = d_rms / pts[k][0] * 100
        rel_eta = d_eta / pts[k][1] * 100
        print(f"  段 {k + 1}: ΔRMS = {d_rms:.2f} mm ({rel_rms:.1f}%), "
              f"Δη = {d_eta:.4f} ({rel_eta:.1f}%)")


if __name__ == "__main__":
    main()
