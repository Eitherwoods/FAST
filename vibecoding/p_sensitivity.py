# -*- coding: utf-8 -*-
"""焦距 p 的敏感性分析：拟合残差与接收比并不在同一处取优。

第2问以“反射面尽量贴近理想抛物面”为准则时，最优焦距 p* 使节点残差 RMS 最小；
但接收比只惩罚面板的斜率（tipt/tilt）误差，对整体平移（活塞）几乎不敏感，
故接收比的最优 p 与残差的最优 p 并不重合。本脚本扫描 p，同时给出
    · 节点残差 RMS（贴近程度）
    · 促动器最大伸缩量、边长最大相对变化（可行性）
    · 馈源舱接收比（接收效果）
供论文“模型检验/敏感性分析”一节直接引用。

用法：python p_sensitivity.py
"""
import sys
import time

import numpy as np
from pathlib import Path

import common as cm
import q3
from common import ALPHA_Q2, BETA_Q2

SAMPLE_M = 20                                  # 接收比采样阶数
P_LIST = (140.20, 140.30, 140.3743, 140.40, 140.45, 140.50, 140.55,
          140.60, 140.80, 141.00, 142.00)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    names, pos, panels = cm.load_data()
    n = cm.axis_vector(ALPHA_Q2, BETA_Q2)
    ctx = cm.build_context(pos, panels, n)
    R = ctx["R"]
    focus = -0.534 * R * n
    # 与照明区相交的面板（边缘跨界面板按采样点逐点截断）
    ap_panels = ctx["in_ap"][panels].any(axis=1)
    bary = cm.triangle_bary(SAMPLE_M)
    verts_base = pos[panels[ap_panels]]
    centers_base = np.zeros((ap_panels.sum(), 3))
    eta_base, _ = q3.trace_ratio(verts_base, centers_base, R, n, focus, bary)

    print(f"基准球面半径 R = {R:.4f} m，口径内节点 {ctx['nap']} 个，"
          f"参与追迹面板 {int(ap_panels.sum())} 块")
    print(f"基准反射球面接收比 = {eta_base:.6f}")
    print()
    head = (f"{'焦距 p (m)':>11} {'理想偏差(中心,m)':>16} {'残差RMS(mm)':>12} "
            f"{'|x|最大(mm)':>11} {'边长变化(%)':>12} {'接收比':>9}")
    rows, t_all = [], time.time()
    print(head)
    print("-" * len(head))

    for p in P_LIST:
        t0 = time.time()
        e_ap, _, _ = cm.radial_extensions(pos[ctx["in_ap"]], n, p)
        x = cm.fit_best(ctx, e_ap)
        rms = float(np.sqrt(np.mean((x - e_ap) ** 2)))
        rel = float(cm.max_rel_change(x, ctx).max())
        pos_adj = cm.adjust_nodes(pos, x, ctx)
        verts = pos_adj[panels[ap_panels]]
        eta, _ = q3.trace_ratio(verts, cm.panel_sphere_centers(verts, R), R, n, focus, bary)
        rows.append((p, 0.466 * R - p, rms, float(np.abs(x).max()), rel, eta))
        print(f"{p:11.4f} {0.466 * R - p:16.3f} {rms * 1000:12.3f} "
              f"{np.abs(x).max() * 1000:11.2f} {rel * 100:12.5f} {eta:9.6f}"
              f"   ({time.time() - t0:.0f}s)")

    best_fit = min(rows, key=lambda r: r[2])
    best_eta = max(rows, key=lambda r: r[5])
    summary = [
        "",
        f"残差最小（贴近优先）：p = {best_fit[0]:.4f} m，RMS = {best_fit[2] * 1000:.3f} mm，"
        f"接收比 = {best_fit[5]:.6f}",
        f"接收比最大（接收优先）：p = {best_eta[0]:.4f} m，RMS = {best_eta[2] * 1000:.3f} mm，"
        f"接收比 = {best_eta[5]:.6f}",
        f"基准反射球面接收比 = {eta_base:.6f}",
        f"总耗时 {time.time() - t_all:.0f} s",
    ]
    print("\n".join(summary))

    out = Path(__file__).resolve().parent / "p_sensitivity_result.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write("焦距 p 敏感性分析\n")
        f.write(f"基准球面半径 R = {R:.4f} m，alpha = {ALPHA_Q2}°，beta = {BETA_Q2}°\n")
        f.write(f"接收比采样阶数 m = {SAMPLE_M}（每面板 {SAMPLE_M * (SAMPLE_M + 1) // 2} 点）\n\n")
        f.write(head + "\n" + "-" * len(head) + "\n")
        for p, ec, rms, xm, rel, eta in rows:
            f.write(f"{p:11.4f} {ec:16.3f} {rms * 1000:12.3f} {xm * 1000:11.2f} "
                    f"{rel * 100:12.5f} {eta:9.6f}\n")
        f.write("\n".join(summary) + "\n")
    print(f"\n结果已写入 {out}")


if __name__ == "__main__":
    main()
