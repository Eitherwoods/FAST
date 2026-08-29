# -*- coding: utf-8 -*-
"""为论文正文抽取中间计算数据（不改动任何求解逻辑，只做统计与打印）。

输出：
1. 数据预处理：基准球面半径统计、节点/面板/索计数、口径内计数
2. 问题一：外层焦距搜索的 RMS(p) 扫描表、典型节点的中间量
3. 问题二：典型节点的理想/可行伸缩量与残差、面板球心反演自检
"""
import sys

import numpy as np

from common import (load_data, axis_vector, radial_extensions, mesh_edges,
                    build_context, solve_adjustment, panel_sphere_centers,
                    fit_best, adjust_nodes)

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

names, pos, panels = load_data()
rad = np.linalg.norm(pos, axis=1)
R = float(rad.mean())
edges = mesh_edges(panels)

print("=" * 72)
print("【1】数据预处理")
print("=" * 72)
print("节点总数 = %d，面板总数 = %d，索（边）总数 = %d" % (len(pos), len(panels), len(edges)))
print("节点模长：均值 R = %.6f m，标准差 = %.6f m，min = %.4f，max = %.4f"
      % (R, rad.std(ddof=1), rad.min(), rad.max()))
print("0.534R = %.6f m，0.466R = %.6f m" % (0.534 * R, 0.466 * R))

e0 = pos[edges[:, 0]]
e1 = pos[edges[:, 1]]
L = np.linalg.norm(e0 - e1, axis=1)
print("索长：均值 = %.4f m，min = %.4f，max = %.4f" % (L.mean(), L.min(), L.max()))
print("0.07%% 索长限值对应绝对量（按均值索长）= %.3f mm" % (0.0007 * L.mean() * 1000))


def rms_at(ctx, n, p):
    e_ap, _, _ = radial_extensions(pos[ctx["in_ap"]], n, p, R)
    x = fit_best(ctx, e_ap)
    return float(np.sqrt(np.mean((x - e_ap) ** 2))), e_ap, x


# ---------------- 问题一 ----------------
print()
print("=" * 72)
print("【2】问题一（alpha=0, beta=90）：外层焦距 p 扫描")
print("=" * 72)
n1 = axis_vector(0.0, 90.0)
res1 = solve_adjustment(pos, panels, 0.0, 90.0)
p1 = res1["p"]
print("最优 p* = %.4f m，RMS = %.6f m，rel_ideal = %.5f%%，rel_max = %.5f%%"
      % (p1, res1["fit_rms"], res1["rel_ideal"] * 100, res1["rel_max"] * 100))
ctx1 = build_context(pos, panels, n1)
print("口径内节点数 = %d，参与约束的边数 = %d，三顶点全在口径内的面板数 = %d"
      % (res1["ext_ideal"].size, ctx1["ie"].size, ctx1["ap_panels"].sum()))

print()
print("  %-10s %-11s %-24s %-24s" % ("p (m)", "RMS(mm)", "理想伸缩量范围(mm)", "可行解范围(mm)"))
for p in [140.00, 140.20, 140.30, 140.3773, 140.45, 140.60, 140.80]:
    v, e_ap, x = rms_at(ctx1, n1, float(p))
    print("  %-10.4f %-11.4f [%8.2f,%8.2f]   [%8.2f,%8.2f]"
          % (p, v * 1000, e_ap.min() * 1000, e_ap.max() * 1000,
             x.min() * 1000, x.max() * 1000))

print()
print("问题一典型节点中间量（按到 z 轴距离 rho 取代表点）：")
ap_idx = res1["ap_idx"]
rho = np.linalg.norm(pos[ap_idx][:, :2], axis=1)
ext_ideal, ext_fit = res1["ext_ideal"], res1["ext_fit"]
print("  %-6s %-10s %-14s %-14s %-10s" % ("节点", "rho(m)", "理想伸缩(mm)", "可行伸缩(mm)", "残差(mm)"))
for tg in [0.0, 30.0, 60.0, 90.0, 120.0, 145.0]:
    j = int(np.argmin(np.abs(rho - tg)))
    print("  %-6s %-10.3f %-14.4f %-14.4f %-10.4f"
          % (names[ap_idx[j]], rho[j], ext_ideal[j] * 1000, ext_fit[j] * 1000,
             (ext_fit[j] - ext_ideal[j]) * 1000))
print("全口径：理想伸缩 [%.2f, %.2f] mm，可行解 [%.2f, %.2f] mm，残差RMS %.2f mm"
      % (ext_ideal.min() * 1000, ext_ideal.max() * 1000,
         ext_fit.min() * 1000, ext_fit.max() * 1000, res1["fit_rms"] * 1000))
print("顶点 V = (%.4f, %.4f, %.4f)，焦点 P = (%.4f, %.4f, %.4f)"
      % (tuple(res1["vertex"]) + tuple(res1["focus"])))

# ---------------- 问题二 ----------------
print()
print("=" * 72)
print("【3】问题二（alpha=36.795, beta=78.169）")
print("=" * 72)
res2 = solve_adjustment(pos, panels, 36.795, 78.169)
p2 = res2["p"]
n2 = axis_vector(36.795, 78.169)
print("最优 p* = %.4f m，RMS = %.4f mm" % (p2, res2["fit_rms"] * 1000))
print("顶点 V = (%.4f, %.4f, %.4f)" % tuple(res2["vertex"]))
print("焦点 P = (%.4f, %.4f, %.4f)" % tuple(res2["focus"]))
print("轴向量 n = (%.6f, %.6f, %.6f)，|n| = %.10f" % (n2[0], n2[1], n2[2], np.linalg.norm(n2)))
ctx2 = build_context(pos, panels, n2)
print("口径内节点数 = %d，参与约束的边数 = %d，三顶点全在口径内的面板数 = %d"
      % (res2["ext_ideal"].size, ctx2["ie"].size, ctx2["ap_panels"].sum()))
print("可行伸缩量 [%.4f, %.4f] m，均值 %.4f mm，|δ| 最大 %.4f mm"
      % (res2["ext_fit"].min(), res2["ext_fit"].max(), res2["ext_fit"].mean() * 1000,
         np.abs(res2["ext_fit"]).max() * 1000))
print("理想伸缩量 [%.4f, %.4f] m，|δ̂| 最大 %.4f mm"
      % (res2["ext_ideal"].min(), res2["ext_ideal"].max(), np.abs(res2["ext_ideal"]).max() * 1000))

pos_adj = adjust_nodes(pos, res2["ext_fit"], ctx2)
O_base = panel_sphere_centers(pos[panels], R)
O_adj = panel_sphere_centers(pos_adj[panels], R)
print("面板球心反演自检：基准态 |O| 最大 = %.3e m，调节后 |O| 最大 = %.4f m，均值 %.4f m"
      % (np.linalg.norm(O_base, axis=1).max(), np.linalg.norm(O_adj, axis=1).max(),
         np.linalg.norm(O_adj, axis=1).mean()))

print()
print("问题二典型节点中间量（按到观测轴距离 d 取代表点）：")
ap2 = res2["ap_idx"]
proj = pos[ap2] @ n2
dist = np.sqrt(np.maximum(rad[ap2] ** 2 - proj ** 2, 0.0))
ei2, ef2 = res2["ext_ideal"], res2["ext_fit"]
print("  %-6s %-10s %-14s %-14s %-10s" % ("节点", "d(m)", "理想伸缩(mm)", "可行伸缩(mm)", "残差(mm)"))
for tg in [0.0, 30.0, 60.0, 90.0, 120.0, 145.0]:
    j = int(np.argmin(np.abs(dist - tg)))
    print("  %-6s %-10.3f %-14.4f %-14.4f %-10.4f"
          % (names[ap2[j]], dist[j], ei2[j] * 1000, ef2[j] * 1000, (ef2[j] - ei2[j]) * 1000))

# 径向求交方程的中间量示例
print()
print("径向求交方程中间量示例：(1-A^2)t^2 - 4pA t + 4p(B-p) = 0，B = -0.534R = %.6f" % (-0.534 * R))
print("  判别式 Δ = 16p(p - B(1-A^2))，取根 t = (4pA + sqrt(Δ)) / (2(1-A^2))，δ̂ = R_i - t")
for tg in [0.0, 30.0, 60.0, 90.0, 120.0, 145.0]:
    j = int(np.argmin(np.abs(dist - tg)))
    u = pos[ap2[j]] / rad[ap2[j]]
    A = float(u @ n2)
    B = -0.534 * R
    a2_ = 1 - A ** 2
    disc = 16 * p2 * (p2 - B * a2_)
    t = (4 * p2 * A + np.sqrt(disc)) / (2 * a2_) if a2_ > 1e-12 else (B - p2) / A
    print("  d=%6.1f m: A=u·n=%9.6f, 1-A^2=%8.6f, Δ=%12.4f, t=%10.6f m, δ̂=%+9.4f mm"
          % (dist[j], A, a2_, disc, t, (rad[ap2[j]] - t) * 1000))
