# -*- coding: utf-8 -*-
"""临时实验：q3 采样阶数收敛性 + 口径覆盖率检查（运行后删除）。"""
import sys
sys.path.insert(0, ".")
import numpy as np
import q3

names, pos, panels = q3.load_data()
adj = q3.adjust_surface(pos, panels, q3.ALPHA, q3.BETA)
R, n, in_ap = adj["R"], adj["n"], adj["in_ap"]
ap_panels = in_ap[panels].all(axis=1)
verts_adj = adj["pos_adj"][panels[ap_panels]]
centers_adj = q3.panel_sphere_centers(verts_adj, R)

for m in (10, 20):
    q3.SAMPLE_M = m
    bary = q3.triangle_samples()
    eta_a = q3.trace_ratio(verts_adj, centers_adj, R, n, adj["focus"], bary)
    verts_base = pos[panels[ap_panels]]
    centers_base = np.zeros((ap_panels.sum(), 3))
    eta_b = q3.trace_ratio(verts_base, centers_base, R, n, adj["focus"], bary)
    print(f"M={m:2d} 采样点={len(bary):3d}: 调节后 eta={eta_a:.6f}, 基准 eta={eta_b:.6f}")

# 口径覆盖率：口径内面板总投影面积 / 口径圆面积
bary = q3.triangle_samples()
areas = 0.5 * np.linalg.norm(np.cross(verts_adj[:, 1] - verts_adj[:, 0],
                                      verts_adj[:, 2] - verts_adj[:, 0]), axis=1)
S = np.einsum("sv,kvx->ksx", bary, verts_adj)
rel = S - centers_adj[:, None, :]
M = centers_adj[:, None, :] + R * rel / np.linalg.norm(rel, axis=2)[:, :, None]
N = (M - centers_adj[:, None, :]) / R
cosi = np.abs(N.reshape(-1, 3) @ n)
proj = np.repeat(areas / len(bary), len(bary)) * cosi
print(f"口径内面板总投影面积 = {proj.sum():.1f} m^2, pi*150^2 = {np.pi*150**2:.1f} m^2, "
      f"覆盖率 = {proj.sum()/(np.pi*150**2)*100:.2f}%")
