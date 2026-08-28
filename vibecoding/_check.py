# -*- coding: utf-8 -*-
"""临时校核脚本：独立验证 q2/q3 结果的正确性（运行后删除）。"""
import sys
sys.path.insert(0, ".")
import numpy as np
import q2

names, pos, panels = q2.load_data()
R = np.linalg.norm(pos, axis=1).mean()
n = q2.axis_vector(q2.ALPHA, q2.BETA)
r = q2.solve()

ap_idx = np.where(r["in_ap"])[0]
x = r["ext"][ap_idx]
print("=== 约束精确校核 ===")
print(f"|ext| max = {np.abs(x).max():.10f}  (限 0.6)")

edges = q2.mesh_edges(panels)
em = r["in_ap"][edges[:, 0]] | r["in_ap"][edges[:, 1]]
ie, je = edges[em, 0], edges[em, 1]
d0 = np.linalg.norm(pos[ie] - pos[je], axis=1)
d1 = np.linalg.norm(r["pos_adj"][ie] - r["pos_adj"][je], axis=1)
rel = np.abs(d1 - d0) / d0
print(f"边数 = {rel.size}, 相对变化 max = {rel.max():.12e}  (限 7e-4)")
print(f"超限边数 = {(rel > 7e-4).sum()}")

# 全部 6525 根主索（含口径外-口径外，应恒为 0 变化）
d0a = np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]], axis=1)
d1a = np.linalg.norm(r["pos_adj"][edges[:, 0]] - r["pos_adj"][edges[:, 1]], axis=1)
print(f"全部主索相对变化 max = {(np.abs(d1a - d0a) / d0a).max():.3e}")

# 坐标与伸缩量自洽性
R_all = np.linalg.norm(pos, axis=1)
u = pos / R_all[:, None]
pos_chk = pos.copy()
pos_chk[ap_idx] = u[ap_idx] * (R_all[ap_idx] - x)[:, None]
print(f"pos_adj 与径向伸缩自洽误差 = {np.abs(pos_chk - r['pos_adj']).max():.2e}")

# 顶点在抛物面上？调整节点到理想抛物面的径向残差
from q2 import radial_extensions
e_ideal, _, _ = radial_extensions(pos[r["in_ap"]], n, r["p"])
print(f"调整后节点到理想抛物面的径向残差 RMS = {np.sqrt(np.mean((x - e_ideal)**2)):.4f} m")

# 顶点校验：到焦点距离 == 到准线距离
V, F, p = r["vertex"], r["focus"], r["p"]
print(f"顶点校核 |V-F| - 2p = {np.linalg.norm(V-F) - 2*p:.2e}")
print(f"|F| = {np.linalg.norm(F):.6f}, 0.534R = {0.534*R:.6f}")
