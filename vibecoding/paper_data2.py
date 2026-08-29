# -*- coding: utf-8 -*-
"""补充：result.xlsx 伸缩量统计 + 接收平面落点分布统计。"""
import sys

import numpy as np

from common import (load_data, panel_sphere_centers, triangle_bary,
                    APERTURE_RADIUS)

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from openpyxl import load_workbook  # noqa: E402
import q3 as Q3  # noqa: E402

names, pos, panels = load_data()
R = float(np.linalg.norm(pos, axis=1).mean())

wb = load_workbook("result.xlsx", data_only=True)
ext = np.array([r[1] for r in wb["促动器顶端伸缩量"].iter_rows(min_row=2, values_only=True)],
               dtype=float)
print("=" * 68)
print("result.xlsx 促动器伸缩量统计")
print("=" * 68)
print("总行数 = %d，非零行数（口径内）= %d" % (ext.size, int((np.abs(ext) > 0).sum())))
print("范围 = [%.6f, %.6f] m，|δ|max = %.6f m，均值 = %.6f mm（仅非零）"
      % (ext.min(), ext.max(), np.abs(ext).max(), ext[np.abs(ext) > 0].mean() * 1000))
print("|δ| ≤ 0.6 m：%s" % ("满足" if np.abs(ext).max() <= 0.6 else "不满足"))

# ---------------- 落点分布 ----------------
adj = Q3.load_adjustment()
n = adj["n"]
P = adj["focus"]
in_ap = adj["in_ap"]
ap_panels = in_ap[panels].any(axis=1)
verts_adj = adj["pos_adj"][panels[ap_panels]]
centers_adj = panel_sphere_centers(verts_adj, R)
bary = triangle_bary(20)


def trace_detail(verts, centers):
    """返回 (权重 w, 落点偏移 dist, 照明区掩膜)。"""
    areas = 0.5 * np.linalg.norm(np.cross(verts[:, 1] - verts[:, 0],
                                          verts[:, 2] - verts[:, 0]), axis=1)
    S = np.einsum("sv,kvx->ksx", bary, verts)
    rel = S - centers[:, None, :]
    M = centers[:, None, :] + R * rel / np.linalg.norm(rel, axis=2)[:, :, None]
    K, Sn = M.shape[:2]
    M = M.reshape(-1, 3)
    O = np.repeat(centers, Sn, axis=0)
    w_area = np.repeat(areas / Sn, Sn)
    rho = np.linalg.norm(M - np.outer(M @ n, n), axis=1)
    keep = rho <= APERTURE_RADIUS
    N = (M - O) / R
    cosi = N @ n
    refl = -n + 2.0 * cosi[:, None] * N
    den = np.where(np.abs(refl @ n) < 1e-12, 1e-12, refl @ n)
    tt = ((P - M) @ n) / den
    Q = M + tt[:, None] * refl
    dist = np.linalg.norm(Q - P, axis=1)
    w = w_area * np.abs(cosi)
    return w[keep], dist[keep]


print()
print("=" * 68)
print("接收平面落点偏移 |Q-P| 分布（投影面积加权）")
print("=" * 68)
for tag, verts, centers in [("调节后", verts_adj, centers_adj),
                            ("基准球面", pos[panels[ap_panels]],
                             np.zeros((ap_panels.sum(), 3)))]:
    w, d = trace_detail(verts, centers)
    tot = w.sum()
    # 加权分位数
    o = np.argsort(d)
    cw = np.cumsum(w[o]) / tot
    qs = [np.interp(q, cw, d[o]) for q in (0.10, 0.25, 0.50, 0.75, 0.90)]
    inside_r = {}
    for rad in (0.5, 1.0, 2.0, 5.0, 10.0, 30.0):
        inside_r[rad] = float(w[d <= rad].sum() / tot)
    print("[%s] 总投影面积 = %.2f m²（覆盖率 %.2f%%）"
          % (tag, tot, tot / (np.pi * APERTURE_RADIUS ** 2) * 100))
    print("   加权分位数 |Q-P|：P10=%.4f  P25=%.4f  P50=%.4f  P75=%.4f  P90=%.4f  (m)"
          % tuple(qs))
    print("   落点 |Q-P| 最大值 = %.4f m" % d.max())
    print("   累积能量占比：" + "  ".join("≤%gm:%.4f" % (k, v) for k, v in inside_r.items()))
