# -*- coding: utf-8 -*-
"""临时实验：用 q2 现有罚函数法在不同焦距下求二级解，确认联合最优焦距（运行后删除）。"""
import sys
sys.path.insert(0, ".")
import numpy as np
from scipy.optimize import minimize
import q2

names, pos, panels = q2.load_data()
R = np.linalg.norm(pos, axis=1).mean()
n = q2.axis_vector(q2.ALPHA, q2.BETA)
dist_to_axis = np.linalg.norm(pos - np.outer(pos @ n, n), axis=1)
in_ap = dist_to_axis <= q2.APERTURE_RADIUS
ap_idx = np.where(in_ap)[0]

R_all = np.linalg.norm(pos, axis=1)
edges = q2.mesh_edges(panels)
em = in_ap[edges[:, 0]] | in_ap[edges[:, 1]]
ie, je = edges[em, 0], edges[em, 1]
Ri, Rj = R_all[ie], R_all[je]
L0 = np.linalg.norm(pos[ie] - pos[je], axis=1)
cospsi = (Ri**2 + Rj**2 - L0**2) / (2.0 * Ri * Rj)

def max_rel(x):
    Rp = R_all.copy()
    Rp[ap_idx] -= x
    L1 = np.sqrt(Rp[ie]**2 + Rp[je]**2 - 2.0 * Rp[ie] * Rp[je] * cospsi)
    return np.abs(L1 - L0) / L0

def penalty_stage2(e_ap):
    x = np.clip(e_ap, -q2.EXT_LIMIT, q2.EXT_LIMIT)
    lam = 1e8
    bnds = [(-q2.EXT_LIMIT, q2.EXT_LIMIT)] * x.size
    for _ in range(4):
        x = minimize(lambda v: np.sum((v - e_ap)**2) + lam * np.sum(
            np.maximum(0.0, max_rel(v) - q2.EDGE_TOL)**2), x,
            method="L-BFGS-B", bounds=bnds).x
        if max_rel(x).max() <= q2.EDGE_TOL:
            break
        lam *= 10.0
    return x

for p_test in [140.30, 140.35, 140.40, 140.4249, 140.45]:
    e_ap, _, _ = q2.radial_extensions(pos[in_ap], n, p_test)
    x = penalty_stage2(e_ap)
    rms = np.sqrt(np.mean((x - e_ap)**2))
    print(f"p={p_test:8.4f}: RMS={rms:.5f} m, max_rel={max_rel(x).max():.4e}, "
          f"|x|max={np.abs(x).max():.4f}", flush=True)
