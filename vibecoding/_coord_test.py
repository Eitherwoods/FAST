# -*- coding: utf-8 -*-
"""临时实验：坐标下降法（精确非线性边约束）检验二级解的最优性与联合优化 p 的收益（运行后删除）。"""
import sys
sys.path.insert(0, ".")
import numpy as np
import q2

names, pos, panels = q2.load_data()
R = np.linalg.norm(pos, axis=1).mean()
n = q2.axis_vector(q2.ALPHA, q2.BETA)
dist_to_axis = np.linalg.norm(pos - np.outer(pos @ n, n), axis=1)
in_ap = dist_to_axis <= q2.APERTURE_RADIUS
ap_idx = np.where(in_ap)[0]
nap = ap_idx.size

R_all = np.linalg.norm(pos, axis=1)
edges = q2.mesh_edges(panels)
em = in_ap[edges[:, 0]] | in_ap[edges[:, 1]]
ie, je = edges[em, 0], edges[em, 1]
Ri0, Rj0 = R_all[ie], R_all[je]
L0 = np.linalg.norm(pos[ie] - pos[je], axis=1)
cospsi = (Ri0**2 + Rj0**2 - L0**2) / (2.0 * Ri0 * Rj0)
sin2 = np.maximum(1.0 - cospsi**2, 0.0)
loc = np.full(len(names), -1, dtype=int)
loc[ap_idx] = np.arange(nap)
li, lj = loc[ie], loc[je]
Lmin, Lmax = L0 * (1 - q2.EDGE_TOL), L0 * (1 + q2.EDGE_TOL)

inc = [[] for _ in range(nap)]
for e in range(ie.size):
    if li[e] >= 0: inc[li[e]].append(e)
    if lj[e] >= 0: inc[lj[e]].append(e)

def max_rel(x):
    Rp = R_all.copy()
    Rp[ap_idx] -= x
    L1 = np.sqrt(Rp[ie]**2 + Rp[je]**2 - 2.0 * Rp[ie] * Rp[je] * cospsi)
    return (np.abs(L1 - L0) / L0).max()

def edge_interval_Rp(e, Rp_other, R_base):
    """给定对端半径与本端基线半径，返回本端半径的可行区间（含基线值的连通分量）。"""
    c, s2 = cospsi[e], sin2[e]
    base = Rp_other * c
    d_hi = Lmax[e]**2 - Rp_other**2 * s2
    r_lo, r_hi = base - np.sqrt(d_hi), base + np.sqrt(d_hi)
    d_lo = Lmin[e]**2 - Rp_other**2 * s2
    if d_lo <= 0:
        return r_lo, r_hi
    s_lo, s_hi = base - np.sqrt(d_lo), base + np.sqrt(d_lo)
    return (s_hi, r_hi) if R_base >= base else (r_lo, s_lo)

def coordinate_descent(e_ap, sweeps=80, seed=0):
    rng = np.random.default_rng(seed)
    x = np.clip(e_ap, -0.6, 0.6)
    Rp = R_all.copy()
    for it in range(sweeps):
        for i in rng.permutation(nap):
            lo, hi = -q2.EXT_LIMIT, q2.EXT_LIMIT
            ni = ap_idx[i]
            for e in inc[i]:
                other = je[e] if ie[e] == ni else ie[e]
                lo_e, hi_e = edge_interval_Rp(e, Rp[other], R_all[ni])
                lo = max(lo, R_all[ni] - hi_e)
                hi = min(hi, R_all[ni] - lo_e)
            x_new = min(max(e_ap[i], lo), hi)
            Rp[ni] = R_all[ni] - x_new
            x[i] = x_new
    return x

for p_test in [139.6, 140.0, 140.4249, 140.8, 141.2]:
    e_ap, _, _ = q2.radial_extensions(pos[in_ap], n, p_test)
    x = coordinate_descent(e_ap)
    rms = np.sqrt(np.mean((x - e_ap)**2))
    print(f"p={p_test:8.4f}: RMS={rms:.5f} m, max_rel={max_rel(x):.4e}, |x|max={np.abs(x).max():.4f}")
