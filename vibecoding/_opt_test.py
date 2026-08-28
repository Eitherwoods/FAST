# -*- coding: utf-8 -*-
"""临时实验：检验二级优化的最优性，以及联合优化焦距 p 的收益（运行后删除）。"""
import sys
sys.path.insert(0, ".")
import numpy as np
from scipy.optimize import minimize, minimize_scalar, LinearConstraint
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
loc = np.full(len(names), -1, dtype=int)
loc[ap_idx] = np.arange(ap_idx.size)
li, lj = loc[ie], loc[je]

def max_rel(x):
    Rp = R_all.copy()
    Rp[ap_idx] -= x
    L1 = np.sqrt(Rp[ie]**2 + Rp[je]**2 - 2.0 * Rp[ie] * Rp[je] * cospsi)
    return np.abs(L1 - L0) / L0

def stage2(e_ap, verbose=False):
    """罚函数法（与 q2.py 相同）。"""
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

def qp_bounds(e_ap):
    """线性化约束的凸 QP（trust-constr），再用罚项精修非线性约束。"""
    # dL ≈ a_e*(x_i + x_j)（一端在口径外时 x_j=0），|dL|/L0 <= 7e-4
    a = (Ri * (1 - cospsi)) / L0          # ∂L/∂Ri 在 Ri=Rj 近似下对两端对称
    c = q2.EDGE_TOL                        # 对 (x_i+x_j) 的限幅 = tol/a*a... 见下
    # 精确一阶：dL = -a*(x_i+x_j)，限 |dL| <= tol*L0 → |x_i+x_j| <= tol*L0/a
    cvec = q2.EDGE_TOL * L0 / a
    M = np.zeros((ie.size, ap_idx.size))
    rows = np.arange(ie.size)
    M[rows, li] = 1.0
    inside_j = lj >= 0
    M[rows[inside_j], lj[inside_j]] = 1.0
    lc = LinearConstraint(M, -cvec, cvec)
    bnds = [(-q2.EXT_LIMIT, q2.EXT_LIMIT)] * e_ap.size
    res = minimize(lambda v: np.sum((v - e_ap)**2), np.clip(e_ap, -0.6, 0.6),
                   method="trust-constr", bounds=bnds, constraints=[lc],
                   options=dict(maxiter=400, gtol=1e-10, xtol=1e-12))
    x = res.x
    # 非线性精修：若真实约束超限，用罚函数从 QP 解出发微调
    if max_rel(x).max() > q2.EDGE_TOL:
        lam = 1e9
        for _ in range(4):
            x = minimize(lambda v: np.sum((v - e_ap)**2) + lam * np.sum(
                np.maximum(0.0, max_rel(v) - q2.EDGE_TOL)**2), x,
                method="L-BFGS-B", bounds=bnds).x
            if max_rel(x).max() <= q2.EDGE_TOL:
                break
            lam *= 10.0
    return x

for p_test in [139.8, 140.1, 140.4249, 140.8]:
    e_ap, _, _ = q2.radial_extensions(pos[in_ap], n, p_test)
    xp = stage2(e_ap)
    xq = qp_bounds(e_ap)
    rp, rq = max_rel(xp).max(), max_rel(xq).max()
    fp = np.sqrt(np.mean((xp - e_ap)**2))
    fq = np.sqrt(np.mean((xq - e_ap)**2))
    print(f"p={p_test:8.4f} | 罚函数: RMS={fp:.4f} rel={rp:.3e} | QP: RMS={fq:.4f} rel={rq:.3e}")
