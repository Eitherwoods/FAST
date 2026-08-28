# -*- coding: utf-8 -*-
"""问题二：α=36.795°, β=78.169°，确定理想抛物面并计算促动器伸缩量，输出 result.xlsx。

模型：
- 观测轴 n = (cosβcosα, cosβsinα, sinβ)，焦点 P = -0.534R·n，抛物面焦距 p 为决策变量，
  顶点 V = -(0.534R + p)·n。
- 节点 i（径向单位向量 u_i，A_i = u_i·n）沿径向与抛物面交点参数 t 满足
      (1-A²)t² - 4pA·t + 4p(B-p) = 0，  B = -0.534R
  其判别式 16p[p + 0.534R(1-A²)] 恒正，径向交点总存在。伸缩量（向球心为正）e = R_i - t。
- 嵌套联合优化（在反射面板调节约束下确定理想抛物面）：
  外层以焦距 p 为决策变量，目标为约束下调节方案相对理想交点的伸缩量均方根残差最小；
  内层对给定 p，在伸缩量限幅 ±0.6 m 与相邻节点间距变化 ≤0.07% 约束下，
  用罚函数法（λ 逐级放大直至可行）求最贴近理想交点的伸缩量。
  仅取 300 m 口径内节点（到观测轴距离 ≤150 m），间距约束含口径边界处的跨接边。
"""
import sys

import numpy as np
import pandas as pd
from pathlib import Path
from openpyxl import Workbook
from scipy.optimize import minimize, minimize_scalar

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "题目与附件与思路"
APERTURE_RADIUS = 150.0
EXT_LIMIT = 0.6
EDGE_TOL = 0.0007
ALPHA, BETA = 36.795, 78.169


def load_data():
    df1 = pd.read_csv(DATA / "附件1.csv", encoding="gbk")
    names = df1.iloc[:, 0].astype(str).tolist()
    pos = df1.iloc[:, 1:4].to_numpy(dtype=float)
    df3 = pd.read_csv(DATA / "附件3.csv", encoding="gbk")
    idx = {nm: i for i, nm in enumerate(names)}
    panels = df3.iloc[:, 0:3].apply(lambda s: s.map(idx)).to_numpy(dtype=int)
    return names, pos, panels


def axis_vector(alpha_deg, beta_deg):
    a, b = np.deg2rad(alpha_deg), np.deg2rad(beta_deg)
    return np.array([np.cos(b) * np.cos(a), np.cos(b) * np.sin(a), np.sin(b)])


def radial_extensions(pos, n, p):
    """各节点沿径向到达抛物面的伸缩量（向球心为正）。"""
    R_i = np.linalg.norm(pos, axis=1)
    u = pos / R_i[:, None]
    B = -0.534 * R_i.mean()
    A = u @ n
    a2 = 1.0 - A * A
    disc = 16.0 * p * (p - B * a2)
    t = np.empty_like(A)
    mask = a2 > 1e-12
    t[mask] = (4.0 * p * A[mask] + np.sqrt(disc[mask])) / (2.0 * a2[mask])
    t[~mask] = (B - p) / A[~mask]
    return R_i - t, u, R_i


def mesh_edges(panels):
    edges = np.vstack([panels[:, [0, 1]], panels[:, [1, 2]], panels[:, [2, 0]]])
    return np.unique(np.sort(edges, axis=1), axis=0)


def solve():
    names, pos, panels = load_data()
    R = np.linalg.norm(pos, axis=1).mean()
    n = axis_vector(ALPHA, BETA)

    dist_to_axis = np.linalg.norm(pos - np.outer(pos @ n, n), axis=1)
    in_ap = dist_to_axis <= APERTURE_RADIUS
    ap_idx = np.where(in_ap)[0]

    # 相邻边几何（径向移动不改变节点间径向夹角 ψ）
    R_all = np.linalg.norm(pos, axis=1)
    edges = mesh_edges(panels)
    em = in_ap[edges[:, 0]] | in_ap[edges[:, 1]]
    ie, je = edges[em, 0], edges[em, 1]
    Ri, Rj = R_all[ie], R_all[je]
    L0 = np.linalg.norm(pos[ie] - pos[je], axis=1)
    cospsi = (Ri ** 2 + Rj ** 2 - L0 ** 2) / (2.0 * Ri * Rj)

    def max_rel(x):
        Rp = R_all.copy()
        Rp[ap_idx] -= x
        L1 = np.sqrt(Rp[ie] ** 2 + Rp[je] ** 2 - 2.0 * Rp[ie] * Rp[je] * cospsi)
        return np.abs(L1 - L0) / L0

    bnds = [(-EXT_LIMIT, EXT_LIMIT)] * ap_idx.size

    def stage2_fit(e_ap):
        """内层：约束下最贴近理想伸缩量，罚函数法逐级放大 λ 直至可行。"""
        x = np.clip(e_ap, -EXT_LIMIT, EXT_LIMIT)
        lam = 1e8
        for _ in range(8):
            x = minimize(lambda v: np.sum((v - e_ap) ** 2)
                         + lam * np.sum(np.maximum(0.0, max_rel(v) - EDGE_TOL) ** 2),
                         x, method="L-BFGS-B", bounds=bnds).x
            if max_rel(x).max() <= EDGE_TOL:
                break
            lam *= 10.0
        return x

    cache = {}

    def fit_rms(p):
        """外层目标：给定焦距时，约束下调节方案的均方根贴合残差。"""
        p = round(float(p), 6)
        if p not in cache:
            e_ap, _, _ = radial_extensions(pos[in_ap], n, p)
            cache[p] = (stage2_fit(e_ap), e_ap)
        x, e_ap = cache[p]
        return float(np.sqrt(np.mean((x - e_ap) ** 2)))

    p_opt = float(minimize_scalar(fit_rms, bounds=(139.9, 140.9), method="bounded",
                                  options={"xatol": 0.02}).x)
    if round(p_opt, 6) not in cache:
        fit_rms(p_opt)
    x, e_ap = cache[round(p_opt, 6)]
    _, u_ap, R_ap = radial_extensions(pos[in_ap], n, p_opt)

    ext = np.zeros(len(names))
    ext[ap_idx] = x
    pos_adj = pos.copy()
    pos_adj[ap_idx] = u_ap * (R_ap - x)[:, None]

    rel = max_rel(x)
    return dict(names=names, pos=pos, pos_adj=pos_adj, ext=ext, in_ap=in_ap,
                R=R, n=n, p=p_opt, vertex=-(0.534 * R + p_opt) * n,
                focus=-0.534 * R * n,
                ext_ideal=e_ap, fit_rms=float(np.sqrt(np.mean((x - e_ap) ** 2))),
                rel_max=rel.max() if rel.size else 0.0, n_edges=rel.size)


def write_result(r, path):
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "理想抛物面顶点坐标"
    ws1.append(["X坐标（米）", "Y坐标（米）", "Z坐标（米）", None, None, "注：至少保留3位小数"])
    ws1.append([round(float(c), 6) for c in r["vertex"]])

    ws2 = wb.create_sheet("调整后主索节点编号及坐标")
    ws2.append(["节点编号", "X坐标（米）", "Y坐标（米）", "Z坐标（米）", None, None, "注：至少保留3位小数"])
    for i in np.where(r["in_ap"])[0]:
        ws2.append([r["names"][i]] + [round(float(c), 6) for c in r["pos_adj"][i]])

    ws3 = wb.create_sheet("促动器顶端伸缩量")
    ws3.append(["对应主索节点编号", "伸缩量（米）", None, "注：至少保留3位小数"])
    for i, nm in enumerate(r["names"]):
        ws3.append([nm, round(float(r["ext"][i]), 6)])

    wb.save(path)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    r = solve()
    print(f"基准球面半径 R = {r['R']:.4f} m")
    print(f"观测轴 n = {np.array2string(r['n'], precision=6)}")
    print(f"最优焦距 p = {r['p']:.4f} m（0.466R = {0.466 * r['R']:.4f} m）")
    v = r["vertex"]
    print(f"理想抛物面顶点 V = ({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f})")
    f = r["focus"]
    print(f"焦点（馈源舱）P = ({f[0]:.4f}, {f[1]:.4f}, {f[2]:.4f})")
    e_ap = r["ext"][r["in_ap"]]
    print(f"300 m 口径内节点数 = {e_ap.size}")
    print(f"理想交点伸缩量范围 = [{r['ext_ideal'].min():.4f}, {r['ext_ideal'].max():.4f}] m")
    print(f"最终伸缩量范围 = [{e_ap.min():.4f}, {e_ap.max():.4f}] m，|e| 最大 = {np.abs(e_ap).max():.4f} m")
    print(f"约束下调节方案相对理想交点的伸缩量均方根残差 = {r['fit_rms']:.4f} m")
    print(f"参与约束的相邻边数 = {r['n_edges']}")
    print(f"相邻节点间距最大相对变化 = {r['rel_max'] * 100:.5f}%（限值 0.07%）")
    out = Path(__file__).resolve().parent / "result.xlsx"
    write_result(r, out)
    print(f"已写入 {out}")
