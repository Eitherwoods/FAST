# -*- coding: utf-8 -*-
"""问题三：基于第2问调节方案计算馈源舱接收比，并与基准反射球面比较。

模型：
- 调节后每块面板仍是半径 R 的球面片（刚体），由其三个调节后顶点反演球心 O_k：
  先求三顶点外接圆心 P0，再沿平面法向偏移 sqrt(R²-r0²)，取与原点同侧（凹侧朝球心）。
- 面板上采样点 M 的法向 N = (M-O_k)/R；入射平行光沿 -n，反射方向
      r = -n + 2(n·N)N
  反射线与过焦点 P、垂直 n 的接收平面交于
      Q = M + t r，t = (P-M)·n / (r·n)
  当 |Q-P| ≤ 0.5 m 时该面元的反射信号被馈源舱有效区域接收。
- 接收比 = 被接收面元的投影面积之和 / 全部面元的投影面积之和，
  面元按 |N·n|·dS（投影面积）计权，用三角形重心坐标网格采样数值积分。
- 基准反射球面：球心在原点，同样方法计算接收比作为对照。
- 验证：对理想抛物面直接采样做射线追踪，接收比应≈1。
"""
import sys

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize, minimize_scalar

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "题目与附件与思路"
APERTURE_RADIUS = 150.0
EXT_LIMIT = 0.6
EDGE_TOL = 0.0007
ALPHA, BETA = 36.795, 78.169
SAMPLE_M = 20            # 每块面板三角形网格采样阶数，采样点数 m(m+1)/2
FEED_RADIUS = 0.5        # 馈源舱有效接收圆盘半径
CONV_M = (10, 20, 30)    # 采样阶数收敛性检验


# ---------- 数据与几何（与 q2.py 相同，独立保存） ----------

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


def adjust_surface(pos, panels, alpha_deg, beta_deg):
    """嵌套联合优化（同第2问）：外层定焦距、内层在约束下修正各节点伸缩量。"""
    R = np.linalg.norm(pos, axis=1).mean()
    n = axis_vector(alpha_deg, beta_deg)
    dist_to_axis = np.linalg.norm(pos - np.outer(pos @ n, n), axis=1)
    in_ap = dist_to_axis <= APERTURE_RADIUS
    ap_idx = np.where(in_ap)[0]

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
        p = round(float(p), 6)
        if p not in cache:
            e_ap, _, _ = radial_extensions(pos[in_ap], n, p)
            cache[p] = stage2_fit(e_ap)
        x = cache[p]
        e_ap, _, _ = radial_extensions(pos[in_ap], n, p)
        return float(np.sqrt(np.mean((x - e_ap) ** 2)))

    p_opt = float(minimize_scalar(fit_rms, bounds=(139.9, 140.9), method="bounded",
                                  options={"xatol": 0.02}).x)
    if round(p_opt, 6) not in cache:
        fit_rms(p_opt)
    x = cache[round(p_opt, 6)]
    _, u_ap, R_ap = radial_extensions(pos[in_ap], n, p_opt)

    ext = np.zeros(len(pos))
    ext[ap_idx] = x
    pos_adj = pos.copy()
    pos_adj[ap_idx] = u_ap * (R_ap - x)[:, None]
    return dict(R=R, n=n, p=p_opt, in_ap=in_ap, pos_adj=pos_adj,
                vertex=-(0.534 * R + p_opt) * n, focus=-0.534 * R * n)


# ---------- 面板几何与射线追踪 ----------

def panel_sphere_centers(verts, R):
    """由每块面板三顶点（K,3,3）反演半径 R 的面板球心（K,3）。"""
    A, B, D = verts[:, 0], verts[:, 1], verts[:, 2]
    u, v = B - A, D - A
    uu = np.einsum("ij,ij->i", u, u)
    vv = np.einsum("ij,ij->i", v, v)
    uv = np.einsum("ij,ij->i", u, v)
    R1 = np.einsum("ij,ij->i", B, B) - np.einsum("ij,ij->i", A, A) - 2 * np.einsum("ij,ij->i", A, u)
    R2 = np.einsum("ij,ij->i", D, D) - np.einsum("ij,ij->i", A, A) - 2 * np.einsum("ij,ij->i", A, v)
    det = 4.0 * (uu * vv - uv * uv)
    s = 2.0 * (R1 * vv - R2 * uv) / det
    t = 2.0 * (uu * R2 - uv * R1) / det
    P0 = A + s[:, None] * u + t[:, None] * v
    np_vec = np.cross(u, v)
    np_vec /= np.linalg.norm(np_vec, axis=1)[:, None]
    r0 = np.linalg.norm(P0 - A, axis=1)
    h = np.sqrt(np.maximum(R ** 2 - r0 ** 2, 0.0))
    side = np.sign(-np.einsum("ij,ij->i", P0, np_vec))
    side[side == 0] = 1.0
    return P0 + side[:, None] * h[:, None] * np_vec


def triangle_samples(m=SAMPLE_M):
    """重心坐标采样点 (S,3)，对应权重均等。"""
    bary = []
    for i in range(m):
        for j in range(m - i):
            b, g = (i + 1 / 3) / m, (j + 1 / 3) / m
            bary.append([1 - b - g, b, g])
    return np.array(bary)


def trace_ratio(verts, centers, R, n, P, bary):
    """顶点 (K,3,3)、球心 (K,3) 的面板集合的接收比。

    仅统计落在 300 m 照明区内（到观测轴距离 ≤150 m）的采样点：口径边界处的
    面板只有部分位于照明区内，按采样点逐点剔除，避免漏计边缘一圈面板的投影面积。

    返回 (接收比, 照明区内总投影面积)。后者用于检验 300 m 口径的面板覆盖率。
    """
    areas = 0.5 * np.linalg.norm(np.cross(verts[:, 1] - verts[:, 0],
                                          verts[:, 2] - verts[:, 0]), axis=1)
    S = np.einsum("sv,kvx->ksx", bary, verts)          # (K,S,3) 平面采样点
    rel = S - centers[:, None, :]
    M = centers[:, None, :] + R * rel / np.linalg.norm(rel, axis=2)[:, :, None]
    K, Sn = M.shape[:2]
    M = M.reshape(-1, 3)
    O = np.repeat(centers, Sn, axis=0)
    w_area = np.repeat(areas / Sn, Sn)

    # 照明区掩膜：采样点到观测轴（过 C 沿 n）的距离 ≤ 150 m
    rho = np.linalg.norm(M - np.outer(M @ n, n), axis=1)
    keep = rho <= APERTURE_RADIUS

    N = (M - O) / R
    cosi = N @ n
    refl = -n + 2.0 * cosi[:, None] * N
    den = refl @ n
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    tt = ((P - M) @ n) / den
    Q = M + tt[:, None] * refl
    dist = np.linalg.norm(Q - P, axis=1)
    w = w_area * np.abs(cosi)
    w, dist = w[keep], dist[keep]
    return float(w[dist <= FEED_RADIUS].sum() / w.sum()), float(w.sum())


def ideal_paraboloid_ratio(R, n, p, P, V):
    """理想抛物面接收比（模型自洽性验证，应≈1）。"""
    e1 = np.cross(n, [0.0, 0.0, 1.0])
    if np.linalg.norm(e1) < 1e-8:
        e1 = np.array([1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    g = np.linspace(-APERTURE_RADIUS, APERTURE_RADIUS, 151)
    x, y = np.meshgrid(g, g, indexing="ij")
    rho2 = x ** 2 + y ** 2
    keep = rho2 <= APERTURE_RADIUS ** 2
    x, y, rho2 = x[keep], y[keep], rho2[keep]
    M = V + (rho2 / (4 * p))[:, None] * n + x[:, None] * e1 + y[:, None] * e2
    F = -0.534 * R * n
    grad = (M - F) / np.linalg.norm(M - F, axis=1)[:, None] - n
    N = grad / np.linalg.norm(grad, axis=1)[:, None]
    cosi = N @ n
    refl = -n + 2.0 * cosi[:, None] * N
    tt = ((P - M) @ n) / (refl @ n)
    Q = M + tt[:, None] * refl
    dist = np.linalg.norm(Q - P, axis=1)
    w = np.abs(cosi)
    return float(w[dist <= FEED_RADIUS].sum() / w.sum())


# ---------- 主流程 ----------

def main():
    names, pos, panels = load_data()
    adj = adjust_surface(pos, panels, ALPHA, BETA)
    R, n, in_ap = adj["R"], adj["n"], adj["in_ap"]
    lines = ["问题三接收比计算结果",
             "",
             f"基准球面半径 R = {R:.4f} m",
             f"观测方向 α = {ALPHA}°, β = {BETA}°",
             f"最优焦距 p = {adj['p']:.4f} m",
             f"馈源舱（焦点）P = ({adj['focus'][0]:.4f}, {adj['focus'][1]:.4f}, {adj['focus'][2]:.4f}) m",
             f"理想抛物面顶点 V = ({adj['vertex'][0]:.4f}, {adj['vertex'][1]:.4f}, {adj['vertex'][2]:.4f}) m",
             ""]
    print(f"调节方案：焦距 p = {adj['p']:.4f} m，口径内节点 {in_ap.sum()} 个")

    # 与照明区相交的面板（至少一个顶点在口径内）：边缘跨界面板按采样点逐点截断
    ap_panels = in_ap[panels].any(axis=1)
    print(f"与照明区相交的面板数 = {ap_panels.sum()} / {len(panels)}"
          f"（三顶点全在口径内的 = {in_ap[panels].all(axis=1).sum()}）")
    verts_adj = adj["pos_adj"][panels[ap_panels]]
    centers_adj = panel_sphere_centers(verts_adj, R)
    centers_check = np.linalg.norm(panel_sphere_centers(pos[panels[ap_panels]], R), axis=1).max()
    print(f"面板球心反演自检：基准态球心到原点最大距离 = {centers_check:.2e} m")
    lines += [f"300 m 口径内节点数 = {int(in_ap.sum())}",
              f"参与光线追迹的面板数 = {int(ap_panels.sum())} / {len(panels)}"
              f"（与照明区相交；三顶点全在口径内的 = {int(in_ap[panels].all(axis=1).sum())}）",
              f"面板球心反演自检：基准态球心到原点最大距离 = {centers_check:.2e} m",
              ""]

    bary = triangle_samples(SAMPLE_M)
    eta_adj, area_adj = trace_ratio(verts_adj, centers_adj, R, n, adj["focus"], bary)

    verts_base = pos[panels[ap_panels]]
    centers_base = np.zeros((ap_panels.sum(), 3))
    eta_base, area_base = trace_ratio(verts_base, centers_base, R, n, adj["focus"], bary)

    eta_ideal = ideal_paraboloid_ratio(R, n, adj["p"], adj["focus"], adj["vertex"])

    print(f"理想抛物面接收比（验证） = {eta_ideal:.6f}")
    print(f"调节后反射面接收比 = {eta_adj:.6f}")
    print(f"基准反射球面接收比 = {eta_base:.6f}")
    if eta_base > 0:
        print(f"接收比提升 = {eta_adj / eta_base:.1f} 倍")

    disk = np.pi * APERTURE_RADIUS ** 2
    lines += [f"总投影面积：调节后 = {area_adj:.2f} m²，基准球面 = {area_base:.2f} m²，"
              f"π·150² = {disk:.2f} m²",
              f"300 m 口径覆盖率：调节后 = {area_adj / disk * 100:.2f}%，"
              f"基准球面 = {area_base / disk * 100:.2f}%",
              "",
              "接收比（采样阶数 m = {:d}）".format(SAMPLE_M),
              f"  理想抛物面（模型自洽性验证） = {eta_ideal:.6f}",
              f"  调节后反射面                 = {eta_adj:.6f}",
              f"  基准反射球面                 = {eta_base:.6f}"]
    if eta_base > 0:
        lines.append(f"  接收比提升 = {eta_adj / eta_base:.1f} 倍")
    lines.append("")

    # 采样阶数收敛性检验
    conv = []
    for m in CONV_M:
        bary_m = triangle_samples(m)
        ea, _ = trace_ratio(verts_adj, centers_adj, R, n, adj["focus"], bary_m)
        eb, _ = trace_ratio(verts_base, centers_base, R, n, adj["focus"], bary_m)
        conv.append((m, m * (m + 1) // 2, ea, eb))
        print(f"M={m:2d}（每面板 {(m * (m + 1)) // 2:3d} 点）："
              f"调节后 = {ea:.6f}，基准 = {eb:.6f}")

    lines += ["采样阶数收敛性检验",
              f"{'阶数 m':<8}{'每面板采样点':<14}{'调节后接收比':<16}{'基准接收比':<16}"]
    for m, s, ea, eb in conv:
        lines.append(f"{m:<8d}{s:<14d}{ea:<16.6f}{eb:<16.6f}")
    lines.append("")

    out = Path(__file__).resolve().parent / "q3_result.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"结果已写入 {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
