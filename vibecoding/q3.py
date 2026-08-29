# -*- coding: utf-8 -*-
"""问题三：基于第2问调节方案计算馈源舱接收比，并与基准反射球面比较。

模型：
- 调节后每块面板仍是半径 R 的球面片（刚体），由其三个调节后顶点反演球心 O_k：
  先求三顶点外接圆心 P0，再沿平面法向偏移 sqrt(R²-r0²)，取与原点同侧（凹侧朝球心）。
- 面板上采样点 M 的法向 N = (M-O_k)/R；入射平行光沿 -n，反射方向
      r = -n + 2(n·N)N
  反射线与过焦点 P、垂直 n 的接收平面交于
      Q = M + t r，t = (P-M)·n / (r·n)
  当 |Q-P| <= 0.5 m 时该面元的反射信号被馈源舱有效区域接收。
- 接收比 = 被接收面元的投影面积之和 / 全部面元的投影面积之和，
  面元按 |N·n|·dS（投影面积）计权，用三角形重心坐标网格采样数值积分。
- 基准反射球面：球心在原点，同样方法计算接收比作为对照。
- 验证：对理想抛物面直接采样做射线追踪，接收比应~=1。
"""
import sys

import numpy as np
from pathlib import Path

from common import (ALPHA_Q2, BETA_Q2, APERTURE_RADIUS, load_data,
                    panel_sphere_centers, triangle_bary)

ALPHA, BETA = ALPHA_Q2, BETA_Q2
SAMPLE_M = 20            # 每块面板三角形网格采样阶数，采样点数 m(m+1)/2
FEED_RADIUS = 0.5        # 馈源舱有效接收圆盘半径
CONV_M = (10, 20, 30)    # 采样阶数收敛性检验


# ---------- 射线追踪 ----------


def trace_ratio(verts, centers, R, n, P, bary):
    """顶点 (K,3,3)、球心 (K,3) 的面板集合的接收比。

    仅统计落在 300 m 照明区内（到观测轴距离 <=150 m）的采样点：口径边界处的
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

    # 照明区掩膜：采样点到观测轴（过 C 沿 n）的距离 <= 150 m
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
    """理想抛物面接收比（模型自洽性验证，应~=1）。"""
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

def load_adjustment(xlsx=None):
    """读取第 2 问的 result.xlsx 作为调节方案，保证问题三与提交结果同源。

    题目要求“基于第 2 问的反射面调节方案”计算接收比，故问题三不再重跑优化，
    而是直接以提交文件中的顶点与调整后节点坐标为准：
      - 顶点 V 反解观测轴 n = -V/|V| 与焦距 p = |V| - 0.534R；
      - 调整后节点坐标表给出口径内节点（口径外节点保持基准态）。
    """
    from openpyxl import load_workbook

    names, pos, panels = load_data()
    R = float(np.linalg.norm(pos, axis=1).mean())
    path = Path(xlsx) if xlsx else Path(__file__).resolve().parent / "result.xlsx"
    if not path.exists():
        sys.exit(f"未找到 {path}：请先运行 q2.py 生成第 2 问的调节方案。")

    wb = load_workbook(path, data_only=True)
    v = np.array(next(wb["理想抛物面顶点坐标"].iter_rows(
        min_row=2, max_row=2, max_col=3, values_only=True)), dtype=float)
    n = -v / np.linalg.norm(v)
    p = float(np.linalg.norm(v) - 0.534 * R)

    idx = {nm: i for i, nm in enumerate(names)}
    pos_adj = pos.copy()
    in_ap = np.zeros(len(names), dtype=bool)
    for row in wb["调整后主索节点编号及坐标"].iter_rows(min_row=2, values_only=True):
        i = idx[str(row[0])]
        pos_adj[i] = (float(row[1]), float(row[2]), float(row[3]))
        in_ap[i] = True

    return dict(R=R, n=n, p=p, in_ap=in_ap, pos_adj=pos_adj, vertex=v,
                focus=-0.534 * R * n, source=path.name)


def main():
    names, pos, panels = load_data()
    adj = load_adjustment()
    R, n, in_ap = adj["R"], adj["n"], adj["in_ap"]
    lines = ["问题三接收比计算结果",
             "",
             f"基准球面半径 R = {R:.4f} m",
             f"观测方向 alpha = {ALPHA}°, beta = {BETA}°",
             f"调节方案来源：{adj['source']}（第 2 问提交结果）",
             f"最优焦距 p = {adj['p']:.4f} m",
             f"馈源舱（焦点）P = ({adj['focus'][0]:.4f}, {adj['focus'][1]:.4f}, {adj['focus'][2]:.4f}) m",
             f"理想抛物面顶点 V = ({adj['vertex'][0]:.4f}, {adj['vertex'][1]:.4f}, {adj['vertex'][2]:.4f}) m",
             ""]
    print(f"调节方案（读自 {adj['source']}）：焦距 p = {adj['p']:.4f} m，"
          f"口径内节点 {in_ap.sum()} 个")

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

    bary = triangle_bary(SAMPLE_M)
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
              f"pi·150² = {disk:.2f} m²",
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
        bary_m = triangle_bary(m)
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
