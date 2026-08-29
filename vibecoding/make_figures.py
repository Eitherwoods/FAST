# -*- coding: utf-8 -*-
"""生成论文配图（输出到 ../figures/，PNG 300dpi + PDF 矢量各一份）。

图 1  fig_p_sensitivity   接收比与残差 RMS 随焦距 p 的变化（读 p_sensitivity_result.txt）
图 2  fig_spot_pattern    反射光落点分布：基准球面 vs 调节后（射线追踪，M=12）
图 3  fig_extension_field 伸缩量分布：理想交点 vs 约束下可行解（读 result.xlsx）
图 4  fig_scheme_compare  三方案综合比较柱状图（读 schemes_compare_result.txt）
图 5  fig_inner_solver    内层求解器对照柱状图（当前代码现算，与提交解同源）

所有数值均来自当前口径的结果文件或由同一代码现算，与 result.xlsx / q3_result.txt 一致。
用法：python make_figures.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (ALPHA_Q2, BETA_Q2, APERTURE_RADIUS, P_BOUNDS,
                    axis_vector, build_context, fit_best, fit_coordinate_descent,
                    fit_penalty, load_data, panel_sphere_centers,
                    radial_extensions, triangle_bary)
import q3

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110

BASE = Path(__file__).resolve().parent.parent
FIG = BASE / "figures"
VIB = Path(__file__).resolve().parent

C_BLUE, C_ORANGE, C_GRAY, C_GREEN, C_RED = "#185FA5", "#D85A30", "#888780", "#3B6D11", "#A32D2D"


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  已生成 figures/{name}.png / .pdf")


def read_sensitivity():
    """解析 p_sensitivity_result.txt 的数据表。"""
    rows = []
    for line in (VIB / "p_sensitivity_result.txt").read_text(encoding="utf-8").splitlines():
        t = line.split()
        if len(t) == 6:
            try:
                rows.append([float(v) for v in t])
            except ValueError:
                pass
    a = np.array(rows)          # 列: p, 理想偏差, RMS(mm), |x|max(mm), 边长%, eta
    return a[:, 0], a[:, 2], a[:, 5]


def read_schemes():
    """解析 schemes_compare_result.txt 的三方案指标（单位 mm）。"""
    keys = ["节点残差 RMS", "面板中心残差 RMS", "整面板积分残差 RMS"]
    data = {}
    for line in (VIB / "schemes_compare_result.txt").read_text(encoding="utf-8").splitlines():
        for k in keys:
            if line.startswith(k):
                vals = [float(line[28 + 16 * i: 44 + 16 * i]) for i in range(3)]
                data[k] = vals
    return data


# ---------- 图 1：焦距敏感性 ----------

def fig_p_sensitivity():
    p, rms, eta = read_sensitivity()
    fig, ax1 = plt.subplots(figsize=(6.6, 4.0))
    ax1.plot(p, rms, "o-", color=C_BLUE, lw=1.8, ms=4, label="残差 RMS（左轴）")
    ax1.set_xlabel("焦距 $p$ (m)")
    ax1.set_ylabel("残差 RMS (mm)", color=C_BLUE)
    ax1.tick_params(axis="y", labelcolor=C_BLUE)
    ax1.set_ylim(0, 700)
    i_fit = int(np.argmin(rms))
    ax1.annotate("贴近优先：RMS 最小\np=140.3743, eta=0.341",
                 xy=(p[i_fit], rms[i_fit]), xytext=(140.45, 330),
                 fontsize=9, color=C_BLUE,
                 arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1))

    ax2 = ax1.twinx()
    ax2.plot(p, eta, "s-", color=C_ORANGE, lw=1.8, ms=4, label="接收比 eta（右轴）")
    ax2.set_ylabel("接收比 eta", color=C_ORANGE)
    ax2.tick_params(axis="y", labelcolor=C_ORANGE)
    ax2.set_ylim(0, 0.6)
    i_eta = int(np.argmax(eta))
    ax2.annotate("接收优先：eta 最大\np~=140.45, eta=0.494",
                 xy=(p[i_eta], eta[i_eta]), xytext=(140.85, 0.45),
                 fontsize=9, color=C_ORANGE,
                 arrowprops=dict(arrowstyle="->", color=C_ORANGE, lw=1))
    ax2.axhline(0.05279, color=C_GRAY, ls="--", lw=1)
    ax2.text(141.55, 0.065, "基准球面 eta=0.053", fontsize=9, color=C_GRAY)

    fig.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "fig_p_sensitivity")


# ---------- 图 2：反射光落点分布 ----------

def spot_scatter(verts, centers, R, n, P, bary, max_pts=40000):
    """返回反射光落点在接收平面内的坐标 (u, v)，以馈源舱 P 为原点 (m)。"""
    S = np.einsum("sv,kvx->ksx", bary, verts)
    rel = S - centers[:, None, :]
    M = centers[:, None, :] + R * rel / np.linalg.norm(rel, axis=2)[:, :, None]
    M = M.reshape(-1, 3)
    O = np.repeat(centers, bary.shape[0], axis=0)
    keep = np.linspace(0, len(M) - 1, min(len(M), max_pts)).astype(int)
    M, O = M[keep], O[keep]

    N = (M - O) / R
    cosi = N @ n
    refl = -n + 2.0 * cosi[:, None] * N
    den = np.where(np.abs(refl @ n) < 1e-12, 1e-12, refl @ n)
    t = ((P - M) @ n) / den
    Q = M + t[:, None] * refl

    e1 = np.cross(n, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    d = Q - P
    return d @ e1, d @ e2


def fig_spot_pattern():
    names, pos, panels = load_data()
    adj = q3.load_adjustment()
    R, n, in_ap, pos_adj = adj["R"], adj["n"], adj["in_ap"], adj["pos_adj"]
    P = adj["focus"]
    ap_panels = in_ap[panels].any(axis=1)
    bary = triangle_bary(12)

    va = pos_adj[panels[ap_panels]]
    ca = panel_sphere_centers(va, R)
    ua, va2 = spot_scatter(va, ca, R, n, P, bary)
    vb = pos[panels[ap_panels]]
    ub, vb3 = spot_scatter(vb, np.zeros((ap_panels.sum(), 3)), R, n, P, bary)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3), sharex=True, sharey=True)
    for ax, u, v, title in [
            (axes[0], ub, vb3, f"基准反射球面（eta = {q3.trace_ratio(vb, np.zeros((ap_panels.sum(), 3)), R, n, P, bary)[0]:.3f}）"),
            (axes[1], ua, va2, f"调节后反射面（eta = {q3.trace_ratio(va, ca, R, n, P, bary)[0]:.3f}）")]:
        ax.scatter(u, v, s=1, alpha=0.12, color=C_BLUE, rasterized=True)
        th = np.linspace(0, 2 * np.pi, 128)
        ax.plot(0.5 * np.cos(th), 0.5 * np.sin(th), color=C_RED, lw=1.5, label="馈源舱有效区域 (r=0.5 m)")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("接收平面内坐标 $u$ (m)")
        ax.set_aspect("equal")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("接收平面内坐标 $v$ (m)")
    axes[0].set_xlim(-30, 30)
    axes[0].set_ylim(-30, 30)
    fig.tight_layout()
    save(fig, "fig_spot_pattern")


# ---------- 图 3：伸缩量分布 ----------

def fig_extension_field():
    names, pos, panels = load_data()
    adj = q3.load_adjustment()
    R, n, p = adj["R"], adj["n"], adj["p"]
    pos_adj = adj["pos_adj"]
    in_ap = adj["in_ap"]
    rho = np.linalg.norm(pos[in_ap] - np.outer(pos[in_ap] @ n, n), axis=1)
    # 调整后节点相对基准球面的径向伸缩量：沿基准球面（球心为原点）径向投影
    rad = pos[in_ap] / np.linalg.norm(pos[in_ap], axis=1, keepdims=True)
    x_fit = np.sum((pos_adj[in_ap] - pos[in_ap]) * rad, axis=1) * 1000  # mm
    e_ideal, _, _ = radial_extensions(pos[in_ap], n, p, R)

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.scatter(rho, e_ideal * 1000, s=6, color=C_GRAY, alpha=0.45, label="理想交点伸缩量 $\\hat\\delta_i$（不可行）")
    ax.scatter(rho, x_fit, s=9, color=C_BLUE, alpha=0.75, label="约束下可行伸缩量 $\\delta_i^{*}$")
    ax.axhline(600, color=C_RED, ls="--", lw=1)
    ax.axhline(-600, color=C_RED, ls="--", lw=1)
    ax.text(151, 640, "促动器限幅 ±0.6 m", fontsize=9, color=C_RED, ha="right")
    ax.set_xlabel("节点到观测轴的距离 $\\rho$ (m)")
    ax.set_ylabel("径向伸缩量 (mm)")
    ax.set_title(f"问题二伸缩量分布（p* = {p:.4f} m，692 个口径内节点）", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    save(fig, "fig_extension_field")


# ---------- 图 4：三方案比较 ----------

def fig_scheme_compare():
    data = read_schemes()
    labels = ["节点残差\nRMS", "面板中心残差\nRMS", "整面板积分残差\nRMS"]
    keys = list(data.keys())
    x = np.arange(3)
    w = 0.26
    colors = [C_BLUE, C_ORANGE, C_GREEN]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for i, name in enumerate(["方案一", "方案二", "方案三"]):
        vals = [data[k][i] for k in keys]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=name, color=colors[i])
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=1)
    ax.set_xticks(x, labels)
    ax.set_ylabel("残差 RMS (mm)")
    ax.set_title("三种调节方案的残差比较（p* = 140.3743 m）", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_ylim(0, 55)
    fig.tight_layout()
    save(fig, "fig_scheme_compare")


# ---------- 图 5：内层求解器对照（当前代码现算） ----------

def fig_inner_solver():
    names, pos, panels = load_data()
    n = axis_vector(ALPHA_Q2, BETA_Q2)
    ctx = build_context(pos, panels, n)
    p = 140.3743
    e_ap, _, _ = radial_extensions(pos[ctx["in_ap"]], n, p, ctx["R"])
    rms = lambda x: float(np.sqrt(np.mean((x - e_ap) ** 2)) * 1000)
    vals = [rms(fit_penalty(ctx, e_ap)),
            rms(fit_coordinate_descent(ctx, e_ap, sweeps=60, seed=0)),
            rms(fit_best(ctx, e_ap))]
    labels = ["罚函数法", "坐标下降法", "混合多起点\n（本文采用）"]

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    bars = ax.bar(labels, vals, 0.5, color=[C_GRAY, C_GRAY, C_BLUE])
    ax.bar_label(bars, fmt="%.2f mm", fontsize=9, padding=2)
    ax.set_ylabel("残差 RMS (mm)")
    ax.set_title(f"内层求解器对照（p = {p} m，方案一目标）", fontsize=11)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_ylim(0, 80)
    fig.tight_layout()
    save(fig, "fig_inner_solver")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    FIG.mkdir(exist_ok=True)
    print("生成论文配图 …")
    fig_p_sensitivity()
    fig_spot_pattern()
    fig_extension_field()
    fig_scheme_compare()
    fig_inner_solver()
    print("全部完成。")
