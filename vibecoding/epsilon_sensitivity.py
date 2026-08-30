# -*- coding: utf-8 -*-
"""索长限值 ε 与行程限幅的敏感性分析（补充“模型检验”一节）。

动机：正文反复断言“索长(相邻节点间距)约束是瓶颈、促动器行程不是瓶颈”，
但此前只有定性论证。本脚本定量检验该论断：

  · 索长限值 ε 扫描：ε ∈ {0.05%, 0.07%, 0.10%, 0.15%, 0.20%}
    固定论文最优焦距 p* = 140.3839 m，仅改变边约束容差，重新求内层可行解，
    观察 贴合残差RMS、|δ|max、接收比 η 随 ε 的变化。
    预期：RMS 随 ε 增大而单调下降并趋于饱和（说明 ε 越小约束越紧、瓶颈越明显）；
          |δ|max 在所有 ε 下均远低于 0.6 m（说明行程不是瓶颈）。
  · 行程限幅扫描：限幅 ∈ {0.4, 0.5, 0.6, 0.7, 0.8} m，ε 固定 0.07%
    预期：RMS 与 |δ|max 几乎不随限幅变化（说明当前解远未触及行程边界）。

为避免重跑昂贵的嵌套联合优化(单次 ~4 min)，p* 固定为论文值；
残差RMS(p) 在 p* 邻域极平缓(p_sensitivity.py 已证实)，固定 p* 不影响结论走向。

接收比 η 的计算与 p_sensitivity.py 完全一致(q3.trace_ratio, m=20)，
故 ε=0.07% 时应得到 η ≈ 0.3563，可与现有结果交叉核验。

用法：python epsilon_sensitivity.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as cm
import q3
from common import (ALPHA_Q2, BETA_Q2, APERTURE_RADIUS, EXT_LIMIT,
                    axis_vector, build_context, fit_best, load_data,
                    radial_extensions, triangle_bary, panel_sphere_centers,
                    adjust_nodes, max_rel_change)

P_STAR = 140.3839          # 论文问题二最优焦距 (m)
SAMPLE_M = 20              # 接收比采样阶数，与 p_sensitivity.py 一致
EPS_LIST = (0.0005, 0.0007, 0.0010, 0.0015, 0.0020)   # 索长限值
EXT_LIST = (0.4, 0.5, 0.6, 0.7, 0.8)                 # 行程限幅 (m)
C_BLUE, C_ORANGE, C_GRAY, C_GREEN, C_RED = (
    "#185FA5", "#D85A30", "#888780", "#3B6D11", "#A32D2D")


def eta_of(pos, panels, x, ctx, n, R, focus, bary, ap_panels):
    pos_adj = adjust_nodes(pos, x, ctx)
    verts = pos_adj[panels[ap_panels]]
    eta, _ = q3.trace_ratio(verts, panel_sphere_centers(verts, R), R, n, focus, bary)
    return eta


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    names, pos, panels = load_data()
    n = axis_vector(ALPHA_Q2, BETA_Q2)
    R = float(np.linalg.norm(pos, axis=1).mean())
    focus = -0.534 * R * n
    bary = triangle_bary(SAMPLE_M)
    t_all = time.time()

    # 基准反射球面接收比（对照）
    ap_any = np.linalg.norm(pos - np.outer(pos @ n, n), axis=1) <= APERTURE_RADIUS
    ap_panels = ap_any[panels].any(axis=1)
    eta_base, _ = q3.trace_ratio(pos[panels[ap_panels]],
                                 np.zeros((ap_panels.sum(), 3)), R, n, focus, bary)
    print(f"基准球面半径 R = {R:.4f} m，固定 p* = {P_STAR:.4f} m")
    print(f"基准反射球面接收比 = {eta_base:.6f}\n")

    # ===== 1. 索长限值 ε 扫描 =====
    print("【1】索长限值 ε 敏感性（行程限幅固定 %.1f m）" % EXT_LIMIT)
    print("-" * 78)
    print("%-10s %-12s %-12s %-12s %-10s" %
          ("ε (%)", "残差RMS(mm)", "|δ|max(m)", "边长变化(%)", "接收比η"))
    rows_e = []
    for eps in EPS_LIST:
        cm.EDGE_TOL = eps
        tol = eps * (1.0 - 1e-3)            # 求解用容差略紧于限值，吸收舍入
        ctx = build_context(pos, panels, n, tol=tol)
        e_ap, _, _ = radial_extensions(pos[ctx["in_ap"]], n, P_STAR, R)
        x = fit_best(ctx, e_ap)
        rms = float(np.sqrt(np.mean((x - e_ap) ** 2))) * 1000.0
        xmax = float(np.abs(x).max())
        rel = float(max_rel_change(x, ctx).max()) * 100.0
        eta = eta_of(pos, panels, x, ctx, n, R, focus, bary, ap_panels)
        rows_e.append((eps * 100, rms, xmax, rel, eta))
        print("%-10.4f %-12.3f %-12.4f %-12.5f %-10.6f" %
              (eps * 100, rms, xmax, rel, eta))
    print()

    # ===== 2. 行程限幅扫描 =====
    print("【2】行程限幅敏感性（索长限值固定 0.070%）")
    print("-" * 78)
    print("%-12s %-12s %-12s %-12s %-10s" %
          ("限幅(m)", "残差RMS(mm)", "|δ|max(m)", "边长变化(%)", "接收比η"))
    rows_x = []
    cm.EDGE_TOL = 0.0007
    for ext in EXT_LIST:
        cm.EXT_LIMIT = ext
        ctx = build_context(pos, panels, n, tol=0.0007 * (1.0 - 1e-3))
        e_ap, _, _ = radial_extensions(pos[ctx["in_ap"]], n, P_STAR, R)
        x = fit_best(ctx, e_ap)
        rms = float(np.sqrt(np.mean((x - e_ap) ** 2))) * 1000.0
        xmax = float(np.abs(x).max())
        rel = float(max_rel_change(x, ctx).max()) * 100.0
        eta = eta_of(pos, panels, x, ctx, n, R, focus, bary, ap_panels)
        rows_x.append((ext, rms, xmax, rel, eta))
        print("%-12.1f %-12.3f %-12.4f %-12.5f %-10.6f" %
              (ext, rms, xmax, rel, eta))
    cm.EXT_LIMIT = EXT_LIMIT
    print("\n总耗时 %.0fs" % (time.time() - t_all))

    # ===== 写结果文件 =====
    out = Path(__file__).resolve().parent / "epsilon_sensitivity_result.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write("索长限值 ε 与行程限幅敏感性分析\n")
        f.write(f"基准球面半径 R = {R:.4f} m，固定 p* = {P_STAR:.4f} m，"
                f"接收比采样阶数 m = {SAMPLE_M}\n")
        f.write(f"基准反射球面接收比 = {eta_base:.6f}\n\n")
        f.write("【1】索长限值 ε 扫描（行程限幅固定 %.1f m）\n" % EXT_LIMIT)
        f.write("%-10s %-12s %-12s %-12s %-10s\n" %
                ("ε(%)", "RMS(mm)", "|δ|max(m)", "边长(%)", "η"))
        for e, r, xm, rl, et in rows_e:
            f.write("%-10.4f %-12.3f %-12.4f %-12.5f %-10.6f\n" % (e, r, xm, rl, et))
        f.write("\n【2】行程限幅扫描（ε 固定 0.070%）\n")
        f.write("%-12s %-12s %-12s %-12s %-10s\n" %
                ("限幅(m)", "RMS(mm)", "|δ|max(m)", "边长(%)", "η"))
        for ex, r, xm, rl, et in rows_x:
            f.write("%-12.1f %-12.3f %-12.4f %-12.5f %-10.6f\n" % (ex, r, xm, rl, et))
    print(f"\n结果已写入 {out}")

    make_figure(rows_e, rows_x, eta_base,
                out.parent.parent / "figures" / "fig_epsilon_sensitivity")
    return 0


def make_figure(rows_e, rows_x, eta_base, base_path):
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.6))
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    eps = [r[0] for r in rows_e]
    rms_e = [r[1] for r in rows_e]
    xmax_e = [r[2] for r in rows_e]
    eta_e = [r[4] for r in rows_e]

    ext = [r[0] for r in rows_x]
    rms_x = [r[1] for r in rows_x]
    xmax_x = [r[2] for r in rows_x]
    eta_x = [r[4] for r in rows_x]

    # ---- Row 1: 索长限值 ε ----
    a = axes[0, 0]
    a.plot(eps, rms_e, "o-", color=C_BLUE, lw=2, ms=6)
    for xi, yi in zip(eps, rms_e):
        a.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                   xytext=(0, 7), ha="center", fontsize=8, color=C_BLUE)
    a.set_xlabel("索长限值 ε (%)")
    a.set_ylabel("贴合残差 RMS (mm)")
    a.set_title("(a) 残差随 ε 的变化")
    a.grid(alpha=0.3)

    a = axes[0, 1]
    a.axhline(0.6, color=C_RED, ls="--", lw=1.5, label="行程限幅 0.6 m")
    a.plot(eps, xmax_e, "s-", color=C_ORANGE, lw=2, ms=6)
    for xi, yi in zip(eps, xmax_e):
        a.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                   xytext=(0, 7), ha="center", fontsize=8, color=C_ORANGE)
    a.set_xlabel("索长限值 ε (%)")
    a.set_ylabel("最大伸缩量 |δ|max (m)")
    a.set_title("(b) |δ|max 始终远低于行程限幅")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    a = axes[0, 2]
    a.axhline(eta_base, color=C_GRAY, ls=":", lw=1.3, label=f"基准球面 {eta_base:.4f}")
    a.plot(eps, eta_e, "^-", color=C_GREEN, lw=2, ms=6)
    for xi, yi in zip(eps, eta_e):
        a.annotate(f"{yi:.4f}", (xi, yi), textcoords="offset points",
                   xytext=(0, 7), ha="center", fontsize=8, color=C_GREEN)
    a.set_xlabel("索长限值 ε (%)")
    a.set_ylabel("接收比 η")
    a.set_title("(c) 接收比随 ε 的变化")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    # ---- Row 2: 行程限幅 ----
    a = axes[1, 0]
    a.plot(ext, rms_x, "o-", color=C_BLUE, lw=2, ms=6)
    for xi, yi in zip(ext, rms_x):
        a.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                   xytext=(0, 7), ha="center", fontsize=8, color=C_BLUE)
    a.set_xlabel("行程限幅 (m)")
    a.set_ylabel("贴合残差 RMS (mm)")
    a.set_title("(d) 残差随行程限幅几乎不变")
    a.grid(alpha=0.3)

    a = axes[1, 1]
    lim = np.linspace(min(ext), max(ext), 50)
    a.plot(lim, lim, "--", color=C_GRAY, lw=1.3, label="限幅 = |δ|max")
    a.plot(ext, xmax_x, "s-", color=C_ORANGE, lw=2, ms=6)
    for xi, yi in zip(ext, xmax_x):
        a.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                   xytext=(0, 7), ha="center", fontsize=8, color=C_ORANGE)
    a.set_xlabel("行程限幅 (m)")
    a.set_ylabel("最大伸缩量 |δ|max (m)")
    a.set_title("(e) |δ|max 远在限幅下方")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    a = axes[1, 2]
    a.axhline(eta_base, color=C_GRAY, ls=":", lw=1.3, label=f"基准球面 {eta_base:.4f}")
    a.plot(ext, eta_x, "^-", color=C_GREEN, lw=2, ms=6)
    for xi, yi in zip(ext, eta_x):
        a.annotate(f"{yi:.4f}", (xi, yi), textcoords="offset points",
                   xytext=(0, 7), ha="center", fontsize=8, color=C_GREEN)
    a.set_xlabel("行程限幅 (m)")
    a.set_ylabel("接收比 η")
    a.set_title("(f) 接收比对行程限幅不敏感")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    fig.suptitle("索长限值 ε 与促动器行程限幅的敏感性分析", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext2 in ("png", "pdf"):
        fig.savefig(str(base_path) + "." + ext2, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  已生成 figures/fig_epsilon_sensitivity.png / .pdf")


if __name__ == "__main__":
    main()
