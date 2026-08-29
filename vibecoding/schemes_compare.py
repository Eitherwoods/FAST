# -*- coding: utf-8 -*-
"""问题二三种调节方案的实验与比较。

方案一：仅考虑节点拟合优度，min Σ_i (e_i - x_i)²
方案二：考虑反射面板中心的拟合优度，min Σ_i (e_i - x_i)² + Σ_k Δ_k²
方案三：考虑整个面板的拟合优度，min Σ_k ∫_{S_k} δ(M)² dS

三方案均在 |x_i|≤0.6 与相邻节点间距变化 ≤0.07% 约束下求解。
方案一用 common.fit_best（罚函数解为初值 + 多起点坐标下降），与第2问提交的
result.xlsx 完全一致；外层 p 搜索也与 common.solve_adjustment 相同（minimize_scalar,
xatol=1e-4），三方案的 p* 与 result.xlsx 严格同源。方案二、三以方案一的解为初值继续坐标下降。
"""
import sys
import time
import numpy as np
from pathlib import Path
from scipy.optimize import minimize_scalar

import common as cm
from common import (EXT_LIMIT, ALPHA_Q2, BETA_Q2, axis_vector, golden_section,
                    load_data, node_feasible_interval, panel_sphere_centers,
                    radial_extensions, triangle_bary)

ALPHA, BETA = ALPHA_Q2, BETA_Q2


# ---------- 面板几何（标量版，方案二/三的逐节点增量计算用） ----------

def panel_sphere_center_single(A, B, D, R):
    """单块面板球心反演（标量版，避免 numpy 向量化开销）。"""
    u = B - A
    v = D - A
    uu = float(u @ u)
    vv = float(v @ v)
    uv = float(u @ v)
    R1 = float(B @ B - A @ A - 2 * A @ u)
    R2 = float(D @ D - A @ A - 2 * A @ v)
    det = 4.0 * (uu * vv - uv * uv)
    s = 2.0 * (R1 * vv - R2 * uv) / det
    t = 2.0 * (uu * R2 - uv * R1) / det
    P0 = A + s * u + t * v
    np_vec = np.cross(u, v)
    np_vec = np_vec / np.linalg.norm(np_vec)
    r0 = float(np.linalg.norm(P0 - A))
    h = np.sqrt(max(R * R - r0 * r0, 0.0))
    side = float(np.sign(-P0 @ np_vec))
    if side == 0.0:
        side = 1.0
    return P0 + side * h * np_vec


def panel_center_single(A, B, D, R):
    """单块面板在球面上的中心点 G（标量版）。"""
    Ok = panel_sphere_center_single(A, B, D, R)
    direction = (A + B + D) - 3 * Ok
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return Ok
    return Ok + R * direction / norm


def panel_centers_on_sphere(verts, R):
    """调节后面板在球面上的中心点 G_k = O_k + R*(A+B+D-3O_k)/|...|。"""
    centers = panel_sphere_centers(verts, R)
    direction = verts.sum(axis=1) - 3 * centers
    norm = np.linalg.norm(direction, axis=1)[:, None]
    return centers + R * direction / np.where(norm > 1e-12, norm, 1.0)


# ---------- 三方案内层求解 ----------

def solve_scheme1(pos, ctx, p, sweeps=60):
    """方案一：min Σ(e_i - x_i)²（与第2问 result.xlsx 同一求解器）。"""
    n = ctx["n"]
    e_ap, _, _ = radial_extensions(pos[ctx["in_ap"]], n, p, ctx["R"])
    x = cm.fit_best(ctx, e_ap, sweeps=sweeps)
    return x, e_ap


def solve_scheme2(pos, ctx, p, lam_panel=1.0, sweeps=12, x_init=None):
    """方案二：min Σ(e_i - x_i)² + lam_panel * Σ_k Δ_k²。

    Δ_k 为面板中心 G_k 到理想抛物面的径向偏差。
    每次更新 x_i 时，只需重算 i 关联面板的中心偏差（标量版函数，避免 numpy 开销）。
    可传入 x_init 做 warm start（如方案一的解）。
    """
    n = ctx["n"]
    R = ctx["R"]
    panels_ap = ctx["panels_ap"]
    node_panels = ctx["node_panels"]
    R_all = ctx["R_all"]
    e_ap, _, _ = radial_extensions(pos[ctx["in_ap"]], n, p, R)
    rng = np.random.default_rng(0)
    nap = ctx["nap"]
    if x_init is not None:
        x = np.clip(x_init.copy(), -EXT_LIMIT, EXT_LIMIT)
    else:
        x = np.clip(e_ap, -EXT_LIMIT, EXT_LIMIT)
    Rp = R_all.copy()
    Rp[ctx["ap_idx"]] -= x
    pos_adj = cm.adjust_nodes(pos, x, ctx)

    def node_obj(i, xi):
        """节点 i 取 xi 时的目标增量。"""
        val = (xi - e_ap[i]) ** 2
        ni = ctx["ap_idx"][i]
        new_pos = (pos[ni] / R_all[ni]) * (R_all[ni] - xi)
        for k in node_panels[ni]:
            pnl = panels_ap[k]
            # 构造临时顶点（只替换 ni 对应的顶点）
            verts = pos_adj[pnl].copy()
            for j in range(3):
                if pnl[j] == ni:
                    verts[j] = new_pos
                    break
            G = panel_center_single(verts[0], verts[1], verts[2], R)
            delta, _, _ = radial_extensions(G[None], n, p, R)
            val += lam_panel * delta[0] ** 2
        return val

    for _ in range(sweeps):
        for i in rng.permutation(nap):
            lo, hi = node_feasible_interval(i, Rp, ctx)
            if lo >= hi:
                continue
            xi_opt = golden_section(lambda xi: node_obj(i, xi), lo, hi, max_iter=30)
            x[i] = xi_opt
            ni = ctx["ap_idx"][i]
            Rp[ni] = R_all[ni] - xi_opt
            pos_adj[ni] = (pos[ni] / R_all[ni]) * Rp[ni]
    return x, e_ap


def solve_scheme3(pos, ctx, p, sample_m=4, sweeps=10, x_init=None):
    """方案三：min Σ_k ∫_{S_k} δ(M)² dS ≈ min Σ_k Σ_s δ_s² · dA_s。

    每块面板用 sample_m 阶重心坐标采样，采样点投影到面板球面后计算径向偏差。
    每次更新 x_i 时，只需重算 i 关联面板的采样点偏差（标量版球心反演）。
    可传入 x_init 做 warm start。
    """
    n = ctx["n"]
    R = ctx["R"]
    panels_ap = ctx["panels_ap"]
    node_panels = ctx["node_panels"]
    R_all = ctx["R_all"]
    e_ap, _, _ = radial_extensions(pos[ctx["in_ap"]], n, p, R)
    rng = np.random.default_rng(0)
    nap = ctx["nap"]
    if x_init is not None:
        x = np.clip(x_init.copy(), -EXT_LIMIT, EXT_LIMIT)
    else:
        x = np.clip(e_ap, -EXT_LIMIT, EXT_LIMIT)
    Rp = R_all.copy()
    Rp[ctx["ap_idx"]] -= x
    pos_adj = cm.adjust_nodes(pos, x, ctx)

    # 预计算面板面积（基准态近似不变）与采样重心坐标
    verts0 = pos[panels_ap]
    areas = 0.5 * np.linalg.norm(np.cross(verts0[:, 1] - verts0[:, 0],
                                          verts0[:, 2] - verts0[:, 0]), axis=1)
    bary = triangle_bary(sample_m)
    S = bary.shape[0]
    weights_per_panel = areas / S

    def node_obj(i, xi):
        """节点 i 取 xi 时的目标增量。"""
        val = 0.0
        ni = ctx["ap_idx"][i]
        new_pos = (pos[ni] / R_all[ni]) * (R_all[ni] - xi)
        for k in node_panels[ni]:
            pnl = panels_ap[k]
            verts = pos_adj[pnl].copy()
            for j in range(3):
                if pnl[j] == ni:
                    verts[j] = new_pos
                    break
            Ok = panel_sphere_center_single(verts[0], verts[1], verts[2], R)
            M_plane = bary @ verts  # (S,3)
            rel = M_plane - Ok[None, :]
            M_sphere = Ok[None, :] + R * rel / np.linalg.norm(rel, axis=1)[:, None]
            deltas, _, _ = radial_extensions(M_sphere, n, p, R)
            val += float(np.sum(deltas ** 2)) * weights_per_panel[k]
        return val

    for _ in range(sweeps):
        for i in rng.permutation(nap):
            lo, hi = node_feasible_interval(i, Rp, ctx)
            if lo >= hi:
                continue
            xi_opt = golden_section(lambda xi: node_obj(i, xi), lo, hi, max_iter=30)
            x[i] = xi_opt
            ni = ctx["ap_idx"][i]
            Rp[ni] = R_all[ni] - xi_opt
            pos_adj[ni] = (pos[ni] / R_all[ni]) * Rp[ni]
    return x, e_ap


# ---------- 评估指标 ----------

def evaluate(pos, ctx, x, e_ap, p, sample_m=8):
    """计算方案的综合评估指标。"""
    n = ctx["n"]
    panels_ap = ctx["panels_ap"]
    R = ctx["R"]

    # 节点残差
    node_rms = float(np.sqrt(np.mean((x - e_ap) ** 2)))
    node_max = float(np.abs(x - e_ap).max())

    # 伸缩量统计
    ext_max = float(np.abs(x).max())
    ext_min = float(x.min())
    ext_mean = float(x.mean())

    # 边长约束
    rel_max = float(cm.max_rel_change(x, ctx).max())

    # 面板中心残差
    pos_adj = cm.adjust_nodes(pos, x, ctx)
    verts = pos_adj[panels_ap]
    G = panel_centers_on_sphere(verts, R)
    deltas_center, _, _ = radial_extensions(G, n, p, R)
    center_rms = float(np.sqrt(np.mean(deltas_center ** 2)))
    center_max = float(np.abs(deltas_center).max())

    # 整个面板积分残差
    bary = triangle_bary(sample_m)
    S = bary.shape[0]
    verts0 = pos[panels_ap]
    areas = 0.5 * np.linalg.norm(np.cross(verts0[:, 1] - verts0[:, 0],
                                          verts0[:, 2] - verts0[:, 0]), axis=1)
    weights = np.repeat(areas / S, S)
    # 采样点投影到调节后面板球面
    centers_adj = panel_sphere_centers(verts, R)
    M_plane = np.einsum("sv,kvx->ksx", bary, verts)  # (K,S,3)
    rel = M_plane - centers_adj[:, None, :]
    M_sphere = centers_adj[:, None, :] + R * rel / np.linalg.norm(rel, axis=2)[:, :, None]
    M_flat = M_sphere.reshape(-1, 3)
    deltas_surf, _, _ = radial_extensions(M_flat, n, p, R)
    integ_rms = float(np.sqrt(np.sum(deltas_surf ** 2 * weights) / np.sum(weights)))
    integ_max = float(np.abs(deltas_surf).max())

    return dict(
        node_rms=node_rms, node_max=node_max,
        ext_min=ext_min, ext_max=ext_max, ext_mean=ext_mean,
        rel_max=rel_max,
        center_rms=center_rms, center_max=center_max,
        integ_rms=integ_rms, integ_max=integ_max,
    )


# ---------- 外层 p 优化 ----------

def optimize_p(pos, ctx, solve_func, p_bounds=cm.P_BOUNDS, xatol=1e-4,
               verbose=True):
    """外层 p 优化：与 common.solve_adjustment 相同的 minimize_scalar 流程。

    保证三方案的 p* 与第 2 问 result.xlsx 由同一搜索方式确定（同一确定性目标）。
    """
    cache = {}

    def fit_rms(p):
        key = round(float(p), 6)
        if key not in cache:
            x, e_ap = solve_func(pos, ctx, key)
            cache[key] = (float(np.sqrt(np.mean((x - e_ap) ** 2))), x, e_ap)
        if verbose:
            print(f"    p={key:.4f}: node_RMS={cache[key][0]:.6f}")
        return cache[key][0]

    p_opt = float(minimize_scalar(fit_rms, bounds=p_bounds, method="bounded",
                                  options={"xatol": xatol}).x)
    key = round(p_opt, 6)
    if key not in cache:
        fit_rms(key)
    rms, x, e_ap = cache[key]
    return rms, p_opt, x, e_ap


# ---------- 主函数 ----------

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    names, pos, panels = load_data()
    n = axis_vector(ALPHA, BETA)
    ctx = cm.build_context(pos, panels, n)
    print(f"基准球面半径 R = {ctx['R']:.4f} m")
    print(f"300 m 口径内节点数 = {ctx['nap']}")
    print(f"300 m 口径内面板数 = {int(ctx['ap_panels'].sum())}")
    print()

    results = {}

    # ===== 方案一 =====
    print("=" * 60)
    print("方案一：仅考虑节点拟合优度  min Σ(e_i - x_i)²")
    print("=" * 60)
    t0 = time.time()
    rms1, p1, x1, e1 = optimize_p(pos, ctx, solve_scheme1, verbose=True)
    t1 = time.time()
    print(f"  => 最优 p* = {p1:.4f}, 节点 RMS = {rms1:.6f}, 耗时 {t1-t0:.1f}s")
    results["方案一"] = dict(p=p1, x=x1, e=e1, time=t1-t0)

    # ===== 方案二 =====
    print("\n" + "=" * 60)
    print("方案二：节点+面板中心拟合  min Σ(e_i-x_i)² + Σ_k Δ_k²")
    print("=" * 60)
    t0 = time.time()
    # 方案二从方案一解 warm start，在方案一最优 p 下求解；再在附近粗扫看是否有改进
    x2, e2 = solve_scheme2(pos, ctx, p1, x_init=x1)
    rms2 = float(np.sqrt(np.mean((x2 - e2) ** 2)))
    p2 = p1
    print(f"    p={p1:.4f}: node_RMS={rms2:.6f}")
    for p_test in [p1 - 0.2, p1 + 0.2]:
        # warm start 用对应 p 的方案一解
        xt1, _ = solve_scheme1(pos, ctx, p_test, sweeps=30)
        xt, et = solve_scheme2(pos, ctx, p_test, x_init=xt1)
        rmst = float(np.sqrt(np.mean((xt - et) ** 2)))
        print(f"    p={p_test:.4f}: node_RMS={rmst:.6f}")
        if rmst < rms2:
            rms2, p2, x2, e2 = rmst, p_test, xt, et
    t2 = time.time()
    print(f"  => 最优 p* = {p2:.4f}, 节点 RMS = {rms2:.6f}, 耗时 {t2-t0:.1f}s")
    results["方案二"] = dict(p=p2, x=x2, e=e2, time=t2-t0)

    # ===== 方案三 =====
    print("\n" + "=" * 60)
    print("方案三：整个面板拟合  min Σ_k ∫_{S_k} δ(M)² dS")
    print("=" * 60)
    t0 = time.time()
    x3, e3 = solve_scheme3(pos, ctx, p1, x_init=x1)
    rms3 = float(np.sqrt(np.mean((x3 - e3) ** 2)))
    p3 = p1
    print(f"    p={p1:.4f}: node_RMS={rms3:.6f}")
    for p_test in [p1 - 0.2, p1 + 0.2]:
        xt1, _ = solve_scheme1(pos, ctx, p_test, sweeps=30)
        xt, et = solve_scheme3(pos, ctx, p_test, x_init=xt1)
        rmst = float(np.sqrt(np.mean((xt - et) ** 2)))
        print(f"    p={p_test:.4f}: node_RMS={rmst:.6f}")
        if rmst < rms3:
            rms3, p3, x3, e3 = rmst, p_test, xt, et
    t3 = time.time()
    print(f"  => 最优 p* = {p3:.4f}, 节点 RMS = {rms3:.6f}, 耗时 {t3-t0:.1f}s")
    results["方案三"] = dict(p=p3, x=x3, e=e3, time=t3-t0)

    # ===== 综合评估 =====
    print("\n" + "=" * 60)
    print("三方案综合比较")
    print("=" * 60)
    metrics = {}
    for name, r in results.items():
        m = evaluate(pos, ctx, r["x"], r["e"], r["p"], sample_m=8)
        m["p"] = r["p"]
        m["time"] = r["time"]
        metrics[name] = m

    headers = ("最优焦距 p* (m)", "节点残差 RMS (mm)", "节点残差最大值 (mm)",
               "面板中心残差 RMS (mm)", "面板中心残差最大值 (mm)",
               "整面板积分残差 RMS (mm)", "整面板积分残差最大值 (mm)",
               "伸缩量范围 [min,max] (mm)", "伸缩量均值 (mm)",
               "边长最大相对变化 (%)", "计算耗时 (s)")
    keys = ("p", "node_rms", "node_max", "center_rms", "center_max",
            "integ_rms", "integ_max", None, "ext_mean", "rel_max", "time")
    fmts = (".4f", ".4f", ".4f", ".4f", ".4f", ".4f", ".4f", None, ".4f", ".5f", ".1f")
    schemes = ("方案一", "方案二", "方案三")

    def cell(label, key, fmt, name):
        if key is None:
            m = metrics[name]
            return f"[{m['ext_min'] * 1000:.2f}, {m['ext_max'] * 1000:.2f}]"
        v = metrics[name][key]
        if "mm" in label:
            v *= 1000
        return f"{v:{fmt}}"

    def table_lines():
        lines = [f"{'指标':<28} {'方案一':<16} {'方案二':<16} {'方案三':<16}",
                 "-" * 76]
        for label, key, fmt in zip(headers, keys, fmts):
            lines.append(f"{label:<28} " + " ".join(
                f"{cell(label, key, fmt, s):<16}" for s in schemes))
        return lines

    print()
    print("\n".join(table_lines()))

    # 保存比较结果到文件
    out_path = Path(__file__).resolve().parent / "schemes_compare_result.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("三方案比较结果\n")
        f.write(f"基准球面半径 R = {ctx['R']:.4f} m\n")
        f.write(f"口径内节点数 = {ctx['nap']}, 面板数 = {int(ctx['ap_panels'].sum())}\n\n")
        f.write("\n".join(table_lines()) + "\n")
    print(f"\n比较结果已保存至 {out_path}")


if __name__ == "__main__":
    main()
