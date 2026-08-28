# -*- coding: utf-8 -*-
"""问题二三种调节方案的实验与比较。

方案一：仅考虑节点拟合优度，min Σ_i (e_i - x_i)²
方案二：考虑反射面板中心的拟合优度，min Σ_i (e_i - x_i)² + Σ_k Δ_k²
方案三：考虑整个面板的拟合优度，min Σ_k ∫_{S_k} δ(M)² dS

三方案均在 |x_i|≤0.6 与相邻节点间距变化 ≤0.07% 约束下求解。
内层用坐标下降法（精确非线性边约束，可行区间解析给出）；
外层以焦距 p 为决策变量（嵌套联合优化，粗扫+细扫）。
"""
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize_scalar

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "题目与附件与思路"
APERTURE_RADIUS = 150.0
EXT_LIMIT = 0.6
EDGE_TOL = 0.0007
ALPHA, BETA = 36.795, 78.169


# ---------- 数据与几何 ----------

def load_data():
    """加载附件1节点坐标与附件3面板拓扑。"""
    df1 = pd.read_csv(DATA / "附件1.csv", encoding="gbk")
    names = df1.iloc[:, 0].astype(str).tolist()
    pos = df1.iloc[:, 1:4].to_numpy(dtype=float)
    df3 = pd.read_csv(DATA / "附件3.csv", encoding="gbk")
    idx = {nm: i for i, nm in enumerate(names)}
    panels = df3.iloc[:, 0:3].apply(lambda s: s.map(idx)).to_numpy(dtype=int)
    return names, pos, panels


def axis_vector(alpha_deg, beta_deg):
    """由方位角、仰角计算观测轴单位向量。"""
    a, b = np.deg2rad(alpha_deg), np.deg2rad(beta_deg)
    return np.array([np.cos(b) * np.cos(a), np.cos(b) * np.sin(a), np.sin(b)])


def radial_extensions(pos, n, p, R):
    """各位置沿径向到达抛物面的伸缩量（向球心为正）。

    对位置 M，径向单位向量 u=M/|M|，A=u·n，求解
        (1-A²)t² - 4pA·t + 4p(B-p) = 0,  B=-0.534R
    取正根 t 为理想交点半径，伸缩量 e = |M| - t。
    """
    R_i = np.linalg.norm(pos, axis=1)
    u = pos / R_i[:, None]
    B = -0.534 * R
    A = u @ n
    a2 = 1.0 - A * A
    disc = 16.0 * p * (p - B * a2)
    t = np.empty_like(A)
    mask = a2 > 1e-12
    t[mask] = (4.0 * p * A[mask] + np.sqrt(disc[mask])) / (2.0 * a2[mask])
    t[~mask] = (B - p) / A[~mask]
    return R_i - t


def mesh_edges(panels):
    """提取面板所有边（去重）。"""
    edges = np.vstack([panels[:, [0, 1]], panels[:, [1, 2]], panels[:, [2, 0]]])
    return np.unique(np.sort(edges, axis=1), axis=0)


def panel_sphere_centers(verts, R):
    """由每块面板三顶点（K,3,3）反演半径 R 的面板球心（K,3）。

    先求三点外接圆心 P0（克莱姆法则），再沿平面法向偏移 sqrt(R²-r0²)，
    取与原点同侧（凹侧朝球心）。
    """
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


def panel_centers_on_sphere(verts, R):
    """调节后面板在球面上的中心点 G_k = O_k + R*(A+B+D-3O_k)/|...|。"""
    centers = panel_sphere_centers(verts, R)
    sum_verts = verts.sum(axis=1)
    direction = sum_verts - 3 * centers
    norm = np.linalg.norm(direction, axis=1)[:, None]
    direction = direction / np.where(norm > 1e-12, norm, 1.0)
    return centers + R * direction


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


def triangle_bary(m):
    """重心坐标采样点 (S,3)，等权。"""
    bary = []
    for i in range(m):
        for j in range(m - i):
            b, g = (i + 1 / 3) / m, (j + 1 / 3) / m
            bary.append([1 - b - g, b, g])
    return np.array(bary)


def setup(pos, panels, n):
    """预计算问题几何常量与节点-面板映射。"""
    R = np.linalg.norm(pos, axis=1).mean()
    dist_to_axis = np.linalg.norm(pos - np.outer(pos @ n, n), axis=1)
    in_ap = dist_to_axis <= APERTURE_RADIUS
    ap_idx = np.where(in_ap)[0]
    nap = ap_idx.size
    R_all = np.linalg.norm(pos, axis=1)
    edges = mesh_edges(panels)
    em = in_ap[edges[:, 0]] | in_ap[edges[:, 1]]
    ie, je = edges[em, 0], edges[em, 1]
    Ri, Rj = R_all[ie], R_all[je]
    L0 = np.linalg.norm(pos[ie] - pos[je], axis=1)
    cospsi = (Ri ** 2 + Rj ** 2 - L0 ** 2) / (2.0 * Ri * Rj)
    sin2 = np.maximum(1.0 - cospsi ** 2, 0.0)
    Lmin, Lmax = L0 * (1 - EDGE_TOL), L0 * (1 + EDGE_TOL)

    # 口径内面板
    ap_panels = in_ap[panels].all(axis=1)
    panels_ap = panels[ap_panels]

    # 节点->局部编号映射
    loc = np.full(len(pos), -1, dtype=int)
    loc[ap_idx] = np.arange(nap)
    li, lj = loc[ie], loc[je]

    # 节点->关联边映射
    inc = [[] for _ in range(nap)]
    for e in range(ie.size):
        if li[e] >= 0:
            inc[li[e]].append(e)
        if lj[e] >= 0:
            inc[lj[e]].append(e)
    inc_arr = [np.array(lst, dtype=int) for lst in inc]

    # 节点->关联面板映射（仅口径内面板）
    node_panels = [[] for _ in range(len(pos))]
    for k, pnl in enumerate(panels_ap):
        for v in pnl:
            node_panels[v].append(k)
    node_panels_arr = [np.array(lst, dtype=int) for lst in node_panels]

    # 口径内面板顶点的局部编号
    panels_ap_local = loc[panels_ap]

    return dict(R=R, n=n, in_ap=in_ap, ap_idx=ap_idx, nap=nap,
                R_all=R_all, ie=ie, je=je, L0=L0, cospsi=cospsi, sin2=sin2,
                Lmin=Lmin, Lmax=Lmax, li=li, lj=lj, inc=inc_arr,
                ap_panels=ap_panels, panels_ap=panels_ap,
                panels_ap_local=panels_ap_local, node_panels=node_panels_arr)


# ---------- 边约束可行区间（坐标下降法用） ----------

def edge_interval_Rp(e, Rp_other, R_base, ctx):
    """给定对端半径与本端基线半径，返回本端半径的可行区间。

    边长约束 |L1-L0|/L0 <= tol 转化为本端半径的可行区间（含基线值的连通分量）。
    """
    c, s2 = ctx["cospsi"][e], ctx["sin2"][e]
    base = Rp_other * c
    d_hi = ctx["Lmax"][e] ** 2 - Rp_other ** 2 * s2
    if d_hi < 0:
        return -np.inf, np.inf  # 对端偏离过大，本端无法补救，不限制
    r_lo, r_hi = base - np.sqrt(d_hi), base + np.sqrt(d_hi)
    d_lo = ctx["Lmin"][e] ** 2 - Rp_other ** 2 * s2
    if d_lo <= 0:
        return r_lo, r_hi
    s_lo, s_hi = base - np.sqrt(d_lo), base + np.sqrt(d_lo)
    return (s_hi, r_hi) if R_base >= base else (r_lo, s_lo)


def node_feasible_interval(i, Rp, ctx):
    """节点 i 的伸缩量 x_i 的可行区间 [lo, hi]（由所有关联边的边约束交集）。"""
    lo, hi = -EXT_LIMIT, EXT_LIMIT
    ni = ctx["ap_idx"][i]
    R_base = ctx["R_all"][ni]
    for e in ctx["inc"][i]:
        other = ctx["je"][e] if ctx["ie"][e] == ni else ctx["ie"][e]
        lo_e, hi_e = edge_interval_Rp(e, Rp[other], R_base, ctx)
        lo = max(lo, R_base - hi_e)
        hi = min(hi, R_base - lo_e)
    if lo > hi:
        lo = hi = 0.5 * (lo + hi)
    return lo, hi


def max_rel(x, ctx):
    """相邻节点间距最大相对变化。"""
    Rp = ctx["R_all"].copy()
    Rp[ctx["ap_idx"]] -= x
    L1 = np.sqrt(Rp[ctx["ie"]] ** 2 + Rp[ctx["je"]] ** 2
                 - 2.0 * Rp[ctx["ie"]] * Rp[ctx["je"]] * ctx["cospsi"])
    return np.abs(L1 - ctx["L0"]) / ctx["L0"]


def adjust_nodes(pos, x, ctx):
    """由伸缩量计算调节后节点位置。"""
    u = pos / ctx["R_all"][:, None]
    pos_adj = pos.copy()
    pos_adj[ctx["ap_idx"]] = u[ctx["ap_idx"]] * (ctx["R_all"][ctx["ap_idx"]] - x)[:, None]
    return pos_adj


# ---------- 坐标下降法内层求解 ----------

def coordinate_descent(e_ap, ctx, obj_node_func, sweeps=60, seed=0):
    """坐标下降法：在边约束下最小化 obj_node_func(i, x_i, state)。

    obj_node_func(i, x_i, Rp, pos_adj): 返回节点 i 取 x_i 时该节点的目标贡献
    （仅与 i 关联的面板/节点项；不包含不依赖 x_i 的常数项）。
    """
    rng = np.random.default_rng(seed)
    nap = ctx["nap"]
    x = np.clip(e_ap, -EXT_LIMIT, EXT_LIMIT)
    Rp = ctx["R_all"].copy()
    pos_adj = adjust_nodes(None, x, ctx)  # 需要 pos，这里先占位

    # 实际需要 pos，从 ctx 取
    # 注：pos 不在 ctx 中，由闭包传入
    raise NotImplementedError  # 由具体方案覆盖


def coordinate_descent_generic(pos, e_ap, ctx, node_obj, sweeps=60, seed=0):
    """通用坐标下降法。

    node_obj(i, x_i, Rp, pos_adj): 返回节点 i 取 x_i 时的目标增量
        （仅该节点影响的项；需与总目标一致的单调函数）。
    返回 x（伸缩量向量）。
    """
    rng = np.random.default_rng(seed)
    nap = ctx["nap"]
    x = np.clip(e_ap, -EXT_LIMIT, EXT_LIMIT)
    Rp = ctx["R_all"].copy()
    Rp[ctx["ap_idx"]] -= x
    pos_adj = adjust_nodes(pos, x, ctx)

    for _ in range(sweeps):
        for i in rng.permutation(nap):
            lo, hi = node_feasible_interval(i, Rp, ctx)
            if lo >= hi:
                continue
            # 在 [lo, hi] 上对 node_obj(i, x_i, Rp, pos_adj) 做一维最小化
            # 用黄金分割法
            x_opt = golden_section(lambda xi: node_obj(i, xi, Rp, pos_adj), lo, hi)
            x[i] = x_opt
            ni = ctx["ap_idx"][i]
            Rp[ni] = ctx["R_all"][ni] - x_opt
            pos_adj[ni] = (pos[ni] / ctx["R_all"][ni]) * Rp[ni]
    return x


def golden_section(f, lo, hi, tol=1e-6, max_iter=50):
    """黄金分割法一维最小化。"""
    gr = (np.sqrt(5) - 1) / 2
    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)
    fc = f(c)
    fd = f(d)
    for _ in range(max_iter):
        if hi - lo < tol:
            break
        if fc < fd:
            hi = d
            d = c
            fd = fc
            c = hi - gr * (hi - lo)
            fc = f(c)
        else:
            lo = c
            c = d
            fc = fd
            d = lo + gr * (hi - lo)
            fd = f(d)
    return 0.5 * (lo + hi)


# ---------- 三方案内层求解 ----------

def solve_scheme1(pos, ctx, p, sweeps=60):
    """方案一：min Σ(e_i - x_i)²。

    一维最优 x_i = clip(e_i, lo, hi)（闭式解）。
    """
    n = ctx["n"]
    e_ap = radial_extensions(pos[ctx["in_ap"]], n, p, ctx["R"])
    rng = np.random.default_rng(0)
    nap = ctx["nap"]
    x = np.clip(e_ap, -EXT_LIMIT, EXT_LIMIT)
    Rp = ctx["R_all"].copy()
    Rp[ctx["ap_idx"]] -= x

    for _ in range(sweeps):
        for i in rng.permutation(nap):
            lo, hi = node_feasible_interval(i, Rp, ctx)
            x[i] = min(max(e_ap[i], lo), hi)
            ni = ctx["ap_idx"][i]
            Rp[ni] = ctx["R_all"][ni] - x[i]
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
    e_ap = radial_extensions(pos[ctx["in_ap"]], n, p, R)
    rng = np.random.default_rng(0)
    nap = ctx["nap"]
    if x_init is not None:
        x = np.clip(x_init.copy(), -EXT_LIMIT, EXT_LIMIT)
    else:
        x = np.clip(e_ap, -EXT_LIMIT, EXT_LIMIT)
    Rp = R_all.copy()
    Rp[ctx["ap_idx"]] -= x
    pos_adj = adjust_nodes(pos, x, ctx)

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
            delta = radial_extensions(G[None], n, p, R)[0]
            val += lam_panel * delta ** 2
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
    e_ap = radial_extensions(pos[ctx["in_ap"]], n, p, R)
    rng = np.random.default_rng(0)
    nap = ctx["nap"]
    if x_init is not None:
        x = np.clip(x_init.copy(), -EXT_LIMIT, EXT_LIMIT)
    else:
        x = np.clip(e_ap, -EXT_LIMIT, EXT_LIMIT)
    Rp = R_all.copy()
    Rp[ctx["ap_idx"]] -= x
    pos_adj = adjust_nodes(pos, x, ctx)

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
            deltas = radial_extensions(M_sphere, n, p, R)
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
    rel_max = float(max_rel(x, ctx).max())

    # 面板中心残差
    pos_adj = adjust_nodes(pos, x, ctx)
    verts = pos_adj[panels_ap]
    G = panel_centers_on_sphere(verts, R)
    deltas_center = radial_extensions(G, n, p, R)
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
    deltas_surf = radial_extensions(M_flat, n, p, R)
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

def optimize_p(pos, ctx, solve_func, p_center=140.42, p_half=0.5, p_step=0.1,
               verbose=True):
    """外层 p 优化：粗扫 + 细扫，目标为节点 RMS。"""
    # 粗扫
    ps = np.arange(p_center - p_half, p_center + p_half + p_step / 2, p_step)
    best = (1e9, None, None, None)
    for p in ps:
        x, e_ap = solve_func(pos, ctx, p)
        rms = float(np.sqrt(np.mean((x - e_ap) ** 2)))
        if verbose:
            print(f"    p={p:.4f}: node_RMS={rms:.6f}")
        if rms < best[0]:
            best = (rms, p, x, e_ap)

    # 细扫
    p0 = best[1]
    p_fine = np.arange(p0 - p_step, p0 + p_step + p_step / 10, p_step / 5)
    for p in p_fine:
        if abs(p - p0) < 1e-9:
            continue
        x, e_ap = solve_func(pos, ctx, p)
        rms = float(np.sqrt(np.mean((x - e_ap) ** 2)))
        if verbose:
            print(f"    p={p:.4f}: node_RMS={rms:.6f}")
        if rms < best[0]:
            best = (rms, p, x, e_ap)

    return best  # (rms, p, x, e_ap)


# ---------- 主函数 ----------

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    names, pos, panels = load_data()
    n = axis_vector(ALPHA, BETA)
    ctx = setup(pos, panels, n)
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

    # 打印表格
    headers = ["指标", "方案一", "方案二", "方案三"]
    rows = [
        ("最优焦距 p* (m)", "p", ".4f"),
        ("节点残差 RMS (mm)", "node_rms", ".4f"),
        ("节点残差最大值 (mm)", "node_max", ".4f"),
        ("面板中心残差 RMS (mm)", "center_rms", ".4f"),
        ("面板中心残差最大值 (mm)", "center_max", ".4f"),
        ("整面板积分残差 RMS (mm)", "integ_rms", ".4f"),
        ("整面板积分残差最大值 (mm)", "integ_max", ".4f"),
        ("伸缩量范围 [min,max] (mm)", None, None),
        ("伸缩量均值 (mm)", "ext_mean", ".4f"),
        ("边长最大相对变化 (%)", "rel_max", ".5f"),
        ("计算耗时 (s)", "time", ".1f"),
    ]
    print(f"\n{'指标':<28} {'方案一':<16} {'方案二':<16} {'方案三':<16}")
    print("-" * 76)
    for label, key, fmt in rows:
        if key is None:
            # 伸缩量范围
            vals = []
            for name in ["方案一", "方案二", "方案三"]:
                m = metrics[name]
                vals.append(f"[{m['ext_min']*1000:.2f}, {m['ext_max']*1000:.2f}]")
            print(f"{label:<28} {vals[0]:<16} {vals[1]:<16} {vals[2]:<16}")
        else:
            vals = []
            for name in ["方案一", "方案二", "方案三"]:
                v = metrics[name][key]
                if "mm" in label:
                    v = v * 1000
                vals.append(f"{v:{fmt}}")
            print(f"{label:<28} {vals[0]:<16} {vals[1]:<16} {vals[2]:<16}")

    # 保存比较结果到文件
    out_path = Path(__file__).resolve().parent / "schemes_compare_result.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("三方案比较结果\n")
        f.write(f"基准球面半径 R = {ctx['R']:.4f} m\n")
        f.write(f"口径内节点数 = {ctx['nap']}, 面板数 = {int(ctx['ap_panels'].sum())}\n\n")
        f.write(f"{'指标':<28} {'方案一':<16} {'方案二':<16} {'方案三':<16}\n")
        f.write("-" * 76 + "\n")
        for label, key, fmt in rows:
            if key is None:
                vals = []
                for name in ["方案一", "方案二", "方案三"]:
                    m = metrics[name]
                    vals.append(f"[{m['ext_min']*1000:.2f}, {m['ext_max']*1000:.2f}]")
                f.write(f"{label:<28} {vals[0]:<16} {vals[1]:<16} {vals[2]:<16}\n")
            else:
                vals = []
                for name in ["方案一", "方案二", "方案三"]:
                    v = metrics[name][key]
                    if "mm" in label:
                        v = v * 1000
                    vals.append(f"{v:{fmt}}")
                f.write(f"{label:<28} {vals[0]:<16} {vals[1]:<16} {vals[2]:<16}\n")
    print(f"\n比较结果已保存至 {out_path}")


if __name__ == "__main__":
    main()
