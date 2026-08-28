# -*- coding: utf-8 -*-
"""FAST 主动反射面：三问共用的数据、几何与优化组件。

几何约定
--------
- 基准球面球心 C 在坐标原点，半径 R = 全部主索节点模长均值（≈300.4 m）。
- 观测轴单位向量 n = (cosβcosα, cosβsinα, sinβ)，焦点（馈源舱）P = -0.534R·n。
- 理想抛物面：焦点 P、对称轴 n、焦距 p，顶点 V = -(0.534R + p)·n。
- 节点沿基准球面径向移动，伸缩量 x 以“向球心”为正；节点 i 径向单位向量 u_i，
  A_i = u_i·n，径向与抛物面交点半径 t 满足
      (1-A²)t² - 4pA·t + 4p(B-p) = 0,   B = -0.534R
  取正根，理想伸缩量 e = R_i - t。

约束
----
- 促动器行程 |x_i| ≤ 0.6 m；
- 相邻节点间距相对变化 ≤ 0.07%。径向移动不改变两节点的径向夹角 ψ，故调节后边长
  L1 = sqrt(Ri'² + Rj'² - 2Ri'Rj'cosψ)。

内层求解器
----------
- ``fit_coordinate_descent``：坐标下降，逐节点取闭式解 clip(e_i, lo, hi)，
  可行区间由边约束解析给出（方案一的目标函数），约束精确满足且残差最小，为默认。
- ``fit_penalty``：罚函数法（λ 逐级放大至可行），保留用于对照。
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize, minimize_scalar

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "题目与附件与思路"

APERTURE_RADIUS = 150.0   # 照明区半径（300 m 口径）
EXT_LIMIT = 0.6           # 促动器径向伸缩限幅
EDGE_TOL = 0.0007         # 相邻节点间距相对变化上限 0.07%
# 求解时按略紧的容差：result.xlsx 坐标只保留 6 位小数，舍入会给边长带来
# 约 1e-7 的相对扰动，直接压线求解会在提交文件中变成 0.07001% 的越界。
EDGE_TOL_SOLVE = EDGE_TOL * (1.0 - 5e-4)
P_BOUNDS = (139.9, 140.9)  # 焦距 p 的外层搜索区间

ALPHA_Q2, BETA_Q2 = 36.795, 78.169   # 问题二、三的观测方向


# ---------- 数据 ----------

def load_data():
    """加载附件1（主索节点坐标）与附件3（反射面板顶点编号）。"""
    df1 = pd.read_csv(DATA / "附件1.csv", encoding="gbk")
    names = df1.iloc[:, 0].astype(str).tolist()
    pos = df1.iloc[:, 1:4].to_numpy(dtype=float)
    df3 = pd.read_csv(DATA / "附件3.csv", encoding="gbk")
    idx = {nm: i for i, nm in enumerate(names)}
    panels = df3.iloc[:, 0:3].apply(lambda s: s.map(idx)).to_numpy(dtype=int)
    return names, pos, panels


def axis_vector(alpha_deg, beta_deg):
    """由方位角 α、仰角 β 计算观测轴单位向量。"""
    a, b = np.deg2rad(alpha_deg), np.deg2rad(beta_deg)
    return np.array([np.cos(b) * np.cos(a), np.cos(b) * np.sin(a), np.sin(b)])


# ---------- 几何 ----------

def radial_extensions(pos, n, p, R=None):
    """各位置沿径向到达抛物面的理想伸缩量（向球心为正）。

    R 为基准球面半径；缺省时取传入位置模长均值（与按节点子集调用保持一致）。

    返回 (e, u, R_i)：理想伸缩量、径向单位向量、各位置到球心距离。
    """
    R_i = np.linalg.norm(pos, axis=1)
    u = pos / R_i[:, None]
    B = -0.534 * (R_i.mean() if R is None else R)
    A = u @ n
    a2 = 1.0 - A * A
    disc = 16.0 * p * (p - B * a2)
    t = np.empty_like(A)
    mask = a2 > 1e-12
    t[mask] = (4.0 * p * A[mask] + np.sqrt(disc[mask])) / (2.0 * a2[mask])
    t[~mask] = (B - p) / A[~mask]
    return R_i - t, u, R_i


def mesh_edges(panels):
    """提取面板全部边（去重）。"""
    edges = np.vstack([panels[:, [0, 1]], panels[:, [1, 2]], panels[:, [2, 0]]])
    return np.unique(np.sort(edges, axis=1), axis=0)


def panel_sphere_centers(verts, R):
    """由每块面板三顶点（K,3,3）反演半径 R 的面板球心（K,3）。

    先求三顶点外接圆心 P0（克莱姆法则），再沿平面法向偏移 sqrt(R²-r0²)，
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


def triangle_bary(m):
    """三角形重心坐标采样点 (S,3)，S = m(m+1)/2，等权。"""
    bary = []
    for i in range(m):
        for j in range(m - i):
            b, g = (i + 1 / 3) / m, (j + 1 / 3) / m
            bary.append([1 - b - g, b, g])
    return np.array(bary)


# ---------- 调节上下文（口径、边几何、关联关系） ----------

def build_context(pos, panels, n, radius=APERTURE_RADIUS, tol=EDGE_TOL_SOLVE):
    """预计算照明区掩码、边约束几何与节点-边/面板关联。

    tol 为求解用的边长容差，默认略紧于 EDGE_TOL，以吸收结果文件的舍入误差。
    """
    R = np.linalg.norm(pos, axis=1).mean()
    dist_to_axis = np.linalg.norm(pos - np.outer(pos @ n, n), axis=1)
    in_ap = dist_to_axis <= radius
    ap_idx = np.where(in_ap)[0]
    nap = ap_idx.size
    R_all = np.linalg.norm(pos, axis=1)

    edges = mesh_edges(panels)
    em = in_ap[edges[:, 0]] | in_ap[edges[:, 1]]   # 含口径边界处的跨接边
    ie, je = edges[em, 0], edges[em, 1]
    L0 = np.linalg.norm(pos[ie] - pos[je], axis=1)
    cospsi = (R_all[ie] ** 2 + R_all[je] ** 2 - L0 ** 2) / (2.0 * R_all[ie] * R_all[je])

    # 节点 -> 局部编号、关联边
    loc = np.full(len(pos), -1, dtype=int)
    loc[ap_idx] = np.arange(nap)
    li, lj = loc[ie], loc[je]
    inc = [[] for _ in range(nap)]
    for e in range(ie.size):
        if li[e] >= 0:
            inc[li[e]].append(e)
        if lj[e] >= 0:
            inc[lj[e]].append(e)

    # 照明区面板与节点 -> 面板关联
    ap_panels = in_ap[panels].all(axis=1)
    panels_ap = panels[ap_panels]
    node_panels = [[] for _ in range(len(pos))]
    for k, pnl in enumerate(panels_ap):
        for v in pnl:
            node_panels[v].append(k)

    return dict(R=R, n=n, radius=radius, in_ap=in_ap, ap_idx=ap_idx, nap=nap,
                R_all=R_all, ie=ie, je=je, L0=L0, cospsi=cospsi,
                sin2=np.maximum(1.0 - cospsi ** 2, 0.0),
                Lmin=L0 * (1 - tol), Lmax=L0 * (1 + tol),
                li=li, lj=lj, inc=[np.asarray(a, dtype=int) for a in inc],
                ap_panels=ap_panels, panels_ap=panels_ap,
                panels_ap_local=loc[panels_ap],
                node_panels=[np.asarray(a, dtype=int) for a in node_panels])


def adjust_nodes(pos, x, ctx):
    """由伸缩量 x（口径内节点）计算调节后的全部节点位置。"""
    u = pos / ctx["R_all"][:, None]
    pos_adj = pos.copy()
    pos_adj[ctx["ap_idx"]] = u[ctx["ap_idx"]] * (ctx["R_all"][ctx["ap_idx"]] - x)[:, None]
    return pos_adj


def max_rel_change(x, ctx):
    """相邻节点间距相对变化（径向移动不改变径向夹角 ψ）。"""
    Rp = ctx["R_all"].copy()
    Rp[ctx["ap_idx"]] -= x
    L1 = np.sqrt(Rp[ctx["ie"]] ** 2 + Rp[ctx["je"]] ** 2
                 - 2.0 * Rp[ctx["ie"]] * Rp[ctx["je"]] * ctx["cospsi"])
    return np.abs(L1 - ctx["L0"]) / ctx["L0"]


# ---------- 边约束的解析可行区间（坐标下降用） ----------

def edge_interval_Rp(e, Rp_other, R_base, ctx):
    """给定边 e 的对端半径与本端基线半径，返回本端半径的可行区间。

    边长约束 |L1-L0|/L0 ≤ tol 化为本端半径的区间（取含基线值的连通分量）。
    """
    c, s2 = ctx["cospsi"][e], ctx["sin2"][e]
    base = Rp_other * c
    d_hi = ctx["Lmax"][e] ** 2 - Rp_other ** 2 * s2
    if d_hi < 0:
        return -np.inf, np.inf   # 对端偏离过大，本端无法补救，不限制
    r_lo, r_hi = base - np.sqrt(d_hi), base + np.sqrt(d_hi)
    d_lo = ctx["Lmin"][e] ** 2 - Rp_other ** 2 * s2
    if d_lo <= 0:
        return r_lo, r_hi
    s_lo, s_hi = base - np.sqrt(d_lo), base + np.sqrt(d_lo)
    return (s_hi, r_hi) if R_base >= base else (r_lo, s_lo)


def node_feasible_interval(i, Rp, ctx):
    """节点 i 的伸缩量 x_i 的可行区间 [lo, hi]（各关联边约束的交集）。"""
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


def is_feasible(x, ctx, rtol=1e-9):
    """边约束可行性判定。

    坐标下降法会把解推到约束边界上（L1 = Lmax 或 Lmin），浮点运算会有 ~1e-11 的
    越界，故判定时留 rtol 的相对裕度。
    """
    return bool(max_rel_change(x, ctx).max() <= EDGE_TOL * (1.0 + rtol))


def golden_section(f, lo, hi, tol=1e-6, max_iter=50):
    """黄金分割法一维最小化（方案二/三的一维子问题求解器）。"""
    gr = (np.sqrt(5) - 1) / 2
    c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if hi - lo < tol:
            break
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo)
            fc = f(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo)
            fd = f(d)
    return 0.5 * (lo + hi)


# ---------- 内层求解器：给定 p，求约束下最贴近理想伸缩量的解 ----------

def fit_coordinate_descent(ctx, e_ap, sweeps=60, seed=0, x_init=None):
    """方案一（min Σ(e_i-x_i)²）的坐标下降解。

    逐节点一维最优即 x_i = clip(e_i, lo, hi)，可行区间由边约束解析给出，
    故每轮扫描单调下降。边约束成对耦合，坐标下降会停在“坐标方向”的局部最优，
    因此可传入 x_init 从其它求解器的解出发再扫描（换一个 basin 继续下降）。
    """
    rng = np.random.default_rng(seed)
    nap = ctx["nap"]
    x = np.clip(e_ap if x_init is None else x_init, -EXT_LIMIT, EXT_LIMIT)
    Rp = ctx["R_all"].copy()
    Rp[ctx["ap_idx"]] -= x
    for _ in range(sweeps):
        for i in rng.permutation(nap):
            lo, hi = node_feasible_interval(i, Rp, ctx)
            x[i] = min(max(e_ap[i], lo), hi)
            Rp[ctx["ap_idx"][i]] = ctx["R_all"][ctx["ap_idx"][i]] - x[i]
    return x


def fit_penalty(ctx, e_ap, lam0=1e8, rounds=8, tol=EDGE_TOL_SOLVE):
    """罚函数法：λ 逐级放大直至边约束可行（保留用于与坐标下降法对照）。

    目标 Σ(x_i-e_i)² + λ Σ max(0, ΔL/L0 - tol)²，box bound 由 L-BFGS-B 处理。
    """
    ap_idx = ctx["ap_idx"]
    bnds = [(-EXT_LIMIT, EXT_LIMIT)] * ap_idx.size
    x = np.clip(e_ap, -EXT_LIMIT, EXT_LIMIT)
    lam = lam0
    for _ in range(rounds):
        x = minimize(lambda v: np.sum((v - e_ap) ** 2)
                     + lam * np.sum(np.maximum(0.0, max_rel_change(v, ctx) - tol) ** 2),
                     x, method="L-BFGS-B", bounds=bnds).x
        if max_rel_change(x, ctx).max() <= tol:
            break
        lam *= 10.0
    return x


def fit_best(ctx, e_ap, sweeps=60, seeds=(0, 1, 2)):
    """多起点混合求解：取所有候选可行解中贴合残差最小者。

    边约束成对耦合，单一启发式都会停在各自的局部最优（实测罚函数法与坐标下降法
    互有胜负，差距约 5%）。以罚函数法的解为初值再作坐标下降，可换一个 basin
    继续下降，实测残差由 64.1 mm 降至 54.9 mm。多种扫描顺序（seeds）多起点后取优。
    """
    x_pen = fit_penalty(ctx, e_ap)
    cands = [x_pen]
    for sd in seeds:
        cands.append(fit_coordinate_descent(ctx, e_ap, sweeps=sweeps, seed=sd))
        cands.append(fit_coordinate_descent(ctx, e_ap, sweeps=sweeps, seed=sd,
                                            x_init=x_pen))
    best, best_rms = None, np.inf
    for x in cands:
        if not is_feasible(x, ctx):
            continue                      # 丢弃不可行解（罚函数法可能微幅越界）
        rms = float(np.sqrt(np.mean((x - e_ap) ** 2)))
        if rms < best_rms:
            best, best_rms = x, rms
    if best is None:                      # 兜底：不可行时退回到坐标下降解
        best = fit_coordinate_descent(ctx, e_ap, sweeps=sweeps, seed=seeds[0])
    return best


INNER_SOLVERS = {"best": fit_best, "coord": fit_coordinate_descent,
                 "penalty": fit_penalty}


# ---------- 外层：嵌套联合优化（同时确定焦距 p 与伸缩量 x） ----------

def solve_adjustment(pos, panels, alpha_deg, beta_deg, inner="best",
                     p_bounds=P_BOUNDS, xatol=0.02):
    """在反射面板调节约束下确定理想抛物面并给出调节方案。

    外层以焦距 p 为决策变量，目标为约束下调节方案相对理想交点的均方根残差；
    内层对给定 p 求约束下最贴近理想伸缩量的可行解。
    """
    n = axis_vector(alpha_deg, beta_deg)
    R = np.linalg.norm(pos, axis=1).mean()
    ctx = build_context(pos, panels, n)
    solver = INNER_SOLVERS[inner]

    def stage(p):
        e_ap, _, _ = radial_extensions(pos[ctx["in_ap"]], n, p)
        return solver(ctx, e_ap), e_ap

    cache = {}

    def fit_rms(p):
        key = round(float(p), 6)
        if key not in cache:
            cache[key] = stage(key)
        x, e_ap = cache[key]
        return float(np.sqrt(np.mean((x - e_ap) ** 2)))

    p_opt = float(minimize_scalar(fit_rms, bounds=p_bounds, method="bounded",
                                  options={"xatol": xatol}).x)
    key = round(p_opt, 6)
    if key not in cache:
        cache[key] = stage(key)
    x, e_ap = cache[key]
    _, u_ap, R_ap = radial_extensions(pos[ctx["in_ap"]], n, p_opt)

    ext = np.zeros(len(pos))
    ext[ctx["ap_idx"]] = x
    pos_adj = pos.copy()
    pos_adj[ctx["ap_idx"]] = u_ap * (R_ap - x)[:, None]

    # 诊断：理想交点（未施加间距约束）下的间距相对变化
    pos_ideal = pos.copy()
    pos_ideal[ctx["ap_idx"]] = (pos[ctx["ap_idx"]] / R_ap[:, None]) * (R_ap - e_ap)[:, None]
    d0 = np.linalg.norm(pos[ctx["ie"]] - pos[ctx["je"]], axis=1)
    d1 = np.linalg.norm(pos_ideal[ctx["ie"]] - pos_ideal[ctx["je"]], axis=1)
    rel_ideal = float((np.abs(d1 - d0) / d0).max())

    rel = max_rel_change(x, ctx)
    return dict(R=R, n=n, p=p_opt, ctx=ctx, in_ap=ctx["in_ap"], ap_idx=ctx["ap_idx"],
                pos=pos, pos_adj=pos_adj, ext=ext, ext_ideal=e_ap, ext_fit=x,
                vertex=-(0.534 * R + p_opt) * n, focus=-0.534 * R * n,
                fit_rms=float(np.sqrt(np.mean((x - e_ap) ** 2))),
                rel_max=float(rel.max()) if rel.size else 0.0,
                rel_ideal=rel_ideal, n_edges=int(rel.size))
