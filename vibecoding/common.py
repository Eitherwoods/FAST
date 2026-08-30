# -*- coding: utf-8 -*-
"""FAST 主动反射面：三问共用的数据、几何与优化组件。

几何约定
--------
- 基准球面球心 C 在坐标原点，半径 R = 全部主索节点模长均值（~=300.4 m）。
- 观测轴单位向量 n = (cos(beta)cos(alpha), cos(beta)sin(alpha), sin(beta))，焦点（馈源舱）P = -0.534R·n。
- 理想抛物面：焦点 P、对称轴 n、焦距 p，顶点 V = -(0.534R + p)·n。
- 节点沿基准球面径向移动，伸缩量 x 以“向球心”为正；节点 i 径向单位向量 u_i，
  A_i = u_i·n，径向与抛物面交点半径 t 满足
      (1-A²)t² - 4pA·t + 4p(B-p) = 0,   B = -0.534R
  取正根，理想伸缩量 e = R_i - t。

约束
----
- 促动器行程 |x_i| <= 0.6 m；
- 相邻节点间距相对变化 <= 0.07%。径向移动不改变两节点的径向夹角 psi，故调节后边长
  L1 = sqrt(Ri'² + Rj'² - 2Ri'Rj'cospsi)。

内层求解器
----------
- ``fit_coordinate_descent``：坐标下降，逐节点取闭式解 clip(e_i, lo, hi)，
  可行区间由边约束解析给出（方案一的目标函数），约束精确满足且残差最小，为默认。
- ``fit_penalty``：罚函数法（lambda 逐级放大至可行），保留用于对照。
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
    """加载附件1（主索节点坐标）与附件3（反射面板顶点编号）。

    附件2（促动器几何）不在此处返回，由 ``get_actuator_geometry`` 单独缓存，
    供 ``build_context`` 在构造上下文时一次性对齐到附件1 节点顺序。
    """
    df1 = pd.read_csv(DATA / "附件1.csv", encoding="gbk")
    names = df1.iloc[:, 0].astype(str).tolist()
    pos = df1.iloc[:, 1:4].to_numpy(dtype=float)
    df3 = pd.read_csv(DATA / "附件3.csv", encoding="gbk")
    idx = {nm: i for i, nm in enumerate(names)}
    panels = df3.iloc[:, 0:3].apply(lambda s: s.map(idx)).to_numpy(dtype=int)
    return names, pos, panels


# 促动器几何缓存（act_dir, L_act_0, pos_upper, pos_lower），避免重复读附件2
_ACT_GEOM_CACHE = None


def get_actuator_geometry():
    """加载附件2：促动器几何，按附件1 节点顺序对齐返回。

    附件2 每行字段：对应主索节点编号 + 下端点（地锚）XYZ + 基准态上端点（顶端）XYZ。
    促动器方向 act_dir = (上端 - 下端)/|...|，实测严格沿基准球面径向反向
    （夹角均值 179.92°），即指向球心方向；下拉索 = 节点 - 上端，长度固定不变。

    返回 (act_dir, L_act_0, pos_upper, pos_lower)，均按附件1 节点索引排列。
    """
    global _ACT_GEOM_CACHE
    if _ACT_GEOM_CACHE is not None:
        return _ACT_GEOM_CACHE
    df1 = pd.read_csv(DATA / "附件1.csv", encoding="gbk")
    names1 = df1.iloc[:, 0].astype(str).tolist()
    idx1 = {nm: i for i, nm in enumerate(names1)}
    df2 = pd.read_csv(DATA / "附件2.csv", encoding="gbk")
    names2 = df2.iloc[:, 0].astype(str).tolist()
    pos_lower = df2.iloc[:, 1:4].to_numpy(dtype=float)
    pos_upper = df2.iloc[:, 4:7].to_numpy(dtype=float)
    # 附件2 行顺序未必与附件1 一致，按"对应主索节点编号"在附件1 中的索引重排
    node_idx = np.array([idx1[nm] for nm in names2])
    order = np.argsort(node_idx)
    pos_lower = pos_lower[order]
    pos_upper = pos_upper[order]
    act_vec = pos_upper - pos_lower
    L_act_0 = np.linalg.norm(act_vec, axis=1)
    act_dir = act_vec / L_act_0[:, None]
    _ACT_GEOM_CACHE = (act_dir, L_act_0, pos_upper, pos_lower)
    return _ACT_GEOM_CACHE


def axis_vector(alpha_deg, beta_deg):
    """由方位角 alpha、仰角 beta 计算观测轴单位向量。"""
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
    """预计算照明区掩码、边约束几何、促动器几何与节点-边/面板关联。

    tol 为求解用的边长容差，默认略紧于 EDGE_TOL，以吸收结果文件的舍入误差。

    促动器方向 act_dir 来自附件2（指向球心方向），用于严格按附件2 几何移动节点：
    节点工作态位置 = pos + δ·act_dir。下拉索长度随之严格保持不变
    （附件2 给定 act_dir 与基准球面径向反向共线，故与原径向假设数值等价）。
    """
    R = np.linalg.norm(pos, axis=1).mean()
    dist_to_axis = np.linalg.norm(pos - np.outer(pos @ n, n), axis=1)
    in_ap = dist_to_axis <= radius
    ap_idx = np.where(in_ap)[0]
    nap = ap_idx.size
    R_all = np.linalg.norm(pos, axis=1)

    # 附件2 促动器几何（act_dir 与 R_all 反向共线，但用附件2 实际方向更严谨）
    act_dir, L_act_0, pos_upper, pos_lower = get_actuator_geometry()
    # 下拉索：上端 → 主索节点（基准态），长度固定不变
    pull_vec = pos - pos_upper
    L_pull_0 = np.linalg.norm(pull_vec, axis=1)

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

    return dict(R=R, n=n, N=len(pos), radius=radius, tol=tol,
                in_ap=in_ap, ap_idx=ap_idx, nap=nap,
                pos=pos, R_all=R_all, ie=ie, je=je, L0=L0, cospsi=cospsi,
                sin2=np.maximum(1.0 - cospsi ** 2, 0.0),
                Lmin=L0 * (1 - tol), Lmax=L0 * (1 + tol),
                li=li, lj=lj, inc=[np.asarray(a, dtype=int) for a in inc],
                ap_panels=ap_panels, panels_ap=panels_ap,
                panels_ap_local=loc[panels_ap],
                node_panels=[np.asarray(a, dtype=int) for a in node_panels],
                act_dir=act_dir, L_act_0=L_act_0,
                pos_upper=pos_upper, pos_lower=pos_lower, L_pull_0=L_pull_0)


def adjust_nodes(pos, x, ctx):
    """由伸缩量 x（口径内节点）计算调节后的全部节点位置。

    节点沿附件2 促动器方向 act_dir 移动：pos' = pos + δ·act_dir。
    δ > 0 表示促动器伸长、节点向球心移动（act_dir 实测指向球心）；
    下拉索长度随之严格保持不变（附件2 几何自洽）。
    """
    pos_adj = pos.copy()
    pos_adj[ctx["ap_idx"]] = pos[ctx["ap_idx"]] + x[:, None] * ctx["act_dir"][ctx["ap_idx"]]
    return pos_adj


def max_rel_change(x, ctx):
    """相邻节点间距相对变化（基于附件2 act_dir 的通用坐标计算）。

    节点工作态位置 = pos + δ·act_dir，直接由新坐标求边长，不再依赖
    "径向移动不改变径向夹角 psi" 的余弦定理近似，故与附件2 几何严格自洽。
    """
    pos_adj = ctx["pos"].copy()
    pos_adj[ctx["ap_idx"]] = ctx["pos"][ctx["ap_idx"]] + x[:, None] * ctx["act_dir"][ctx["ap_idx"]]
    L1 = np.linalg.norm(pos_adj[ctx["ie"]] - pos_adj[ctx["je"]], axis=1)
    return np.abs(L1 - ctx["L0"]) / ctx["L0"]


# ---------- 边约束的解析可行区间（坐标下降用，基于附件2 act_dir） ----------

def edge_interval_delta(e, this_end, delta_other, ctx):
    """边 e 上本端节点 this_end 的 δ 可行区间（对端 δ = delta_other）。

    基于 act_dir 的一般二次方程。设 d = pos_i - pos_j（本端减对端），
    a_i, a_j 为两端 act_dir，则边长
        L1² = |d + δ_i·a_i - δ_j·a_j|²
            = δ_i² + 2·c1·δ_i + c0
    其中
        c1 = (a_i·d) - δ_j·(a_i·a_j)
        c0 = L0² - 2·δ_j·(a_j·d) + δ_j²
    边长容差 |L1-L0|/L0 ≤ tol 等价于 Lmin² ≤ (δ_i+c1)² + (c0-c1²) ≤ Lmax²，
    化为 δ_i 的区间（取包含基线值 0 的连通分量）。
    """
    ie_arr, je_arr = ctx["ie"], ctx["je"]
    other = je_arr[e] if ie_arr[e] == this_end else ie_arr[e]
    ai = ctx["act_dir"][this_end]
    aj = ctx["act_dir"][other]
    d = ctx["pos"][this_end] - ctx["pos"][other]
    L0 = ctx["L0"][e]
    tol = ctx["tol"]
    Lmin = L0 * (1 - tol)
    Lmax = L0 * (1 + tol)

    ai_d = float(ai @ d)
    aj_d = float(aj @ d)
    ai_aj = float(ai @ aj)
    dj = float(delta_other)

    c1 = ai_d - dj * ai_aj
    c0 = L0 * L0 - 2.0 * dj * aj_d + dj * dj

    # 上界 L1² ≤ Lmax²：(δ + c1)² ≤ Lmax² - c0 + c1²
    R2_hi = Lmax * Lmax - c0 + c1 * c1
    if R2_hi < 0:
        # 对端已使边长超容差且本端无法补救，返回空区间（强迫本端 δ=0，由 is_feasible 过滤）
        return 0.0, 0.0
    root_hi = np.sqrt(R2_hi)
    upper_lo = -c1 - root_hi
    upper_hi = -c1 + root_hi

    # 下界 L1² ≥ Lmin²：(δ + c1)² ≥ Lmin² - c0 + c1²
    R2_lo = Lmin * Lmin - c0 + c1 * c1
    if R2_lo <= 0:
        return upper_lo, upper_hi            # 下界恒成立
    root_lo = np.sqrt(R2_lo)
    # 取包含基线值 0 的分量（当对端可行时 |c1| ≥ root_lo，故 c1 符号决定选段）
    if c1 < 0:
        return upper_lo, min(upper_hi, -c1 - root_lo)   # 左段
    return max(upper_lo, -c1 + root_lo), upper_hi        # 右段


def node_feasible_interval(i, delta_global, ctx):
    """节点 i 的 δ 可行区间 [lo, hi]（各关联边约束的交集 + 行程限幅）。

    delta_global 为全局 δ 数组（口径外为 0），用于取对端 δ；与原径向版本
    相比，直接用 act_dir 几何，无需径向夹角 psi。
    """
    lo, hi = -EXT_LIMIT, EXT_LIMIT
    ni = ctx["ap_idx"][i]
    for e in ctx["inc"][i]:
        other = ctx["je"][e] if ctx["ie"][e] == ni else ctx["ie"][e]
        lo_e, hi_e = edge_interval_delta(e, ni, delta_global[other], ctx)
        lo = max(lo, lo_e)
        hi = min(hi, hi_e)
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

    维护全局 δ 数组 delta_global（口径外为 0），供 node_feasible_interval
    取对端 δ；与原径向版本相比，无需维护 Rp（调节后半径）。
    """
    rng = np.random.default_rng(seed)
    nap = ctx["nap"]
    x = np.clip(e_ap if x_init is None else x_init, -EXT_LIMIT, EXT_LIMIT)
    delta_global = np.zeros(ctx["N"])
    delta_global[ctx["ap_idx"]] = x
    for _ in range(sweeps):
        for i in rng.permutation(nap):
            lo, hi = node_feasible_interval(i, delta_global, ctx)
            new = min(max(e_ap[i], lo), hi)
            delta_global[ctx["ap_idx"][i]] = new
            x[i] = new
    return x


def fit_penalty(ctx, e_ap, lam0=1e8, rounds=8, tol=EDGE_TOL_SOLVE):
    """罚函数法：lambda 逐级放大直至边约束可行（保留用于与坐标下降法对照）。

    目标 Σ(x_i-e_i)² + lambda Σ max(0, ΔL/L0 - tol)²，box bound 由 L-BFGS-B 处理。
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
                     p_bounds=P_BOUNDS, xatol=1e-4):
    """在反射面板调节约束下确定理想抛物面并给出调节方案。

    外层以焦距 p 为决策变量，目标为约束下调节方案相对理想交点的均方根残差；
    内层对给定 p 求约束下最贴近理想伸缩量的可行解。

    xatol 为外层一维搜索的焦距精度：内层解相对理想交点的残差在 p 的
    0.01 m 量级上已有起伏，故取 1e-4 而非默认的 0.02，避免 p* 漂移。
    """
    n = axis_vector(alpha_deg, beta_deg)
    R = np.linalg.norm(pos, axis=1).mean()
    ctx = build_context(pos, panels, n)
    solver = INNER_SOLVERS[inner]

    def stage(p):
        # 显式传全体节点的基准半径 R（与 schemes_compare 一致），
        # 避免用口径内子集均值带来的口径差异
        e_ap, _, _ = radial_extensions(pos[ctx["in_ap"]], n, p, R)
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
    _, u_ap, R_ap = radial_extensions(pos[ctx["in_ap"]], n, p_opt, R)

    ext = np.zeros(len(pos))
    ext[ctx["ap_idx"]] = x
    pos_adj = pos.copy()
    # 节点沿附件2 act_dir 移动 δ（与 adjust_nodes 一致，下拉索长度严格不变）
    pos_adj[ctx["ap_idx"]] = pos[ctx["ap_idx"]] + x[:, None] * ctx["act_dir"][ctx["ap_idx"]]

    # 诊断：理想交点（未施加间距约束）下的间距相对变化
    pos_ideal = pos.copy()
    pos_ideal[ctx["ap_idx"]] = pos[ctx["ap_idx"]] + e_ap[:, None] * ctx["act_dir"][ctx["ap_idx"]]
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


def check_attachment2(x, ctx):
    """附件2 几何自洽性校核（独立第七项校核）。

    严格按附件2 几何：促动器沿 act_dir 伸缩 δ，节点同步沿 act_dir 移动，
    下拉索长度随之严格保持不变。本函数校核：
    1. 下拉索长度相对变化（应 ≈ 0，因两端同步移动）；
    2. 工作态促动器长度 = 基准长度 + δ，是否落入 [L_act_0 - EXT_LIMIT, L_act_0 + EXT_LIMIT]；
    3. 促动器方向与基准球面径向（反向）的夹角（验证附件2 几何一致性）。

    返回 dict，键：
      pull_rod_rel_max / _mean / _violations  下拉索长度相对变化统计
      L_act_min / _max / _0_mean              工作态/基准态促动器长度
      ext_in_range                            伸缩量是否全部在 ±EXT_LIMIT 内
      act_radial_ang_mean / _max              促动器与径向反向的夹角（°）
    """
    pos = ctx["pos"]
    act_dir = ctx["act_dir"]
    L_act_0 = ctx["L_act_0"]
    pos_upper = ctx["pos_upper"]
    L_pull_0 = ctx["L_pull_0"]
    ap_idx = ctx["ap_idx"]

    # 全局 δ（口径外为 0）
    delta_global = np.zeros(ctx["N"])
    delta_global[ap_idx] = x

    # 工作态促动器上端点：基准态上端 + δ·act_dir
    pos_upper_adj = pos_upper + delta_global[:, None] * act_dir
    # 工作态节点位置：基准态节点 + δ·act_dir
    pos_adj = pos + delta_global[:, None] * act_dir
    # 工作态下拉索长度（应与基准态严格相等）
    L_pull_adj = np.linalg.norm(pos_adj - pos_upper_adj, axis=1)
    rel_pull = np.abs(L_pull_adj - L_pull_0) / L_pull_0

    # 工作态促动器长度（= 基准长度 + δ，因下端点不动、上端点沿 act_dir 移动）
    L_act_adj = L_act_0 + delta_global

    # 促动器方向 vs 基准球面径向反向（C→节点取反，即指向球心方向）
    R_all = ctx["R_all"]
    inward = -pos / R_all[:, None]
    cos_ang = np.einsum("ij,ij->i", act_dir, inward)
    ang_deg = np.rad2deg(np.arccos(np.clip(cos_ang, -1, 1)))

    return dict(
        pull_rod_rel_max=float(rel_pull.max()),
        pull_rod_rel_mean=float(rel_pull.mean()),
        pull_rod_violations=int((rel_pull > EDGE_TOL).sum()),
        L_act_min=float(L_act_adj[ap_idx].min()),
        L_act_max=float(L_act_adj[ap_idx].max()),
        L_act_0_mean=float(L_act_0.mean()),
        ext_in_range=bool(np.abs(delta_global).max() <= EXT_LIMIT + 1e-9),
        act_radial_ang_mean=float(ang_deg.mean()),
        act_radial_ang_max=float(ang_deg.max()),
    )
