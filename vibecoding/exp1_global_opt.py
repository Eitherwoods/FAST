# -*- coding: utf-8 -*-
"""实验 1：内层 QCQP 的全局最优性对照（问题二方向，固定 p=140.3839 m）。

对照三个求解途径，回答"混合多起点解 52.2 mm 是否接近全局最优"：
  A. 混合多起点（论文采用的 fit_best）
  B. 30 个随机种子的坐标下降统计（min/mean/max）
  C. scipy SLSQP 直接求解原非凸 QCQP（两个起点：clip(e) 与混合解精修）
"""
import sys
import time
import numpy as np
from scipy.optimize import minimize

import common as cm
from common import (ALPHA_Q2, BETA_Q2, axis_vector, load_data,
                    radial_extensions, is_feasible, fit_best,
                    fit_coordinate_descent)

P_FIX = 140.3839


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    names, pos, panels = load_data()
    n = axis_vector(ALPHA_Q2, BETA_Q2)
    ctx = cm.build_context(pos, panels, n)
    R = ctx["R"]
    e_ap, _, _ = radial_extensions(pos[ctx["in_ap"]], n, P_FIX, R)
    ap, nap = ctx["ap_idx"], ctx["nap"]
    rms = lambda x: float(np.sqrt(np.mean((x - e_ap) ** 2)))

    print(f"p = {P_FIX} m，口径内节点 {nap}，参与约束边 {ctx['ie'].size}")
    print(f"理想交点解（截断到行程）最大边长变化 = "
          f"{cm.max_rel_change(np.clip(e_ap, -0.6, 0.6), ctx).max():.6f}")

    # ---- A. 混合多起点（论文方案） ----
    t0 = time.time()
    x_best = fit_best(ctx, e_ap, sweeps=60, seeds=(0, 1, 2))
    print(f"\n[A] 混合多起点 fit_best : RMS = {rms(x_best) * 1000:.3f} mm, "
          f"可行 = {is_feasible(x_best, ctx)}, 耗时 {time.time() - t0:.1f}s")

    # ---- B. 30 个随机种子的坐标下降统计 ----
    vals = []
    t0 = time.time()
    for sd in range(30):
        x = fit_coordinate_descent(ctx, e_ap, sweeps=60, seed=sd)
        if is_feasible(x, ctx):
            vals.append(rms(x))
    a = np.array(vals) * 1000
    print(f"[B] 30 个种子坐标下降 : min={a.min():.3f}  mean={a.mean():.3f}  "
          f"max={a.max():.3f}  std={a.std():.3f} mm, 可行解数={a.size}, "
          f"耗时 {time.time() - t0:.1f}s")

    # ---- C. SLSQP 直接求解原问题（多起点） ----
    ie, je, cospsi = ctx["ie"], ctx["je"], ctx["cospsi"]
    Rp_all = ctx["R_all"].copy()
    Lmin, Lmax = ctx["Lmin"], ctx["Lmax"]
    bnds = [(-0.6, 0.6)] * nap

    def con_fun(v):
        Rp = Rp_all.copy()
        Rp[ap] -= v
        L1 = np.sqrt(Rp[ie] ** 2 + Rp[je] ** 2 - 2.0 * Rp[ie] * Rp[je] * cospsi)
        return np.concatenate([Lmax - L1, L1 - Lmin])   # >= 0 均满足

    cons = [{"type": "ineq", "fun": con_fun}]

    starts = {"clip(e)": np.clip(e_ap, -0.6, 0.6), "混合解精修": x_best}
    for nm, x0 in starts.items():
        t0 = time.time()
        try:
            res = minimize(lambda v: np.sum((v - e_ap) ** 2), x0,
                           method="SLSQP", bounds=bnds, constraints=cons,
                           options={"maxiter": 80, "ftol": 1e-10})
            x = np.clip(res.x, -0.6, 0.6)
            print(f"[C] SLSQP({nm}) : RMS = {rms(x) * 1000:.3f} mm, "
                  f"可行 = {is_feasible(x, ctx)}, success = {res.success}, "
                  f"迭代 {res.nit}, 最大边长变化 "
                  f"{cm.max_rel_change(x, ctx).max() * 100:.5f}%, "
                  f"耗时 {time.time() - t0:.1f}s")
        except Exception as ex:
            print(f"[C] SLSQP({nm}) 失败: {ex}")


if __name__ == "__main__":
    main()
