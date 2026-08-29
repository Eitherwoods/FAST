# -*- coding: utf-8 -*-
"""内层求解器对照：罚函数法 vs 坐标下降法（固定焦距 p 下的公平比较）。

两种内层求解器都要在 |x_i| <= 0.6 与相邻节点间距变化 <=0.07% 约束下最小化
Σ(x_i - e_i(p))²。比较指标为相对理想交点的均方根残差与约束满足情况。
"""
import sys

import numpy as np

import common as cm


def report(tag, ctx, pos, p):
    e_ap, _, _ = cm.radial_extensions(pos[ctx["in_ap"]], ctx["n"], p)
    print(f"\n[{tag}] p = {p:.4f} m，口径内节点 {ctx['nap']} 个")
    rows = []
    for name, solver, kw in [("罚函数法", cm.fit_penalty, {}),
                             ("坐标下降 60 轮", cm.fit_coordinate_descent, {"sweeps": 60}),
                             ("坐标下降 300 轮", cm.fit_coordinate_descent, {"sweeps": 300}),
                             ("混合多起点", cm.fit_best, {})]:
        x = solver(ctx, e_ap, **kw)
        rms = float(np.sqrt(np.mean((x - e_ap) ** 2)))
        rel = float(cm.max_rel_change(x, ctx).max())
        rows.append((name, rms, rel, float(np.abs(x).max())))
    for name, rms, rel, ext in rows:
        ok = "" if rel <= cm.EDGE_TOL * (1 + 1e-9) else "  <- 违反约束"
        print(f"  {name:<16} 残差 RMS = {rms * 1000:8.4f} mm   "
              f"|x| 最大 = {ext * 1000:7.2f} mm   边长最大相对变化 = {rel * 100:.5f}%{ok}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    names, pos, panels = cm.load_data()

    # 问题二、三的观测方向
    n = cm.axis_vector(cm.ALPHA_Q2, cm.BETA_Q2)
    ctx = cm.build_context(pos, panels, n)
    for p in (140.30, 140.40, 140.50):
        report("问题二/三", ctx, pos, p)

    # 问题一的观测方向（正上方）
    n1 = cm.axis_vector(0.0, 90.0)
    ctx1 = cm.build_context(pos, panels, n1)
    for p in (140.30, 140.40, 140.50):
        report("问题一", ctx1, pos, p)


if __name__ == "__main__":
    main()
