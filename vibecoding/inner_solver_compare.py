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
    return rows


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    names, pos, panels = cm.load_data()

    # 问题二、三的观测方向
    n = cm.axis_vector(cm.ALPHA_Q2, cm.BETA_Q2)
    ctx = cm.build_context(pos, panels, n)

    # 论文主表：p=140.3839 m（问题二方向），结果保存到文件
    rows_main = report("问题二/三 主表", ctx, pos, 140.3839)
    with open("inner_solver_compare_result.txt", "w", encoding="utf-8") as fh:
        fh.write(f"内层求解器对照（问题二方向，p=140.3839 m，"
                 f"口径内 {ctx['nap']} 节点、{ctx['ie'].size} 条边）\n")
        fh.write(f"{'求解器':<20} {'残差RMS(mm)':>12} {'|x|max(mm)':>12} "
                 f"{'边长最大相对变化(%)':>20}\n")
        for name, rms, rel, ext in rows_main:
            fh.write(f"{name:<20} {rms * 1000:>12.4f} {ext * 1000:>12.2f} "
                     f"{rel * 100:>20.5f}\n")
    print("结果已保存至 inner_solver_compare_result.txt")

    # 额外焦距对照（论文正文"在 p∈{140.30,140.40,140.50} 三个额外焦距上重复"）
    for p in (140.30, 140.40, 140.50):
        report("问题二/三 额外", ctx, pos, p)

    # 问题一的观测方向（正上方）
    n1 = cm.axis_vector(0.0, 90.0)
    ctx1 = cm.build_context(pos, panels, n1)
    for p in (140.30, 140.40, 140.50):
        report("问题一", ctx1, pos, p)


if __name__ == "__main__":
    main()
