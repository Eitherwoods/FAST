# -*- coding: utf-8 -*-
"""实验 2：三种方案在多个焦距 p 下各自目标值的对比。

用每种方案**各自的真实目标函数**在 p 网格上独立求解，检验三者的 p 极值
位置是否一致，为论文"三种方案最优焦距完全一致（p* 由几何与约束主导、
与拟合目标细化无关）"提供定量证据。

  f1(p) = mean((x-e)^2)                          （方案一目标，m^2）
  f2(p) = mean((x-e)^2) + mean(Delta_k^2)        （方案二目标，m^2）
  f3(p) = mean(积分残差^2) = integ_rms^2         （方案三目标，m^2）

各 f 的极小位置即为该方案意义下的最优焦距。
"""
import sys
import time
import numpy as np

import common as cm
from common import ALPHA_Q2, BETA_Q2, axis_vector, load_data
import schemes_compare as sc

P_GRID = [140.20, 140.30, 140.3839, 140.45, 140.60]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    names, pos, panels = load_data()
    n = axis_vector(ALPHA_Q2, BETA_Q2)
    ctx = cm.build_context(pos, panels, n)

    rows = []
    for p in P_GRID:
        t0 = time.time()
        x1, e1 = sc.solve_scheme1(pos, ctx, p)          # 方案一
        m1 = sc.evaluate(pos, ctx, x1, e1, p, sample_m=8)
        x2, e2 = sc.solve_scheme2(pos, ctx, p, x_init=x1)   # 方案二（方案一解 warm start）
        m2 = sc.evaluate(pos, ctx, x2, e2, p, sample_m=8)
        x3, e3 = sc.solve_scheme3(pos, ctx, p, x_init=x1)   # 方案三
        m3 = sc.evaluate(pos, ctx, x3, e3, p, sample_m=8)

        f1 = m1["node_rms"] ** 2
        f2 = m2["node_rms"] ** 2 + m2["center_rms"] ** 2
        f3 = m3["integ_rms"] ** 2
        rows.append((p, f1, f2, f3, m1, m2, m3))
        print(f"p = {p:.4f}:  f1 = {f1*1e3:.4f}  f2 = {f2*1e3:.4f}  "
              f"f3 = {f3*1e3:.4f}  (x1e-3 m^2), 耗时 {time.time()-t0:.0f}s")

    print("\n各方案目标取最小的 p 及其它信息：")
    for k, name in ((1, "方案一"), (2, "方案二"), (3, "方案三")):
        best = min(rows, key=lambda r: r[k])
        mk = best[k + 3]
        print(f"  {name}: min f{k} 在 p = {best[0]:.4f}，f{k} = {best[k]*1e3:.4f} x1e-3 m^2, "
              f"node_rms = {mk['node_rms']*1000:.2f} mm, "
              f"center_rms = {mk['center_rms']*1000:.2f} mm, "
              f"integ_rms = {mk['integ_rms']*1000:.2f} mm")

    print("\n同一 p 下三方案的各项残差（mm）：")
    print(f"{'p(m)':>10} {'f1min?':>7} {'node1':>7} {'node2':>7} {'node3':>7} "
          f"{'cent1':>7} {'cent2':>7} {'cent3':>7} {'int1':>7} {'int2':>7} {'int3':>7}")
    for p, f1, f2, f3, m1, m2, m3 in rows:
        print(f"{p:>10.4f} {'*' if f1==min(r[1] for r in rows) else '':>7} "
              f"{m1['node_rms']*1000:>7.2f} {m2['node_rms']*1000:>7.2f} {m3['node_rms']*1000:>7.2f} "
              f"{m1['center_rms']*1000:>7.2f} {m2['center_rms']*1000:>7.2f} {m3['center_rms']*1000:>7.2f} "
              f"{m1['integ_rms']*1000:>7.2f} {m2['integ_rms']*1000:>7.2f} {m3['integ_rms']*1000:>7.2f}")

    with open("exp2_p_curve_result.txt", "w", encoding="utf-8") as fh:
        fh.write("三方案各自目标随 p 的变化（目标值 x1e-3 m^2）\n")
        fh.write(f"{'p(m)':>10} {'f1':>10} {'f2':>10} {'f3':>10}\n")
        for p, f1, f2, f3, *_ in rows:
            fh.write(f"{p:>10.4f} {f1*1e3:>10.4f} {f2*1e3:>10.4f} {f3*1e3:>10.4f}\n")
    print("\n结果已保存至 exp2_p_curve_result.txt")


if __name__ == "__main__":
    main()
