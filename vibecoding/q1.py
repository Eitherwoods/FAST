# -*- coding: utf-8 -*-
"""问题一：α=0°, β=90°（天体位于基准球面正上方），确定理想抛物面。

模型：
- 基准球面球心为原点，半径 R 由附件1节点坐标均值确定。
- 观测轴单位向量 n = (cosβcosα, cosβsinα, sinβ)，焦点 P = -0.534R·n（焦面与基准球面同心，焦径比 0.466）。
- 理想抛物面：焦点 P、对称轴 n、焦距 p（决策变量），隐式方程 |M-P| = (M-P)·n + 2p，
  顶点 V = -(0.534R + p)·n。
- 在反射面板调节约束下确定理想抛物面：嵌套联合优化，外层以焦距 p 为决策变量，
  目标为约束下调节方案相对理想交点的伸缩量均方根残差最小；内层对给定 p，
  在 |e_i| ≤ 0.6 m 与相邻节点间距变化 ≤0.07% 约束下求最贴近理想交点的伸缩量
  （仅统计 300 m 口径内节点，即距观测轴 ≤150 m；间距约束含口径边界处的跨接边）。

α=0°、β=90° 时 n=(0,0,1)，抛物面退化为以 z 轴为对称轴的旋转抛物面
    x² + y² = 4p(z + 0.534R + p)
"""
import sys

import numpy as np

from common import load_data, solve_adjustment


def solve(alpha_deg, beta_deg):
    names, pos, panels = load_data()
    r = solve_adjustment(pos, panels, alpha_deg, beta_deg)
    r["names"] = [names[i] for i in r["ap_idx"]]
    return r


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    r = solve(0.0, 90.0)
    e = r["ext_ideal"]
    print(f"基准球面半径 R = {r['R']:.4f} m")
    print(f"观测轴 n = {np.array2string(r['n'], precision=6)}")
    print(f"最优焦距 p = {r['p']:.4f} m（0.466R = {0.466 * r['R']:.4f} m）")
    print(f"理想抛物面顶点 V = ({r['vertex'][0]:.4f}, {r['vertex'][1]:.4f}, {r['vertex'][2]:.4f})")
    print(f"焦点（馈源舱）P = ({r['focus'][0]:.4f}, {r['focus'][1]:.4f}, {r['focus'][2]:.4f})")
    print(f"300 m 口径内节点数 = {e.size}")
    print(f"理想交点伸缩量范围 = [{e.min():.4f}, {e.max():.4f}] m，|e| 最大 = {np.abs(e).max():.4f} m")
    print(f"理想交点伸缩量均方根 = {np.sqrt(np.mean(e ** 2)):.4f} m")
    print(f"约束 |e| ≤ 0.6 m {'满足' if np.abs(r['ext_fit']).max() <= 0.6 else '不满足'}")
    print(f"理想交点下相邻节点间距最大相对变化 = {r['rel_ideal'] * 100:.5f}%（限值 0.07%）")
    print(f"约束下调节方案贴合残差均方根 = {r['fit_rms']:.4f} m，"
          f"间距最大相对变化 = {r['rel_max'] * 100:.5f}%")
