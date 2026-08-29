# -*- coding: utf-8 -*-
"""问题二：alpha=36.795°, beta=78.169°，确定理想抛物面并计算促动器伸缩量，输出 result.xlsx。

模型：
- 观测轴 n = (cos(beta)cos(alpha), cos(beta)sin(alpha), sin(beta))，焦点 P = -0.534R·n，抛物面焦距 p 为决策变量，
  顶点 V = -(0.534R + p)·n。
- 节点 i（径向单位向量 u_i，A_i = u_i·n）沿径向与抛物面交点参数 t 满足
      (1-A²)t² - 4pA·t + 4p(B-p) = 0，  B = -0.534R
  其判别式 16p[p + 0.534R(1-A²)] 恒正，径向交点总存在。伸缩量（向球心为正）e = R_i - t。
- 嵌套联合优化（在反射面板调节约束下确定理想抛物面）：
  外层以焦距 p 为决策变量，目标为约束下调节方案相对理想交点的伸缩量均方根残差最小；
  内层对给定 p，在伸缩量限幅 ±0.6 m 与相邻节点间距变化 <=0.07% 约束下，
  用坐标下降法（边约束可行区间解析给出）求最贴近理想交点的伸缩量。
  仅取 300 m 口径内节点（到观测轴距离 <=150 m），间距约束含口径边界处的跨接边。
"""
import sys

import numpy as np
from pathlib import Path
from openpyxl import Workbook

from common import ALPHA_Q2, BETA_Q2, load_data, solve_adjustment

ALPHA, BETA = ALPHA_Q2, BETA_Q2


def solve():
    names, pos, panels = load_data()
    r = solve_adjustment(pos, panels, ALPHA, BETA)
    r["names"] = names
    return r


def write_result(r, path):
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "理想抛物面顶点坐标"
    ws1.append(["X坐标（米）", "Y坐标（米）", "Z坐标（米）", None, None, "注：至少保留3位小数"])
    ws1.append([round(float(c), 6) for c in r["vertex"]])

    ws2 = wb.create_sheet("调整后主索节点编号及坐标")
    ws2.append(["节点编号", "X坐标（米）", "Y坐标（米）", "Z坐标（米）", None, None, "注：至少保留3位小数"])
    for i in r["ap_idx"]:
        ws2.append([r["names"][i]] + [round(float(c), 6) for c in r["pos_adj"][i]])

    ws3 = wb.create_sheet("促动器顶端伸缩量")
    ws3.append(["对应主索节点编号", "伸缩量（米）", None, "注：至少保留3位小数"])
    for i, nm in enumerate(r["names"]):
        ws3.append([nm, round(float(r["ext"][i]), 6)])

    wb.save(path)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    r = solve()
    print(f"基准球面半径 R = {r['R']:.4f} m")
    print(f"观测轴 n = {np.array2string(r['n'], precision=6)}")
    print(f"最优焦距 p = {r['p']:.4f} m（0.466R = {0.466 * r['R']:.4f} m）")
    v = r["vertex"]
    print(f"理想抛物面顶点 V = ({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f})")
    f = r["focus"]
    print(f"焦点（馈源舱）P = ({f[0]:.4f}, {f[1]:.4f}, {f[2]:.4f})")
    e_ap = r["ext"][r["in_ap"]]
    print(f"300 m 口径内节点数 = {e_ap.size}")
    print(f"理想交点伸缩量范围 = [{r['ext_ideal'].min():.4f}, {r['ext_ideal'].max():.4f}] m")
    print(f"最终伸缩量范围 = [{e_ap.min():.4f}, {e_ap.max():.4f}] m，|e| 最大 = {np.abs(e_ap).max():.4f} m")
    print(f"约束下调节方案相对理想交点的伸缩量均方根残差 = {r['fit_rms']:.4f} m")
    print(f"参与约束的相邻边数 = {r['n_edges']}")
    print(f"相邻节点间距最大相对变化 = {r['rel_max'] * 100:.5f}%（限值 0.07%）")
    out = Path(__file__).resolve().parent / "result.xlsx"
    write_result(r, out)
    print(f"已写入 {out}")
