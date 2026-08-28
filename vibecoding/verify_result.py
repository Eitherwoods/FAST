# -*- coding: utf-8 -*-
"""校验 result.xlsx（第2问提交成果）是否满足全部调节约束。

检查项：
1. 工作表齐全、口径内节点数与附件1一致；
2. 促动器伸缩量 |x| ≤ 0.6 m，口径外为 0；
3. 调整后节点坐标与“基准位置沿径向移动伸缩量”自洽；
4. 相邻节点间距相对变化 ≤ 0.07%（含口径边界处的跨接边）；
5. 理想抛物面顶点落在观测轴上，焦距落在搜索区间内。
"""
import sys

import numpy as np
from openpyxl import load_workbook

import common as cm
from common import ALPHA_Q2, BETA_Q2, EXT_LIMIT, build_context, load_data


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    out = cm.Path(__file__).resolve().parent / "result.xlsx"
    wb = load_workbook(out, data_only=True)
    if wb.sheetnames != ["理想抛物面顶点坐标", "调整后主索节点编号及坐标", "促动器顶端伸缩量"]:
        print("工作表名与附件4模板不一致：", wb.sheetnames)

    names, pos, panels = load_data()
    n = cm.axis_vector(ALPHA_Q2, BETA_Q2)
    R = np.linalg.norm(pos, axis=1).mean()
    idx = {nm: i for i, nm in enumerate(names)}

    # ---- 工作表1：顶点 ----
    v = np.array([c for c in next(wb["理想抛物面顶点坐标"].iter_rows(
        min_row=2, max_row=2, max_col=3, values_only=True))], dtype=float)
    p_from_v = np.linalg.norm(v) - 0.534 * R
    axis_err = np.linalg.norm(np.cross(v, n))
    print(f"基准球面半径 R = {R:.4f} m，观测轴 n = {np.array2string(n, precision=6)}")
    print(f"抛物面顶点 V = ({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f})，|V| = {np.linalg.norm(v):.4f} m")
    print(f"  [1] 顶点在观测轴上：|V×n| = {axis_err:.2e} "
          f"{'OK' if axis_err < 1e-6 else '不满足'}")
    print(f"  [2] 反解焦距 |V|-0.534R = {p_from_v:.4f} m "
          f"{'OK' if cm.P_BOUNDS[0] <= p_from_v <= cm.P_BOUNDS[1] else '超出搜索区间'}")

    # ---- 工作表2/3：节点坐标与伸缩量 ----
    rows2 = list(wb["调整后主索节点编号及坐标"].iter_rows(min_row=2, values_only=True))
    rows3 = list(wb["促动器顶端伸缩量"].iter_rows(min_row=2, values_only=True))

    ext = np.zeros(len(names))
    for row in rows3:                       # 表尾有空列与“注”列，按下标取值
        ext[idx[str(row[0])]] = float(row[1])
    ok_sheet3 = len(rows3) == len(names)
    print(f"伸缩量表 {len(rows3)} 行（附件1 共 {len(names)} 个节点）"
          f"{'OK' if ok_sheet3 else '行数不符'}")

    pos_adj = pos.copy()
    dist_to_axis = np.linalg.norm(pos - np.outer(pos @ n, n), axis=1)
    in_ap = dist_to_axis <= cm.APERTURE_RADIUS
    for row in rows2:
        pos_adj[idx[str(row[0])]] = (float(row[1]), float(row[2]), float(row[3]))
    n_ap = int(in_ap.sum())
    ok_count = len(rows2) == n_ap
    print(f"节点坐标表 {len(rows2)} 行（口径内节点 {n_ap} 个）{'OK' if ok_count else '行数不符'}")

    ext_max = float(np.abs(ext).max())
    print(f"  [3] |伸缩量| ≤ 0.6 m：最大 = {ext_max:.6f} m "
          f"{'OK' if ext_max <= EXT_LIMIT else '不满足'}")
    out_zeros = int(np.abs(ext[~in_ap]).max())
    print(f"  [4] 口径外节点伸缩量为 0：{'OK' if out_zeros == 0 else '存在非零值'}")

    # 口径内节点：坐标与径向自洽性
    u = pos[in_ap] / np.linalg.norm(pos[in_ap], axis=1)[:, None]
    expect = u * (np.linalg.norm(pos[in_ap], axis=1) - ext[in_ap])[:, None]
    dev = float(np.abs(pos_adj[in_ap] - expect).max())
    print(f"  [5] 节点沿径向变位自洽：坐标最大偏差 = {dev:.2e} m "
          f"{'OK' if dev < 1e-5 else '不满足'}")

    # ---- 边长约束 ----
    ctx = build_context(pos, panels, n)
    L0 = np.linalg.norm(pos[ctx["ie"]] - pos[ctx["je"]], axis=1)
    L1 = np.linalg.norm(pos_adj[ctx["ie"]] - pos_adj[ctx["je"]], axis=1)
    rel = np.abs(L1 - L0) / L0
    print(f"  [6] 相邻节点间距相对变化 ≤ 0.07%（{rel.size} 条边，含口径边界跨接边）："
          f"最大 = {rel.max() * 100:.5f}% "
          f"{'OK' if cm.is_feasible(ext[in_ap], ctx) else '不满足'}")

    ok = (axis_err < 1e-6 and cm.P_BOUNDS[0] <= p_from_v <= cm.P_BOUNDS[1]
          and ok_sheet3 and ok_count and ext_max <= EXT_LIMIT and out_zeros == 0
          and dev < 1e-5 and cm.is_feasible(ext[in_ap], ctx))
    print("\n结论：" + ("全部约束满足，result.xlsx 可用于提交。" if ok else "存在未满足项。"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
