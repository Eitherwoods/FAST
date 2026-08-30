# -*- coding: utf-8 -*-
"""从新 result.xlsx 读取问题二结果表的统计量（RMS、最大残差、伸缩量均值等）。"""
import sys
import numpy as np
from openpyxl import load_workbook
import common as cm
from common import ALPHA_Q2, BETA_Q2, load_data

sys.stdout.reconfigure(encoding="utf-8")


def main():
    names, pos, panels = load_data()
    n = cm.axis_vector(ALPHA_Q2, BETA_Q2)
    ctx = cm.build_context(pos, panels, n)
    R = ctx["R"]

    out = cm.Path(__file__).resolve().parent / "result.xlsx"
    wb = load_workbook(out, data_only=True)
    idx = {nm: i for i, nm in enumerate(names)}

    # 顶点
    v = np.array([c for c in next(wb["理想抛物面顶点坐标"].iter_rows(
        min_row=2, max_row=2, max_col=3, values_only=True))], dtype=float)
    p = float(np.linalg.norm(v) - 0.534 * R)

    # 伸缩量
    rows3 = list(wb["促动器顶端伸缩量"].iter_rows(min_row=2, values_only=True))
    ext = np.zeros(len(names))
    for row in rows3:
        ext[idx[str(row[0])]] = float(row[1])

    ap_idx = ctx["ap_idx"]
    e_ap, _, _ = cm.radial_extensions(pos[ctx["in_ap"]], n, p, R)
    x = ext[ap_idx]
    resid = x - e_ap
    rms = float(np.sqrt(np.mean(resid ** 2)))
    max_resid = float(np.abs(resid).max())
    ext_mean = float(np.abs(x[x != 0]).mean()) if (x != 0).any() else 0.0
    ext_max = float(np.abs(x).max())
    # 多个候选均值，供论文表格选用
    ext_mean_signed_all = float(x.mean())
    ext_mean_signed_nonzero = float(x[x != 0].mean()) if (x != 0).any() else 0.0
    ext_mean_abs_all = float(np.abs(x).mean())
    ext_mean_abs_nonzero = ext_mean  # 即上方已算

    # 附件2 校核
    chk = cm.check_attachment2(x, ctx)

    print(f"最优焦距 p* = {p:.4f} m")
    print(f"顶点 V = ({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f})")
    print(f"|V| = {np.linalg.norm(v):.4f} m")
    print(f"残差 RMS = {rms*1000:.1f} mm")
    print(f"残差最大值 = {max_resid*1000:.1f} mm")
    print(f"伸缩量范围 = [{x.min():.4f}, {x.max():.4f}] m")
    print(f"|δ|max = {ext_max:.4f} m")
    print(f"伸缩量均值(|非零|) = {ext_mean*1000:.1f} mm")
    print(f"伸缩量均值(带符号, 全部) = {ext_mean_signed_all*1000:.1f} mm")
    print(f"伸缩量均值(带符号, 非零) = {ext_mean_signed_nonzero*1000:.1f} mm")
    print(f"伸缩量均值(|全部|) = {ext_mean_abs_all*1000:.1f} mm")
    print(f"边长最大相对变化 = {cm.max_rel_change(x, ctx).max()*100:.5f}%")
    print(f"\n附件2 校核：")
    print(f"  下拉索相对变化最大 = {chk['pull_rod_rel_max']*100:.6f}%")
    print(f"  促动器长度范围 = [{chk['L_act_min']:.4f}, {chk['L_act_max']:.4f}] m")
    print(f"  促动器方向 vs 径向反向: 均值 {chk['act_radial_ang_mean']:.4f}°, "
          f"最大 {chk['act_radial_ang_max']:.4f}°")


if __name__ == "__main__":
    main()
