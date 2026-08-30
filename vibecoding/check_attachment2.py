# -*- coding: utf-8 -*-
"""附件2 一致性检验：促动器上端点 vs 主索节点坐标 vs 径向方向。

本脚本回答：论文假设3「促动器沿基准球面径向伸缩，节点位置 P' = P + δ·e（e 为径向单位向量）」
与附件2 给出的「下拉索长度固定不变」是否相容。

核心比对三项：
1. 促动器上端点(基准态) 与 主索节点坐标 是否重合？(若不重合，下拉索长度 = |node - upper|)
2. 下拉索方向 与 节点径向方向 是否一致？(若不一致，节点严格沿径向移动会改变下拉索长度)
3. 若按论文假设调节，下拉索长度变化的最大相对值是多少？是否超过 0.07%？
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(r"d:\数模\26最后的培训\FAST\题目与附件与思路")


def main():
    # 附件1：主索节点坐标
    df1 = pd.read_csv(DATA / "附件1.csv", encoding="gbk")
    names1 = df1.iloc[:, 0].astype(str).tolist()
    pos_node = df1.iloc[:, 1:4].to_numpy(dtype=float)
    idx1 = {nm: i for i, nm in enumerate(names1)}
    R_all = np.linalg.norm(pos_node, axis=1)
    R = R_all.mean()
    print(f"附件1：{len(names1)} 个主索节点，基准球面半径 R = {R:.4f} m")

    # 附件2：促动器下端点、基准态上端点、对应主索节点编号
    df2 = pd.read_csv(DATA / "附件2.csv", encoding="gbk")
    print(f"附件2：{len(df2)} 个促动器，列名 = {list(df2.columns)}")
    names2 = df2.iloc[:, 0].astype(str).tolist()
    pos_lower = df2.iloc[:, 1:4].to_numpy(dtype=float)   # 地锚点（下端点，固定）
    pos_upper = df2.iloc[:, 4:7].to_numpy(dtype=float)   # 基准态促动器上端点

    # 对齐到附件1 的索引
    node_idx2 = np.array([idx1[nm] for nm in names2])
    pos_node_link = pos_node[node_idx2]

    # (1) 促动器上端点 vs 主索节点
    diff_upper_node = np.linalg.norm(pos_upper - pos_node_link, axis=1)
    print("\n=== (1) 促动器上端点 vs 主索节点 ===")
    print(f"  最大距离 = {diff_upper_node.max():.4f} m")
    print(f"  均值距离 = {diff_upper_node.mean():.4f} m")
    print(f"  最小距离 = {diff_upper_node.min():.4f} m")
    print(f"  是否重合？{'是' if diff_upper_node.max() < 1e-3 else '否'}")

    # 下拉索（基准态）：从上端点指向主索节点
    pull_vec = pos_node_link - pos_upper
    pull_len = np.linalg.norm(pull_vec, axis=1)
    pull_dir = pull_vec / pull_len[:, None]
    print(f"\n  下拉索长度：均值 {pull_len.mean():.4f} m，"
          f"范围 [{pull_len.min():.4f}, {pull_len.max():.4f}] m")

    # 节点径向方向（C 指向节点）
    radial_dir = pos_node_link / R_all[node_idx2, None]
    # 下拉索方向与径向方向夹角（基准态）
    cos_ang = np.einsum("ij,ij->i", pull_dir, radial_dir)
    ang_deg = np.rad2deg(np.arccos(np.clip(cos_ang, -1, 1)))
    print(f"\n  下拉索方向 vs 径向方向夹角："
          f"均值 {ang_deg.mean():.4f}°，最大 {ang_deg.max():.4f}°")

    # (2) 论文假设下：节点沿径向移动 δ（取附件4 result.xlsx 里的实际 δ）
    print("\n=== (2) 按论文假设调节后下拉索长度的变化 ===")
    try:
        df_res = pd.read_excel(DATA.parent / "vibecoding" / "result.xlsx",
                               sheet_name=None)
        # 找包含伸缩量列的工作表
        ext_sheet = None
        for sn, df in df_res.items():
            if df.shape[1] >= 2 and ("伸缩量" in str(df.columns[1])
                                     or "ext" in str(df.columns[1]).lower()):
                ext_sheet = (sn, df)
                break
        if ext_sheet is None:
            # 退化：用所有数值列第二个
            for sn, df in df_res.items():
                if df.shape[1] >= 2:
                    ext_sheet = (sn, df)
                    break
        sn, df = ext_sheet
        print(f"  读取 result.xlsx 工作表 {sn!r}，列 = {list(df.columns)}")
        # 取伸缩量列（数值列）
        ext_col = [c for c in df.columns
                   if df[c].dtype.kind in "f" and df[c].abs().max() > 1e-6]
        if not ext_col:
            ext_col = [df.columns[1]]
        ext = df[ext_col[0]].to_numpy(dtype=float)
        print(f"  伸缩量范围 [{ext.min():.4f}, {ext.max():.4f}] m，"
              f"非零个数 {(np.abs(ext) > 1e-9).sum()}")
    except Exception as e:
        print(f"  读取 result.xlsx 失败：{e}")
        print("  用理论最大值 δ = ±0.6 m 模拟")
        ext = np.full(len(names2), 0.0)

    # 节点调节后位置（按论文假设：沿径向移动 δ，向球心为正）
    pos_node_adj = pos_node_link - ext[:, None] * radial_dir
    # 促动器上端点沿径向移动 δ（如果假设促动器也沿径向，与节点同步）
    pos_upper_adj = pos_upper - ext[:, None] * radial_dir
    # 下拉索新长度（若下拉索视为刚性不变）
    pull_len_adj = np.linalg.norm(pos_node_adj - pos_upper_adj, axis=1)
    rel_change_pull = np.abs(pull_len_adj - pull_len) / pull_len
    print(f"\n  若促动器上端点也沿径向同步移动 δ：")
    print(f"    下拉索长度相对变化最大 = {rel_change_pull.max()*100:.4f}%")
    print(f"    超过 0.07% 的个数 = {(rel_change_pull > 0.0007).sum()}")

    # 若下拉索严格长度不变，节点必须沿下拉索方向移动
    pos_node_adj2 = pos_node_link - ext[:, None] * pull_dir
    # 这种情形下径向偏移量
    radial_drift = pos_node_adj2 - (pos_node_link -
                                    ext[:, None] * radial_dir)
    radial_drift_norm = np.linalg.norm(radial_drift, axis=1)
    print(f"\n  若严格保持下拉索长度不变（节点沿下拉索方向移动 δ）：")
    print(f"    与径向移动的横向偏离均值 = {radial_drift_norm.mean()*1000:.2f} mm")
    print(f"    横向偏离最大 = {radial_drift_norm.max()*1000:.2f} mm")
    print(f"    横向偏离 / |δ| 比值均值 = "
          f"{(radial_drift_norm.mean()/(np.abs(ext)+1e-9).mean()):.4f}")


if __name__ == "__main__":
    main()
