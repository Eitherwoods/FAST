# -*- coding: utf-8 -*-
"""附件2 几何一致性检验（补充）：促动器方向 vs 径向方向。

回答：论文假设3「促动器沿基准球面径向伸缩」是否与附件2 给出的几何一致？
促动器方向 = 上端点 - 下端点（地锚→顶端），若该方向与节点径向方向近似共线，
则「促动器沿径向伸缩」是合理的几何简化。
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(r"d:\数模\26最后的培训\FAST\题目与附件与思路")


def main():
    df1 = pd.read_csv(DATA / "附件1.csv", encoding="gbk")
    names1 = df1.iloc[:, 0].astype(str).tolist()
    pos_node = df1.iloc[:, 1:4].to_numpy(dtype=float)
    idx1 = {nm: i for i, nm in enumerate(names1)}
    R_all = np.linalg.norm(pos_node, axis=1)
    R = R_all.mean()

    df2 = pd.read_csv(DATA / "附件2.csv", encoding="gbk")
    names2 = df2.iloc[:, 0].astype(str).tolist()
    pos_lower = df2.iloc[:, 1:4].to_numpy(dtype=float)
    pos_upper = df2.iloc[:, 4:7].to_numpy(dtype=float)
    node_idx2 = np.array([idx1[nm] for nm in names2])
    pos_node_link = pos_node[node_idx2]

    # 促动器方向（下端→上端）+ 长度
    act_vec = pos_upper - pos_lower
    act_len = np.linalg.norm(act_vec, axis=1)
    act_dir = act_vec / act_len[:, None]

    # 节点径向方向（C→节点）
    radial_dir = pos_node_link / R_all[node_idx2, None]

    # 促动器方向 vs 径向方向
    cos_act_rad = np.einsum("ij,ij->i", act_dir, radial_dir)
    # 注意：促动器从下端（更靠近球心，z 更负）到上端（更靠外，z 更正）
    # 径向方向是 C→节点，节点在球面外（z 更负？不一定）
    # 我们要看绝对夹角
    cos_abs = np.abs(cos_act_rad)
    ang_act_rad = np.rad2deg(np.arccos(np.clip(cos_abs, -1, 1)))

    # 下拉索方向（上端→节点）
    pull_vec = pos_node_link - pos_upper
    pull_len = np.linalg.norm(pull_vec, axis=1)
    pull_dir = pull_vec / pull_len[:, None]
    cos_pull_rad = np.einsum("ij,ij->i", pull_dir, radial_dir)
    ang_pull_rad = np.rad2deg(np.arccos(np.clip(cos_pull_rad, -1, 1)))

    # 三点共线检验：下端、上端、节点是否在径向直线上
    # 即 pull_dir 与 act_dir 是否反向（180°）
    cos_pull_act = np.einsum("ij,ij->i", pull_dir, act_dir)
    ang_pull_act = np.rad2deg(np.arccos(np.clip(cos_pull_act, -1, 1)))

    print("=== 附件2 几何一致性 ===\n")
    print(f"促动器长度：均值 {act_len.mean():.4f} m，"
          f"范围 [{act_len.min():.4f}, {act_len.max():.4f}] m")
    print(f"下拉索长度：均值 {pull_len.mean():.4f} m，"
          f"范围 [{pull_len.min():.4f}, {pull_len.max():.4f}] m")

    print(f"\n促动器方向 vs 节点径向方向：")
    print(f"  夹角均值 {ang_act_rad.mean():.4f}°，"
          f"最大 {ang_act_rad.max():.4f}°，"
          f"最小 {ang_act_rad.min():.4f}°")
    print(f"  -> 促动器严格沿径向？"
          f"{'是' if ang_act_rad.max() < 0.1 else '否，但近似'}")

    print(f"\n下拉索方向 vs 节点径向方向：")
    print(f"  夹角均值 {ang_pull_rad.mean():.4f}°，"
          f"最大 {ang_pull_rad.max():.4f}°")
    print(f"  -> 下拉索严格沿径向反向？"
          f"{'是' if abs(ang_pull_rad.mean()-180) < 0.1 else '否，但近似'}")

    print(f"\n下拉索方向 vs 促动器方向：")
    print(f"  夹角均值 {ang_pull_act.mean():.4f}°，"
          f"最大 {ang_pull_act.max():.4f}°")
    print(f"  -> 三点共线？"
          f"{'是' if abs(ang_pull_act.mean()-180) < 0.1 else '否'}")

    # 严格按附件2 建模：促动器沿 act_dir 方向伸缩 δ
    # 则 upper_new = upper + δ·act_dir，要求 pull_len 不变
    # 解出节点新的位置（保持下拉索长度）
    # 简化：节点沿 radial 方向移动 δ'，解 δ' 使 pull_len 保持
    # 设 upper_new = upper + δ_upper·act_dir, node_new = node + δ_node·radial_dir
    # |node_new - upper_new|² = |node - upper|²
    # 展开: pull_vec + δ_node·radial - δ_upper·act_dir 的模² = pull_len²
    # 取 δ_upper = δ_node = δ（论文同步假设）
    # = pull_len² + 2δ (pull·radial - pull·act_dir) + δ²(|radial|²+|act_dir|²-2 radial·act_dir)
    # 若要严格 = pull_len²，δ 应满足一元二次方程
    # 这里只算: 用论文同步假设的 δ 同步移动后，pull_len 相对变化最大值

    # 用附件4 result.xlsx 的实际 δ
    df_res = pd.read_excel(DATA.parent / "vibecoding" / "result.xlsx",
                            sheet_name="促动器顶端伸缩量")
    ext_col = df_res.columns[1]
    ext = df_res[ext_col].to_numpy(dtype=float)
    # 把 ext 对齐到附件2 顺序
    idx2_to_res = {nm: i for i, nm in enumerate(df_res.iloc[:, 0].astype(str))}
    ext_aligned = np.array([ext[idx2_to_res[nm]] if nm in idx2_to_res else 0.0
                            for nm in names2])

    # 严格按附件2: upper 沿 act_dir 移动 δ_upper，node 沿 radial 移动 δ_node
    # 论文假设 δ_upper = δ_node = δ_aligned，但严格情形两者应解耦
    # 这里用论文假设同步移动 δ，测 pull_len 变化
    pos_upper_adj = pos_upper + ext_aligned[:, None] * act_dir
    pos_node_adj = pos_node_link - ext_aligned[:, None] * radial_dir
    pull_len_adj = np.linalg.norm(pos_node_adj - pos_upper_adj, axis=1)
    rel_pull = np.abs(pull_len_adj - pull_len) / pull_len

    print(f"\n=== 论文同步假设（δ_upper = δ_node = δ）下下拉索长度变化 ===")
    print(f"  δ 范围 [{ext_aligned.min():.4f}, {ext_aligned.max():.4f}] m")
    print(f"  下拉索相对变化最大 = {rel_pull.max()*100:.6f}%")
    print(f"  下拉索相对变化均值 = {rel_pull.mean()*100:.6f}%")
    print(f"  超过 0.07% 的个数 = {(rel_pull > 0.0007).sum()}")

    # 若严格下拉索不变，节点应沿 pull_dir 同步移动（与 upper 同 δ）
    pos_node_adj2 = pos_upper + ext_aligned[:, None] * act_dir + pull_len[:, None] * pull_dir
    # 节点位移 = pos_node_adj2 - pos_node
    disp = pos_node_adj2 - pos_node_link
    disp_radial = np.einsum("ij,ij->i", disp, radial_dir)
    disp_perp = disp - disp_radial[:, None] * radial_dir
    disp_perp_norm = np.linalg.norm(disp_perp, axis=1)
    nonzero = np.abs(ext_aligned) > 1e-6
    print(f"\n=== 严格按附件2 建模（下拉索不变）下节点位移分解 ===")
    print(f"  非零节点数 {(nonzero).sum()}")
    print(f"  径向分量均值 = {disp_radial[nonzero].mean()*1000:.4f} mm")
    print(f"  横向分量均值 = {disp_perp_norm[nonzero].mean()*1000:.4f} mm")
    print(f"  横向/径向 比值均值 = "
          f"{(disp_perp_norm[nonzero]/(np.abs(disp_radial[nonzero])+1e-9)).mean():.4f}")
    print(f"  -> 论文「节点沿径向移动」假设的偏差")
    print(f"     相对横向偏差最大 = {disp_perp_norm.max()*1000:.4f} mm"
          f"（对应 |δ| = {np.abs(ext_aligned).max():.4f} m）")


if __name__ == "__main__":
    main()
