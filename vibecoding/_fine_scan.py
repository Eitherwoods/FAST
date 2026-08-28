# -*- coding: utf-8 -*-
"""临时实验：最优焦距细扫描 + 坐标下降收敛性复核（运行后删除）。"""
import sys
sys.path.insert(0, ".")
import numpy as np
from _coord_test import coordinate_descent, max_rel
import q2

names, pos, panels = q2.load_data()
n = q2.axis_vector(q2.ALPHA, q2.BETA)
dist_to_axis = np.linalg.norm(pos - np.outer(pos @ n, n), axis=1)
in_ap = dist_to_axis <= q2.APERTURE_RADIUS

best = (1e9, None)
for p_test in [140.30, 140.35, 140.40, 140.4249, 140.45, 140.50, 140.55]:
    e_ap, _, _ = q2.radial_extensions(pos[in_ap], n, p_test)
    x = coordinate_descent(e_ap, sweeps=120)
    rms = np.sqrt(np.mean((x - e_ap)**2))
    mr = max_rel(x)
    flag = " <== best" if rms < best[0] else ""
    if rms < best[0]:
        best = (rms, p_test)
    print(f"p={p_test:8.4f}: RMS={rms:.5f} m, max_rel={mr:.4e}{flag}")
print(f"最优 p ≈ {best[1]}, RMS = {best[0]:.5f} m")
