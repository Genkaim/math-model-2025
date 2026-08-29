# -*- coding: utf-8 -*-
"""问题三预研：多光束干涉判别
判据1：相邻峰谷之间位于中线以下的点占比（正弦≈50%，尖峰>50%，尖谷<50%）
判据2：第三束光振幅比 q = |r10*r12|，r12 由包络法 B 反推
"""
import numpy as np
import openpyxl
from scipy.signal import find_peaks

def load(path):
    wb = openpyxl.load_workbook(path, read_only=True)
    rows = list(wb.active.iter_rows(values_only=True))
    wb.close()
    arr = np.array([[float(r[0]), float(r[1])] for r in rows[1:]
                    if r[0] is not None and r[1] is not None])
    return arr[:, 0], arr[:, 1]

files = [('附件1 SiC 10°', '题目/附件/附件1.xlsx', 2.55, (787., 968.)),
         ('附件2 SiC 15°', '题目/附件/附件2.xlsx', 2.55, (787., 968.)),
         ('附件3 Si  10°', '题目/附件/附件3.xlsx', 3.42, None),
         ('附件4 Si  15°', '题目/附件/附件4.xlsx', 3.42, None)]

print(f"{'附件':<14}{'波数范围':<20}{'峰谷对':>6}{'低于中线占比':>12}{'q=r10*r12':>10}")
print('-' * 66)
for name, path, n1, restr in files:
    s, R = load(path)
    m = R > 0
    if restr:
        m &= ~((s > restr[0]) & (s < restr[1]))   # 剔除残余射线带
    s, R = s[m], R[m]
    pk, _ = find_peaks(R, prominence=1.0)
    va, _ = find_peaks(-R, prominence=1.0)
    E = sorted(list(pk) + list(va))
    fracs, qs = [], []
    n0 = 1.0
    r10 = (n1 - n0) / (n1 + n0)
    for a, b in zip(E[:-1], E[1:]):
        seg = (s > s[a]) & (s < s[b])
        if seg.sum() < 20:
            continue
        mid = (R[a] + R[b]) / 2.
        fracs.append((R[seg] < mid).mean())
        Rmax, Rmin = max(R[a], R[b]), min(R[a], R[b])
        B = (np.sqrt(Rmax / 100.) - np.sqrt(Rmin / 100.)) / 2.
        t01, t10 = 2 * n0 / (n0 + n1), 2 * n1 / (n1 + n0)
        r12 = B / (t01 * t10)
        qs.append(r10 * r12)
    f = np.mean(fracs)
    print(f"{name:<16}{s.min():7.1f}~{s.max():7.1f}  {len(fracs):>5}"
          f"{f*100:>10.1f}%  {np.median(qs):>10.3f}")
