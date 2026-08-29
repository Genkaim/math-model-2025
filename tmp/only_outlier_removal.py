# -*- coding: utf-8 -*-
"""
对照实验：只做「异常点剔除」、不做有效干涉区划分，直接全谱求解
==============================================================
论文 5.2.2 给出了三步预处理：异常点剔除 → 有效干涉区划分 → 峰谷检测。
本脚本只保留第一步，考察若把全谱（剔除首点伪零后）都拿来算，结果会偏离多少。

算法步骤（与 5.2.3 完全相同，只是把区间 [1400,2500] 换成全谱 [400,4000]）：
  1. 按剔除规则删除 R<=0 且与相邻点突变 >10 个百分点的孤立点
  2. 全谱峰谷检测（显著度 1%）
  3. 逐对相邻峰谷用包络线法反演 n(σp)、n(σv)（问题一式(13)(19)）
  4. 代入式(3)求 d_j，取中位数与 MAD
"""
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import median_abs_deviation
import openpyxl
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
N0 = 1.0
REGION_C = (1400.0, 2500.0)          # 原方案的有效干涉区，用于对照
PROM = 1.0                            # 显著度（百分点）
JUMP = 10.0                           # 突变阈门（百分点）


def load(path):
    wb = openpyxl.load_workbook(path, read_only=True)
    rows = list(wb.active.iter_rows(values_only=True))
    wb.close()
    a = np.array([[float(r[0]), float(r[1])] for r in rows[1:]
                  if r[0] is not None and r[1] is not None])
    return a[:, 0], a[:, 1]


def remove_outliers(s, R):
    """剔除规则：R<=0 且与相邻测量点突变超过 JUMP 个百分点。"""
    keep = np.ones(len(R), bool)
    for i in np.where(R <= 0)[0]:
        # 与左右相邻点的偏差
        dev = []
        if i > 0:
            dev.append(abs(R[i] - R[i - 1]))
        if i < len(R) - 1:
            dev.append(abs(R[i] - R[i + 1]))
        if max(dev) > JUMP:
            keep[i] = False
    return s[keep], R[keep]


def solve(s, R, theta, lo, hi, filt=False):
    """在 [lo,hi] 内：峰谷检测 → 包络线法反演 n(σ) → 逐对厚度。

    与论文 5.2.3 一致：先把各对反演的 n 按中心波数排成 n(σ)（此处用线性插值，
    论文的可靠区内用一次拟合，二者在 n 平滑时等价），再在 σp、σv 处分别取 n，
    代入式(3)。filt=True 时启用包络线法内部的数值有效性检查（Rmin>0.5、Rmax<90）。
    """
    pk, _ = find_peaks(R, prominence=PROM)
    va, _ = find_peaks(-R, prominence=PROM)
    E = sorted([(s[i], R[i], 'max') for i in pk] + [(s[i], R[i], 'min') for i in va])
    sin2 = np.sin(theta) ** 2
    rows = []
    for a, b in zip(E[:-1], E[1:]):
        if a[2] == b[2]:
            continue
        s_p, s_v = min(a[0], b[0]), max(a[0], b[0])
        if not (lo < s_p and s_v < hi):
            continue
        Rmax, Rmin = max(a[1], b[1]), min(a[1], b[1])
        if filt and (Rmin < 0.5 or Rmax > 90):
            continue
        sr, srm = np.sqrt(Rmax / 100.), np.sqrt(Rmin / 100.)
        u = (sr + srm) / 2.                       # 问题一式(13)
        if not (0 < u < 1):
            continue
        n = N0 * (1 + u) / (1 - u)                # 问题一式(19)
        rows.append((s_p, s_v, (s_p + s_v) / 2., n))
    P = np.array(rows)
    o = np.argsort(P[:, 2])
    n_p = np.interp(P[:, 0], P[o, 2], P[o, 3])    # n(σp)
    n_v = np.interp(P[:, 1], P[o, 2], P[o, 3])    # n(σv)
    d = 1.0 / (4.0 * np.abs(P[:, 1] * np.sqrt(n_v ** 2 - sin2)
                            - P[:, 0] * np.sqrt(n_p ** 2 - sin2))) * 1e4
    return np.column_stack([P, n_p, n_v, d]), P[:, [2, 3]], len(E)


def report(name, ang, s_raw, R_raw):
    s, R = remove_outliers(s_raw, R_raw)
    th = np.radians(ang)
    print('=' * 72)
    print(f'【{name}】入射角 {ang}°   原始 {len(R_raw)} 点 → 剔除后 {len(R)} 点')
    print(f'  剔除点：σ={s_raw[R_raw <= 0]}  R={R_raw[R_raw <= 0]}')

    # ---------- 全谱（只做异常点剔除） ----------
    P_all, _, nex = solve(s, R, th, 400., 4000.)
    print(f'\n  全谱峰谷检测：极值 {nex} 个，可用峰谷对 {len(P_all)} 个')
    print(f'  {"σp":>8} {"σv":>8} {"n(σp)":>7} {"n(σv)":>7} {"d_j/μm":>9}')
    for sp, sv, mid, n, np_, nv, d in P_all:
        print(f'  {sp:8.2f} {sv:8.2f} {np_:7.3f} {nv:7.3f} {d:9.2f}')
    med_all = np.median(P_all[:, 6])
    mad_all = median_abs_deviation(P_all[:, 6], scale=1.0)
    print(f'  >> 全谱中位数 d = {med_all:.2f} μm，MAD = {mad_all:.2f} μm '
          f'（{mad_all / med_all * 100:.1f}%），d_j 范围 {P_all[:, 6].min():.2f}~{P_all[:, 6].max():.2f} μm')

    q = np.percentile(P_all[:, 6], [25, 50, 75])
    print(f'  >> 全谱 d_j 四分位：Q1={q[0]:.2f}，中位数={q[1]:.2f}，Q3={q[2]:.2f}；'
          f'算术平均={P_all[:, 6].mean():.2f}（中位数对离群值稳健，均值不稳健）')

    # ---------- 全谱 + 一次拟合 n(σ)（照搬论文实现，仅把区间换成全谱） ----------
    P_b, N_b, _ = solve(s, R, th, 400., 4000.)
    cb = np.polyfit(N_b[:, 0], N_b[:, 1], 1)
    nb_p, nb_v = np.polyval(cb, P_b[:, 0]), np.polyval(cb, P_b[:, 1])
    sin2 = np.sin(th) ** 2
    d_b = 1.0 / (4.0 * np.abs(P_b[:, 1] * np.sqrt(nb_v ** 2 - sin2)
                              - P_b[:, 0] * np.sqrt(nb_p ** 2 - sin2))) * 1e4
    print(f'  >> 若 n(σ) 仍按论文做法取一次拟合：中位数 d = {np.median(d_b):.2f} μm，'
          f'MAD = {median_abs_deviation(d_b, scale=1.0):.2f} μm，'
          f'd_j 范围 {d_b.min():.2f}~{d_b.max():.2f} μm')

    # ---------- 原方案（有效干涉区 1400~2500） ----------
    P_c, _, _ = solve(s, R, th, *REGION_C)
    med_c = np.median(P_c[:, 6])
    mad_c = median_abs_deviation(P_c[:, 6], scale=1.0)
    print(f'\n  【对照】有效干涉区 1400~2500：{len(P_c)} 对，'
          f'中位数 d = {med_c:.2f} μm，MAD = {mad_c:.2f} μm（{mad_c / med_c * 100:.1f}%）')
    print(f'  >> 全谱结果相对有效区结果偏差 {abs(med_all - med_c) / med_c * 100:.1f}%')

    # ---------- 分波段看全谱的 d_j ----------
    bands = [(400, 780), (780, 968), (968, 1400), (1400, 2500), (2500, 4000)]
    print('\n  分波段 d_j（全谱中位数）：')
    for lo, hi in bands:
        m = (P_all[:, 2] >= lo) & (P_all[:, 2] < hi)
        if m.sum() == 0:
            print(f'    {lo:5}~{hi:5}: 无峰谷对')
            continue
        print(f'    {lo:5}~{hi:5}: {m.sum():2d} 对，d_j 中位数 '
              f'{np.median(P_all[m, 6]):6.2f} μm，n 范围 {P_all[m, 4].min():.2f}~{P_all[m, 5].max():.2f}')
    return med_all, med_c


def main():
    res = {}
    for name, ang in [('附件1', 10.), ('附件2', 15.)]:
        s, R = load(str(BASE / f'题目/附件/{name}.xlsx'))
        res[name] = report(name, ang, s, R)
    print('\n' + '=' * 72)
    a, b = res['附件1'][0], res['附件2'][0]
    c, d = res['附件1'][1], res['附件2'][1]
    print(f'双入射角一致性：')
    print(f'  全谱方案    10°={a:.2f} μm，15°={b:.2f} μm，相对差异 {abs(a - b) / ((a + b) / 2) * 100:.2f}%')
    print(f'  有效区方案  10°={c:.2f} μm，15°={d:.2f} μm，相对差异 {abs(c - d) / ((c + d) / 2) * 100:.2f}%')


if __name__ == '__main__':
    main()
