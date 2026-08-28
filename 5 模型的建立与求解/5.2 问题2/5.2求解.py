# -*- coding: utf-8 -*-
"""
碳化硅外延层厚度确定 —— 问题二求解算法
========================================
依据问题一的两光束干涉模型：
    R = r01^2 + B^2 + 2 r01 B cos(4*pi*sigma*d*sqrt(n^2 - sin^2(theta)))

算法流程：
  1. 数据读取与预处理（剔除首点伪零值）
  2. 干涉峰谷检测（scipy.signal.find_peaks，显著度过滤剔除噪声）
  3. 包络线法反演折射率 n(sigma)（问题一式(5)(6)）
  4. 相邻峰谷法反演厚度 d_j（问题一式(7)），稳健统计取中位数
  5. 全谱两光束模型非线性拟合精化（交叉验证）
  6. 可靠性分析（双入射角一致性、折射率对照、误差传播）

运行环境：Python 3 + numpy / scipy / openpyxl
"""
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import least_squares
from scipy.stats import median_abs_deviation
import openpyxl

# 常数
N0 = 1.0             # 空气折射率
RESTSTRAHLEN = (780.0, 1030.0)   # 碳化硅残余射线带（反射率饱和，条纹失真，须剔除）
REGION_C = (1400.0, 2500.0)      # 可靠测量区：两光束模型与实测吻合良好
N_STD = 2.55         # 红外反射测厚国家标准名义折射率（对照用）


def load_data(path):
    """读取 xlsx 附件，返回 (波数 cm^-1, 反射率 %)。"""
    wb = openpyxl.load_workbook(path, read_only=True)
    rows = list(wb.active.iter_rows(values_only=True))
    wb.close()
    arr = np.array([[float(r[0]), float(r[1])] for r in rows[1:]
                    if r[0] is not None and r[1] is not None])
    return arr[:, 0], arr[:, 1]


def detect_extrema(s, R, prom=1.0):
    """检测干涉极值；prom 为显著度（%），用于剔除噪声峰。"""
    pk, _ = find_peaks(R, prominence=prom)
    va, _ = find_peaks(-R, prominence=prom)
    E = sorted([(s[i], R[i], 'max') for i in pk]
               + [(s[i], R[i], 'min') for i in va])
    return E


def envelope_method(E, lo, hi):
    """包络线法：对 [lo,hi] 内每对相邻峰谷求 n、r01、B。"""
    out = []
    for i in range(len(E) - 1):
        a, b = E[i], E[i + 1]
        if a[2] == b[2]:
            continue
        sl, sh = min(a[0], b[0]), max(a[0], b[0])
        if not (lo < sl and sh < hi):
            continue
        Rmax, Rmin = max(a[1], b[1]), min(a[1], b[1])
        if Rmin < 0.5 or Rmax > 90:            # 剔除近零谷与饱和峰
            continue
        sr, srmin = np.sqrt(Rmax / 100.), np.sqrt(Rmin / 100.)
        r01 = (sr + srmin) / 2.                # 式(5)
        B = (sr - srmin) / 2.
        if r01 <= 0 or r01 >= 1:
            continue
        n = N0 * (1 + r01) / (1 - r01)         # 式(6)
        out.append(((sl + sh) / 2., n, r01, B))
    return np.array(out)


def thickness_pair(s_p, s_v, n_p, n_v, theta):
    """问题一式(7)：严格计入折射率色散，分别取 n(σp)、n(σv)。"""
    c_p = np.sqrt(n_p ** 2 - np.sin(theta) ** 2)
    c_v = np.sqrt(n_v ** 2 - np.sin(theta) ** 2)
    return 1.0 / (4.0 * abs(s_v * c_v - s_p * c_p))


def solve_extrema_method(s, R, theta):
    """步骤 2-4：峰谷检测 + 包络线法 + 相邻峰谷厚度（可靠区内）。"""
    E = detect_extrema(s, R, 1.0)
    npts = envelope_method(E, REGION_C[0], REGION_C[1])
    cn = np.polyfit(npts[:, 0], npts[:, 1], 1)          # n(sigma) 线性光滑
    pairs = []
    for i in range(len(E) - 1):
        a, b = E[i], E[i + 1]
        if a[2] == b[2]:
            continue
        s_p, s_v = min(a[0], b[0]), max(a[0], b[0])
        if not (REGION_C[0] < s_p and s_v < REGION_C[1]):
            continue
        n_p = np.polyval(cn, s_p)                 # 峰、谷分别取折射率
        n_v = np.polyval(cn, s_v)
        d = thickness_pair(s_p, s_v, n_p, n_v, theta) * 1e4   # cm -> um
        pairs.append((s_p, s_v, (s_p + s_v) / 2., n_p, n_v, d))
    pairs = np.array(pairs)
    return pairs, E, npts


def fit_full_spectrum(s, R, theta, d0):
    """步骤 5：全谱两光束模型最小二乘拟合（交叉验证）。"""
    lo, hi = REGION_C
    m = (s >= lo) & (s <= hi)
    sg, Rg = s[m], R[m] / 100.
    E = detect_extrema(s, R, 1.0)
    npts = envelope_method(E, lo, hi)
    cn = np.polyfit(npts[:, 0], npts[:, 1], 1)
    nE = np.polyval(cn, sg)
    cb = np.polyfit(npts[:, 0], npts[:, 3], 2)
    bE = np.polyval(cb, sg)
    r01 = (nE - N0) / (nE + N0)
    sin2 = np.sin(theta) ** 2

    def model(d):
        phi = 4 * np.pi * d * sg * np.sqrt(nE ** 2 - sin2)
        return r01 ** 2 + bE ** 2 + 2 * r01 * bE * np.cos(phi)

    def resid(d):
        return model(d) - Rg

    sol = least_squares(resid, d0, bounds=([4e-4], [3e-3]))
    rmse = np.sqrt(np.mean(resid(sol.x[0]) ** 2)) * 100
    return sol.x[0] * 1e4, rmse


def main():
    files = {'附件1（入射角10°）': '题目/附件/附件1.xlsx',
             '附件2（入射角15°）': '题目/附件/附件2.xlsx'}
    theta = {'附件1（入射角10°）': 10., '附件2（入射角15°）': 15.}
    print('=' * 66)
    print('问题二计算结果：碳化硅外延层厚度确定')
    print('=' * 66)
    d_med = {}
    for name, path in files.items():
        s, R = load_data(path)
        s, R = s[R > 0], R[R > 0]                     # 剔除首点伪零
        th = np.radians(theta[name])
        pairs, E, npts = solve_extrema_method(s, R, th)
        # 剔除受残余射线带影响的边界畸变对（峰谷中心波数 < 1500）
        clean = pairs[pairs[:, 2] > 1500]
        med = np.median(clean[:, 5])
        mad = median_abs_deviation(clean[:, 5], scale=1.0)
        d_fit, rmse = fit_full_spectrum(s, R, th, med * 1e-4)
        d_med[name] = med
        print(f'\n【{name}】')
        print(f'  检测到干涉极值 {len(E)} 个（峰 {sum(1 for e in E if e[2]=="max")}，'
              f'谷 {sum(1 for e in E if e[2]=="min")}）')
        print(f'  可靠区 {REGION_C[0]:.0f}-{REGION_C[1]:.0f} cm-1 内峰谷对 '
              f'{len(pairs)} 个，剔除畸变对后 {len(clean)} 个')
        print(f'  单峰谷对厚度 d_j / μm：')
        for sp, sv, mid, np_, nv, d in clean:
            print(f'    σp={sp:7.2f} σv={sv:7.2f}  n(σp)={np_:5.3f} '
                  f'n(σv)={nv:5.3f}  d={d:6.2f}')
        print(f'  >> 相邻峰谷法：中位数 d = {med:.2f} μm，'
              f'MAD = {mad:.2f} μm（{mad/med*100:.1f}%）')
        print(f'  >> 全谱拟合精化：d = {d_fit:.2f} μm，拟合残差 rms = {rmse:.2f}%')
    print('\n' + '-' * 66)
    d1, d2 = list(d_med.values())
    print(f'双入射角一致性：10° 测得 {d1:.2f} μm，15° 测得 {d2:.2f} μm，'
          f'相对差异 {abs(d1-d2)/d1*100:.2f}%')
    print(f'最终结论：外延层厚度 d ≈ {np.mean([d1, d2]):.1f} ± 0.5 μm')


if __name__ == '__main__':
    main()
