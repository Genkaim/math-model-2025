# -*- coding: utf-8 -*-
"""5.3.3 两光束模型数值模拟（对应正文式(34)附近的判别阈值）。

模拟内容：
1. 按硅的参数合成两光束反射谱，考察多光束分母因子（振幅比 ρ 取
   0.03、0.05、0.10）给峰谷反射率读数、反推折射率与 η 带来的改变，
   用于确定式(33)的两个阈值；
2. 在两光束谱上叠加与实测相当水平的扰动，重复统计 η，
   用于给出"η 正常波动范围约 50%±2%"的结论。

运行环境：Python 3 + numpy / scipy
用法：在项目根目录运行  python "5 模型的建立与求解/5.3 问题3/5.3判别模拟.py"
"""
import numpy as np
from scipy.signal import find_peaks

# 硅参数：|r01| = 0.548（n = 3.42），B 由实测峰谷反射率
# Rmax ≈ 39%、Rmin ≈ 21% 确定；入射角 10°，厚度 3.7 μm。
R01, B = -0.548, 0.083
N1, THETA, D = 3.42, np.radians(10.), 3.7e-4
C = np.sqrt(N1 ** 2 - np.sin(THETA) ** 2)
S = np.linspace(600., 3000., 24001)
PHI = 4 * np.pi * D * C * S


def R_of(rho):
    """多光束反射率，ρ=0 时退化为两光束。"""
    return np.abs(R01 + B * np.exp(1j * PHI)
                  / (1 - rho * np.exp(1j * PHI))) ** 2 * 100


def eta_of(R):
    """全部峰谷区间低于中线的点占比 η（式(34)）。"""
    pk, _ = find_peaks(R, prominence=0.5)
    va, _ = find_peaks(-R, prominence=0.5)
    E = sorted(list(pk) + list(va))
    fr = [(R[a:b] < (R[a] + R[b]) / 2).mean()
          for a, b in zip(E[:-1], E[1:]) if b - a > 50]
    return np.mean(fr) * 100


def main():
    print('一、振幅比 ρ 的影响（无扰动）')
    print('-' * 66)
    R_base = R_of(0.0)
    pk0, _ = find_peaks(R_base, prominence=0.5)
    va0, _ = find_peaks(-R_base, prominence=0.5)
    Rmax0, Rmin0 = R_base[pk0].mean(), R_base[va0].mean()
    u0 = (np.sqrt(Rmax0 / 100) + np.sqrt(Rmin0 / 100)) / 2
    print(f'基准（两光束）: Rmax = {Rmax0:.2f}%  Rmin = {Rmin0:.2f}%  '
          f'η = {eta_of(R_base):.1f}%')
    for rho in (0.03, 0.05, 0.10):
        R = R_of(rho)
        pk, _ = find_peaks(R, prominence=0.5)
        va, _ = find_peaks(-R, prominence=0.5)
        Rmax, Rmin = R[pk].mean(), R[va].mean()
        u = (np.sqrt(Rmax / 100) + np.sqrt(Rmin / 100)) / 2
        print(f'ρ = {rho:.2f}: Rmax {Rmax - Rmax0:+.2f} 个百分点, '
              f'Rmin {Rmin - Rmin0:+.2f} 个百分点, '
              f'反推折射率偏差 {(u - u0) / u0 * 100:+.2f}%, '
              f'η = {eta_of(R):.1f}%')

    print()
    print('二、两光束谱叠加扰动后 η 的波动（各 200 次）')
    print('-' * 66)
    rng = np.random.default_rng(0)
    for sig in (0.05, 0.1):
        vals = [eta_of(R_base + rng.normal(0., sig, len(R_base)))
                for _ in range(200)]
        vals = np.array(vals)
        print(f'扰动 {sig:.2f} 个百分点: η 在 '
              f'{vals.min():.1f}% ~ {vals.max():.1f}% 之间 '
              f'（标准差 {vals.std():.2f} 个百分点）')


if __name__ == '__main__':
    main()
