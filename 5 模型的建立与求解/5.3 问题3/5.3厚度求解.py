# -*- coding: utf-8 -*-
"""
问题三：硅外延层厚度的多光束模型计算（方案一，与论文 5.3.5 节一致）
================================================================

附件 3、4 已确认存在多光束干涉。方案一以式(30)的多光束反射率表达式
对全谱作最小二乘拟合，显式计入后续各束反射光的贡献：

    R(σ) = | r01 − B(σ)·e^{iφ} / (1 − q·e^{iφ}) |²,
    φ = 4πσ·d·sqrt(n² − n0²·sin²θ).

硅谱的第二束光振幅随波数变化明显（低波数端谷接近零，高波数端条纹
变浅），故 B 不作常数，而取若干节点的线性插值；r01 与 q 仍为常数。
波数低于约 1000 cm⁻¹ 的低频段，重掺杂衬底的折射率偏离名义值较多，
常数折射率的前提不再成立，拟合取 1000~3400 cm⁻¹ 的有效区间。

数据预处理与 5.2 节相同：删除反射率非正的点。

计算流程：
  1. 折射率取硅的文献名义值 n = 3.42（红外透明区）；
  2. 厚度初值由相邻峰间距估算，并在其附近扫描残差精化；
  3. 以式(30)作最小二乘拟合，拟合参数为厚度 d、
     B 的节点值与组合量 q = r10·r12·A；
  4. 输出拟合厚度、|q| 与残差均方根。

运行环境：Python 3 + numpy / scipy / openpyxl
用法：在项目根目录运行  python "5 模型的建立与求解/5.3 问题3/5.3厚度求解.py"
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks
import openpyxl

N0 = 1.0
N_SI = 3.42                 # 硅的文献名义折射率（红外透明区）
D_BOUND = (5e-5, 3e-3)      # 厚度拟合边界（cm），即 0.5~30 μm
KNOTS = 6                   # B(σ) 插值节点数
FIT_RANGE = (1000., 3400.)  # 有效拟合区间（cm⁻¹）
PROM = 1.0                  # 峰谷检测显著度门槛（百分点），与 5.2 节相同


def load_data(path):
    """读取 xlsx 附件，返回 (波数 cm^-1, 反射率 %)。"""
    wb = openpyxl.load_workbook(path, read_only=True)
    rows = list(wb.active.iter_rows(values_only=True))
    wb.close()
    arr = np.array([[float(r[0]), float(r[1])] for r in rows[1:]
                    if r[0] is not None and r[1] is not None])
    return arr[:, 0], arr[:, 1]


def phase(s, d, theta):
    """往返相位 φ = 4πσd·sqrt(n² − sin²θ)。"""
    return 4 * np.pi * s * d * np.sqrt(N_SI ** 2 - np.sin(theta) ** 2)


def r_multi(s, d, knots, b_nodes, q, r01, theta):
    """多光束模型，式(30)：R = |r01 − B(σ) e^{iφ}/(1 − q e^{iφ})|²，
    B(σ) 由节点值线性插值。"""
    phi = phase(s, d, theta)
    b = np.interp(s, knots, b_nodes)
    return np.abs(r01 - b * np.exp(1j * phi)
                  / (1.0 - q * np.exp(1j * phi))) ** 2 * 100.


def fit(s, R, theta, model, x0, bounds=(-np.inf, np.inf)):
    """最小二乘拟合，返回参数向量与残差均方根（百分点）。"""
    resid = lambda x: model(s, *x, theta) - R
    sol = least_squares(resid, x0, bounds=bounds, x_scale='jac')
    rmse = np.sqrt(np.mean(resid(sol.x) ** 2))
    return sol.x, rmse


def d0_from_peaks(s, R, theta):
    """相邻峰间距估厚度初值：Δσ ≈ 1/(2d·sqrt(n²−sin²θ))。"""
    pk, _ = find_peaks(R, prominence=PROM)
    dσ = np.median(np.diff(s[pk]))
    return 1.0 / (2.0 * dσ * np.sqrt(N_SI ** 2 - np.sin(theta) ** 2))


def solve(path, theta, rng=FIT_RANGE):
    """一份附件：式(30)拟合，返回 (厚度/μm, |q|, rms, 参数, 节点, σ, R)。"""
    s, R = load_data(path)
    keep = R > 0                                # 预处理：删除反射率非正的点
    s, R = s[keep], R[keep]
    if rng:
        msk = (s >= rng[0]) & (s <= rng[1])
        s, R = s[msk], R[msk]

    r01 = (N_SI - N0) / (N_SI + N0)
    knots = np.linspace(s.min(), s.max(), KNOTS)

    # 厚度初值：由相邻峰间距估算，再在其附近扫描残差取最小者精化；
    # 扫描时 B 暂取常数，只用于定厚度，不参与最终结果。
    d_init = d0_from_peaks(s, R, theta)
    b_const = (np.sqrt(R.max() / 100.) - np.sqrt(R.min() / 100.)) / 2.
    grid = np.linspace(0.9 * d_init, 1.1 * d_init, 201)
    d_scan = grid[np.argmin([np.mean((r_multi(s, d, knots,
                                              np.full(KNOTS, b_const),
                                              0.0, r01, theta) - R) ** 2)
                             for d in grid])]

    # 式(30)拟合：参数为 d、B 的节点值与 q；
    # B 节点初值取包络振幅的典型值。
    x0 = np.concatenate([[d_scan], np.full(KNOTS, b_const), [0.05]])
    lo = np.concatenate([[D_BOUND[0]], np.zeros(KNOTS), [-0.3]])
    hi = np.concatenate([[D_BOUND[1]], np.full(KNOTS, 1.0), [0.3]])
    x, rms = fit(s, R, theta,
                 lambda sg, d, *rest: r_multi(sg, d, knots, rest[:KNOTS],
                                              rest[KNOTS], r01, theta),
                 x0, (lo, hi))
    return x[0] * 1e4, abs(x[-1]), rms, x, knots, s, R


def main():
    files = [('附件 3（Si，10°）', '题目/附件/附件3.xlsx', np.radians(10.)),
             ('附件 4（Si，15°）', '题目/附件/附件4.xlsx', np.radians(15.))]

    print('=' * 72)
    print(f'硅外延层厚度（方案一：多光束模型式(30)拟合；n = 3.42，'
          f'有效区间 {FIT_RANGE[0]:.0f}~{FIT_RANGE[1]:.0f} cm⁻¹）')
    print('=' * 72)
    ds = []
    for name, path, theta in files:
        d, q, rms, _x, _knots, _s, _R = solve(path, theta)
        ds.append(d)
        print(f'{name:<16} d = {d:6.3f} μm   |q| = {q:.3f}   rms = {rms:5.2f}%')
    rel = abs(ds[0] - ds[1]) / np.mean(ds) * 100
    print('-' * 72)
    print(f'双入射角一致性：{ds[0]:.3f} μm 与 {ds[1]:.3f} μm，'
          f'相对差异 {rel:.2f}%；平均 d ≈ {np.mean(ds):.3f} μm')
    print('=' * 72)


if __name__ == '__main__':
    main()
