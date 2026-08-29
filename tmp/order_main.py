# -*- coding: utf-8 -*-
"""级次回归定厚：完整实现 + 自检表数据 + 方案一段内拟合对照"""
import numpy as np
from scipy.stats import median_abs_deviation
import importlib.util, os

ROOT = r"c:/Users/cxh20/Documents/AAACode/MathModel/优秀论文/复现2"
_spec = importlib.util.spec_from_file_location(
    "sol52", os.path.join(ROOT, "5 模型的建立与求解", "5.2 问题2", "5.2求解.py"))
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

FILES = {'附件1（10°）': ('题目/附件/附件1.xlsx', 10.),
         '附件2（15°）': ('题目/附件/附件2.xlsx', 15.)}


def prepare(name):
    """返回 (sig 极值波数升序, sc 散点中心波数, nv 散点反演 n, theta)。"""
    rel, thdeg = FILES[name]
    s_raw, R_raw = S.load_data(os.path.join(ROOT, *rel.split('/')))
    s, R = S.remove_outliers(s_raw, R_raw)
    E = S.detect_extrema(s, R)
    P = S.envelope_pairs(E)
    o = np.argsort(P[:, 2])
    return np.array([e[0] for e in E]), P[o, 2], P[o, 3], np.radians(thdeg)


def x_series(sig, sc, nv, th):
    """极值处的 n_j 由散点线性插值，构造 x_j。"""
    n_j = np.interp(sig, sc, nv)
    x = 2 * sig * np.sqrt(n_j ** 2 - np.sin(th) ** 2)
    return n_j, x


def best_segment(x, min_len=8):
    """枚举长度≥min_len 的连续段，按标准误 rms/√N 最小取最优。
    返回 (标准误, rms, R², i0, i1, b)。"""
    out = []
    for i0 in range(len(x)):
        for i1 in range(i0 + min_len, len(x) + 1):
            j = np.arange(i1 - i0)
            xs = x[i0:i1]
            b, a = np.polyfit(j, xs, 1)
            r = xs - (a + b * j)
            rms = float(np.sqrt(np.mean(r ** 2)))
            ss = np.sum((xs - xs.mean()) ** 2)
            out.append((rms / np.sqrt(i1 - i0), rms,
                        1 - np.sum(r ** 2) / ss, i0, i1, b))
    out.sort()
    return out[0]


def seg_fit(s, R, th, lo, hi, d0):
    """方案一：在 [lo,hi] 段内最小二乘拟合厚度。"""
    from scipy.optimize import least_squares
    m = (s >= lo) & (s <= hi)
    sg, Rg = s[m], R[m] / 100.
    E = S.detect_extrema(s, R)
    pt = np.array([r for r in S.full_npts(E)
                   if lo <= r[0] <= hi])
    pt = pt[np.argsort(pt[:, 0])]
    nE = np.interp(sg, pt[:, 0], pt[:, 1])
    bE = np.interp(sg, pt[:, 0], pt[:, 2])
    sin2 = np.sin(th) ** 2
    rs, rp = S._fresnel_mod(nE, th)

    def model(d):
        phi = 4 * np.pi * d * sg * np.sqrt(nE ** 2 - sin2)
        return 0.5 * (rs ** 2 + bE ** 2 + 2 * rs * bE * np.cos(phi)) \
             + 0.5 * (rp ** 2 + bE ** 2 + 2 * rp * bE * np.cos(phi))

    sol = least_squares(lambda d: model(d) - Rg, d0, bounds=S.D_BOUND)
    rmse = np.sqrt(np.mean((model(sol.x[0]) - Rg) ** 2)) * 100
    return sol.x[0] * 1e4, rmse


def main():
    results = {}
    for name in FILES:
        sig, sc, nv, th = prepare(name)
        n_j, x = x_series(sig, sc, nv, th)
        dx = np.diff(x)
        se, rms, r2, i0, i1, b = best_segment(x)
        d = 1e4 / (2 * b)

        print('=' * 78)
        print(f'【{name}】全谱极值 {len(sig)} 个')
        print('=' * 78)
        print(f'  {"j":>3}{"σ_j/cm⁻¹":>10}{"n_j":>8}{"x_j":>10}{"Δx":>9}')
        for k in range(len(sig)):
            ddx = f'{dx[k]:8.1f}' if k < len(dx) else '  —'
            print(f'  {k:3d}{sig[k]:10.1f}{n_j[k]:8.3f}{x[k]:10.1f}{ddx:>9}')

        print(f'\n  Δx ≤ 0 的断点位置：',
              [k for k in range(len(dx)) if dx[k] <= 0] or '无')
        print(f'  自检最优段：j∈[{i0},{i1 - 1}]，波数 {sig[i0]:.0f}~{sig[i1-1]:.0f} cm⁻¹，'
              f'{i1 - i0} 点')
        print(f'  斜率 b = {b:.6f} cm⁻¹，d = {d:.4f} μm')
        print(f'  R² = {r2:.5f}，rms = {rms:.1f}，标准误 = {se:.1f}')

        # 最优段逐点自检表
        j = np.arange(i1 - i0)
        xs = x[i0:i1]
        bj, aj = np.polyfit(j, xs, 1)
        res = xs - (aj + bj * j)
        dxm = (xs[-1] - xs[0]) / (len(j) - 1)
        print(f'\n  最优段自检表（逐点）：')
        print(f'    {"序号":>4}{"σ_j":>9}{"n_j":>8}{"x_j":>10}{"残差":>9}{"残差/Δx":>10}')
        for k in range(len(j)):
            print(f'    {j[k]:4d}{sig[i0+k]:9.1f}{n_j[i0+k]:8.3f}{xs[k]:10.1f}'
                  f'{res[k]:9.1f}{res[k] / dxm:9.3f}')

        # 方案一：在自检段内拟合
        rel, thdeg = FILES[name]
        s_raw, R_raw = S.load_data(os.path.join(ROOT, *rel.split('/')))
        s, R = S.remove_outliers(s_raw, R_raw)
        d_fit, rmse = seg_fit(s, R, th, sig[i0], sig[i1 - 1], d * 1e-4)
        print(f'\n  方案一（段内拟合）：d_fit = {d_fit:.4f} μm，rms = {rmse:.2f}%')
        results[name] = dict(d=d, r2=r2, i0=i0, i1=i1, lo=sig[i0], hi=sig[i1 - 1],
                             npts=i1 - i0, d_fit=d_fit)
        print()

    a, b_ = list(results.values())
    rel = abs(a['d'] - b_['d']) / ((a['d'] + b_['d']) / 2) * 100
    relf = abs(a['d_fit'] - b_['d_fit']) / ((a['d_fit'] + b_['d_fit']) / 2) * 100
    print('=' * 78)
    print('汇总：')
    print(f'  方案二（级次回归）：10° = {a["d"]:.4f} μm（段 {a["lo"]:.0f}~{a["hi"]:.0f}，'
          f'{a["npts"]}点，R²={a["r2"]:.5f}），15° = {b_["d"]:.4f} μm'
          f'（段 {b_["lo"]:.0f}~{b_["hi"]:.0f}，{b_["npts"]}点，R²={b_["r2"]:.5f}）')
    print(f'  双角度差（方案二）：{rel:.2f}%')
    print(f'  方案一（段内拟合）：10° = {a["d_fit"]:.4f} μm，15° = {b_["d_fit"]:.4f} μm，'
          f'双角度差 {relf:.2f}%')
    print('=' * 78)


if __name__ == '__main__':
    main()
