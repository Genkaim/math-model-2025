# -*- coding: utf-8 -*-
"""诊断：折射率—中心波数曲线拟合 vs 散点插值，在级次回归中的稳健性。

改变自检的最短段长 min_len，观察各色散模型给出的 d 是否稳定。
"""
import numpy as np
import importlib.util, sys, os

ROOT = r"c:/Users/cxh20/Documents/AAACode/MathModel/优秀论文/复现2"
sys.path.insert(0, os.path.join(ROOT, "tmp"))
import order_fit_n as O

_sp = importlib.util.spec_from_file_location(
    "sol52", os.path.join(ROOT, "5 模型的建立与求解", "5.2 问题2", "5.2求解.py"))
S = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(S)

SHORT = {'一次   n=a+bσ': '一次', '二次   n=a+bσ+cσ²': '二次',
         'Cauchy  n=A+Bσ²': 'Cauchy', 'Sellmeier1  n²=A+Bσ²': 'Sellmeier'}


def prepare(name):
    rel, thdeg = O.FILES[name]
    s_raw, R_raw = S.load_data(os.path.join(ROOT, *rel.split('/')))
    s, R = S.remove_outliers(s_raw, R_raw)
    E = S.detect_extrema(s, R)
    P = S.envelope_pairs(E)
    sc, nv = P[:, 2], P[:, 3]
    o = np.argsort(sc)
    return (np.array([e[0] for e in E]), sc[o], nv[o], np.radians(thdeg))


def run(min_len):
    out = {}
    for name in O.FILES:
        sig, sc, nv, th = prepare(name)
        n_int = np.interp(sig, sc, nv)
        se, rms, r2, i0, i1, b = O.enumerate_segments(sig, n_int, th, min_len)[0]
        lo, hi = sig[i0], sig[i1 - 1]
        mk = (sc >= lo) & (sc <= hi)
        r = {'插值': O.order_regression(sig, n_int, th, seg=(i0, i1))['d']}
        for mdl in O.MODELS:
            c, f, _ = O.fit_dispersion(sc[mk], nv[mk], mdl)
            r[mdl] = O.order_regression(sig, f(sig, c), th, seg=(i0, i1))['d']
        out[name] = (r, lo, hi, i1 - i0)
    return out


def main():
    A, B = list(O.FILES)
    print('改变自检最短段长 min_len，观察各模型给出的 d（μm）')
    print('=' * 92)
    for ml in (6, 8, 10, 12, 14):
        o = run(ml)
        r1, lo1, hi1, k1 = o[A]
        r2, lo2, hi2, k2 = o[B]
        print(f'\nmin_len = {ml}   选用段：附件1 {lo1:.0f}~{hi1:.0f}（{k1}点）  '
              f'附件2 {lo2:.0f}~{hi2:.0f}（{k2}点）')
        print(f'    {"模型":<12}{"附件1 10°":>11}{"附件2 15°":>11}{"相对差异":>11}')
        print('    ' + '-' * 46)
        for k in ['插值'] + list(O.MODELS):
            d1, d2 = r1[k], r2[k]
            lab = SHORT.get(k, k)
            print(f'    {lab:<12}{d1:11.3f}{d2:11.3f}'
                  f'{abs(d1 - d2) / ((d1 + d2) / 2) * 100:10.2f}%')

    # ---- 自检段内逐点残差诊断（min_len = 8）
    print('\n\n' + '=' * 92)
    print('自检段内逐点残差诊断（min_len = 8，n 取散点插值）')
    print('=' * 92)
    for name in O.FILES:
        sig, sc, nv, th = prepare(name)
        n_int = np.interp(sig, sc, nv)
        _, _, _, i0, i1, _ = O.enumerate_segments(sig, n_int, th, 8)[0]
        sin2 = np.sin(th) ** 2
        x = 2 * sig * np.sqrt(n_int ** 2 - sin2)
        j = np.arange(i1 - i0)
        xs = x[i0:i1]
        b, a = np.polyfit(j, xs, 1)
        res = xs - (a + b * j)
        n_seg = n_int[i0:i1]
        print(f'\n【{name}】段 {sig[i0]:.0f}~{sig[i1-1]:.0f} cm⁻¹，d = {1e4/(2*b):.3f} μm')
        print(f'    {"序号":>4}{"σ_j/cm⁻¹":>11}{"n_j":>8}{"x_j":>10}{"残差":>9}{"残差/Δx":>10}')
        print('    ' + '-' * 52)
        for k in range(len(j)):
            dxm = (xs[-1] - xs[0]) / (len(j) - 1)
            print(f'    {j[k]:4d}{sig[i0+k]:11.1f}{n_seg[k]:8.3f}{xs[k]:10.1f}'
                  f'{res[k]:9.1f}{res[k]/dxm:9.3f}')
    print()


if __name__ == '__main__':
    main()
