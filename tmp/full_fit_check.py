# -*- coding: utf-8 -*-
"""验证：方案一在"全谱（剔除异常点后）"上的最小二乘拟合。
n(σ)、B(σ) 由全谱包络线法反演散点线性插值得到，唯一待定参数 d。
"""
import numpy as np
import importlib.util, os

ROOT = r"c:/Users/cxh20/Documents/AAACode/MathModel/优秀论文/复现2"
_spec = importlib.util.spec_from_file_location(
    "sol52", os.path.join(ROOT, "5 模型的建立与求解", "5.2 问题2", "5.2求解.py"))
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

FILES = {'附件1（10°）': ('题目/附件/附件1.xlsx', 10.),
         '附件2（15°）': ('题目/附件/附件2.xlsx', 15.)}


def full_npts(E, lo=400., hi=4000.):
    """全谱逐对反演散点，列为 [σ中心, n, B]。"""
    out = []
    for a, b in zip(E[:-1], E[1:]):
        if a[2] == b[2]:
            continue
        sp, sv = min(a[0], b[0]), max(a[0], b[0])
        if not (lo < sp and sv < hi):
            continue
        Rmax, Rmin = max(a[1], b[1]), min(a[1], b[1])
        sr, srm = np.sqrt(Rmax / 100.), np.sqrt(Rmin / 100.)
        u = (sr + srm) / 2.
        if not (0 < u < 1):
            continue
        n = S.N0 * (1 + u) / (1 - u)
        B = (sr - srm) / 2.
        out.append(((sp + sv) / 2., n, B))
    return np.array(out)


def full_fit(s, R, theta, d0, lo=400., hi=4000.):
    """全谱拟合，返回 (d_fit/μm, rms/%, (sg, Rm, Rg))。"""
    from scipy.optimize import least_squares
    E = S.detect_extrema(s, R)
    pt = full_npts(E, lo, hi)
    o = np.argsort(pt[:, 0])
    sc, nv, bv = pt[o, 0], pt[o, 1], pt[o, 2]
    sg, Rg = s, R / 100.
    nE = np.interp(sg, sc, nv)
    bE = np.interp(sg, sc, bv)
    sin2 = np.sin(theta) ** 2
    rs, rp = S._fresnel_mod(nE, theta)

    def model(d):
        phi = 4 * np.pi * d * sg * np.sqrt(nE ** 2 - sin2)
        return 0.5 * (rs ** 2 + bE ** 2 + 2 * rs * bE * np.cos(phi)) \
             + 0.5 * (rp ** 2 + bE ** 2 + 2 * rp * bE * np.cos(phi))

    sol = least_squares(lambda d: model(d) - Rg, d0, bounds=S.D_BOUND)
    rmse = np.sqrt(np.mean((model(sol.x[0]) - Rg) ** 2)) * 100
    return sol.x[0] * 1e4, rmse, (sg, model(sol.x[0]) * 100., Rg * 100.)


def main():
    for name, (rel, thdeg) in FILES.items():
        s_raw, R_raw = S.load_data(os.path.join(ROOT, *rel.split('/')))
        s, R = S.remove_outliers(s_raw, R_raw)
        th = np.radians(thdeg)
        P2, _ = S.solve(s, R, th)
        med = np.median(P2[:, 6])
        print(f'【{name}】方案二（插值）中位数 d = {med:.3f} μm')
        for d0 in (med * 1e-4, 6e-4, 7e-4, 8e-4):
            d_fit, rmse, _ = full_fit(s, R, th, d0)
            print(f'  全谱拟合 初值 d0={d0*1e4:.1f} μm → d_fit={d_fit:.3f} μm, '
                  f'rms={rmse:.2f}%')
        print()


if __name__ == '__main__':
    main()
