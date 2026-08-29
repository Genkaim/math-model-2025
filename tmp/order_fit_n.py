# -*- coding: utf-8 -*-
"""
级次回归定厚 —— 折射率由"折射率—中心波数"散点回归拟合给出
============================================================
流程：
  1. 只执行一条剔除规则 R ≤ 0%（5.2.2），全谱 400~4000 cm⁻¹ 检测干涉峰谷
  2. 逐对相邻峰谷用包络线法反演 n，得散点 (σ_c, n)
  3. 用候选色散模型回归拟合 n(σ) 连续曲线
  4. 由该曲线给出每个极值处的 n_j，构造 x_j = 2σ_j·sqrt(n_j² − sin²θ)
  5. x_j 关于极值序号 j 线性，斜率 b = 1/(2d) → d = 1/(2b)·10⁴ μm
"""
import numpy as np
import importlib.util, os

ROOT = r"c:/Users/cxh20/Documents/AAACode/MathModel/优秀论文/复现2"
_spec = importlib.util.spec_from_file_location(
    "sol52", os.path.join(ROOT, "5 模型的建立与求解", "5.2 问题2", "5.2求解.py"))
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)


# ------------------------------------------------------------ 色散模型
def _ls(A, y):
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return c


MODELS = {
    '一次   n = a + bσ': (
        lambda s: np.column_stack([np.ones_like(s), s]),
        lambda s, c: c[0] + c[1] * s),
    '二次   n = a + bσ + cσ²': (
        lambda s: np.column_stack([np.ones_like(s), s, s ** 2]),
        lambda s, c: c[0] + c[1] * s + c[2] * s ** 2),
    'Cauchy  n = A + Bσ²': (
        lambda s: np.column_stack([np.ones_like(s), s ** 2]),
        lambda s, c: c[0] + c[1] * s ** 2),
    'Sellmeier1  n² = A + Bσ²': (
        lambda s: np.column_stack([np.ones_like(s), s ** 2]),
        lambda s, c: np.sqrt(np.clip(c[0] + c[1] * s ** 2, 1e-6, None))),
}


def fit_dispersion(sc, n, model):
    """拟合 n(σ) 曲线。Sellmeier 以 n² 为因变量。"""
    A, f = MODELS[model]
    if model.startswith('Sellmeier'):
        c = _ls(A(sc), n ** 2)
    else:
        c = _ls(A(sc), n)
    resid = f(sc, c) - n
    return c, f, resid


# ------------------------------------------------------------ 级次回归
def longest_run(x):
    """取 Δx > 0 的最长连续段，返回索引切片 (i0, i1)（含端点）。"""
    dx = np.diff(x)
    best = cur = 0
    bi = ci = 0
    for k, v in enumerate(dx):
        if v > 0:
            cur += 1
            if cur > best:
                best, bi = cur, ci
        else:
            cur, ci = 0, k + 1
    return bi, bi + best + 1


def enumerate_segments(sig, n, theta, min_len=8):
    """枚举所有长度≥min_len 的连续极值段，逐段做级次回归。

    返回列表，元素为 (标准误, rms, R², i0, i1, 斜率b)。
    标准误 = rms/√N，用于兼顾"拟合得好"与"用点足够多"。
    """
    sin2 = np.sin(theta) ** 2
    x = 2 * sig * np.sqrt(np.clip(n ** 2 - sin2, 1e-9, None))
    out = []
    for i0 in range(len(x)):
        for i1 in range(i0 + min_len, len(x) + 1):
            j = np.arange(i1 - i0)
            xs = x[i0:i1]
            b, a = np.polyfit(j, xs, 1)
            r = xs - (a + b * j)
            rms = float(np.sqrt(np.mean(r ** 2)))
            ss = np.sum((xs - xs.mean()) ** 2)
            out.append((rms / np.sqrt(i1 - i0), rms, 1 - np.sum(r ** 2) / ss,
                        i0, i1, b))
    out.sort()
    return out


def order_regression(sig, n, theta, auto_segment=True, seg=None):
    """级次回归求厚度。seg 给定则只在指定索引段 [i0,i1) 内回归。"""
    sin2 = np.sin(theta) ** 2
    x = 2 * sig * np.sqrt(np.clip(n ** 2 - sin2, 1e-9, None))
    if seg is not None:
        i0, i1 = seg
    elif auto_segment:
        i0, i1 = longest_run(x)
    else:
        i0, i1 = 0, len(x)
    j = np.arange(i1 - i0)
    xs = x[i0:i1]
    b, a = np.polyfit(j, xs, 1)
    pred = a + b * j
    r = xs - pred
    ss = np.sum((xs - xs.mean()) ** 2)
    return dict(d=1e4 / (2 * b), b=b, r2=1 - np.sum(r ** 2) / ss,
                rms=np.sqrt(np.mean(r ** 2)), npts=len(j),
                lo=sig[i0], hi=sig[i1 - 1], nmax=len(sig))


# ------------------------------------------------------------ 主程序
FILES = {'附件1（10°）': ('题目/附件/附件1.xlsx', 10.),
         '附件2（15°）': ('题目/附件/附件2.xlsx', 15.)}


def main():
    results = {}
    for name, (rel, thdeg) in FILES.items():
        s_raw, R_raw = S.load_data(os.path.join(ROOT, *rel.split('/')))
        s, R = S.remove_outliers(s_raw, R_raw)
        th = np.radians(thdeg)
        E = S.detect_extrema(s, R)
        P = S.envelope_pairs(E)                     # 全谱峰谷对 → (σ_c, n) 散点
        sc, nv = P[:, 2], P[:, 3]
        o = np.argsort(sc)
        sc, nv = sc[o], nv[o]

        print('=' * 78)
        print(f'【{name}】  剔除后 {len(R)} 点，全谱检出 {len(E)} 个极值，{len(P)} 个峰谷对')
        print('=' * 78)
        print('\n(1) 包络线法反演的散点 (中心波数 σ_c, 折射率 n)')
        print(f'    {"σ_c/cm⁻¹":>10} {"n":>8}   |   {"σ_c":>10} {"n":>8}')
        for k in range(0, len(sc), 2):
            row = f'    {sc[k]:10.1f} {nv[k]:8.3f}   |'
            if k + 1 < len(sc):
                row += f'   {sc[k+1]:10.1f} {nv[k+1]:8.3f}'
            print(row)

        print('\n(2) 各色散模型的回归拟合（对全部散点）')
        print(f'    {"模型":<26}{"残差rms":>10}{"R²":>9}'
              f'{"n(400)":>9}{"n(1500)":>9}{"n(4000)":>9}')
        print('    ' + '-' * 72)
        for m in MODELS:
            c, f, res = fit_dispersion(sc, nv, m)
            r2 = 1 - np.sum(res ** 2) / np.sum((nv - nv.mean()) ** 2)
            print(f'    {m:<26}{np.sqrt(np.mean(res**2)):10.4f}{r2:9.4f}'
                  f'{f(400., c):9.3f}{f(1500., c):9.3f}{f(4000., c):9.3f}')

        print('\n(3) 级次线性度自检：枚举连续极值段，按标准误排序（前 5）')
        sig = np.array([e[0] for e in E])
        n_int = np.interp(sig, sc, nv)          # 先用散点插值作初值
        segs = enumerate_segments(sig, n_int, th)
        print(f'    {"d/μm":>8}{"R²":>9}{"rms":>8}{"点数":>6}  波数段/cm⁻¹')
        print('    ' + '-' * 58)
        for se, rms_, r2_, i0, i1, bb in segs[:5]:
            print(f'    {1e4 / (2 * bb):8.3f}{r2_:9.5f}{rms_:8.1f}{i1 - i0:6d}  '
                  f'{sig[i0]:.0f}~{sig[i1-1]:.0f}')
        _, _, _, i0, i1, _ = segs[0]
        lo, hi = sig[i0], sig[i1 - 1]

        m_seg = (sc >= lo) & (sc <= hi)
        sc2, nv2 = sc[m_seg], nv[m_seg]
        print(f'\n(4) 在自检选出的段内（{lo:.0f}~{hi:.0f} cm⁻¹，'
              f'{len(sc2)} 个散点）拟合折射率—中心波数曲线')
        print(f'    {"模型":<26}{"残差rms":>10}{"R²":>9}{"n(1200)":>9}{"n(2500)":>9}')
        print('    ' + '-' * 62)
        curves = {}
        for m in MODELS:
            c, f, res = fit_dispersion(sc2, nv2, m)
            r2f = 1 - np.sum(res ** 2) / np.sum((nv2 - nv2.mean()) ** 2)
            curves[m] = (c, f)
            print(f'    {m:<26}{np.sqrt(np.mean(res**2)):10.4f}{r2f:9.4f}'
                  f'{f(1200., c):9.3f}{f(2500., c):9.3f}')

        print('\n(5) 由色散曲线重赋 n_j 后的级次回归')
        print(f'    {"模型":<26}{"d/μm":>9}{"R²":>9}{"rms":>8}'
              f'{"点数":>6}  波数段/cm⁻¹')
        print('    ' + '-' * 74)
        res_all = {}
        for m, (c, f) in curves.items():
            r = order_regression(sig, f(sig, c), th, seg=(i0, i1))
            res_all[m] = r
            print(f'    {m:<26}{r["d"]:9.3f}{r["r2"]:9.5f}{r["rms"]:8.1f}'
                  f'{r["npts"]:6d}  {r["lo"]:.0f}~{r["hi"]:.0f}')
        r_int = order_regression(sig, n_int, th, seg=(i0, i1))
        res_all['（对照）散点线性插值'] = r_int
        print(f'    {"（对照）散点线性插值":<24}{r_int["d"]:9.3f}{r_int["r2"]:9.5f}'
              f'{r_int["rms"]:8.1f}{r_int["npts"]:6d}  '
              f'{r_int["lo"]:.0f}~{r_int["hi"]:.0f}')
        results[name] = res_all
        print()

    print('=' * 78)
    print('双入射角一致性对比')
    print('=' * 78)
    a, b = list(results.values())
    print(f'    {"模型":<26}{"附件1 10°":>11}{"附件2 15°":>11}{"相对差异":>11}')
    print('    ' + '-' * 60)
    for m in a:
        d1, d2 = a[m]['d'], b[m]['d']
        print(f'    {m:<26}{d1:11.3f}{d2:11.3f}'
              f'{abs(d1 - d2) / ((d1 + d2) / 2) * 100:10.2f}%')


if __name__ == '__main__':
    main()
