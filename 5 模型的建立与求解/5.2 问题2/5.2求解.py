# -*- coding: utf-8 -*-
"""
碳化硅外延层厚度确定 —— 问题二求解算法（与论文 5.2 节一致）
================================================================

数据预处理（5.2.2）执行两条剔除规则：

  ① 删除反射率 R ≤ 0% 的数据点
     反射率不可能取零或负值，故 R≤0 的点不可能是任何波段上的真实读数。
     本批数据中这样的点每份附件恰有一个（σ=399.68 cm⁻¹，R=0.0000%）。

  ② 删除反推折射率超出理论范围 2.20~2.60 的峰谷对
     包络线法反推的 n 必须落在碳化硅红外透明区的理论折射率范围内
     （下界 2.20、上界 2.60，由文献给出，见论文 5.2.2）；
     超出该范围的反推 n 在物理上不可能，是残余射线带及其邻域两光束
     模型前提失效的产物，对应的峰谷对不参与后续求解。

方案二（5.2.3，相邻峰谷法）在通过上述筛选的峰谷对上进行：
  1. 逐对相邻峰谷用包络线法反推折射率（问题一式(13)(19)）
  2. 剔除 n 超出理论范围的峰谷对
  3. 由各对反推值按中心波数线性插值构造 n(σ)，在峰、谷波数处分别取 n(σp)、n(σv)
  4. 代入厚度公式（问题一式(5)）得 d_j，取中位数与 MAD

方案一（5.2.4，全谱拟合）：以方案二所得 d 为初值，对全谱作两光束模型
最小二乘拟合，n(σ)、B(σ) 由包络线法散点线性插值给出，作相互验证。

运行环境：Python 3 + numpy / scipy / openpyxl
用法：在项目根目录运行  python "5 模型的建立与求解/5.2 问题2/5.2求解.py"
"""
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import median_abs_deviation
import openpyxl

# 常数
N0 = 1.0                 # 空气折射率 n0
PROM = 1.0               # 峰谷检测的显著度门槛（百分点）
FULL_RANGE = (400., 4000.)   # 参与峰谷检测的波数范围（剔除异常点后的全谱）
N_RANGE = (2.20, 2.60)   # 折射率的理论范围（文献给出的 SiC 红外折射率，见 5.2.2）
FIT_RANGE = (1400.0, 2500.0)  # 方案一拟合的波数范围（5.2.2 筛选后的有效光谱区）
RESTSTRAHLEN = (787.0, 968.0)    # 碳化硅残余射线带（仅绘图/讨论用）
D_BOUND = (4e-4, 3e-3)   # 厚度拟合边界（cm），即 4~30 μm


# ------------------------------------------------------------ 数据读取与预处理
def load_data(path):
    """读取 xlsx 附件，返回 (波数 cm^-1, 反射率 %)。"""
    wb = openpyxl.load_workbook(path, read_only=True)
    rows = list(wb.active.iter_rows(values_only=True))
    wb.close()
    arr = np.array([[float(r[0]), float(r[1])] for r in rows[1:]
                    if r[0] is not None and r[1] is not None])
    return arr[:, 0], arr[:, 1]


def remove_outliers(s, R):
    """剔除规则：删除反射率 R ≤ 0% 的数据点（论文 5.2.2）。"""
    keep = R > 0
    return s[keep], R[keep]


# ------------------------------------------------------------ 峰谷检测
def detect_extrema(s, R, prom=PROM):
    """检测干涉极值；prom 为显著度（%），用于剔除噪声峰。"""
    pk, _ = find_peaks(R, prominence=prom)
    va, _ = find_peaks(-R, prominence=prom)
    E = sorted([(s[i], R[i], 'max') for i in pk]
               + [(s[i], R[i], 'min') for i in va])
    return E


# ------------------------------------------------------------ 包络线法与厚度
def envelope_pairs(E, lo=FULL_RANGE[0], hi=FULL_RANGE[1]):
    """逐对相邻峰谷用包络线法反推折射率。

    依据问题一式(13)(19)：
        u = (sqrt(Rmax)+sqrt(Rmin))/2 ,  n = n0*(1+u)/(1-u)
    返回数组，列为 [σp, σv, σ中心, n(σ中心)]。
    """
    rows = []
    for a, b in zip(E[:-1], E[1:]):
        if a[2] == b[2]:                      # 必须峰谷相邻
            continue
        s_p, s_v = min(a[0], b[0]), max(a[0], b[0])
        if not (lo < s_p and s_v < hi):       # 该对须完整落在范围内
            continue
        Rmax, Rmin = max(a[1], b[1]), min(a[1], b[1])
        u = (np.sqrt(Rmax / 100.) + np.sqrt(Rmin / 100.)) / 2.   # 式(13)
        if not (0 < u < 1):
            continue
        n = N0 * (1 + u) / (1 - u)            # 式(19)
        rows.append((s_p, s_v, (s_p + s_v) / 2., n))
    return np.array(rows)


def envelope_method(E, lo=FULL_RANGE[0], hi=FULL_RANGE[1]):
    """包络线法（含 |B| 反推），列为 [σ中心, n, r01, B]。"""
    out = []
    for a, b in zip(E[:-1], E[1:]):
        if a[2] == b[2]:
            continue
        sl, sh = min(a[0], b[0]), max(a[0], b[0])
        if not (lo < sl and sh < hi):
            continue
        Rmax, Rmin = max(a[1], b[1]), min(a[1], b[1])
        if Rmin < 0.5 or Rmax > 90:                # 剔除近零谷与饱和峰
            continue
        sr, srm = np.sqrt(Rmax / 100.), np.sqrt(Rmin / 100.)
        r01 = (sr + srm) / 2.                      # 式(13)
        B = (sr - srm) / 2.                        # 式(12)
        if not (0 < r01 < 1):
            continue
        n = N0 * (1 + r01) / (1 - r01)             # 式(19)
        out.append(((sl + sh) / 2., n, r01, B))
    return np.array(out)


def thickness_pair(s_p, s_v, n_p, n_v, theta):
    """问题一式(5)：严格计入折射率色散，分别取 n(σp)、n(σv)。

    d = 1 / (4*|σv*sqrt(n(σv)^2-sin^2θ) - σp*sqrt(n(σp)^2-sin^2θ)|)
    返回厚度，单位 cm。
    """
    c_p = np.sqrt(n_p ** 2 - np.sin(theta) ** 2)
    c_v = np.sqrt(n_v ** 2 - np.sin(theta) ** 2)
    return 1.0 / (4.0 * abs(s_v * c_v - s_p * c_p))


def filter_pairs(P, nrange=N_RANGE):
    """剔除反推折射率超出理论范围的峰谷对（论文 5.2.2 第二条规则）。

    P 列为 [σp, σv, σ中心, n反推]。反推 n 必须落在碳化硅红外透明区的
    理论折射率范围 [2.20, 2.60] 内，超出即判为包络线法前提失效。
    """
    n = P[:, 3]
    keep = (n >= nrange[0]) & (n <= nrange[1])
    return P[keep], np.count_nonzero(~keep)


def solve(s, R, theta, lo=FULL_RANGE[0], hi=FULL_RANGE[1]):
    """方案二完整求解：峰谷检测 → 包络线法 → 理论范围筛选 → 逐对厚度。

    n(σ) 由各对反推值按中心波数线性插值得到，再在 σp、σv 处分别取值。
    返回数组，列为 [σp, σv, σ中心, n反推, n(σp), n(σv), d_j/μm]。
    """
    E = detect_extrema(s, R)
    P = envelope_pairs(E, lo, hi)
    P, _n_drop = filter_pairs(P)
    o = np.argsort(P[:, 2])
    n_p = np.interp(P[:, 0], P[o, 2], P[o, 3])     # n(σp)
    n_v = np.interp(P[:, 1], P[o, 2], P[o, 3])     # n(σv)
    d = thickness_pair(P[:, 0], P[:, 1], n_p, n_v, theta) * 1e4   # cm → μm
    return np.column_stack([P, n_p, n_v, d]), E


# ------------------------------------------------------------ 方案一：有效区全谱拟合
def _fresnel_mod(nE, theta):
    """双偏振菲涅耳反射系数模值（问题一式(15)(16)），返回 (|rs|, |rp|)。

    干涉项统一取负（−2|r|·B·cosφ）：空气—外延层界面的反射系数与衬底
    反射的符号共同给出第二束光相对第一束光的 π 相位，使反射率极大出现在
    cosφ=−1 处（见 5.1 式(5) 的符号讨论）。
    """
    sin2 = np.sin(theta) ** 2
    cos1 = np.sqrt(nE ** 2 - sin2) / nE
    c0 = np.cos(theta)
    n0c, nc1 = N0 * c0, nE * cos1
    nc0, n0c1 = nE * c0, N0 * cos1
    rs = np.abs((n0c - nc1) / (n0c + nc1))
    rp = np.abs((nc0 - n0c1) / (nc0 + n0c1))
    return rs, rp


def envelope_pairs_full(E, lo=FIT_RANGE[0], hi=FIT_RANGE[1]):
    """逐对反推并通过理论范围筛选，返回 [σp, σv, σ中心, n, B]。"""
    rows = []
    for a, b in zip(E[:-1], E[1:]):
        if a[2] == b[2]:
            continue
        sp, sv = min(a[0], b[0]), max(a[0], b[0])
        if not (lo < sp and sv < hi):
            continue
        Rmax, Rmin = max(a[1], b[1]), min(a[1], b[1])
        sr, srm = np.sqrt(Rmax / 100.), np.sqrt(Rmin / 100.)
        u = (sr + srm) / 2.                       # 式(13)
        if not (0 < u < 1):
            continue
        n = N0 * (1 + u) / (1 - u)                # 式(19)
        if not (N_RANGE[0] <= n <= N_RANGE[1]):   # 规则② 筛选
            continue
        B = (sr - srm) / 2.                       # 式(12)
        rows.append((sp, sv, (sp + sv) / 2., n, B))
    return np.array(rows)


def fit_region_params(s, R, lo=FIT_RANGE[0], hi=FIT_RANGE[1]):
    """有效区拟合参数：(σ, R_obs/100, n(σ), B(σ))，由筛选后散点线性插值。"""
    E = detect_extrema(s, R)
    pt = envelope_pairs_full(E, lo, hi)
    pt = pt[np.argsort(pt[:, 2])]
    m = (s >= lo) & (s <= hi)
    sg, Rg = s[m], R[m] / 100.
    nE = np.interp(sg, pt[:, 2], pt[:, 3])
    bE = np.interp(sg, pt[:, 2], pt[:, 4])
    return sg, Rg, nE, bE


def two_beam_model_curve(s, R, theta, d, polarized=True):
    """给定厚度 d(cm)，返回有效区 (σ, R_model/%, R_obs/%)。"""
    sg, Rg, nE, bE = fit_region_params(s, R)
    sin2 = np.sin(theta) ** 2
    phi = 4 * np.pi * d * sg * np.sqrt(nE ** 2 - sin2)
    if polarized:
        rs, rp = _fresnel_mod(nE, theta)
        Rm = 0.5 * (rs ** 2 + bE ** 2 - 2 * rs * bE * np.cos(phi)) \
           + 0.5 * (rp ** 2 + bE ** 2 - 2 * rp * bE * np.cos(phi))
    else:
        r01 = (nE - N0) / (nE + N0)
        Rm = r01 ** 2 + bE ** 2 - 2 * r01 * bE * np.cos(phi)
    return sg, Rm * 100., Rg * 100.


def fit_full_spectrum(s, R, theta, d0, polarized=True):
    """（方案一）有效区最小二乘拟合，返回 (d_fit/μm, rms/%)。"""
    sg, Rg, nE, bE = fit_region_params(s, R)
    sin2 = np.sin(theta) ** 2
    if polarized:
        rs, rp = _fresnel_mod(nE, theta)

        def model(d):
            phi = 4 * np.pi * d * sg * np.sqrt(nE ** 2 - sin2)
            return 0.5 * (rs ** 2 + bE ** 2 - 2 * rs * bE * np.cos(phi)) \
                 + 0.5 * (rp ** 2 + bE ** 2 - 2 * rp * bE * np.cos(phi))
    else:
        r01 = (nE - N0) / (nE + N0)

        def model(d):
            phi = 4 * np.pi * d * sg * np.sqrt(nE ** 2 - sin2)
            return r01 ** 2 + bE ** 2 - 2 * r01 * bE * np.cos(phi)

    from scipy.optimize import least_squares
    resid = lambda d: model(d) - Rg
    sol = least_squares(resid, d0, bounds=D_BOUND)
    rmse = np.sqrt(np.mean(resid(sol.x[0]) ** 2)) * 100
    return sol.x[0] * 1e4, rmse


# ------------------------------------------------------------ 主程序
def main():
    files = {'附件1（入射角10°）': '题目/附件/附件1.xlsx',
             '附件2（入射角15°）': '题目/附件/附件2.xlsx'}
    theta_deg = {'附件1（入射角10°）': 10., '附件2（入射角15°）': 15.}

    print('=' * 74)
    print('问题二计算结果（数据预处理：①删除 R≤0% 的点；②剔除 n 超出理论范围的峰谷对）')
    print('=' * 74)

    summary = {}
    for name, path in files.items():
        s_raw, R_raw = load_data(path)
        s, R = remove_outliers(s_raw, R_raw)
        th = np.radians(theta_deg[name])

        P, E = solve(s, R, th)                  # 方案二：相邻峰谷法
        # 重新统计筛选剔除的对数
        Pp = envelope_pairs(E)
        _, n_drop = filter_pairs(Pp)
        d_j = P[:, 6]
        med = np.median(d_j)
        mad = median_abs_deviation(d_j, scale=1.0)
        d_fit, rmse = fit_full_spectrum(s, R, th, med * 1e-4)   # 方案一：全谱拟合
        summary[name] = (med, mad, d_fit, rmse)

        print(f'\n【{name}】')
        print(f'  数据点 {len(R_raw)} → {len(R)}'
              f'（剔除 σ={s_raw[R_raw <= 0][0]:.2f} cm⁻¹，R={R_raw[R_raw <= 0][0]:.4f}%）')
        print(f'  全谱检出干涉极值 {len(E)} 个'
              f'（峰 {sum(1 for e in E if e[2] == "max")}，'
              f'谷 {sum(1 for e in E if e[2] == "min")}），'
              f'峰谷对 {len(Pp)} 对 → 筛选（n∈[2.20,2.60]）剔除 {n_drop} 对，'
              f'保留 {len(P)} 对')
        print(f'\n  {"σp":>9} {"σv":>9} {"n(σp)":>7} {"n(σv)":>7} {"d_j/μm":>9}')
        print(f'  {"-" * 9} {"-" * 9} {"-" * 7} {"-" * 7} {"-" * 9}')
        for sp, sv, mid, n, np_, nv, d in P:
            print(f'  {sp:9.2f} {sv:9.2f} {np_:7.3f} {nv:7.3f} {d:9.2f}')
        print(f'\n  >> 方案二（相邻峰谷法，中位数）d = {med:.2f} μm')
        print(f'  >> 分散程度   MAD = {mad:.2f} μm（{mad / med * 100:.1f}%）')
        print(f'  >> d_j 范围 {d_j.min():.2f} ~ {d_j.max():.2f} μm')
        print(f'  >> 方案一（全谱拟合）d = {d_fit:.2f} μm，rms = {rmse:.2f}%')

    print('\n' + '=' * 74)
    (d1, m1, f1, _), (d2, m2, f2, _) = list(summary.values())
    rel = abs(d1 - d2) / ((d1 + d2) / 2) * 100
    print(f'双入射角一致性（方案二）：10° 测得 {d1:.2f} μm，15° 测得 {d2:.2f} μm，'
          f'相对差异 {rel:.2f}%')
    print(f'方案一拟合：{f1:.2f} μm / {f2:.2f} μm')
    print(f'最终结论：外延层厚度 d ≈ {np.mean([d1, d2]):.1f} μm'
          f'（单次测量的分散程度约 ±{np.mean([m1, m2]):.1f} μm）')
    print('=' * 74)


if __name__ == '__main__':
    main()
