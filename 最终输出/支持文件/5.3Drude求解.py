# -*- coding: utf-8 -*-
"""
问题三：引入衬底 Drude 色散模型的硅外延层厚度求解
================================================================

评议指出两处缺陷：(1) 方案一以常数折射率拟合，低波数端失效后被直接
放弃；(2) 方案二的相邻峰谷公式未计入衬底界面反射相移的色散
dδ12/dσ，造成峰谷法与 FFT 之间约 1.2% 的系统偏差。本脚本一并修复。

物理模型
--------
外延层折射率取文献名义值 n = 3.42（弱吸收，A ≈ 1）。重掺杂衬底的
介电函数用 Drude 模型描述（波数 σ 以 cm⁻¹ 计）：

    ε2(σ) = ε∞ − σp² / (σ² + iγσ),

σp 为等离子体波数，γ 为阻尼波数。衬底与外延层同为硅，其高频
介电常数与外延层一致，取 ε∞ ≤ n² = 11.7；若放任 ε∞ > n²，
r12 会在谱段内过零，δ12 在过零点陡变，修正后的峰谷值在高波数端
发散，故将 ε∞ 约束在该物理上限之内。代入三层膜 Airy 公式（即论文
式(30)），s 偏振反射率为：

    r = (r01 + r12 e^{iφ}) / (1 + r01 r12 e^{iφ}),
    φ = 4πσ d √(n² − sin²θ),   r12 = (β1 − β2)/(β1 + β2),
    β1 = √(n² − sin²θ),   β2 = √(ε2 − sin²θ).

方案一（Drude 全谱拟合）：厚度初值不取峰间距估计（低波数端条纹被
δ12 色散拉宽，该估计有偏），而用 FFT 基频独立锚定；拟合参数为
d、ε∞、σp、γ，拟合区间取全谱（400~4000 cm⁻¹），不再截断低波数端。

方案二（相移修正的相邻峰谷法）：条纹极值条件为

    φ(σ) + δ12(σ) = kπ,   δ12 = arg r12,

相邻峰、谷之间相位差为 π，逐对解出

    d_j = [π − δ12(σ2) + δ12(σ1)] / [4π c (σ2 − σ1)],
    c = √(n² − sin²θ),

δ12(σ) 由方案一拟合的 Drude 参数算出。δ12 为常数时即退化为原式。

相干长度定量校验：采样间隔 Δσ 由数据估算，相干长度按
L_c ≈ 1/Δσ 计算，并与相邻光束光程差 ΔL = 2dc 对比。

运行环境：Python 3 + numpy / scipy / openpyxl
用法：在项目根目录运行  python "5 模型的建立与求解/5.3 问题3/5.3Drude求解.py"
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from scipy.stats import median_abs_deviation
import openpyxl

N0 = 1.0
N_SI = 3.42                 # 外延层折射率文献名义值（红外透明区）
EI_INF = N_SI ** 2          # 衬底高频介电常数上限（同为硅，≈11.7）
PROM = 1.0                  # 峰谷检测显著度门槛（百分点），与 5.2 节相同
D_BOUND = (5e-5, 3e-3)      # 厚度拟合边界（cm），即 0.5~30 μm
S_CUT = 1500.               # 峰谷统计的低波数截断（cm⁻¹）


def load_data(path):
    """读取 xlsx 附件，返回 (波数 cm^-1, 反射率 %)。"""
    wb = openpyxl.load_workbook(path, read_only=True)
    rows = list(wb.active.iter_rows(values_only=True))
    wb.close()
    arr = np.array([[float(r[0]), float(r[1])] for r in rows[1:]
                    if r[0] is not None and r[1] is not None])
    return arr[:, 0], arr[:, 1]


def eps_sub(s, eps_inf, sigma_p, gamma):
    """衬底介电函数的 Drude 模型，σ 以 cm⁻¹ 计。"""
    return eps_inf - sigma_p ** 2 / (s ** 2 + 1j * gamma * s)


def r_airy(s, d, eps_inf, sigma_p, gamma, theta):
    """三层膜 Airy 反射振幅与衬底界面反射系数 (r, r12)。"""
    sin2 = np.sin(theta) ** 2
    beta1 = np.sqrt(N_SI ** 2 - sin2)
    beta2 = np.sqrt(eps_sub(s, eps_inf, sigma_p, gamma) - sin2 + 0j)
    r01 = (np.cos(theta) - beta1) / (np.cos(theta) + beta1)
    r12 = (beta1 - beta2) / (beta1 + beta2)
    e = np.exp(4j * np.pi * s * d * beta1)
    return (r01 + r12 * e) / (1 + r01 * r12 * e), r12


def R_model(s, d, eps_inf, sigma_p, gamma, theta):
    """Drude 多光束反射率（%）。"""
    r, _ = r_airy(s, d, eps_inf, sigma_p, gamma, theta)
    return np.abs(r) ** 2 * 100.


def d0_from_fft(s, R, theta):
    """FFT 基频锚定厚度初值（cm）：截断、均匀化、去趋势、加窗、零填充。"""
    m = (R > 0) & (s >= S_CUT)
    sg = np.linspace(s[m].min(), s[m].max(), m.sum())
    Rg = np.interp(sg, s[m], R[m])
    Rg = Rg - np.polyval(np.polyfit(sg, Rg, 3), sg)
    n = len(Rg)
    sp = np.fft.rfft(Rg * np.hanning(n), n=8 * n)
    f0 = (np.argmax(np.abs(sp[1:])) + 1) / ((sg[-1] - sg[0]) * 8)
    c = np.sqrt(N_SI ** 2 - np.sin(theta) ** 2)
    return f0 / (2 * c)


def fit_drude(s, R, theta):
    """方案一：全谱 Drude 拟合，返回 (参数向量, 残差均方根/百分点)。
    参数顺序：d/cm, ε∞, σp, γ。
    分两步：先以 FFT 厚度为锚，在 (σp, γ, ε∞) 网格上多初值搜索，
    确定衬底色散背景；再全参数精修。"""
    d0 = d0_from_fft(s, R, theta)
    resid = lambda x: R_model(s, *x, theta) - R
    best = None
    for sp in (800., 1500., 2500., 3500., 4000.):
        for ga in (150., 400., 1000.):
            sol = least_squares(lambda y: R_model(s, d0, y[0], y[1], y[2],
                                                  theta) - R,
                                [EI_INF - 0.05, sp, ga],
                                bounds=([5., 200., 20.],
                                        [EI_INF, 4000., 3000.]),
                                x_scale='jac')
            r = np.sqrt(np.mean((R_model(s, d0, *sol.x, theta) - R) ** 2))
            if best is None or r < best[0]:
                best = (r, d0, sol.x)
    x0 = [best[1], *best[2]]
    lo = [D_BOUND[0], 5., 200., 20.]
    hi = [D_BOUND[1], EI_INF, 4000., 3000.]
    sol = least_squares(resid, x0, bounds=(lo, hi), x_scale='jac')
    rms = np.sqrt(np.mean(resid(sol.x) ** 2))
    return sol.x, rms


def delta12(s, x, theta):
    """衬底界面反射相移 δ12(σ)，按波数展开。"""
    sg = np.linspace(s.min(), s.max(), 4001)
    _, r12 = r_airy(sg, x[0], x[1], x[2], x[3], theta)
    return np.interp(s, sg, np.unwrap(np.angle(r12)))


def pv_pairs(path, theta, x):
    """一份附件的相邻峰谷对：返回数组
    [σ1, σ2, d_j未修正/μm, d_j修正/μm]。"""
    s, R = load_data(path)
    m = R > 0
    s, R = s[m], R[m]
    pk, _ = find_peaks(R, prominence=PROM)
    va, _ = find_peaks(-R, prominence=PROM)
    E = sorted([(s[i], 'max') for i in pk] + [(s[i], 'min') for i in va])
    c = np.sqrt(N_SI ** 2 - np.sin(theta) ** 2)
    dl = delta12(s, x, theta)
    rows = []
    for a, b in zip(E[:-1], E[1:]):
        if a[1] == b[1]:
            continue
        s1, s2 = min(a[0], b[0]), max(a[0], b[0])
        if s1 < S_CUT:
            continue
        i1, i2 = np.argmin(np.abs(s - s1)), np.argmin(np.abs(s - s2))
        d_unc = 1e4 / (4. * c * (s2 - s1))
        d_cor = 1e4 * (np.pi - dl[i2] + dl[i1]) / (4. * np.pi * c * (s2 - s1))
        rows.append((s1, s2, d_unc, d_cor))
    return np.array(rows)


def coherence(s, d_um, theta):
    """相干长度定量校验：返回 (采样间隔, L_c/μm, 光程差/μm)。"""
    ds = np.median(np.diff(s))
    Lc = 1.0 / ds * 1e4                     # cm → μm
    c = np.sqrt(N_SI ** 2 - np.sin(theta) ** 2)
    return ds, Lc, 2. * d_um * c


def carrier_density(sigma_p):
    """由等离子体波数估算衬底掺杂浓度（cm⁻³）。
    N = (2πcσp)²ε0m*/e²，电导有效质量取 m* = 0.26 m_e。"""
    eps0, e, me = 8.854e-12, 1.602e-19, 9.109e-31
    wp = 2. * np.pi * 2.998e10 * sigma_p  # cm⁻¹ → rad/s
    return wp ** 2 * eps0 * 0.26 * me / e ** 2 / 1e6


def main():
    files = [('附件 3（Si，10°）', '题目/附件/附件3.xlsx', np.radians(10.)),
             ('附件 4（Si，15°）', '题目/附件/附件4.xlsx', np.radians(15.))]

    print('=' * 76)
    print('方案一（Drude 色散模型全谱拟合，400~4000 cm⁻¹）')
    print('=' * 76)
    fits = {}
    for name, path, theta in files:
        s, R = load_data(path)
        m = R > 0
        x, rms = fit_drude(s[m], R[m], theta)
        fits[name] = (s, x, theta)
        print(f'{name}')
        print(f'  d = {x[0] * 1e4:.3f} μm   ε∞ = {x[1]:.2f}   '
              f'σp = {x[2]:.0f} cm⁻¹   γ = {x[3]:.0f} cm⁻¹   rms = {rms:.2f}%')
        print(f'  屏蔽等离子体边缘 σp/√ε∞ = {x[2] / np.sqrt(x[1]):.0f} cm⁻¹'
              f'   衬底掺杂浓度 N ≈ {carrier_density(x[2]):.2e} cm⁻³')

    print('\n' + '=' * 76)
    print('方案二（相邻峰谷法，含相移色散修正；δ12 取自 Drude 拟合）')
    print('=' * 76)
    for name, path, theta in files:
        _s, x, _t = fits[name]
        P = pv_pairs(path, theta, x)
        med_u, mad_u = np.median(P[:, 2]), median_abs_deviation(P[:, 2])
        med_c, mad_c = np.median(P[:, 3]), median_abs_deviation(P[:, 3])
        print(f'\n【{name}】 峰谷对 {len(P)} 对')
        print(f'  {"σ1":>9} {"σ2":>9} {"d未修正":>8} {"d修正":>8}')
        for s1, s2, du, dc in P:
            print(f'  {s1:9.2f} {s2:9.2f} {du:8.3f} {dc:8.3f}')
        print(f'  未修正：中位数 {med_u:.3f} μm，MAD {mad_u:.3f} μm'
              f'（{mad_u / med_u * 100:.1f}%）')
        print(f'  修正后：中位数 {med_c:.3f} μm，MAD {mad_c:.3f} μm'
              f'（{mad_c / med_c * 100:.1f}%）')

    print('\n' + '=' * 76)
    print('相干长度定量校验（L_c ≈ 1/Δσ，Δσ 为采样间隔）')
    print('=' * 76)
    for name, path, theta in files:
        s, x, _t = fits[name]
        ds, Lc, dL = coherence(s, x[0] * 1e4, theta)
        print(f'{name:<16} Δσ = {ds:.3f} cm⁻¹   L_c ≈ {Lc:.0f} μm'
              f'   ΔL = 2dc ≈ {dL:.1f} μm   ΔL/L_c ≈ {dL / Lc:.1e}')
    print('=' * 76)


if __name__ == '__main__':
    main()
