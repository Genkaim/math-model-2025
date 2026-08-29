# -*- coding: utf-8 -*-
"""
5.1 画图：5.1 节全部插图
========================
  图2  干涉条纹与相邻峰谷相位差示意图（5.1.3，说明峰谷位置携带厚度信息）
  图3  双偏振反射系数模值随入射角的变化（5.1.2，支撑方案二的垂直入射近似）
  图9  包络线法示意图（5.1.2，说明包络线消去光 2 的贡献、留下界面反射振幅 u）

模型（问题一式(5)，垂直入射近似下的示意形式）：
    R(σ) = |r01|^2 + B^2 + 2|r01| B cos φ,   φ = 4πσd·√(n^2 - n0^2 sin^2θ)
相邻波峰与波谷对应 cosφ = +1 与 -1，相位差 Δφ = π，其波数间隔 Δσ = 1/(4d√(n^2-sin^2θ))。

用法：python "5.1画图.py"（输出至 ../../图片/ 目录）
运行环境：Python 3 + numpy / matplotlib
字体：中文 SimSun（宋体），西文与数学符号 Times New Roman
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

# ---------------- 示意参数（取碳化硅的典型值） ----------------
N0 = 1.0           # 空气折射率
N_SIC = 2.40       # 外延层折射率（示意取常数，忽略色散）
D_UM = 7.30        # 外延层厚度 μm
THETA_DEG = 10.0   # 入射角（度）
B_AMP = 0.05       # 光 2 振幅系数（远小于光 1，两光束近似成立）
SIGMA_LO, SIGMA_HI = 1400.0, 2500.0   # 波数范围 cm^-1

OUT_DIR = Path(__file__).resolve().parent.parent.parent / '图片'

# 标注文字的白底衬底：即使压到曲线上也能清晰阅读
WHITE_BOX = dict(boxstyle='round,pad=0.3', facecolor='white',
                 edgecolor='none', alpha=0.85)


def phase_factor():
    """返回 K = n·cosθ₁ = √(n² - n₀²sin²θ)。"""
    th = np.radians(THETA_DEG)
    return np.sqrt(N_SIC ** 2 - (N0 * np.sin(th)) ** 2)


def reflectance(sigma_cm):
    """两光束模型的反射率 R(σ)（返回百分数）。"""
    d_cm = D_UM * 1e-4
    r01 = (N_SIC - N0) / (N_SIC + N0)              # 垂直入射菲涅耳系数
    phi = 4 * np.pi * sigma_cm * d_cm * phase_factor()
    R = r01 ** 2 + B_AMP ** 2 + 2 * r01 * B_AMP * np.cos(phi)
    return R * 100.


def peak_valley_sigma(m=6):
    """第 m 个波峰与其后相邻波谷的波数（cm^-1）。

    峰：φ = 2mπ      →  σ = m/(2dK)
    谷：φ = (2m+1)π  →  σ = (2m+1)/(4dK)
    """
    d_cm, K = D_UM * 1e-4, phase_factor()
    sigma_p = m / (2 * d_cm * K)
    sigma_v = (2 * m + 1) / (4 * d_cm * K)
    return sigma_p, sigma_v


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sg = np.linspace(SIGMA_LO, SIGMA_HI, 2000)
    Rv = reflectance(sg)
    s_p, s_v = peak_valley_sigma()

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(sg, Rv, lw=1.4, color='#1f4e79', zorder=2)

    # 相邻峰、谷标记（用图例说明，不再单独加指示箭头与文字）
    ax.plot([s_p], [reflectance(s_p)], '^', ms=9, color='tab:red',
            label='相邻波峰', zorder=4)
    ax.plot([s_v], [reflectance(s_v)], 'v', ms=9, color='tab:green',
            label='相邻波谷', zorder=4)

    # 峰谷之间的相位差标注：整条标注线抬到曲线最高点之上，留出明显间隙避免与曲线重叠
    rng = Rv.max() - Rv.min()
    y_arrow = Rv.max() + rng * 0.34
    # 标注线两端向下引虚线，指明它对应的是哪两个波数
    ax.vlines([s_p, s_v], [reflectance(s_p), reflectance(s_v)], y_arrow,
              color='0.55', ls=':', lw=1.0, zorder=3)
    ax.annotate('', xy=(s_p, y_arrow), xytext=(s_v, y_arrow),
                arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.2),
                zorder=5)
    ax.text((s_p + s_v) / 2, y_arrow + rng * 0.05,
            '相邻峰谷相位差 Δφ = π', ha='center', va='bottom', fontsize=12,
            bbox=WHITE_BOX)

    ax.text((s_p + s_v) / 2, Rv.min() - rng * 0.20,
            r'$\Delta\sigma = \dfrac{1}{4d\sqrt{n^2-n_0^2\sin^2\theta}}$'
            f' ≈ {s_v - s_p:.0f} ' + r'$\mathrm{cm^{-1}}$',
            ha='center', fontsize=11, bbox=WHITE_BOX)

    ax.set_xlabel('波数 σ / cm⁻¹')
    ax.set_ylabel('反射率 R / %')
    ax.set_xlim(SIGMA_LO, SIGMA_HI)
    ax.set_ylim(Rv.min() - rng * 0.40, Rv.max() + rng * 0.72)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc='upper right')
    fig.tight_layout()

    name = '图2 干涉条纹与相邻峰谷相位差示意图'
    fig.savefig(OUT_DIR / f'{name}.svg', dpi=300)
    fig.savefig(OUT_DIR / f'{name}.png', dpi=200)
    plt.close(fig)
    print(f'已保存 {OUT_DIR / name}.svg')
    print(f'  峰 σ_p = {s_p:.1f} cm^-1，谷 σ_v = {s_v:.1f} cm^-1，'
          f'间隔 Δσ = {s_v - s_p:.1f} cm^-1')


# ---------------------------------------------------------------- 图 3
def plot_polarization():
    """图3：双偏振反射系数模值随入射角的变化（对应问题一式(10)~(14)）。

    取 n≈2.4 画出 |rs(θ)|、|rp(θ)| 与垂直入射值 |r01|；再画出按光强平均后的
    等效反射振幅 √((|rs|²+|rp|²)/2)。可见单独看某种偏振，10°、15° 处的偏差约
    1%~3%；但两种偏振按光强平均后，等效振幅几乎不随角度变化，与垂直入射值重合，
    这正是方案二（垂直入射近似）可以成立的依据。
    """
    th = np.linspace(0.0, 30.0, 601)
    c = np.cos(np.radians(th))
    sin1 = N0 * np.sin(np.radians(th)) / N_SIC
    cos1 = np.sqrt(1.0 - sin1 ** 2)
    rs = np.abs((N0 * c - N_SIC * cos1) / (N0 * c + N_SIC * cos1))
    rp = np.abs((N_SIC * c - N0 * cos1) / (N_SIC * c + N0 * cos1))
    r_unpol = np.sqrt((rs ** 2 + rp ** 2) / 2.0)      # 光强平均的等效反射振幅
    r0 = (N_SIC - N0) / (N_SIC + N0)                  # 垂直入射值

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(th, rs, lw=1.6, color='#1f77b4', label='s 偏振 |rs(θ)|')
    ax.plot(th, rp, lw=1.6, color='#d62728', label='p 偏振 |rp(θ)|')
    ax.plot(th, r_unpol, lw=2.0, ls='-', color='#2ca02c',
            label='光强平均等效振幅 √((rs²+rp²)/2)')
    ax.axhline(r0, color='k', lw=1.2, ls='--',
               label=f'垂直入射 |r01| = {r0:.4f}')
    # 标出本题的两个工作角
    for ang, col in ((10.0, '#1f77b4'), (15.0, '#d62728')):
        ax.axvline(ang, color='0.6', ls=':', lw=1.0)
        ax.text(ang + 0.4, r0 + 0.011, f'{ang:.0f}°', fontsize=11, color='0.35')
    ax.set_xlabel('入射角 θ / °')
    ax.set_ylabel('反射系数模值')
    ax.set_xlim(0, 30)
    ax.legend(fontsize=9, loc='center left')
    ax.grid(alpha=0.3)

    name = '图3 双偏振反射系数模值随入射角的变化'
    fig.tight_layout()
    fig.savefig(OUT_DIR / f'{name}.svg', dpi=300)
    fig.savefig(OUT_DIR / f'{name}.png', dpi=200)
    plt.close(fig)
    print(f'已保存 {OUT_DIR / name}.svg')


# ---------------------------------------------------------------- 图 9
def plot_envelope_demo():
    """图9：包络线法示意图（对应问题一式(5)~(8)）。

    上图：两光束干涉的振荡谱 R(σ) 与上、下包络线 Rmax、Rmin；
    下图：把上、下包络开方，画出 √Rmax、√Rmin 及二者的中线 u=(√Rmax+√Rmin)/2。
          开方之后两束光的振幅变成一次量，相加即得界面反射振幅 |r01|、
          相减即得光 2 的振幅 |B| —— 这正是包络线法能把两个振幅拆开的原因。
    """
    sg = np.linspace(1250.0, 2650.0, 4000)
    n = 2.30 + 2.0e-5 * (sg - 1250.0)                       # 缓慢的正常色散
    K = np.sqrt(n ** 2 - (N0 * np.sin(np.radians(THETA_DEG))) ** 2)
    r01 = (n - N0) / (n + N0)                               # 界面反射振幅
    B = 0.11 * np.exp(-(sg - 1250.0) / 2600.0)              # 光 2 振幅，随波数缓降
    phi = 4.0 * np.pi * sg * (D_UM * 1e-4) * K
    R = (r01 ** 2 + B ** 2 + 2 * r01 * B * np.cos(phi)) * 100
    Rmax = (r01 + B) ** 2 * 100
    Rmin = (r01 - B) ** 2 * 100

    s0 = 1900.0                                             # 取一处作标注
    i0 = int(np.argmin(np.abs(sg - s0)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.6), sharex=True)

    # 上图：反射率与包络线
    ax1.plot(sg, R, lw=0.9, color='#1f4e79', label='两光束干涉谱 R(σ)', zorder=2)
    ax1.plot(sg, Rmax, lw=1.6, ls='--', color='#d62728', label='上包络线 Rmax(σ)')
    ax1.plot(sg, Rmin, lw=1.6, ls='--', color='#2ca02c', label='下包络线 Rmin(σ)')
    ax1.plot([sg[i0]] * 2, [Rmin[i0], Rmax[i0]], color='0.45', ls=':', lw=1.0)
    ax1.plot(sg[i0], Rmax[i0], 'o', ms=7, color='#d62728', zorder=4,
             markeredgecolor='white', markeredgewidth=1.2)
    ax1.plot(sg[i0], Rmin[i0], 'o', ms=7, color='#2ca02c', zorder=4,
             markeredgecolor='white', markeredgewidth=1.2)
    ax1.text(0.985, 0.035, '竖虚线：在同一波数读取一对包络值',
             transform=ax1.transAxes, ha='right', va='bottom',
             fontsize=10, color='0.25', bbox=WHITE_BOX)
    ax1.set_ylabel('反射率 R / %')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(alpha=0.3)

    # 下图：开方后的上下包络与中线 u
    sq_max, sq_min = np.sqrt(Rmax / 100), np.sqrt(Rmin / 100)
    u = (sq_max + sq_min) / 2.0
    ax2.plot(sg, sq_max, lw=1.6, ls='--', color='#d62728', label='√Rmax(σ)')
    ax2.plot(sg, sq_min, lw=1.6, ls='--', color='#2ca02c', label='√Rmin(σ)')
    ax2.plot(sg, u, lw=2.0, color='#1f4e79',
             label='u(σ) = (√Rmax+√Rmin)/2 = |r01|')
    ax2.plot([sg[i0]] * 2, [sq_min[i0], sq_max[i0]], color='0.45', ls=':', lw=1.0)
    ax2.plot(sg[i0], u[i0], 'o', ms=7, color='#1f4e79', zorder=4,
             markeredgecolor='white', markeredgewidth=1.2)
    ax2.set_xlabel('波数 σ / cm⁻¹')
    ax2.set_ylabel('反射振幅')
    ax2.legend(fontsize=9, loc='center right')
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    name = '图9 包络线法（由干涉条纹的幅度提取界面反射信息）'
    fig.savefig(OUT_DIR / f'{name}.svg', dpi=300)
    fig.savefig(OUT_DIR / f'{name}.png', dpi=200)
    plt.close(fig)
    print(f'已保存 {OUT_DIR / name}.svg')
    print(f'  σ={sg[i0]:.0f} cm^-1 处：u={u[i0]:.4f}，'
          f'|B|={(sq_max[i0]-sq_min[i0])/2:.4f}')


if __name__ == '__main__':
    main()                    # 图2（5.1.3）
    plot_polarization()       # 图3（5.1.2）
    plot_envelope_demo()      # 图9（5.1.2）
