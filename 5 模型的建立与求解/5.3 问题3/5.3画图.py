# -*- coding: utf-8 -*-
"""
问题三结果可视化 —— 独立绘图脚本
================================
生成论文 5.3 节的插图：

  图13  两光束与多光束干涉条纹对比示意图（5.3.3）
  图14  方案一（Drude 色散模型）全谱拟合结果对比（拟合曲线与残差，附件 3、4，5.3.5）
  图15  反射谱的傅里叶功率谱（基频与谐波，附件 3、4，5.3.5）

注：硅光谱绘图函数 plot_si_spectra 对应旧图11，已从正文移除，默认不再调用；
    图11 现为问题三求解流程图（Graphviz dot 生成）。
    判别结果绘图函数 plot_discrimination 对应旧图13，已从正文移除，默认不再调用；
    图13 现为两光束与多光束干涉条纹对比。

判别指标（与论文 5.3.2 一致）：
  η —— 相邻峰谷之间位于中线以下的数据点占比：两光束正弦条纹 ≈50%，
       多光束（谷尖深、峰宽平）显著低于 50%；
  ρ —— 相邻出射光束振幅比 ρ=|r10·r12·A|，由包络线法反推的界面反射
       系数 r12 与空气/外延层界面系数 r10 相乘得到（式(28)）。

用法：
  python 5.3画图.py            # 输出到 ../../图片/（复现2/图片/）
  python 5.3画图.py DIR        # 输出到指定目录

运行环境：Python 3 + numpy / scipy / openpyxl / matplotlib
"""
import sys
from pathlib import Path
import numpy as np
import openpyxl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def _load_checker():
    """按文件名加载 5.3检验.py（文件名以数字开头，不能用普通 import）。"""
    import importlib.util
    p = Path(__file__).resolve().parent / '5.3检验.py'
    spec = importlib.util.spec_from_file_location('q3_check', p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['q3_check'] = mod
    spec.loader.exec_module(mod)
    return mod


_ck = _load_checker()


def _load_drude():
    """按文件名加载 5.3Drude求解.py（文件名以数字开头，不能用普通 import）。"""
    import importlib.util
    p = Path(__file__).resolve().parent / '5.3Drude求解.py'
    spec = importlib.util.spec_from_file_location('q3_drude', p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['q3_drude'] = mod
    spec.loader.exec_module(mod)
    return mod


_du = _load_drude()

# 字体设置：中文 SimSun（宋体），西文/数字 Times New Roman，数学 stix
plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path(__file__).resolve().parent.parent.parent    # 复现2 根目录
PROM = 1.0                 # 峰谷检测显著度门槛（百分点），与 5.2 一致

# 附件：名称 -> (路径, 外延层折射率名义值, 残余射线带(剔除后参与统计))
FILES = {
    '附件1（SiC，10°）': ('题目/附件/附件1.xlsx', 2.55, (787., 968.)),
    '附件2（SiC，15°）': ('题目/附件/附件2.xlsx', 2.55, (787., 968.)),
    '附件3（Si，10°）':  ('题目/附件/附件3.xlsx', 3.42, None),
    '附件4（Si，15°）':  ('题目/附件/附件4.xlsx', 3.42, None),
}
SI_COLOR = {'附件3（Si，10°）': '#1f77b4', '附件4（Si，15°）': '#d62728'}
ZOOM = {'附件3（Si，10°）': (1400., 1800.), '附件4（Si，15°）': (1300., 1700.)}


def load(path):
    """读取 xlsx 附件，返回 (波数 cm^-1, 反射率 %)。"""
    wb = openpyxl.load_workbook(path, read_only=True)
    rows = list(wb.active.iter_rows(values_only=True))
    wb.close()
    arr = np.array([[float(r[0]), float(r[1])] for r in rows[1:]
                    if r[0] is not None and r[1] is not None])
    return arr[:, 0], arr[:, 1]


def discriminate(s, R, n1, restrahlen):
    """逐段统计低于中线占比 η 与振幅比 ρ，返回 (η均值, ρ中位数, 段数)。

    η：相邻峰谷之间 R 低于中线 (Rmax+Rmin)/2 的点占比；
    ρ：由包络振幅 B=(√Rmax-√Rmin)/2 反推
        r12 = B/(t01·t10)，ρ = |r10·r12|（吸收衰减 A≈1）。
    """
    m = R > 0
    if restrahlen:
        m &= ~((s > restrahlen[0]) & (s < restrahlen[1]))
    s, R = s[m], R[m]
    pk, _ = find_peaks(R, prominence=PROM)
    va, _ = find_peaks(-R, prominence=PROM)
    E = sorted(list(pk) + list(va))
    n0 = 1.0
    r10 = (n1 - n0) / (n1 + n0)
    t01, t10 = 2 * n0 / (n0 + n1), 2 * n1 / (n1 + n0)
    fracs, rhos = [], []
    for a, b in zip(E[:-1], E[1:]):
        seg = (s > s[a]) & (s < s[b])
        if seg.sum() < 20:
            continue
        mid = (R[a] + R[b]) / 2.
        fracs.append((R[seg] < mid).mean())
        Rmax, Rmin = max(R[a], R[b]), min(R[a], R[b])
        B = (np.sqrt(Rmax / 100.) - np.sqrt(Rmin / 100.)) / 2.
        rhos.append(r10 * B / (t01 * t10))
    return np.mean(fracs), np.median(rhos), len(fracs)


# ------------------------------------------------- 硅光谱（旧图11，已移出正文，备用）
def plot_si_spectra(out_dir):
    """附件3、附件4 的反射光谱全谱，上方各置一个放大子图展示局部
    条纹，直观呈现硅晶圆干涉条纹“谷尖深、峰宽平”的多光束特征。"""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    zin_pos = {'附件3（Si，10°）': [0.14, 0.44, 0.34, 0.50],
               '附件4（Si，15°）': [0.56, 0.44, 0.34, 0.50]}
    for name, (path, _n1, _rb) in FILES.items():
        if name not in SI_COLOR:
            continue
        s, R = load(str(BASE_DIR / path))
        m = R > 0
        ax.plot(s[m], R[m], lw=0.7, color=SI_COLOR[name], label=name, zorder=3)
        # 局部放大子图：标注峰谷位置以展示峰形
        zlo, zhi = ZOOM[name]
        zm = m & (s >= zlo) & (s <= zhi)
        zin = ax.inset_axes(zin_pos[name])
        zin.plot(s[zm], R[zm], lw=0.8, color=SI_COLOR[name], zorder=3)
        pk, _ = find_peaks(R[zm], prominence=PROM)
        va, _ = find_peaks(-R[zm], prominence=PROM)
        zin.plot(s[zm][pk], R[zm][pk], '^', ms=3.2, color='tab:red', zorder=4)
        zin.plot(s[zm][va], R[zm][va], 'v', ms=3.2, color='tab:green', zorder=4)
        # 相邻峰谷对的中线：可见大部分数据点位于中线上方（η<50%）
        E = sorted(list(pk) + list(va))
        for a, b in zip(E[:-1], E[1:]):
            mid = (R[zm][a] + R[zm][b]) / 2.
            zin.hlines(mid, s[zm][a], s[zm][b], color='0.3', lw=0.8,
                       ls='--', zorder=2)
        zin.set_xlim(zlo, zhi)
        zin.set_xticks(np.linspace(zlo, zhi, 3))
        zin.tick_params(labelsize=7)
        zin.grid(alpha=0.3)
        zin.set_title(f'{name.split("（")[0]}局部放大', fontsize=8)
    ax.set_xlabel('波数 σ / cm⁻¹')
    ax.set_ylabel('反射率 R / %')
    ax.set_xlim(400, 4000)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, '硅晶圆片的反射光谱（附件3、附件4，未入文）')


# ---------------------------------------------------------------- 图 13
def plot_model_comparison(out_dir):
    """图13：两光束与多光束干涉条纹对比示意图。取硅的界面参数合成曲线；
    为清晰展示形态差异，ρ 取 0.20（大于硅晶圆实际值 ≈0.05，差异方向相同）。
    左：两条曲线全谱对比；右：放大一个峰谷区间并绘出多光束曲线的中线。"""
    r01, B = -0.548, 0.083          # 硅：|r01|=0.548（n=3.42），B 由 Rmax≈39%、Rmin≈21% 定
    g = 0.20                          # 放大的振幅比，仅用于示意
    # 用波数作横轴：d=3.7 μm、n=3.42、θ=10°，条纹周期与附件 3 相近
    k = 4 * np.pi * 3.7e-4 * np.sqrt(3.42**2 - np.sin(np.radians(10))**2)
    s = np.linspace(1200., 2000., 8001)
    phi = k * s
    R2 = np.abs(r01 + B * np.exp(1j * phi))**2 * 100
    Rm = np.abs(r01 + B * np.exp(1j * phi)
                / (1 - g * np.exp(1j * phi)))**2 * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.0))
    ax1.plot(s, R2, lw=1.1, color='0.45', label='两光束模型（式(5)）')
    ax1.plot(s, Rm, lw=1.1, color='#d62728', label='多光束模型（式(30)）')
    ax1.set_xlabel('波数 σ / cm⁻¹')
    ax1.set_ylabel('反射率 R / %')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # 右：取居中的一个谷—峰区间放大（谷变尖处），绘多光束曲线的中线与标注
    pk, _ = find_peaks(Rm, prominence=0.5)
    va, _ = find_peaks(-Rm, prominence=0.5)
    pairs = [(i, pk[pk > i][0]) for i in va if (pk > i).any()]
    a, b = pairs[len(pairs) // 2]
    ax2.plot(s[a:b], Rm[a:b], lw=1.1, color='#d62728', label='多光束模型')
    ax2.plot(s[a:b], R2[a:b], lw=1.1, ls='--', color='0.45', label='两光束模型')
    mid = (Rm[a] + Rm[b]) / 2
    ax2.hlines(mid, s[a], s[b], color='0.3', lw=0.8, ls=':', label='中线')
    ax2.set_xlim(s[a], s[b])
    ax2.set_xlabel('波数 σ / cm⁻¹')
    ax2.set_ylabel('反射率 R / %')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, '图13 两光束与多光束干涉条纹对比')


# ---------------------------------------------------------------- 旧图13（已移出正文，备用）
def plot_discrimination(stats, out_dir):
    """多光束干涉定量判别（旧图13，已移出正文）——左图为低于中线占比 η（两光束≈50%），
    右图为振幅比 ρ（ρ≤0.03 两光束、ρ≥0.05 多光束显著）。"""
    names = list(stats.keys())
    eta = np.array([stats[k][0] * 100 for k in names])
    rho = np.array([stats[k][1] for k in names])
    cols = ['#7f7f7f', '#7f7f7f', '#1f77b4', '#d62728']
    x = np.arange(len(names))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.0))
    ax = axes[0]
    bars = ax.bar(x, eta, width=0.55, color=cols, edgecolor='k', lw=0.5, zorder=3)
    ax.axhline(50, color='k', lw=1.0, ls='--', label='两光束干涉参考线 50%')
    ax.axhspan(48, 52, color='#cde7c0', alpha=0.5, label='两光束容差 ±2%')
    for b, v in zip(bars, eta):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.7, f'{v:.1f}%',
                ha='center', fontsize=8)
    ax.set_ylabel('低于中线占比 η / %')
    ax.set_ylim(0, 60)

    ax = axes[1]
    bars = ax.bar(x, rho, width=0.55, color=cols, edgecolor='k', lw=0.5, zorder=3)
    ax.axhline(0.03, color='#2ca02c', lw=1.0, ls='--', label='两光束上限 ρ=0.03')
    ax.axhline(0.05, color='#d62728', lw=1.0, ls='--', label='多光束显著 ρ=0.05')
    for b, v in zip(bars, rho):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.001, f'{v:.3f}',
                ha='center', fontsize=8)
    ax.set_ylabel('振幅比 ρ')
    ax.set_ylim(0, 0.07)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=7.5, rotation=15)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    _save(fig, out_dir, '多光束干涉的定量判别结果（未入文）')


# ---------------------------------------------------------------- 图 14
def plot_multibeam_fit(name, ang, path, theta, out_dir):
    """图14（5.3.5）：方案一（Drude 色散模型）全谱拟合结果对比。
    实测曲线取全谱（预处理后），拟合曲线由衬底 Drude 介电函数代入
    多光束反射率公式拟合所得参数画出，残差绘于右轴。"""
    s, R = _du.load_data(str(BASE_DIR / path))
    m = R > 0
    s, R = s[m], R[m]
    x, rms = _du.fit_drude(s, R, theta)
    Rm = _du.R_model(s, *x, theta)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(s, R, lw=0.9, color='0.35', label='实测反射率')
    ax.plot(s, Rm, lw=1.3, ls='--', color='#d62728',
            label='Drude 色散模型拟合')
    ax.set_xlabel('波数 σ / cm⁻¹')
    ax.set_ylabel('反射率 R / %')
    ax.set_xlim(s.min(), s.max())
    ax.legend(fontsize=9,
              title=(f'd={x[0] * 1e4:.2f} μm，σ_p={x[2]:.0f} cm⁻¹，'
                     f'γ={x[3]:.0f} cm⁻¹，残差 rms={rms:.2f}%'),
              title_fontsize=9)
    ax.grid(alpha=0.3)
    # 残差子图（右侧纵轴，浅色弱化，与图 8 同风格）
    ax2 = ax.twinx()
    ax2.plot(s, R - Rm, lw=0.7, color='#c8c8c8')
    ax2.set_ylabel('残差 R_obs - R_fit / %', color='#a8a8a8')
    ax2.tick_params(axis='y', labelcolor='#a8a8a8')
    ax2.axhline(0, color='#dcdcdc', lw=0.6, ls=':')
    rmax = np.max(np.abs(R - Rm)) * 1.3
    ax2.set_ylim(-rmax, rmax)
    fig.tight_layout()
    _save(fig, out_dir, f'图14 方案一拟合结果（{name.split("（")[0]}，{ang:.0f}°）')


# ---------------------------------------------------------------- 图 15
def plot_fft_spectrum(name, ang, path, theta, out_dir):
    """图15（5.3.5）：截断区间内反射谱的傅里叶功率谱。
    标出基频与前两次谐波，多光束干涉的谐波结构清晰可见。"""
    s, R = _ck.load_data(str(BASE_DIR / path))
    f, power, f0 = _ck.fft_spectrum(s, R)
    d = _ck.fft_thickness(s, R, theta)[0]
    m = f <= 4 * f0                               # 只画到四次基频以内
    f, power = f[m] * 1e3, power[m]             # 频率刻度 ×10³ 周期/cm⁻¹
    power = power / power.max()

    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(f, power, lw=1.1, color='#1f77b4')
    ax.fill_between(f, power, alpha=0.12, color='#1f77b4')
    for k, lab in ((1, '基频 f₀'), (2, '二次谐波'), (3, '三次谐波')):
        ax.axvline(k * f0 * 1e3, color='#d62728', lw=0.9, ls='--')
        ax.annotate(lab, (k * f0 * 1e3, 0.98), xytext=(k * f0 * 1e3 + 0.12, 0.93),
                    fontsize=9, color='#d62728')
    ax.set_xlabel('频率 f /（10⁻³ 周期·cm⁻¹）')
    ax.set_ylabel('归一化功率')
    ax.set_xlim(0, 4 * f0 * 1e3)
    ax.set_ylim(0, 1.08)
    ax.annotate(f'基频 f₀={f0 * 1e3:.2f}，对应厚度 d={d:.2f} μm',
                xy=(0.03, 0.86), xycoords='axes fraction',
                ha='left', fontsize=9.5,
                bbox=dict(fc='white', ec='#c8c8c8', lw=0.6, alpha=0.9))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir,
          f'图15 反射谱的傅里叶功率谱（{name.split("（")[0]}，{ang:.0f}°）')


# ---------------------------------------------------------------- 工具
def _save(fig, out_dir, name):
    """同时保存 svg（论文嵌入）与 png（快速预览）。"""
    fig.savefig(out_dir / f'{name}.svg', dpi=300)
    fig.savefig(out_dir / f'{name}.png', dpi=200)
    plt.close(fig)
    print(f'  已保存 {out_dir / name}.svg')


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE_DIR / '图片'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'问题三结果可视化：输出目录 {out_dir}')

    stats = {}
    print(f"{'附件':<18}{'峰谷段数':>8}{'低于中线占比':>12}{'振幅比ρ':>10}")
    for name, (path, n1, rb) in FILES.items():
        s, R = load(str(BASE_DIR / path))
        eta, rho, nseg = discriminate(s, R, n1, rb)
        stats[name] = (eta, rho, nseg)
        print(f'{name:<20}{nseg:>8}{eta*100:>11.1f}%{rho:>10.3f}')

    # 旧图11（硅光谱）已移出正文，默认不生成；如需重新插入请启用并按出现位置重新编号
    # plot_si_spectra(out_dir)
    plot_model_comparison(out_dir)    # 图13（合成示意图，不依赖实测数据）
    # 旧图13（判别结果）已移出正文，默认不生成；如需重新插入请启用并重新编号
    # plot_discrimination(stats, out_dir)
    si_files = {'附件3（Si，10°）': ('题目/附件/附件3.xlsx', 10.),
                '附件4（Si，15°）': ('题目/附件/附件4.xlsx', 15.)}
    for name, (path, ang) in si_files.items():
        plot_multibeam_fit(name, ang, path, np.radians(ang), out_dir)  # 图14
        plot_fft_spectrum(name, ang, path, np.radians(ang), out_dir)  # 图15
    print('全部插图生成完毕。')


if __name__ == '__main__':
    main()
