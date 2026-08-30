# -*- coding: utf-8 -*-
"""
问题二结果可视化 —— 独立绘图脚本
================================
调用 5.2求解.py 的算法函数，生成论文 5.2 节建议的插图：

  图5  原始反射光谱（含异常首点，5.2.1）
  图6  剔除异常点后的反射光谱及干涉峰谷标记（5.2.2）
  图7  反推折射率 n(σ) 随波数变化（含理论范围筛选，5.2.3）
  图8  全谱模型拟合结果对比（含残差，5.2.4）
  图9  双入射角厚度估计对比（散点 + 中位数 ± MAD，5.2.6）

用法：
  python 5.2画图.py            # 输出到 ../../图片/（复现2/图片/）
  python 5.2画图.py DIR        # 输出到指定目录

运行环境：Python 3 + numpy / scipy / matplotlib
"""
import sys
import importlib.util
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _load_solver():
    """按文件名加载 5.2求解.py（文件名以数字开头，不能用普通 import）。"""
    p = Path(__file__).resolve().parent / '5.2求解.py'
    spec = importlib.util.spec_from_file_location('q2_solver', p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['q2_solver'] = mod
    spec.loader.exec_module(mod)
    return mod


_q = _load_solver()
load_data = _q.load_data
remove_outliers = _q.remove_outliers
detect_extrema = _q.detect_extrema
solve = _q.solve
fit_full_spectrum = _q.fit_full_spectrum
two_beam_model_curve = _q.two_beam_model_curve

# 字体设置：中文 SimSun（宋体），西文/数字 Times New Roman，数学 stix
plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

# 附件与入射角
FILES = {'附件1（入射角10°）': '题目/附件/附件1.xlsx',
         '附件2（入射角15°）': '题目/附件/附件2.xlsx'}
THETA = {'附件1（入射角10°）': 10., '附件2（入射角15°）': 15.}
COLORS = {'附件1（入射角10°）': '#1f77b4', '附件2（入射角15°）': '#d62728'}

BASE_DIR = Path(__file__).resolve().parent.parent.parent    # 复现2 根目录


# ---------------------------------------------------------------- 图 5
def plot_raw_spectrum(raws, out_dir):
    """图5（5.2.1）：原始反射光谱（全谱趋势），黑色圆点标出异常首点位置。"""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for name, (s_raw, R_raw) in raws.items():
        ax.plot(s_raw, R_raw, lw=0.7, color=COLORS[name], label=name, zorder=3)
    # 标注首点零值（R=0.0000%）
    for name, (s_raw, R_raw) in raws.items():
        i0 = np.where(R_raw <= 0)[0]
        if len(i0):
            ax.plot(s_raw[i0[0]], 0, 'ko', ms=6, zorder=5)
    ax.set_xlabel('波数 σ / cm⁻¹')
    ax.set_ylabel('反射率 R / %')
    ax.set_xlim(400, 4000)
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, '图5 原始反射光谱')


# ---------------------------------------------------------------- 图 6
def plot_clean_spectrum(results, out_dir):
    """图6（5.2.2）：聚焦有效区的反射光谱，峰谷标记仅标出
    通过理论范围筛选（规则②）后参与计算的峰谷对。"""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for name, (_s, _R, E, P) in results.items():
        m = (_s >= 1200) & (_s <= 2600)
        ax.plot(_s[m], _R[m], lw=0.7, color=COLORS[name], label=name, zorder=3)
        typ = {sig: t for sig, r, t in E}
        pm = {'max': ('^', 'tab:red'), 'min': ('v', 'tab:green')}
        for sp, sv in P[:, [0, 1]]:
            for s0 in (sp, sv):
                mk, c = pm[typ.get(s0, 'min')]
                ax.plot(s0, np.interp(s0, _s, _R), mk, ms=4.0,
                        color=c, zorder=4)
    ax.set_xlabel('波数 σ / cm⁻¹')
    ax.set_ylabel('反射率 R / %')
    ax.set_xlim(1200, 2600)
    ax.set_ylim(0, 25)
    ax.legend(fontsize=8, ncol=2, loc='lower right')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, '图6 剔除异常点后的反射光谱及干涉峰谷标记')


# ---------------------------------------------------------------- 图 7
def plot_refractive_index(results, out_dir):
    """图7（5.2.3）：通过理论范围筛选后的反推折射率 n(σ)——散点为表 6
    对应数据，实线为线性插值曲线，绿色横带为理论折射率范围 2.20~2.60，
    虚线为文献给出的下限 2.20 与国标名义值 2.55。"""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.axhspan(2.20, 2.60, color='#cde7c0', alpha=0.55,
               label='理论折射率范围 2.20~2.60')
    for name, (_s, _R, _E, P) in results.items():
        o2 = np.argsort(P[:, 2])
        sc, nv = P[o2, 2], P[o2, 3]
        ax.scatter(sc, nv, s=26, color=COLORS[name], edgecolor='k', lw=0.4,
                   label=f'{name}（计算值）', zorder=3)
        sg = np.linspace(sc.min(), sc.max(), 200)
        ax.plot(sg, np.interp(sg, sc, nv), color=COLORS[name], lw=1.3,
                label=f'{name}（推测值）', zorder=4)
    ax.axhline(2.55, color='k', lw=1, ls='--', label='国家标准名义折射率 n=2.55')
    ax.set_xlabel('波数 σ / cm⁻¹')
    ax.set_ylabel('折射率 n')
    ax.set_xlim(1200, 2600)
    ax.set_ylim(2.1, 2.65)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, '图7 反推折射率随波数的变化')


# ---------------------------------------------------------------- 图 8
def plot_full_fit(name, ang, s, R, theta, d_fit, rmse, out_dir):
    """图8（5.2.4）：全谱模型拟合结果对比（附残差）。实测反射率取
    5.2.2 剔除异常点后的数据；d_fit 与 rms 放入图例标题。"""
    sg, Rm, Rg = two_beam_model_curve(s, R, theta, d_fit * 1e-4)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(sg, Rg, lw=1.0, color='0.35', label='实测反射率')
    ax.plot(sg, Rm, lw=1.4, ls='--', color='#d62728', label='平均近似模型拟合（式(5)）')
    ax.set_xlabel('波数 σ / cm⁻¹')
    ax.set_ylabel('反射率 R / %')
    ax.set_xlim(sg.min(), sg.max())
    ax.legend(fontsize=9, title=f'拟合厚度 d={d_fit:.2f} μm，残差 rms={rmse:.2f}%',
              title_fontsize=9)
    ax.grid(alpha=0.3)
    # 残差子图（右侧纵轴，浅色弱化，避免影响主图观感）
    ax2 = ax.twinx()
    ax2.plot(sg, Rg - Rm, lw=0.7, color='#c8c8c8')
    ax2.set_ylabel('残差 R_obs - R_fit / %', color='#a8a8a8')
    ax2.tick_params(axis='y', labelcolor='#a8a8a8')
    ax2.axhline(0, color='#dcdcdc', lw=0.6, ls=':')
    rmax = np.max(np.abs(Rg - Rm)) * 1.3
    ax2.set_ylim(-rmax, rmax)
    fig.tight_layout()
    _save(fig, out_dir, f'图8 全谱拟合结果（{name.split("（")[0]}，{ang:.0f}°）')


# ---------------------------------------------------------------- 图 9
def plot_two_angle(results, out_dir):
    """图9（5.2.6）：双入射角厚度估计对比——散点为各峰谷对 d_j，
    实线为中位数，阴影带为中位数 ± MAD；两组带重合直观展示结果一致。"""
    from scipy.stats import median_abs_deviation
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    rng = np.random.default_rng(0)          # 固定抖动种子，保证可复现
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for i, (name, (_s, _R, _E, P)) in enumerate(results.items()):
        d = P[:, 6]
        med = np.median(d)
        mad = median_abs_deviation(d, scale=1.0)
        x = i + 1
        # 中位数 ± MAD 阴影带
        ax.fill_between([x - 0.28, x + 0.28], med - mad, med + mad,
                        color=COLORS[name], alpha=0.18)
        # 各峰谷对 d_j 散点（水平抖动避免重叠）
        jitter = rng.uniform(-0.09, 0.09, len(d))
        ax.scatter(x + jitter, d, s=32, color=COLORS[name],
                   edgecolor='k', lw=0.4, zorder=3)
        # 中位数横线与数值标注
        ax.hlines(med, x - 0.28, x + 0.28, color=COLORS[name], lw=2.2,
                  zorder=4)
        ax.text(x, med + 0.3, f'{med:.2f}', ha='center',
                fontsize=9, color=COLORS[name])
    ax.set_xticks([1, 2])
    ax.set_xticklabels(list(results.keys()))
    ax.set_ylabel('厚度估计 d_j / μm')
    ax.set_xlim(0.5, 2.5)
    ax.set_ylim(6, 14.5)
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='0.5',
               markeredgecolor='k', markersize=7, label='各峰谷对厚度估计 d_j'),
        Line2D([0], [0], color='0.5', lw=2, label='中位数'),
        Patch(facecolor='0.5', alpha=0.25, label='中位数 ± MAD'),
    ]
    ax.legend(handles=handles, fontsize=8, loc='upper right')
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    _save(fig, out_dir, '图9 双入射角厚度估计对比')


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
    print(f'问题二结果可视化：输出目录 {out_dir}')

    raws = {}        # 原始数据（图5）
    results = {}     # 处理后数据 + 求解结果（图6/7/9）
    for name, path in FILES.items():
        path = str(BASE_DIR / path)
        s_raw, R_raw = load_data(path)
        s, R = remove_outliers(s_raw, R_raw)
        th = np.radians(THETA[name])
        P, E = solve(s, R, th)                 # 方案二：筛选 + 插值
        med = np.median(P[:, 6])
        d_fit, rmse = fit_full_spectrum(s, R, th, med * 1e-4)   # 方案一
        raws[name] = (s_raw, R_raw)
        results[name] = (s, R, E, P)
        plot_full_fit(name, THETA[name], s, R, th, d_fit, rmse, out_dir)  # 图8

    plot_raw_spectrum(raws, out_dir)           # 图5（5.2.1）
    plot_clean_spectrum(results, out_dir)      # 图6（5.2.2）
    plot_refractive_index(results, out_dir)    # 图7（5.2.3）
    plot_two_angle(results, out_dir)           # 图9（5.2.6）
    print('全部插图生成完毕。')


if __name__ == '__main__':
    main()
