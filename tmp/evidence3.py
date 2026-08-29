# -*- coding: utf-8 -*-
"""5.2.2 证据提取 III：全谱峰谷统计 + Lorentz 振子模型（修正符号）"""
import numpy as np, openpyxl, os
from scipy.signal import find_peaks

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(path):
    wb = openpyxl.load_workbook(path, read_only=True)
    rows = list(wb.active.iter_rows(values_only=True))
    wb.close()
    arr = np.array([[float(r[0]), float(r[1])] for r in rows[1:]
                    if r[0] is not None and r[1] is not None])
    return arr[:, 0], arr[:, 1]


print('=' * 80)
print('证据 H：全谱检测峰谷（显著度 1%），统计各波段的条纹幅度与个数')
print('=' * 80)
for nm, p in [('附件1', '题目/附件/附件1.xlsx'), ('附件2', '题目/附件/附件2.xlsx')]:
    s, R = load(p)
    s, R = s[R > 0], R[R > 0]          # 剔除首点
    pk, _ = find_peaks(R, prominence=1.0)
    va, _ = find_peaks(-R, prominence=1.0)
    E = sorted([(s[i], R[i], 'max') for i in pk] + [(s[i], R[i], 'min') for i in va])
    print(f'\n【{nm}】全谱极值 {len(E)} 个（峰 {len(pk)}、谷 {len(va)}）')
    print('  %-14s %6s %12s %12s' % ('波段/cm^-1', '极值数', '相邻峰谷幅度均值%', '可见度'))
    for lo, hi in [(400, 780), (780, 880), (880, 1030), (1030, 1200), (1200, 1400),
                   (1400, 1800), (1800, 2500), (2500, 4000)]:
        amps, vis = [], []
        sub = [e for e in E if lo <= e[0] <= hi]
        for i in range(len(sub) - 1):
            if sub[i][2] != sub[i + 1][2]:
                a = abs(sub[i + 1][1] - sub[i][1])
                amps.append(a)
                vis.append(a / (sub[i][1] + sub[i + 1][1]))
        print('  %-14s %6d %12s %12s'
              % (f'{lo}-{hi}', len(sub),
                 ('%.3f' % np.mean(amps)) if amps else '—',
                 ('%.4f' % np.mean(vis)) if vis else '—'))
    # 相邻极值波数间隔（条纹周期）
    sg = np.array([e[0] for e in E])
    gaps = np.diff(sg)
    print('  相邻极值波数间隔: 全谱 min=%.1f max=%.1f' % (gaps.min(), gaps.max()))
    for lo, hi in [(400, 780), (1030, 1400), (1400, 2500), (2500, 4000)]:
        g = [gaps[i] for i in range(len(gaps)) if lo <= sg[i] <= hi]
        if g:
            print(f'    {lo}-{hi}: 间隔 {min(g):.1f}~{max(g):.1f} cm^-1, 均值 {np.mean(g):.1f}')

print('\n' + '=' * 80)
print('证据 I：Lorentz 振子模型（修正符号）—— 残余射线带内光穿透深度')
print('=' * 80)
print('  ε(ω) = ε∞ · (ω_LO²−ω²−iγω)/(ω_TO²−ω²−iγω)')
EPS_INF, WTO, WLO, GAMMA = 6.5, 783.0, 964.0, 4.0
D_UM = 7.3
print('  %6s %10s %8s %8s %9s %12s %14s'
      % ('σ/cm^-1', 'ε', 'n', 'κ', 'R0/%', '穿透深度/μm', '往返衰减'))
for sgm in (700, 780, 800, 830, 880, 920, 964, 1000, 1036, 1100, 1200, 1400, 2000):
    num = WLO ** 2 - sgm ** 2 - 1j * GAMMA * sgm
    den = WTO ** 2 - sgm ** 2 - 1j * GAMMA * sgm
    eps = EPS_INF * num / den
    N = np.sqrt(eps)
    n, k = N.real, abs(N.imag)
    R0 = abs((1 - N) / (1 + N)) ** 2 * 100
    delta_um = 1.0 / (2 * np.pi * sgm * k) * 1e4 if k > 1e-6 else float('inf')
    att = np.exp(-2 * D_UM / delta_um) if np.isfinite(delta_um) else 1.0
    print('  %6d %10s %8.3f %8.3f %9.2f %12s %14s'
          % (sgm, f'{eps.real:.2f}{eps.imag:+.2f}j', n, k, R0,
             ('%.4f' % delta_um) if np.isfinite(delta_um) else '∞',
             ('%.2e' % att) if att > 0 else '0'))

print('\n' + '=' * 80)
print('证据 J：由数据反推 TO/LO，与文献值对照')
print('=' * 80)
for nm, p in [('附件1', '题目/附件/附件1.xlsx'), ('附件2', '题目/附件/附件2.xlsx')]:
    s, R = load(p)
    s, R = s[R > 0], R[R > 0]
    m = (s >= 700) & (s <= 1300)
    ss, RR = s[m], R[m]
    i_pk = int(np.argmax(RR))
    Rmx = RR[i_pk]
    base = np.median(RR[:30])
    half = base + 0.5 * (Rmx - base)
    lo_side = ss[:i_pk][int(np.argmin(np.abs(RR[:i_pk] - half)))]
    hi_side = ss[i_pk:][int(np.argmin(np.abs(RR[i_pk:] - half)))]
    m2 = (s >= 1000) & (s <= 1300)
    j = int(np.argmin(R[m2]))
    print(f'  {nm}: 峰值 {Rmx:.2f}% @ {ss[i_pk]:.1f}; '
          f'半高低频边 {lo_side:.1f} | 高频边 {hi_side:.1f} cm^-1; '
          f'高频侧极小 {R[m2].min():.3f}% @ {s[m2][j]:.1f}')
print('  4H-SiC 文献值: ω_TO ≈ 783 cm^-1, ω_LO ≈ 964 cm^-1（E1/A1 模，多文献一致）')
