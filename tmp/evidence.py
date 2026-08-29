# -*- coding: utf-8 -*-
"""5.2.2 证据提取：首点零点假数据 + 残余射线带的数据依据"""
import numpy as np, openpyxl, os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(path):
    wb = openpyxl.load_workbook(path, read_only=True)
    rows = list(wb.active.iter_rows(values_only=True))
    wb.close()
    arr = np.array([[float(r[0]), float(r[1])] for r in rows[1:]
                    if r[0] is not None and r[1] is not None])
    return arr[:, 0], arr[:, 1]


print('=' * 74)
print('证据 A：首点零点假数据')
print('=' * 74)
for nm, p in [('附件1', '题目/附件/附件1.xlsx'), ('附件2', '题目/附件/附件2.xlsx'),
              ('附件3', '题目/附件/附件3.xlsx'), ('附件4', '题目/附件/附件4.xlsx')]:
    s, R = load(p)
    d = np.diff(s)
    print(f'\n【{nm}】 点数={len(s)}, σ范围 {s[0]:.2f}~{s[-1]:.2f} cm^-1')
    print('  前6点 (σ, R%):', [(round(s[i], 2), round(R[i], 4)) for i in range(6)])
    print(f'  步长: 首步长={d[0]:.4f}, 其余 min={d[1:].min():.4f} max={d[1:].max():.4f} '
          f'mean={d[1:].mean():.6f} std={d[1:].std():.2e}')
    print(f'  R(剔除首点): min={R[1:].min():.4f}  max={R[1:].max():.4f}  mean={R[1:].mean():.3f}')
    print(f'  R==0 点数={int((R == 0).sum())}, 索引={np.where(R == 0)[0][:5]}')
    # 二阶外推：用后 5 点二次拟合外推到 σ0
    c = np.polyfit(s[1:6], R[1:6], 2)
    print(f'  二次外推 R(σ0)={np.polyval(c, s[0]):.3f}%  vs 实测 {R[0]:.3f}%')
    # 理论下限：两光束干涉极小值 (r01-B)^2
    for n_ in (2.3, 2.55):
        r01 = (n_ - 1) / (n_ + 1)
        for B in (0.05, 0.10):
            print(f'    理论干涉极小 R_min=(r01-B)^2: n={n_}, B={B} -> '
                  f'{((r01 - B) ** 2) * 100:.2f}%')

print('\n' + '=' * 74)
print('证据 B：残余射线带（分段统计反射率与条纹可见度）')
print('=' * 74)
from scipy.signal import find_peaks

for nm, p in [('附件1', '题目/附件/附件1.xlsx'), ('附件2', '题目/附件/附件2.xlsx')]:
    s, R = load(p)
    s, R = s[R > 0], R[R > 0]
    print(f'\n【{nm}】')
    bands = [(400, 780), (780, 880), (880, 1030), (1030, 1200), (1200, 1400),
             (1400, 1800), (1800, 2500), (2500, 4000)]
    print('  %-16s %8s %8s %8s %10s %10s' % ('波段/cm^-1', 'R均值%', 'Rmin%', 'Rmax%', '峰谷跨度%', '可见度'))
    for lo, hi in bands:
        m = (s >= lo) & (s <= hi)
        if m.sum() < 20:
            continue
        ss, RR = s[m], R[m]
        pk, _ = find_peaks(RR, prominence=0.3)
        va, _ = find_peaks(-RR, prominence=0.3)
        if len(pk) and len(va):
            span = RR[pk].mean() - RR[va].mean()
            vis = span / (RR[pk].mean() + RR[va].mean())
        else:
            span, vis = 0.0, 0.0
        print('  %-16s %8.2f %8.2f %8.2f %10.3f %10.4f'
              % (f'{lo}-{hi}', RR.mean(), RR.min(), RR.max(), span, vis))

print('\n' + '=' * 74)
print('证据 C：残余射线带内条纹能否被检测（显著度阈值 1%）')
print('=' * 74)
for nm, p in [('附件1', '题目/附件/附件1.xlsx'), ('附件2', '题目/附件/附件2.xlsx')]:
    s, R = load(p)
    s, R = s[R > 0], R[R > 0]
    for lo, hi in [(780, 1030), (1030, 1400), (1400, 2500)]:
        m = (s >= lo) & (s <= hi)
        ss, RR = s[m], R[m]
        pk, _ = find_peaks(RR, prominence=1.0)
        va, _ = find_peaks(-RR, prominence=1.0)
        # 局部噪声水平：一阶差分的稳健标准差
        noise = 1.4826 * np.median(np.abs(np.diff(RR) - np.median(np.diff(RR))))
        print(f'  {nm} {lo}-{hi}: 峰={len(pk)} 谷={len(va)} '
              f'极差={RR.max()-RR.min():.3f}% 局部噪声={noise:.4f}% '
              f'信噪比={(RR.max()-RR.min())/(noise+1e-12):.2f}')

print('\n' + '=' * 74)
print('证据 D：残余射线带中心位置（反射率极大）与 SiC 声子频率')
print('=' * 74)
for nm, p in [('附件1', '题目/附件/附件1.xlsx'), ('附件2', '题目/附件/附件2.xlsx')]:
    s, R = load(p)
    s, R = s[R > 0], R[R > 0]
    m = (s >= 700) & (s <= 1200)
    i = np.argmax(R[m])
    print(f'  {nm}: 700-1200 内反射率极大 {R[m][i]:.2f}% @ σ={s[m][i]:.2f} cm^-1')
    # 半高宽
    half = (R[m].max() + R[m].min()) / 2
    idx = np.where(R[((s >= 700) & (s <= 1200))] >= half)[0]
    print(f'    半高位置: {s[((s >= 700) & (s <= 1200))][idx[0]]:.2f} ~ '
          f'{s[((s >= 700) & (s <= 1200))][idx[-1]]:.2f} cm^-1')
    # 饱和平台：R > 0.9*Rmax 的波数区间
    thr = 0.98 * R[m].max()
    idx2 = np.where(R[m] >= thr)[0]
    print(f'    R ≥ 98%Rmax 的平台: {s[m][idx2][0]:.2f} ~ {s[m][idx2][-1]:.2f} cm^-1 '
          f'（宽度 {s[m][idx2][-1]-s[m][idx2][0]:.1f}）')
