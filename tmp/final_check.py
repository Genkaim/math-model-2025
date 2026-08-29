# -*- coding: utf-8 -*-
"""5.2 重写：核对全部表格数字"""
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


D = {}
for nm, p in [('附件1', '题目/附件/附件1.xlsx'), ('附件2', '题目/附件/附件2.xlsx')]:
    s, R = load(p)
    D[nm] = (s[R > 0], R[R > 0])

print('== 表2 各波段反射率统计 ==')
BANDS = [(400, 780), (780, 880), (880, 1030), (1030, 1200), (1200, 1400),
         (1400, 2500), (2500, 4000)]
for lo, hi in BANDS:
    line = f'{lo}-{hi:5d} '
    for nm in ('附件1', '附件2'):
        s, R = D[nm]
        m = (s >= lo) & (s <= hi)
        line += f'| {R[m].mean():6.2f} {R[m].min():6.2f} {R[m].max():6.2f} '
    print(line)

print('\n== 表5 各波段峰谷检测统计（显著度 1%）==')
for nm in ('附件1', '附件2'):
    s, R = D[nm]
    pk, _ = find_peaks(R, prominence=1.0)
    va, _ = find_peaks(-R, prominence=1.0)
    E = sorted([(s[i], R[i], 'max') for i in pk] + [(s[i], R[i], 'min') for i in va])
    print(f'{nm}: 全谱 {len(E)} 个（峰 {len(pk)}、谷 {len(va)}）')
    print('  %-13s %5s %10s %9s' % ('波段', '极值数', '峰谷幅度均值%', '可见度'))
    for lo, hi in BANDS:
        sub = [e for e in E if lo <= e[0] <= hi]
        amps, vis = [], []
        for i in range(len(sub) - 1):
            if sub[i][2] != sub[i + 1][2]:
                a = abs(sub[i + 1][1] - sub[i][1])
                amps.append(a)
                vis.append(a / (sub[i][1] + sub[i + 1][1]))
        print('  %-13s %5d %10s %9s' % (f'{lo}-{hi}', len(sub),
              ('%.2f' % np.mean(amps)) if amps else '—',
              ('%.3f' % np.mean(vis)) if vis else '—'))
    g = np.diff([e[0] for e in E])
    print(f'  相邻极值间隔 {g.min():.1f}~{g.max():.1f} cm^-1')

print('\n== 表6 各波段折射率反演与外推偏离 ==')
for nm in ('附件1', '附件2'):
    s, R = D[nm]
    pk, _ = find_peaks(R, prominence=1.0)
    va, _ = find_peaks(-R, prominence=1.0)
    E = sorted([(s[i], R[i], 'max') for i in pk] + [(s[i], R[i], 'min') for i in va])
    rows = []
    for i in range(len(E) - 1):
        a, b = E[i], E[i + 1]
        if a[2] == b[2]:
            continue
        Rmx, Rmn = max(a[1], b[1]), min(a[1], b[1])
        sl, sh = min(a[0], b[0]), max(a[0], b[0])
        if Rmx > 90:
            continue
        u = (np.sqrt(Rmx / 100) + np.sqrt(Rmn / 100)) / 2
        if not 0 < u < 1:
            continue
        rows.append(((sl + sh) / 2, sl, sh, (1 + u) / (1 - u)))
    good = [r for r in rows if 1400 < r[0] < 2500]
    sc = np.array([r[0] for r in good])
    nc = np.array([r[3] for r in good])
    c = np.polyfit(sc, nc, 1)
    rms = float(np.sqrt(np.mean((np.polyval(c, sc) - nc) ** 2)))
    print(f'{nm}: n = {c[0]:.4e}·σ + {c[1]:.4f}, rms={rms:.4f}, '
          f'3rms={3*rms:.4f}, 可靠区对数={len(good)}')
    for lo, hi, lab in [(400, 780, '<780'), (780, 980, '带内'),
                        (980, 1220, '980~1220'), (1220, 1400, '1220~1400'),
                        (1400, 2500, '1400~2500')]:
        sel = [r for r in rows if lo <= r[0] <= hi]
        if not sel:
            print(f'  {lab:12s} 无峰谷对')
            continue
        ns = np.array([r[3] for r in sel])
        dev = ns - np.polyval(c, np.array([r[0] for r in sel]))
        print(f'  {lab:12s} n={ns.min():.2f}~{ns.max():.2f}  '
              f'偏离={dev.min():+.3f}~{dev.max():+.3f}  最大|偏离|/rms={np.abs(dev).max()/rms:.1f}')

print('\n== 表7 可靠区反演折射率（中心波数, n）==')
for nm in ('附件1', '附件2'):
    s, R = D[nm]
    pk, _ = find_peaks(R, prominence=1.0)
    va, _ = find_peaks(-R, prominence=1.0)
    E = sorted([(s[i], R[i], 'max') for i in pk] + [(s[i], R[i], 'min') for i in va])
    out = []
    for i in range(len(E) - 1):
        a, b = E[i], E[i + 1]
        if a[2] == b[2]:
            continue
        Rmx, Rmn = max(a[1], b[1]), min(a[1], b[1])
        sl, sh = min(a[0], b[0]), max(a[0], b[0])
        if not (1400 < sl and sh < 2500) or Rmx > 90:
            continue
        u = (np.sqrt(Rmx / 100) + np.sqrt(Rmn / 100)) / 2
        out.append(((sl + sh) / 2, (1 + u) / (1 - u)))
    print(nm + ': ' + '  '.join(f'({a:.0f}, {b:.3f})' for a, b in out))

print('\n== 表8 厚度 d_j（含/不含边界对）==')
for nm, th in [('附件1', 10.), ('附件2', 15.)]:
    s, R = D[nm]
    pk, _ = find_peaks(R, prominence=1.0)
    va, _ = find_peaks(-R, prominence=1.0)
    E = sorted([(s[i], R[i], 'max') for i in pk] + [(s[i], R[i], 'min') for i in va])
    rows = []
    for i in range(len(E) - 1):
        a, b = E[i], E[i + 1]
        if a[2] == b[2]:
            continue
        Rmx, Rmn = max(a[1], b[1]), min(a[1], b[1])
        sl, sh = min(a[0], b[0]), max(a[0], b[0])
        if not (1400 < sl and sh < 2500) or Rmx > 90:
            continue
        u = (np.sqrt(Rmx / 100) + np.sqrt(Rmn / 100)) / 2
        rows.append(((sl + sh) / 2, sl, sh, (1 + u) / (1 - u)))
    sc = np.array([r[0] for r in rows]); nc = np.array([r[3] for r in rows])
    c = np.polyfit(sc, nc, 1)
    st = np.sin(np.radians(th)) ** 2
    ds = []
    for mid, sl, sh, _n in rows:
        nv = np.polyval(c, sh); npp = np.polyval(c, sl)
        d = 1 / (4 * abs(sh * np.sqrt(nv ** 2 - st) - sl * np.sqrt(npp ** 2 - st))) * 1e4
        ds.append((mid, sl, sh, npp, nv, d))
    print(nm)
    for mid, sl, sh, npp, nv, d in ds:
        print(f'  σp={sl:7.2f} σv={sh:7.2f} n(σp)={npp:.3f} n(σv)={nv:.3f} '
              f'd={d:6.2f}  {"剔除" if mid <= 1500 else "保留"}')
    keep = [x[5] for x in ds if x[0] > 1500]
    allv = [x[5] for x in ds]
    print(f'  中位数(保留)={np.median(keep):.2f}  MAD={np.median([abs(v-np.median(keep)) for v in keep]):.2f}')
    print(f'  中位数(全部)={np.median(allv):.2f}')
