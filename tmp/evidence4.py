# -*- coding: utf-8 -*-
"""5.2.2 核心自洽性检验：全谱包络线法反演 n(σ)，找出模型失效波段"""
import numpy as np, openpyxl, os
from scipy.signal import find_peaks

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
N0 = 1.0


def load(path):
    wb = openpyxl.load_workbook(path, read_only=True)
    rows = list(wb.active.iter_rows(values_only=True))
    wb.close()
    arr = np.array([[float(r[0]), float(r[1])] for r in rows[1:]
                    if r[0] is not None and r[1] is not None])
    return arr[:, 0], arr[:, 1]


for nm, p in [('附件1 (10°)', '题目/附件/附件1.xlsx'), ('附件2 (15°)', '题目/附件/附件2.xlsx')]:
    s, R = load(p)
    s, R = s[R > 0], R[R > 0]
    pk, _ = find_peaks(R, prominence=1.0)
    va, _ = find_peaks(-R, prominence=1.0)
    E = sorted([(s[i], R[i], 'max') for i in pk] + [(s[i], R[i], 'min') for i in va])
    print('=' * 84)
    print(f'【{nm}】全谱相邻峰谷的包络线法反演（R<90% 且 R>0.5% 才保留）')
    print('=' * 84)
    print('  %-16s %9s %9s %8s %8s %9s' % ('峰谷波数区间', 'Rmax/%', 'Rmin/%', 'n', '|B|', '备注'))
    rows = []
    for i in range(len(E) - 1):
        a, b = E[i], E[i + 1]
        if a[2] == b[2]:
            continue
        Rmx, Rmn = max(a[1], b[1]), min(a[1], b[1])
        sl, sh = min(a[0], b[0]), max(a[0], b[0])
        sq_x, sq_n = np.sqrt(Rmx / 100), np.sqrt(Rmn / 100)
        u = (sq_x + sq_n) / 2
        B = (sq_x - sq_n) / 2
        if not (0 < u < 1):
            continue
        n = N0 * (1 + u) / (1 - u)
        tag = ''
        if Rmx > 90:
            tag = '峰值饱和，弃用'
        elif Rmn < 0.5:
            tag = '谷值近零，弃用'
        rows.append(((sl + sh) / 2, sl, sh, Rmx, Rmn, n, B, tag))
        print('  %-16s %9.2f %9.3f %8.3f %8.4f  %s'
              % (f'{sl:.1f}-{sh:.1f}', Rmx, Rmn, n, B, tag))

    # 自洽性：用 1400-2500 的 n 做一次拟合，外推到低波数，比较偏离
    good = np.array([r for r in rows if 1400 < r[0] < 2500 and not r[7]], dtype=object)
    print(f'\n  —— 用 1400~2500 cm^-1 的 {len(good)} 个点拟合 n(σ) 一次式，检验其余波段的偏离 ——')
    if len(good) >= 3:
        sc = np.array([r[0] for r in good], dtype=float)
        nc = np.array([r[5] for r in good], dtype=float)
        c = np.polyfit(sc, nc, 1)
        rms = np.sqrt(np.mean((np.polyval(c, sc) - nc) ** 2))
        print(f'     n(σ) = {c[0]:.6e}·σ + {c[1]:.4f},  拟合残差 rms = {rms:.4f}')
        print('  %-16s %9s %9s %11s' % ('峰谷区间', 'n实测', 'n外推', '偏离'))
        for r in rows:
            if r[7]:
                continue
            n_pred = np.polyval(c, r[0])
            dev = r[5] - n_pred
            flag = '  ← 严重偏离' if abs(dev) > 3 * rms else ''
            print('  %-16s %9.3f %9.3f %+11.4f%s'
                  % (f'{r[1]:.1f}-{r[2]:.1f}', r[5], n_pred, dev, flag))
