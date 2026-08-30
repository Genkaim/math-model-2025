# -*- coding: utf-8 -*-
"""
问题三：多光束干涉的定量判别（与论文 5.3.3、5.3.4 节一致）
================================================================

按 5.3.3 给出的两个互补判据对四份附件逐一统计：

  判据一（振幅比，式(33)）：
      第二束光振幅 B = (sqrt(Rmax) - sqrt(Rmin))/2（问题一式(6)），
      与式(28)对比反推衬底界面反射系数，
          r12_hat = B / (t01*t10*A)，  取 A≈1，
      振幅比估计值  rho_hat = |r10| * r12_hat，
      其中 r10、t01、t10 按垂直入射菲涅耳公式由外延层折射率名义值算出。

  判据二（条纹形态，式(34)）：
      第 j 对相邻峰、谷之间的中线取
          R_mid = (Rmax + Rmin)/2，
      统计区间内位于中线以下的数据点占比 eta_j，
      对全部峰谷对取平均得 eta。

数据预处理与 5.2 节相同：删除 R≤0 的异常点；峰、谷检测的显著度门槛
取 1.0 个百分点。附件 1、2 的统计中剔除两束光模型失效的残余射线带
787~968 cm⁻¹。

运行环境：Python 3 + numpy / scipy / openpyxl
用法：在项目根目录运行  python "5 模型的建立与求解/5.3 问题3/5.3判别.py"
"""
import numpy as np
from scipy.signal import find_peaks
import openpyxl

N0 = 1.0                    # 空气折射率
PROM = 1.0                  # 峰谷检测显著度门槛（百分点），与 5.2 节相同
MIN_PTS = 20                # 参与统计的峰谷区间最少数据点数
RESTR = (787.0, 968.0)      # 碳化硅残余射线带（附件 1、2 剔除）
RHO_LO, RHO_HI = 0.03, 0.05   # 振幅比阈值（数值模拟确定，见 5.3.3）
ETA_TOL = 0.02              # eta 的正常波动半径（50%±2%）


def load_data(path):
    """读取 xlsx 附件，返回 (波数 cm^-1, 反射率 %)。"""
    wb = openpyxl.load_workbook(path, read_only=True)
    rows = list(wb.active.iter_rows(values_only=True))
    wb.close()
    arr = np.array([[float(r[0]), float(r[1])] for r in rows[1:]
                    if r[0] is not None and r[1] is not None])
    return arr[:, 0], arr[:, 1]


def discriminate(path, n1, restr=None):
    """对一份附件统计 (峰谷对数, eta, rho_hat)。"""
    s, R = load_data(path)
    keep = R > 0                                # 预处理规则①
    if restr:
        keep &= ~((s > restr[0]) & (s < restr[1]))
    s, R = s[keep], R[keep]

    pk, _ = find_peaks(R, prominence=PROM)
    va, _ = find_peaks(-R, prominence=PROM)
    E = sorted(list(pk) + list(va))

    # 垂直入射菲涅耳系数（名义折射率）
    r10 = (n1 - N0) / (n1 + N0)                 # 膜内→空气
    t01 = 2 * N0 / (N0 + n1)                    # 空气→膜
    t10 = 2 * n1 / (n1 + N0)                    # 膜→空气

    etas, rhos = [], []
    for a, b in zip(E[:-1], E[1:]):
        seg = (s > s[a]) & (s < s[b])
        if seg.sum() < MIN_PTS:
            continue
        Rmax, Rmin = max(R[a], R[b]), min(R[a], R[b])
        mid = (Rmax + Rmin) / 2.                # 中线，式(34)前的定义
        etas.append((R[seg] < mid).mean())
        B = (np.sqrt(Rmax / 100.) - np.sqrt(Rmin / 100.)) / 2.   # 问题一式(6)
        r12_hat = B / (t01 * t10)               # 取 A≈1，式(33)
        rhos.append(abs(r10) * r12_hat)
    return len(etas), np.mean(etas), np.median(rhos)


def judge(eta, rho):
    """按 5.3.3 的阈值给出判别结论。"""
    if rho <= RHO_LO and abs(eta - 0.5) <= ETA_TOL:
        return '两光束模型适用'
    if rho >= RHO_HI or abs(eta - 0.5) > ETA_TOL:
        return '存在多光束干涉'
    return '介于阈值之间，需结合数据检验'


def main():
    files = [('附件 1（SiC，10°）', '题目/附件/附件1.xlsx', 2.55, RESTR),
             ('附件 2（SiC，15°）', '题目/附件/附件2.xlsx', 2.55, RESTR),
             ('附件 3（Si，10°）', '题目/附件/附件3.xlsx', 3.42, None),
             ('附件 4（Si，15°）', '题目/附件/附件4.xlsx', 3.42, None)]

    print('=' * 72)
    print('多光束干涉定量判别：峰值段数 / 低于中线占比 eta / 振幅比 rho_hat')
    print('=' * 72)
    for name, path, n1, restr in files:
        J, eta, rho = discriminate(path, n1, restr)
        print(f'{name:<18} 峰谷段数 {J:>3}   '
              f'eta = {eta*100:5.1f}%   rho_hat = {rho:.3f}   '
              f'→ {judge(eta, rho)}')
    print('=' * 72)


if __name__ == '__main__':
    main()
