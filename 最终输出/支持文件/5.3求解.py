# -*- coding: utf-8 -*-
"""
问题三：硅外延层厚度的计算（与论文 5.3.5 节一致）
================================================================

附件 3、4 已确认存在多光束干涉。多光束干涉的分母因子以 2π 为周期，
峰谷位置与两光束情形一致、条纹周期不变，因此仍按问题一的方案二
（相邻峰谷法）由峰谷波数求厚度；但峰谷反射率读数在多光束干涉下
有偏，不能再由包络线法反推折射率，折射率改取硅的文献名义值
n = 3.42，逐对代入问题一式(5)：

    d_j = 1 / (4·|σv·c − σp·c|),   c = sqrt(n² − n0²·sin²θ).

数据预处理与 5.2 节相同：删除反射率非正的点；峰、谷检测显著度门槛
取 1.0 个百分点。此外，波数低于约 1500 cm⁻¹ 的低频段，重掺杂衬底
的折射率偏离名义值较多，常数折射率的前提不再成立，该段的峰谷对
不参与统计。

运行环境：Python 3 + numpy / scipy / openpyxl
用法：在项目根目录运行  python "5 模型的建立与求解/5.3 问题3/5.3求解.py"
"""
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import median_abs_deviation
import openpyxl

N0 = 1.0
N_SI = 3.42                 # 硅的文献名义折射率（红外透明区）
PROM = 1.0                  # 峰谷检测显著度门槛（百分点），与 5.2 节相同
D_BOUND = (0.5, 30.0)       # 峰谷对厚度的合理范围（μm）
S_CUT = 1500.               # 低频截断波数（cm⁻¹），低于该值的峰谷对不参与统计


def load_data(path):
    """读取 xlsx 附件，返回 (波数 cm^-1, 反射率 %)。"""
    wb = openpyxl.load_workbook(path, read_only=True)
    rows = list(wb.active.iter_rows(values_only=True))
    wb.close()
    arr = np.array([[float(r[0]), float(r[1])] for r in rows[1:]
                    if r[0] is not None and r[1] is not None])
    return arr[:, 0], arr[:, 1]


def solve(path, theta):
    """一份附件：峰谷检测 → 逐对厚度 → 筛选，返回峰谷对数组。

    数组列为 [σ1, σ2, d_j/μm]，σ1、σ2 为该对两极值按波数排列。
    """
    s, R = load_data(path)
    keep = R > 0                                # 预处理：删除反射率非正的点
    s, R = s[keep], R[keep]
    pk, _ = find_peaks(R, prominence=PROM)
    va, _ = find_peaks(-R, prominence=PROM)
    E = sorted([(s[i], R[i], 'max') for i in pk]
               + [(s[i], R[i], 'min') for i in va])
    c = np.sqrt(N_SI ** 2 - (N0 * np.sin(theta)) ** 2)
    rows = []
    for a, b in zip(E[:-1], E[1:]):
        if a[2] == b[2]:                        # 必须峰谷相邻
            continue
        s1, s2 = min(a[0], b[0]), max(a[0], b[0])
        if min(s1, s2) < S_CUT:
            continue                            # 低频段折射率偏离名义值，截断
        d_j = 1e4 / (4.0 * c * (s2 - s1))       # cm → μm
        if not (D_BOUND[0] <= d_j <= D_BOUND[1]):
            continue                            # 谱线起突处的孤立极值
        rows.append((s1, s2, d_j))
    return np.array(rows)


def main():
    files = [('附件 3（Si，10°）', '题目/附件/附件3.xlsx', np.radians(10.)),
             ('附件 4（Si，15°）', '题目/附件/附件4.xlsx', np.radians(15.))]

    print('=' * 66)
    print(f'硅外延层厚度（相邻峰谷法，n = {N_SI}）')
    print('=' * 66)
    meds = []
    for name, path, theta in files:
        P = solve(path, theta)
        d_j = P[:, 2]
        med = np.median(d_j)
        mad = median_abs_deviation(d_j, scale=1.0)
        meds.append(med)
        print(f'\n【{name}】')
        print(f'  {"σ1":>9} {"σ2":>9} {"d_j/μm":>8}')
        for s1, s2, dj in P:
            print(f'  {s1:9.2f} {s2:9.2f} {dj:8.3f}')
        print(f'  峰谷对 {len(P)} 对，中位数 {med:.2f} μm，'
              f'MAD {mad:.2f} μm（{mad / med * 100:.1f}%）')
    rel = abs(meds[0] - meds[1]) / np.mean(meds) * 100
    print('\n' + '-' * 66)
    print(f'双入射角一致性：{meds[0]:.2f} μm 与 {meds[1]:.2f} μm，'
          f'相对差异 {rel:.1f}%')
    print(f'最终结果：硅外延层厚度 d ≈ {np.mean(meds):.2f} μm')
    print('=' * 66)


if __name__ == '__main__':
    main()
