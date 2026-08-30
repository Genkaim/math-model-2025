# -*- coding: utf-8 -*-
"""
问题三：硅外延层厚度结果的 FFT 检验（与论文 5.3.5 节一致）
================================================================

主结果为相邻峰谷法（5.3求解.py）。这里用快速傅里叶变换频谱分析作
独立检验：干涉条纹等价于反射率随波数以基频

    f = 2·d·c  （周期数每 cm⁻¹）

振荡，对谱线作傅里叶变换后由功率谱主峰确定基频，即得

    d = 1e4·f / (2·c)  μm。

处理步骤依次为：截取波数不低于 1500 cm⁻¹ 的区间；插值到均匀网格；
三次多项式去趋势，扣除谱线的缓慢背景；乘汉宁窗抑制频谱泄漏；
零填充细化频率刻度后变换。多光束干涉使条纹波形偏离正弦，只在频谱上
产生整倍频的谐波，不移动基频，故该检验不受峰谷识别的影响。

数据预处理与 5.3.5 节相同：删除反射率非正的点；波数低于 1500 cm⁻¹
的低频段不参与统计。

运行环境：Python 3 + numpy / openpyxl
用法：在项目根目录运行  python "5 模型的建立与求解/5.3 问题3/5.3检验.py"
"""
import numpy as np
import openpyxl

N0 = 1.0
N_SI = 3.42                 # 硅的文献名义折射率（红外透明区）
S_CUT = 1500.               # 低频截断波数（cm⁻¹），与 5.3求解.py 一致
N_PAD = 8                   # 零填充倍数，细化频率刻度


def load_data(path):
    """读取 xlsx 附件，返回 (波数 cm^-1, 反射率 %)。"""
    wb = openpyxl.load_workbook(path, read_only=True)
    rows = list(wb.active.iter_rows(values_only=True))
    wb.close()
    arr = np.array([[float(r[0]), float(r[1])] for r in rows[1:]
                    if r[0] is not None and r[1] is not None])
    return arr[:, 0], arr[:, 1]


def preprocess(s, R):
    """截断、均匀化、去趋势，返回 (均匀网格, 去趋势后的反射率)。"""
    m = (R > 0) & (s >= S_CUT)
    s, R = s[m], R[m]
    sg = np.linspace(s.min(), s.max(), len(s))
    R = np.interp(sg, s, R)
    R = R - np.polyval(np.polyfit(sg, R, 3), sg)  # 三次多项式去趋势
    return sg, R


def fft_spectrum(s, R):
    """对预处理后的谱线作加窗、零填充的 FFT，
    返回 (频率刻度/周期数每 cm⁻¹, 功率, 基频)。"""
    sg, R = preprocess(s, R)
    sp = np.fft.rfft(R * np.hanning(len(R)), n=N_PAD * len(R))
    df = 1.0 / ((sg[-1] - sg[0]) * N_PAD)
    f = np.arange(len(sp)) * df
    power = np.abs(sp)
    k = np.argmax(power[1:]) + 1                # 跳过零频
    return f, power, f[k]


def fft_thickness(s, R, theta):
    """FFT 检验给出的厚度（μm）。"""
    c = np.sqrt(N_SI ** 2 - (N0 * np.sin(theta)) ** 2)
    _f, _power, f0 = fft_spectrum(s, R)
    return 1e4 * f0 / (2 * c), f0


def main():
    files = [('附件 3（Si，10°）', '题目/附件/附件3.xlsx', np.radians(10.)),
             ('附件 4（Si，15°）', '题目/附件/附件4.xlsx', np.radians(15.))]

    print('=' * 70)
    print('硅外延层厚度结果的 FFT 检验（主结果：相邻峰谷法 3.41 / 3.40 μm）')
    print('=' * 70)
    for name, path, theta in files:
        s, R = load_data(path)
        d, f0 = fft_thickness(s, R, theta)
        print(f'{name:<16} d = {d:.3f} μm   基频 f0 = {f0:.5f} 周期/cm⁻¹'
              f'（2f0 = {2 * f0:.5f}，3f0 = {3 * f0:.5f}）')
    print('=' * 70)


if __name__ == '__main__':
    main()
