# -*- coding: utf-8 -*-
"""5.2.6 可靠性分析的核查脚本：
1) 稳健统计量（中位数 / MAD / 极值）
2) 共同波段两角折射率差 Δn 的均值与方向一致性
3) 全谱拟合残差：RMS、线性趋势斜率、滞后1自相关（结构性检验）
4) 联合拟合：两角共享厚度 d，与独立拟合比较（残差与 AIC）
"""
import importlib.util
import numpy as np
from pathlib import Path
from scipy.optimize import least_squares
from scipy.stats import median_abs_deviation

here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('sol', here / '5.2求解.py')
sol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol)

ROOT = here.parent.parent
FILES = {'附件1': ('题目/附件/附件1.xlsx', 10.),
         '附件2': ('题目/附件/附件2.xlsx', 15.)}

def prep(name, path, deg):
    s_raw, R_raw = sol.load_data(str(ROOT / path))
    s, R = sol.remove_outliers(s_raw, R_raw)
    th = np.radians(deg)
    P, E = sol.solve(s, R, th)
    return s, R, th, P

# ---------------- 1) 稳健统计量 ----------------
res = {}
print('=' * 70)
print('1) 方案二稳健统计量')
for name, (path, deg) in FILES.items():
    s, R, th, P = prep(name, path, deg)
    d_j = P[:, 6]
    med, mad = np.median(d_j), median_abs_deviation(d_j, scale=1.0)
    res[name] = dict(s=s, R=R, th=th, P=P, d_j=d_j, med=med, mad=mad)
    print(f'  {name}({deg:.0f}°): 中位数 {med:.2f} μm, MAD {mad:.2f} μm '
          f'({mad/med*100:.1f}%), d_j 范围 {d_j.min():.2f}~{d_j.max():.2f} μm')
m1, m2 = res['附件1']['med'], res['附件2']['med']
print(f'  两中位数相对差异: {abs(m1-m2)/((m1+m2)/2)*100:.2f}%')

# ---------------- 2) 折射率一致性 Δn ----------------
print('=' * 70)
print('2) 共同波段折射率差 Δn = n(15°) - n(10°)')
P1, P2 = res['附件1']['P'], res['附件2']['P']
c1, n1 = P1[:, 2], P1[:, 3]
c2, n2 = P2[:, 2], P2[:, 3]
lo = max(c1.min(), c2.min())
hi = min(c1.max(), c2.max())
grid = np.linspace(lo, hi, 200)
dn = np.interp(grid, c2, n2) - np.interp(grid, c1, n1)
base = np.interp(grid, c1, n1)
print(f'  共同波段: {lo:.1f} ~ {hi:.1f} cm^-1')
print(f'  平均 Δn = {dn.mean():.4f} ({dn.mean()/base.mean()*100:.2f}%)')
print(f'  Δn 符号一致(全为正)的比例: {(dn > 0).mean()*100:.0f}%')

# ---------------- 3) 残差结构检验（独立拟合） ----------------
print('=' * 70)
print('3) 方案一独立拟合残差结构')
fits = {}
for name, (path, deg) in FILES.items():
    s, R, th, P = res[name]['s'], res[name]['R'], res[name]['th'], res[name]['P']
    med = res[name]['med']
    d_fit, rmse = sol.fit_full_spectrum(s, R, th, med * 1e-4)
    sg, Rm, Ro = sol.two_beam_model_curve(s, R, th, d_fit * 1e-4)
    resid = Ro - Rm
    # 线性趋势
    k = np.polyfit(sg, resid, 1)
    # 去趋势后滞后 1 自相关
    rt = resid - np.polyval(k, sg)
    ac = np.corrcoef(rt[:-1], rt[1:])[0, 1]
    fits[name] = (d_fit, rmse)
    print(f'  {name}: d_fit = {d_fit:.2f} um, RMS = {rmse:.3f}%, '
          f'趋势斜率 = {k[0]:.3e} %/cm^-1, 滞后1自相关 = {ac:.3f}')

# ---------------- 4) 联合拟合：共享厚度 ----------------
print('=' * 70)
print('4) 联合拟合（两角共享 d）vs 独立拟合')
reg = {n: sol.fit_region_params(res[n]['s'], res[n]['R']) for n in FILES}

def joint_resid(d):
    out = []
    for (name, (path, deg)), (sg, Rg, nE, bE) in zip(FILES.items(), reg.values()):
        th = np.radians(deg)
        rs, rp = sol._fresnel_mod(nE, th)
        phi = 4 * np.pi * d * sg * np.sqrt(nE ** 2 - np.sin(th) ** 2)
        Rm = 0.5 * (rs**2 + bE**2 - 2*rs*bE*np.cos(phi)) \
           + 0.5 * (rp**2 + bE**2 - 2*rp*bE*np.cos(phi))
        out.append(Rm - Rg)
    return np.concatenate(out)

d0 = np.mean([fits[n][0] for n in FILES]) * 1e-4
solj = least_squares(joint_resid, d0, bounds=sol.D_BOUND)
d_joint = solj.x[0] * 1e4
r_all = joint_resid(solj.x[0])
N = len(r_all)
n1p = len(reg['附件1'][0]); n2p = len(reg['附件2'][0])
rms_j = np.sqrt(np.mean(r_all[:n1p]**2)) * 100, np.sqrt(np.mean(r_all[n1p:]**2)) * 100
print(f'  联合拟合: d = {d_joint:.2f} μm')
print(f'  联合拟合分角 RMS: 附件1 = {rms_j[0]:.3f}%, 附件2 = {rms_j[1]:.3f}%')
print(f'  独立拟合分角 RMS: 附件1 = {fits["附件1"][1]:.3f}%, 附件2 = {fits["附件2"][1]:.3f}%')
rss_j = np.sum(r_all**2)
rss_i = sum((fits[n][1]/100)**2 * len(reg[n][0]) for n in FILES)
aic = lambda rss, k: N * np.log(rss / N) + 2 * k
bic = lambda rss, k: N * np.log(rss / N) + k * np.log(N)
print(f'  独立拟合(2个d): AIC = {aic(rss_i, 3):.1f}, BIC = {bic(rss_i, 3):.1f}')
print(f'  联合拟合(1个d): AIC = {aic(rss_j, 2):.1f}, BIC = {bic(rss_j, 2):.1f}')
print(f'  联合拟合 d 与两角独立值({fits["附件1"][0]:.2f}, {fits["附件2"][0]:.2f}) 的偏差: '
      f'{abs(d_joint-fits["附件1"][0])/d_joint*100:.2f}%, '
      f'{abs(d_joint-fits["附件2"][0])/d_joint*100:.2f}%')
