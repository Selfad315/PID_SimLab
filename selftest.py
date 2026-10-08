# -*- coding: utf-8 -*-
"""核心算法自检脚本（不依赖 streamlit，可在命令行直接运行）。"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from pidlab import (SecondOrderPlant, PID, simulate, simulate_open_loop, compute_metrics,
                    second_order_theory, phase_crossover, zn_closed_loop, zn_open_loop,
                    decay_curve, auto_optimize, fopdt_from_step)

ok = True
def check(name, cond, info=""):
    global ok
    flag = "PASS" if cond else "FAIL"
    if not cond: ok = False
    print(f"[{flag}] {name}  {info}")

print("=" * 78)
print("PID_SimLab 核心算法自检")
print("=" * 78)
print("numpy", np.__version__)

# ---------- 1. 二阶系统仿真 vs 理论公式 ----------
print("\n--- 1) 二阶系统：仿真 vs 理论 ---")
plant = SecondOrderPlant(K=1.0, wn=2.0, zeta=0.3, delay=0.0)
res = simulate_open_loop(plant, t_end=12.0, n_samples=6000)
m = compute_metrics(res.t, res.y, res.r)
th = second_order_theory(plant.zeta, plant.wn)
print(f"  仿真: sigma={m['overshoot']:.3f}%  tp={m['tp']:.4f}s  tr(10-90%)={m['tr']:.4f}s  tr(0-100%)={m['tr_full']:.4f}s  ts2={m['ts_2']:.4f}s")
print(f"  理论: sigma={th['overshoot']:.3f}%  tp={th['tp']:.4f}s  tr(0-100%)={th['tr']:.4f}s  ts上界={th['ts']:.4f}s  ts近似={th['ts_approx']:.4f}s")
check("超调量仿真≈理论", abs(m["overshoot"] - th["overshoot"]) < 0.5,
      f"误差 {abs(m['overshoot']-th['overshoot']):.4f}%")
check("峰值时间仿真≈理论", abs(m["tp"] - th["tp"]) < 0.02)
check("上升时间(0-100%)仿真≈理论", abs(m["tr_full"] - th["tr"]) < 0.02,
      f"仿真 {m['tr_full']:.4f} vs 理论 {th['tr']:.4f}")
check("实际调节时间优于包络上界且接近工程近似",
      th["ts_approx"] * 0.8 <= m["ts_2"] <= th["ts"] * 1.05,
      f"{th['ts_approx']:.3f} <= {m['ts_2']:.3f} <= {th['ts']:.3f}")
check("稳态值≈1", abs(m["y_ss"] - 1.0) < 1e-3, f"yss={m['y_ss']:.6f}")
check("稳态误差≈0", abs(m["ess"]) < 1e-3)

# ---------- 2. 纯滞后环节 ----------
print("\n--- 2) 纯滞后对象（延迟 0.5s 应整体平移） ---")
# 开环阶跃响应：纯滞后只应造成整体时间平移（闭环则不然！）
p0 = SecondOrderPlant(1.0, 2.0, 0.4, delay=0.0)
p1 = SecondOrderPlant(1.0, 2.0, 0.4, delay=0.5)
r0 = simulate_open_loop(p0, t_end=10, n_samples=6000)
r1 = simulate_open_loop(p1, t_end=10, n_samples=6000)
# 在 t 处比较 y1(t) 与 y0(t-0.5)
tt = np.linspace(1.5, 8.0, 200)
y0i = np.interp(tt - 0.5, r0.t, r0.y)
y1i = np.interp(tt, r1.t, r1.y)
err = float(np.max(np.abs(y0i - y1i)))
check("带滞后响应 = 无滞后响应平移 tau", err < 5e-3, f"最大偏差 {err:.5f}")

# ---------- 3. PID 三种模式 ----------
print("\n--- 3) PID 模式与稳态误差 ---")
plant3 = SecondOrderPlant(1.0, 1.0, 0.5, delay=0.0)
mm = {}
for mode, kw in (("P", dict(kp=2.0)), ("PI", dict(kp=2.0, ki=2.0)), ("PD", dict(kp=2.0, kd=0.2)),
                 ("PID", dict(kp=2.0, ki=2.0, kd=0.2))):
    rr = simulate(plant3, PID(mode=mode, **kw), t_end=80, n_samples=8000)
    mm[mode] = compute_metrics(rr.t, rr.y, rr.r, label=mode)
    print(f"  {mode:4s} sigma={mm[mode]['overshoot']:7.3f}%  ts2={mm[mode]['ts_2']:7.3f}s  ess={mm[mode]['ess']:.2e}  IAE={mm[mode]['iae']:.4f}")
check("P 控制存在稳态误差", abs(mm["P"]["ess"]) > 1e-3)
check("PI 消除稳态误差", abs(mm["PI"]["ess"]) < 1e-3)
check("PID 消除稳态误差", abs(mm["PID"]["ess"]) < 1e-3)

# ---------- 4. ZN 临界比例度法 ----------
print("\n--- 4) ZN 临界比例度法 ---")
p4 = SecondOrderPlant(1.0, 1.0, 0.5, delay=0.1)
cr = phase_crossover(p4)
check("存在 -180° 穿越点", cr is not None)
if cr:
    wu, Ku, Pu = cr[0], cr[1], cr[2]
    print(f"  wu={wu:.4f} rad/s  Ku={Ku:.4f}  Pu={Pu:.4f} s")
    # 用 Ku 做纯比例闭环，应处于等幅振荡（临界稳定）
    rk = simulate(p4, PID(kp=Ku, mode="P"), t_end=40, n_samples=8000, anti_windup=False)
    amp1 = compute_metrics(rk.t, rk.y, rk.r)
    mid = np.abs(rk.y[int(len(rk.y)*0.4):int(len(rk.y)*0.8)])
    print(f"  临界增益下振荡幅度 {mid.max():.4f} / {mid.min():.4f}（近似等幅）")
    check("临界增益下不发散", np.isfinite(rk.y).all() and np.max(np.abs(rk.y)) < 50)
zn = zn_closed_loop(p4, "PID")
check("ZN(PID) 整定成功", zn.valid, f"Kp={zn.kp:.4f}, Ki={zn.ki:.4f}, Kd={zn.kd:.4f}")

# ---------- 5. ZN 阶跃响应法 / FOPDT ----------
print("\n--- 5) ZN 阶跃响应法（FOPDT 辨识） ---")
f = fopdt_from_step(p4)
print(f"  辨识 K={f['K']:.4f}  T={f['T']:.4f}s  L={f['L']:.4f}s")
check("FOPDT 辨识成功", f["valid"])
check("辨识增益≈1", abs(f["K"] - 1.0) < 0.02)
zo = zn_open_loop(p4, "PID")
check("ZN 阶跃响应法成功", zo.valid, f"Kp={zo.kp:.4f}, Ti={zo.ti:.4f}, Td={zo.td:.4f}")

# ---------- 6. 衰减曲线法 ----------
print("\n--- 6) 衰减曲线法（4:1） ---")
t0 = time.time()
dc = decay_curve(p4, "PID", ratio=0.25, n_samples=2000)
el = time.time() - t0
print(f"  Kc={dc.kc:.4f}  Ts={dc.ts_osc:.4f}s  实测衰减比={dc.extra.get('decay_ratio_achieved'):.4f}")
print(f"  PID: Kp={dc.kp:.4f}, Ki={dc.ki:.4f}, Kd={dc.kd:.4f}  ({el:.2f}s)")
check("衰减曲线法成功", dc.valid)
check("搜索到的衰减比≈0.25", abs(dc.extra.get("decay_ratio_achieved", 9) - 0.25) < 0.05,
      f"实测 {dc.extra.get('decay_ratio_achieved'):.4f}")

# ---------- 7. 抗积分饱和 ----------
print("\n--- 7) 执行器饱和 + 抗积分饱和 ---")
from pidlab import Actuator
act = Actuator(enabled=True, u_min=-1.0, u_max=1.0)
sa = simulate(p4, PID(kp=5, ki=8, kd=0.5, mode="PID"), t_end=30, n_samples=3000,
              actuator=act, anti_windup=True)
sb = simulate(p4, PID(kp=5, ki=8, kd=0.5, mode="PID"), t_end=30, n_samples=3000,
              actuator=act, anti_windup=False)
ma = compute_metrics(sa.t, sa.y, sa.r); mb = compute_metrics(sb.t, sb.y, sb.r)
print(f"  抗饱和: sigma={ma['overshoot']:.2f}%  ts2={ma['ts_2']:.3f}s  IAE={ma['iae']:.3f}")
print(f"  无抗饱和: sigma={mb['overshoot']:.2f}%  ts2={mb['ts_2']:.3f}s  IAE={mb['iae']:.3f}")
check("抗积分饱和改善性能", ma["iae"] <= mb["iae"] + 1e-9)

# ---------- 8. 扰动 ----------
print("\n--- 8) 抗干扰仿真 ---")
from pidlab import step_disturbance
rd = simulate(p4, PID(kp=2, ki=2, kd=0.2, mode="PID"), t_end=30, n_samples=4000,
              disturbance=step_disturbance(12.0, 0.3))
from pidlab import disturbance_metrics
mdd = disturbance_metrics(rd.t, rd.y, 1.0, 12.0)
print(f"  最大动态偏差={mdd['max_dev']:.4f}  恢复时间={mdd['recovery']:.4f}s  扰动后IAE={mdd['iae_dist']:.4f}")
check("扰动指标可计算", np.isfinite(mdd["max_dev"]))

# ---------- 9. 死区 ----------
print("\n--- 9) 死区非线性 ---")
act2 = Actuator(enabled_dead_zone=True, dead_zone=0.15)
lin = simulate(p4, PID(kp=1.2, ki=0.6, kd=0.1, mode="PID"), t_end=30, n_samples=3000)
non = simulate(p4, PID(kp=1.2, ki=0.6, kd=0.1, mode="PID"), t_end=30, n_samples=3000, actuator=act2)
ml = compute_metrics(lin.t, lin.y, lin.r); mn = compute_metrics(non.t, non.y, non.r)
print(f"  线性: sigma={ml['overshoot']:.3f}% ts2={ml['ts_2']:.4f}s ess={ml['ess']:.3e}")
print(f"  死区: sigma={mn['overshoot']:.3f}% ts2={mn['ts_2']:.4f}s ess={mn['ess']:.3e}")
check("死区导致性能变化", abs(mn["ess"] - ml["ess"]) > 1e-9 or abs(mn["ts_2"] - ml["ts_2"]) > 1e-9)

# ---------- 10. 数值寻优 ----------
print("\n--- 10) 数值寻优法（PID） ---")
t0 = time.time()
op = auto_optimize(p4, "PID", n_samples=800, maxiter=60)
el = time.time() - t0
mo = compute_metrics(*(lambda r: (r.t, r.y, r.r))(simulate(p4, op.to_pid(), t_end=30, n_samples=3000)))
print(f"  Kp={op.kp:.4f} Ki={op.ki:.4f} Kd={op.kd:.4f}  ({el:.2f}s)")
print(f"  sigma={mo['overshoot']:.3f}%  ts2={mo['ts_2']:.3f}s  IAE={mo['iae']:.4f}  J={op.extra.get('fun'):.4f}")
check("寻优得到有效参数", op.valid and op.kp > 0)

# ---------- 11. 通用对象模型库 / 自定义传递函数 ----------
print("\n--- 11) 通用对象模型库 / 自定义传递函数 ---")
from pidlab import LTIPlant, MODEL_SPECS, build_plant

# (a) 同一传递函数两种构造方式必须一致
p_so = SecondOrderPlant(1.0, 2.0, 0.3, 0.1)
p_lg = LTIPlant(num=[4.0], den=[1.0, 1.2, 4.0], delay=0.1)
check("LTIPlant 与 SecondOrderPlant 传递函数一致",
      np.allclose(p_so.num, p_lg.num) and np.allclose(p_so.den, p_lg.den))
check("second_order_params() 反解正确",
      np.allclose(p_lg.second_order_params(), (1.0, 2.0, 0.3), atol=1e-9),
      f"反解 K/wn/zeta = {tuple(round(v, 4) for v in p_lg.second_order_params())}")

# (b) 模型库所有条目都能建模并完成闭环仿真
lib_ok, lib_info = True, []
for nm, sp_ in MODEL_SPECS.items():
    if sp_["kind"] == "custom":
        continue
    pv = {a[0]: a[2] for a in sp_["params"]}
    pl = build_plant(nm, pv, delay=0.1)
    rr = simulate(pl, PID(kp=2, ki=1, kd=0.2, mode="PID"), t_end=30, n_samples=1200)
    mm = compute_metrics(rr.t, rr.y, rr.r)
    lib_ok &= bool(rr.y.size == 1201) and ("score" in mm)
    lib_info.append(f"{nm}({pl.order}阶/{pl.stability_label()})")
print("  模型库:", "、".join(lib_info))
check("模型库全部条目可建模 + 闭环仿真 + 出指标", lib_ok)

# (c) 自定义三阶传递函数
p3 = build_plant("自定义传递函数", {}, delay=0.05, num=[1.0], den=[1.0, 3.0, 3.0, 1.0])
r3 = simulate(p3, PID(kp=2, ki=1, kd=0.2, mode="PID"), t_end=30, n_samples=1500)
m3 = compute_metrics(r3.t, r3.y, r3.r)
check("自定义三阶传递函数可仿真", p3.order == 3 and np.isfinite(m3["score"]),
      f"阶次={p3.order}, y(∞)={m3['y_ss']:.4f}")

# (d) 非真传递函数必须被拒绝
try:
    LTIPlant(num=[1, 2, 3, 4], den=[1, 1]).ss()
    improper_rejected = False
except Exception:
    improper_rejected = True
check("非真传递函数被正确拒绝", improper_rejected)

# (e) 不稳定对象的指标仍带 score 键（保证排序比较不炸）
pu = LTIPlant(num=[1.0], den=[1.0, -1.0, 1.0])
ru = simulate(pu, PID(kp=1, ki=0, kd=0, mode="P"), t_end=10, n_samples=800)
mu = compute_metrics(ru.t, ru.y, ru.r)
check("不稳定对象指标包含 score 键", ("score" in mu) and mu["score"] == 0.0)

# ---------- 12. 频域稳定裕度 / Nyquist 判据 / 根轨迹 ----------
print("\n--- 12) 频域稳定裕度 / Nyquist 判据 / 根轨迹 ---")
from pidlab import freq as fq
from pidlab import closed_loop_tf

# (a) 解析可验证算例：L(s) = 2/(s+1)^3
#     相位穿越 ωpc = tan60° = √3，|L(j√3)| = 2/8 → h = 4 → 12.041 dB
pl12 = LTIPlant(num=[2.0], den=[1.0, 3.0, 3.0, 1.0], label="三阶")
pid12 = PID(kp=1.0, ki=0.0, kd=0.0, mode="P")
mm12 = fq.stability_margins(pl12, pid12)
gm_th = 20 * np.log10(8.0 / 2.0)
wpc_th = np.sqrt(3.0)
wgc_th = np.sqrt(2.0 ** (2 / 3) - 1.0)
pm_th = 180.0 - 3.0 * np.degrees(np.arctan(wgc_th))
check("幅值裕度与解析值一致", abs(mm12["gm_db"] - gm_th) < 0.05,
      f"{mm12['gm_db']:.3f} dB vs 理论 {gm_th:.3f} dB")
check("相位穿越频率与解析值一致", abs(mm12["w_pc"] - wpc_th) < 1e-3,
      f"{mm12['w_pc']:.5f} vs {wpc_th:.5f}")
check("幅值穿越频率与解析值一致", abs(mm12["w_gc"] - wgc_th) < 1e-3,
      f"{mm12['w_gc']:.5f} vs {wgc_th:.5f}")
check("相位裕度与解析值一致", abs(mm12["pm"] - pm_th) < 0.1,
      f"{mm12['pm']:.3f}° vs 理论 {pm_th:.3f}°")

# (b) Nyquist 判据 Z = P − N 必须与直接求闭环极点一致（含积分环节与不稳定工况）
cases12 = [
    ("二阶+PID（稳定）", SecondOrderPlant(1.0, 1.0, 0.5, 0.0), PID(kp=2, ki=1, kd=0.2, mode="PID")),
    ("三阶+P（稳定）", LTIPlant([1.0], [1.0, 3.0, 3.0, 1.0]), PID(kp=1, ki=0, kd=0, mode="P")),
    ("三阶+P（不稳定）", LTIPlant([20.0], [1.0, 3.0, 3.0, 1.0]), PID(kp=1, ki=0, kd=0, mode="P")),
    ("三阶+PID（不稳定）", LTIPlant([1.0], [1.0, 3.0, 3.0, 1.0]), PID(kp=40, ki=40, kd=0.2, mode="PID")),
    ("带积分对象+PID", LTIPlant([1.0], [0.5, 1.0, 0.0]), PID(kp=3, ki=2, kd=0.2, mode="PID")),
]
ok12 = True
for nm12, pl_, pid_ in cases12:
    rr12 = fq.nyquist_analysis(pl_, pid_, n=12000)
    ok12 &= bool(rr12["consistent"])
    print(f"    {nm12:<20} P={rr12['P']} N={rr12['N']:>2} "
          f"Z(P-N)={rr12['Z_nyquist']} Z(求根)={rr12['Z_actual']}")
check("Nyquist 判据 Z=P−N 与直接求闭环极点一致", ok12)

# (c) 根轨迹 k=1 的闭环极点应等于直接求解闭环特征方程的根
pl13 = SecondOrderPlant(1.0, 1.0, 0.5, 0.0)
pid13 = PID(kp=2, ki=1, kd=0.2, mode="PID")
rl13 = fq.root_locus(pl13, pid13, k_max=10.0, n=60, use_pade=False)
num13, den13 = closed_loop_tf(pl13, pid13)
check("根轨迹 k=1 的极点 = 闭环特征根",
      np.allclose(np.sort_complex(rl13["current"]), np.sort_complex(np.roots(den13)), atol=1e-6),
      f"极点 {np.round(np.sort_complex(rl13['current']), 4)}")

# (d) 裕度评价与稳定性一致
from pidlab import MODEL_SPECS as MS12
pl14 = build_plant("标准二阶对象", {"K": 1.0, "wn": 1.0, "zeta": 0.5}, delay=0.1)
g_ok = fq.margin_grade(fq.stability_margins(pl14, PID(kp=2, ki=1, kd=0.2, mode="PID")))[0]
g_bad = fq.margin_grade(fq.stability_margins(
    build_plant("三阶惯性对象", {"K": 1.0, "T1": 1.0, "T2": 0.5, "T3": 0.2}, delay=0.1),
    PID(kp=40, ki=40, kd=0.2, mode="PID")))[0]
check("裕度评价能区分稳定/不稳定", g_ok in ("良好", "合格", "临界") and g_bad == "不稳定",
      f"正常参数→{g_ok}，大增益→{g_bad}")

# ---------- 13. 改进型 PID：积分分离 / 不完全微分 / 抗饱和 ----------
print("\n--- 13) 改进型 PID（积分分离 / 不完全微分 / 抗饱和）---")
from pidlab import control_quality, Actuator

pl13b = SecondOrderPlant(1.0, 1.0, 0.5, 0.1)

# (a) 积分分离：大阶跃 + 无抗饱和时，应显著压低超调
res_sep = {}
for flag in (False, True):
    pid_s = PID(kp=2, ki=4, kd=0.2, mode="PID",
                integral_separation=flag, sep_threshold=0.5)
    rr = simulate(pl13b, pid_s, t_end=40, n_samples=3000, ref=6.0, anti_windup_mode="none")
    res_sep[flag] = compute_metrics(rr.t, rr.y, rr.r)
print(f"    大阶跃 ref=6：无分离 峰值={res_sep[False]['peak']:.2f} / σ={res_sep[False]['overshoot']:.1f}%"
      f"　有分离 峰值={res_sep[True]['peak']:.2f} / σ={res_sep[True]['overshoot']:.1f}%")
check("积分分离显著降低大阶跃超调",
      res_sep[True]["peak"] < res_sep[False]["peak"] * 0.5,
      f"峰值 {res_sep[False]['peak']:.1f} → {res_sep[True]['peak']:.1f}")

# (b) 不完全微分：噪声下应显著降低控制量总变差 TV
tv = {}
for flag in (False, True):
    pid_d = PID(kp=4, ki=2, kd=0.6, mode="PID", derivative_filter=flag, N=10)
    rr = simulate(pl13b, pid_d, t_end=40, n_samples=3000, noise_std=0.05, seed=1)
    tv[flag] = control_quality(rr.u, rr.t[1] - rr.t[0])["tv"]
print(f"    噪声 σ=0.05：完全微分 TV={tv[False]:.0f}　不完全微分 TV={tv[True]:.0f}")
check("不完全微分显著降低控制量抖动", tv[True] < tv[False] * 0.8,
      f"TV 降低 {(1 - tv[True] / tv[False]) * 100:.1f}%")

# (c) 三种抗饱和方案都应优于「无抗饱和」
act13 = Actuator(enabled=True, u_min=-1.0, u_max=1.0)
ov = {}
for mkey in ("none", "back", "conditional", "clamping"):
    pid_a = PID(kp=3, ki=8, kd=0.2, mode="PID")
    rr = simulate(pl13b, pid_a, t_end=40, n_samples=3000, actuator=act13, anti_windup_mode=mkey)
    ov[mkey] = compute_metrics(rr.t, rr.y, rr.r)["overshoot"]
    print(f"    抗饱和 {mkey:<12} σ={ov[mkey]:7.2f}%")
check("三种抗饱和方案均优于无抗饱和",
      all(ov[k] < ov["none"] for k in ("back", "conditional", "clamping")),
      f"无抗饱和 {ov['none']:.1f}% → 反算 {ov['back']:.1f}% / 条件 {ov['conditional']:.1f}% / 限幅 {ov['clamping']:.1f}%")

# (d) 量测噪声确实进入控制器，但指标仍按真值计算
rr = simulate(pl13b, PID(kp=4, ki=2, kd=0.5, mode="PID"), t_end=10, n_samples=1500,
              noise_std=0.02, seed=3)
check("量测噪声生效且与真值分离",
      rr.y_meas is not None and not np.allclose(rr.y, rr.y_meas),
      f"y 与 y_meas 最大差 {np.max(np.abs(rr.y - rr.y_meas)):.4f}")

print("\n" + "=" * 78)
print("自检结果：", "全部通过 [OK]" if ok else "存在失败项 [FAIL]")
print("=" * 78)
sys.exit(0 if ok else 1)
