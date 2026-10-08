# -*- coding: utf-8 -*-
"""频域分析与稳定性判据：稳定裕度、Nyquist 判据、根轨迹。

术语与约定
----------
* 开环传递函数 L(s) = C(s)·G(s)，含纯滞后 e^(-τs)；
* 幅值裕度  h = 1/|L(jω_pc)|，用分贝表示 h_dB = -20lg|L(jω_pc)|，
  其中 ω_pc 为相位穿越频率（∠L = -180°）。h_dB > 0 表示稳定；
* 相位裕度  γ = 180° + ∠L(jω_gc)，其中 ω_gc 为幅值穿越频率（|L| = 1）。γ > 0 表示稳定；
* Nyquist 判据  Z = P - N：P 为开环右半平面极点数，N 为 Nyquist 曲线绕 (-1, j0)
  的净逆时针圈数，Z 为闭环右半平面极点数。Z = 0 即闭环稳定。
* 根轨迹只对有理传递函数有定义，纯滞后用一阶 Padé 近似
  e^(-τs) ≈ (1 - τs/2)/(1 + τs/2) 计入。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .control import PID, closed_loop_tf
from .models import LTIPlant, poly_add, poly_mul


# --------------------------------------------------------------------------- #
#  多项式与近似工具
# --------------------------------------------------------------------------- #
def poly_shift(p, n: int = 1) -> np.ndarray:
    """多项式乘以 s^n（降幂系数在尾部补零）。"""
    return np.concatenate([np.asarray(p, dtype=float), np.zeros(n, dtype=float)])


def poly_scale(p, k: float) -> np.ndarray:
    return np.asarray(p, dtype=float) * float(k)


def pade1(delay: float):
    """一阶 Padé 近似 e^(-τs) ≈ (1 - τs/2)/(1 + τs/2)，返回 (num, den)。"""
    if delay <= 1e-12:
        return np.array([1.0]), np.array([1.0])
    a = float(delay) / 2.0
    return np.array([-a, 1.0]), np.array([a, 1.0])


def freq_scale(plant: LTIPlant, pid: PID | None = None) -> float:
    """特征频率尺度：由对象与控制器的零极点模长决定。"""
    pts = list(np.abs(plant.poles())) + list(np.abs(plant.zeros()))
    if pid is not None:
        nc, dc = pid.tf()
        pts += [float(np.abs(np.roots(nc)).min()) if nc.size > 1 else np.nan]
        pts += [float(np.abs(np.roots(dc)).min()) if dc.size > 1 else np.nan]
    vals = [p for p in pts if np.isfinite(p) and p > 1e-9]
    return float(np.min(vals)) if vals else 1.0


def freq_grid(plant: LTIPlant, pid: PID | None = None, n: int = 30000, span: float = 6.0):
    ref = max(freq_scale(plant, pid), 1e-6)
    return np.logspace(np.log10(ref) - span, np.log10(ref) + span, n)


# --------------------------------------------------------------------------- #
#  开环频率特性
# --------------------------------------------------------------------------- #
def loop_tf(plant: LTIPlant, pid: PID):
    """开环 L(s)=C(s)G(s) 的有理部分 (num, den)，不含纯滞后。"""
    nc, dc = pid.tf()
    return poly_mul(nc, plant.num), poly_mul(dc, plant.den)


def loop_freqresp(plant: LTIPlant, pid: PID, w) -> np.ndarray:
    """开环频率特性 L(jω)，含纯滞后相位。"""
    w = np.asarray(w, dtype=float)
    num, den = loop_tf(plant, pid)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.polyval(num, 1j * w) / np.polyval(den, 1j * w)
    if plant.delay:
        L = L * np.exp(-1j * w * plant.delay)
    return L


def _first_crossing(w: np.ndarray, f: np.ndarray):
    """找 f 由正变负的首个零点，线性插值返回对应的 w。"""
    f = np.asarray(f, dtype=float)
    for i in range(f.size - 1):
        if np.isfinite(f[i]) and np.isfinite(f[i + 1]) and f[i] > 0.0 >= f[i + 1]:
            t = f[i] / (f[i] - f[i + 1])
            return float(w[i] + t * (w[i + 1] - w[i]))
    return None


# --------------------------------------------------------------------------- #
#  稳定裕度
# --------------------------------------------------------------------------- #
def stability_margins(plant: LTIPlant, pid: PID, n: int = 40000) -> dict:
    """计算幅值裕度、相位裕度及其穿越频率，并给出稳定判定。"""
    w = freq_grid(plant, pid, n=n)
    L = loop_freqresp(plant, pid, w)
    mag = np.abs(L)
    ph = np.unwrap(np.angle(L))
    ph_deg = np.degrees(ph)

    w_gc = _first_crossing(w, mag - 1.0)          # |L| 由 >1 降到 <1
    w_pc = _first_crossing(w, ph + np.pi)          # 相位由 >-180° 降到 <-180°

    out = {"w": w, "mag_db": 20 * np.log10(np.maximum(mag, 1e-300)),
           "phase_deg": ph_deg, "w_gc": w_gc, "w_pc": w_pc,
           "pm": None, "gm": None, "gm_db": None, "stable": None}

    if w_gc is not None:
        ph_gc = float(np.interp(np.log(w_gc), np.log(w), ph))
        out["pm"] = float(np.degrees(ph_gc + np.pi))
    if w_pc is not None:
        mag_pc = float(np.interp(np.log(w_pc), np.log(w), mag))
        if mag_pc > 1e-15:
            out["gm"] = float(1.0 / mag_pc)
            out["gm_db"] = float(-20.0 * np.log10(mag_pc))
        else:
            out["gm"], out["gm_db"] = np.inf, np.inf

    gm_ok = True if out["gm_db"] is None else (out["gm_db"] > 0.0)
    pm_ok = True if out["pm"] is None else (out["pm"] > 0.0)
    if (out["gm_db"] is not None) or (out["pm"] is not None):
        out["stable"] = bool(gm_ok and pm_ok)
    return out


# --------------------------------------------------------------------------- #
#  Nyquist 判据
# --------------------------------------------------------------------------- #
def nyquist_analysis(plant: LTIPlant, pid: PID, n: int = 30000, use_pade: bool = True) -> dict:
    """Nyquist 判据 Z = P - N，并与直接求闭环极点的结果互验。"""
    if use_pade and plant.delay > 1e-12:
        pn, pd = pade1(plant.delay)
        ng, dg = poly_mul(plant.num, pn), poly_mul(plant.den, pd)
    else:
        ng, dg = plant.num, plant.den

    nc, dc = pid.tf()
    num_L = poly_mul(nc, ng)
    den_L = poly_mul(dc, dg)

    def _trailing_zeros(p):
        """多项式在原点的零点重数（尾部零的个数）。"""
        c = 0
        for v in np.asarray(p, float)[::-1]:
            if abs(v) < 1e-12:
                c += 1
            else:
                break
        return c

    # 原点处的净积分环节数（先做零极点对消：P/PD 控制器的 s/s 不算积分环节）
    n_origin = max(0, _trailing_zeros(den_L) - _trailing_zeros(num_L))

    poles_L = np.roots(den_L)
    P = int(np.sum(np.real(poles_L) > 1e-9))            # 开环右半平面极点数

    w = freq_grid(plant, pid, n=n)
    L = np.polyval(num_L, 1j * w) / np.polyval(den_L, 1j * w)
    F = L + 1.0
    ph = np.unwrap(np.angle(F))
    phi_pos = float(ph[-1] - ph[0])                     # ω: 0+ → +∞ 的相角变化
    # 完整 Nyquist 围线 = 正频段 + 镜像负频段 + 原点绕行小圆弧
    # 小圆弧对应 L(ε·e^{jθ}) ≈ (K/ε^m)·e^{-jmθ}，相角变化 -m·π
    total = 2.0 * phi_pos - n_origin * np.pi
    N_raw = total / (2.0 * np.pi)
    N = int(np.round(N_raw))
    Z_nyq = int(P - N)

    den_cl = poly_add(den_L, num_L)
    cl_poles = np.roots(den_cl)
    Z_actual = int(np.sum(np.real(cl_poles) > 1e-9))

    return {"w": w, "L": L, "num_L": num_L, "den_L": den_L,
            "P": P, "N": N, "N_raw": N_raw, "Z_nyquist": Z_nyq, "Z_actual": Z_actual,
            "cl_poles": cl_poles, "open_poles": poles_L, "n_origin": n_origin,
            "consistent": bool(Z_nyq == Z_actual), "stable": bool(Z_actual == 0),
            "used_pade": bool(use_pade and plant.delay > 1e-12)}


# --------------------------------------------------------------------------- #
#  根轨迹
# --------------------------------------------------------------------------- #
def root_locus(plant: LTIPlant, pid: PID, k_max: float = 20.0, n: int = 600,
               use_pade: bool = True) -> dict:
    """经典根轨迹：对开环 L(s)=C(s)G(s) 引入标量增益 k（k=1 即当前参数）。

    特征方程 1 + k·L(s) = 0  →  den_L(s) + k·num_L(s) = 0
    k = 0 时闭环极点 = 开环极点；k → ∞ 时趋向开环零点。
    """
    if use_pade and plant.delay > 1e-12:
        pn, pd = pade1(plant.delay)
        ng, dg = poly_mul(plant.num, pn), poly_mul(plant.den, pd)
    else:
        ng, dg = plant.num, plant.den
    nc, dc = pid.tf()
    num_L = poly_mul(nc, ng)
    den_L = poly_mul(dc, dg)

    ks = np.concatenate([np.linspace(0.0, 1.0, int(n * 0.35), endpoint=False),
                         np.linspace(1.0, float(k_max), int(n * 0.65))])
    poles = [np.roots(poly_add(den_L, poly_scale(num_L, float(k)))) for k in ks]
    cur = np.roots(poly_add(den_L, num_L))
    return {"k": ks, "poles": poles,
            "ol_poles": np.roots(den_L), "ol_zeros": np.roots(num_L) if num_L.size > 1 else np.array([]),
            "current": cur, "current_rhp": int(np.sum(np.real(cur) > 1e-9)),
            "used_pade": bool(use_pade and plant.delay > 1e-12)}


# --------------------------------------------------------------------------- #
#  裕度评价与工程建议
# --------------------------------------------------------------------------- #
def margin_grade(margins: dict) -> tuple:
    """按工程经验给稳定裕度分级：返回 (等级, 说明, 颜色键)。"""
    gm = margins.get("gm_db")
    pm = margins.get("pm")
    if (gm is None) and (pm is None):
        return "无法判定", "开环频率特性不存在穿越点，无法给出裕度", "gray"
    bad = []
    if pm is not None and pm <= 0:
        bad.append(f"相位裕度 {pm:.1f}° ≤ 0°")
    if gm is not None and gm <= 0:
        bad.append(f"幅值裕度 {gm:.1f} dB ≤ 0 dB")
    if bad:
        return "不稳定", "；".join(bad) + "，闭环不稳定", "red"

    warn = []
    if pm is not None and pm < 30:
        warn.append(f"相位裕度偏小（{pm:.1f}° < 30°）")
    if gm is not None and gm < 6:
        warn.append(f"幅值裕度偏小（{gm:.1f} dB < 6 dB）")
    if warn:
        return "临界", "；".join(warn) + "，超调大、抗扰差", "orange"

    good = (pm is not None and pm >= 45) and (gm is None or gm >= 10)
    if good:
        return "良好", "幅值/相位裕度充足，动态性能与鲁棒性折中较好", "green"
    return "合格", "稳定裕度满足基本要求（PM ≥ 30° 且 GM ≥ 6 dB）", "green"


def margin_advice(margins: dict) -> list:
    """由裕度反推参数调整方向。"""
    tips = []
    pm = margins.get("pm")
    gm = margins.get("gm_db")
    if pm is not None and pm < 30:
        tips.append("相位裕度偏小 → 适当**减小 Kp**（或增大 Kd 的微分超前作用），可使穿越频率左移、相位裕度回升")
    if pm is not None and pm > 70:
        tips.append("相位裕度很大（> 70°）→ 响应偏保守、上升慢，可**适当增大 Kp** 提高快速性")
    if gm is not None and gm < 6:
        tips.append("幅值裕度偏小 → 系统接近临界，**减小 Kp 或增大 Ki 的积分时间**（减小 Ki）更安全")
    if (gm is None) and (pm is not None) and pm > 0:
        tips.append("相位不穿越 −180° → 幅值裕度无穷大，对象特性好、稳定性充裕")
    if not tips:
        tips.append("当前裕度处于合理区间，可在超调量与调节时间之间继续微调")
    return tips
