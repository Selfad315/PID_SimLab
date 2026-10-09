# -*- coding: utf-8 -*-
"""PID 参数整定算法：

1. 试凑法（人工整定，界面提供滑杆 + 参考初值）
2. Ziegler-Nichols 法
   - 临界比例度法（闭环振荡法）：由频率特性求临界增益 Ku、临界周期 Pu
   - 阶跃响应法（开环反应曲线法）：由阶跃响应辨识 FOPDT 模型 (K, T, L)
3. 衰减曲线法（4:1 / 10:1 衰减比，数值搜索满足衰减比的纯比例增益 Kc）
4. 附加：基于数值寻优的多目标自动整定（对照组 / 参考最优）

所有经验公式都在界面上以"整定公式卡"形式展示，便于论文与答辩引用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import brentq, minimize

from .control import PID, simulate, simulate_open_loop
from .metrics import compute_metrics
from .models import LTIPlant, SecondOrderPlant

# --------------------------------------------------------------------------- #
#  整定公式表（一页纸，可直接抄进论文）
# --------------------------------------------------------------------------- #
ZN_TABLE = {
    "P":   {"kp": "0.5 Ku",          "ti": "—",       "td": "—"},
    "PI":  {"kp": "0.45 Ku",         "ti": "Pu/1.2",  "td": "—"},
    "PD":  {"kp": "0.8 Ku",          "ti": "—",       "td": "Pu/8"},
    "PID": {"kp": "0.6 Ku",          "ti": "Pu/2",    "td": "Pu/8"},
}
ZN_OPEN_TABLE = {
    "P":   {"kp": "T/(K·L)",         "ti": "—",       "td": "—"},
    "PI":  {"kp": "0.9·T/(K·L)",     "ti": "L/0.3",   "td": "—"},
    "PD":  {"kp": "1.2·T/(K·L)",     "ti": "—",       "td": "0.5L"},
    "PID": {"kp": "1.2·T/(K·L)",     "ti": "2L",      "td": "0.5L"},
}
DECAY_TABLE_41 = {
    "P":   {"kp": "Kc",              "ti": "—",       "td": "—"},
    "PI":  {"kp": "0.8 Kc",          "ti": "0.5 Ts",  "td": "—"},
    "PD":  {"kp": "0.8 Kc",          "ti": "—",       "td": "0.1 Ts"},
    "PID": {"kp": "0.8 Kc",          "ti": "0.3 Ts",  "td": "0.1 Ts"},
}
DECAY_TABLE_101 = {
    "P":   {"kp": "Kc",              "ti": "—",       "td": "—"},
    "PI":  {"kp": "1.25 Kc",         "ti": "2 Ts",    "td": "—"},
    "PD":  {"kp": "1.25 Kc",         "ti": "—",       "td": "0.2 Ts"},
    "PID": {"kp": "1.25 Kc",         "ti": "0.8 Ts",  "td": "0.2 Ts"},
}


@dataclass
class TuningResult:
    method: str
    mode: str
    kp: float
    ki: float = 0.0
    kd: float = 0.0
    ti: float = float("nan")
    td: float = float("nan")
    valid: bool = True
    note: str = ""
    ku: float = float("nan")
    pu: float = float("nan")
    kc: float = float("nan")
    ts_osc: float = float("nan")
    extra: dict = field(default_factory=dict)

    def to_pid(self, name: Optional[str] = None, N: float = 10.0) -> PID:
        return PID(kp=self.kp, ki=self.ki, kd=self.kd, mode=self.mode,
                   N=N, name=name or self.method)

    def as_row(self) -> dict:
        return {"整定方法": self.method, "控制模式": self.mode,
                "Kp": round(self.kp, 4) if self.valid else None,
                "Ki": round(self.ki, 4) if self.valid else None,
                "Kd": round(self.kd, 4) if self.valid else None,
                "Ti": round(self.ti, 4) if np.isfinite(self.ti) else None,
                "Td": round(self.td, 4) if np.isfinite(self.td) else None,
                "状态": "成功" if self.valid else "不适用",
                "说明": self.note}


def _freq_scale(plant: LTIPlant) -> float:
    """由极点模长给出该对象的特征频率尺度（用于自动选择频率网格与仿真时长）。"""
    mags = np.abs(plant.poles())
    mags = mags[np.isfinite(mags) & (mags > 1e-9)]
    return float(np.min(mags)) if mags.size else 1.0


def _default_horizon(plant: LTIPlant) -> float:
    """自动仿真时长：按最慢极点的时间常数估计，含积分环节时额外延长。"""
    re = np.abs(np.real(plant.poles()))
    nz = re[re > 1e-9]
    base = 8.0 / float(np.min(nz)) if nz.size else 20.0
    if np.any(re <= 1e-9):
        base = max(base * 3.0, 30.0)          # 含积分环节，观测窗口要更长
    return float(np.clip(base + 6.0 * plant.delay, 5.0, 600.0))


# --------------------------------------------------------------------------- #
#  频域：临界点 (Ku, Pu)
# --------------------------------------------------------------------------- #
def phase_crossover(plant: LTIPlant, n: int = 6000):
    """求开环频率特性相角穿越 -180° 处的频率 wu、临界增益 Ku、临界周期 Pu。

    返回 (wu, Ku, Pu, w, phase_deg, mag_db)；若不存在穿越点则返回 None。
    """
    ref = max(_freq_scale(plant), 1e-6)
    w = np.logspace(np.log10(max(ref / 1e4, 1e-6)), np.log10(ref * 1e4), n)
    g = plant.freqresp(w)
    phase = np.degrees(np.unwrap(np.angle(g)))
    mag_db = 20 * np.log10(np.maximum(np.abs(g), 1e-300))

    target = -180.0
    d = phase - target
    idx = np.nonzero(np.diff(np.sign(d)) != 0)[0]
    for i in idx:
        wa, wb = w[i], w[i + 1]
        ws = np.linspace(wa, wb, 80)
        phs = np.unwrap(np.angle(plant.freqresp(ws)))
        if (phs[0] - np.radians(target)) * (phs[-1] - np.radians(target)) <= 0:
            wu = float(np.interp(np.radians(target), -phs, ws))
            Ku = float(1.0 / abs(plant.freqresp(np.array([wu]))[0]))
            Pu = float(2 * np.pi / wu)
            return wu, Ku, Pu, w, phase, mag_db
    return None


def zn_closed_loop(plant: LTIPlant, mode: str = "PID") -> TuningResult:
    """Z-N 临界比例度法（闭环振荡法）。"""
    cr = phase_crossover(plant)
    if cr is None:
        return TuningResult("ZN 临界比例度法", mode, kp=0, valid=False,
                            note="开环相角未穿越 -180°，不存在临界增益 Ku（纯二阶无滞后系统即属于此类）。"
                                 "请给对象加入纯滞后 τ>0 或提高系统阶数后重试。")
    wu, Ku, Pu, *_ = cr
    rule = ZN_TABLE[mode]
    if mode == "P":
        kp, ti, td = 0.5 * Ku, np.nan, np.nan
    elif mode == "PI":
        kp, ti, td = 0.45 * Ku, Pu / 1.2, np.nan
    elif mode == "PD":
        kp, ti, td = 0.8 * Ku, np.nan, Pu / 8.0
    else:
        kp, ti, td = 0.6 * Ku, Pu / 2.0, Pu / 8.0
    ki = kp / ti if np.isfinite(ti) and ti > 0 else 0.0
    kd = kp * td if np.isfinite(td) else 0.0
    return TuningResult("ZN 临界比例度法", mode, kp, ki, kd, ti, td,
                        ku=Ku, pu=Pu,
                        note=f"wu={wu:.4g} rad/s, Ku={Ku:.4g}, Pu={Pu:.4g} s；{rule}",
                        extra={"wu": wu, "crossover": cr})


# --------------------------------------------------------------------------- #
#  开环反应曲线法：FOPDT 辨识
# --------------------------------------------------------------------------- #
def fopdt_from_step(plant: LTIPlant, t_end: Optional[float] = None,
                    n_samples: int = 4000) -> dict:
    """由阶跃响应切线法辨识一阶惯性加纯滞后模型 K/(Ts+1)·e^{-Ls}。

    做法：找到阶跃响应最大斜率点(拐点)作切线，
          切线与时间轴交点即纯滞后 L，与稳态值交点得时间常数 T。
    """
    t_end = t_end or _default_horizon(plant)
    res = simulate_open_loop(plant, t_end=t_end, n_samples=n_samples)
    t, y = res.t, res.y
    dy = np.gradient(y, t)
    i = int(np.argmax(dy))
    S = float(dy[i])
    y_i = float(y[i])
    t_i = float(t[i])
    K = float(np.mean(y[-max(3, n_samples // 50):]))
    if S <= 1e-12:
        return {"valid": False, "K": K, "T": np.nan, "L": np.nan,
                "t": t, "y": y, "t_i": t_i, "y_i": y_i, "S": S}
    L = t_i - y_i / S
    T = (K - y_i) / S
    L = max(0.0, L)
    return {"valid": bool(T > 0 and K != 0), "K": K, "T": float(T), "L": float(L),
            "t": t, "y": y, "t_i": t_i, "y_i": y_i, "S": S}


def zn_open_loop(plant: LTIPlant, mode: str = "PID") -> TuningResult:
    """Z-N 阶跃响应法（开环反应曲线法）。"""
    m = fopdt_from_step(plant)
    if not m["valid"]:
        return TuningResult("ZN 阶跃响应法", mode, kp=0, valid=False,
                            note="阶跃响应切线法辨识失败（响应无单调上升段或增益为 0）。")
    K, T, L = m["K"], m["T"], m["L"]
    if L <= 1e-6:
        return TuningResult("ZN 阶跃响应法", mode, kp=0, valid=False,
                            note="辨识得到的纯滞后 L≈0，反应曲线法公式 T/(K·L) 发散；"
                                 "请给对象加入少量纯滞后 τ>0。",
                            extra={"fopdt": m})
    if mode == "P":
        kp, ti, td = T / (K * L), np.nan, np.nan
    elif mode == "PI":
        kp, ti, td = 0.9 * T / (K * L), L / 0.3, np.nan
    elif mode == "PD":
        kp, ti, td = 1.2 * T / (K * L), np.nan, 0.5 * L
    else:
        kp, ti, td = 1.2 * T / (K * L), 2 * L, 0.5 * L
    ki = kp / ti if np.isfinite(ti) and ti > 0 else 0.0
    kd = kp * td if np.isfinite(td) else 0.0
    return TuningResult("ZN 阶跃响应法", mode, kp, ki, kd, ti, td,
                        note=f"FOPDT 辨识：K={K:.4g}, T={T:.4g} s, L={L:.4g} s；{ZN_OPEN_TABLE[mode]}",
                        extra={"fopdt": m})


# --------------------------------------------------------------------------- #
#  衰减曲线法
# --------------------------------------------------------------------------- #
def decay_curve(plant: LTIPlant, mode: str = "PID", ratio: float = 0.25,
                t_end: Optional[float] = None, n_samples: int = 2600,
                kc_lo: float = 1e-3, kc_hi: float = 500.0) -> TuningResult:
    """衰减曲线法：数值搜索使闭环纯比例控制达到指定衰减比的 Kc，再套经验公式。"""
    t_end = t_end or _default_horizon(plant)
    label = "4:1 衰减曲线法" if abs(ratio - 0.25) < 1e-9 else (
        "10:1 衰减曲线法" if abs(ratio - 0.1) < 1e-9 else f"衰减比 {ratio:g} 曲线法")

    def sim_at(kc: float):
        pid = PID(kp=kc, ki=0.0, kd=0.0, mode="P")
        res = simulate(plant, pid, t_end=t_end, n_samples=n_samples)
        m = compute_metrics(res.t, res.y, res.r, label=f"Kc={kc:.4g}")
        return m, res

    def f(kc: float) -> float:
        m, _ = sim_at(kc)
        d = m.get("decay_ratio", np.nan)
        if not np.isfinite(d):
            return 2.0 - ratio
        return float(d - ratio)

    grid = np.logspace(np.log10(kc_lo), np.log10(kc_hi), 24)
    vals = np.array([f(k) for k in grid])
    bracket = None
    for i in range(len(grid) - 1):
        if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i] * vals[i + 1] <= 0:
            bracket = (grid[i], grid[i + 1])
            break

    if bracket is not None:
        try:
            kc = float(brentq(f, bracket[0], bracket[1], xtol=1e-4, rtol=1e-6, maxiter=60))
            mode_note = ""
        except Exception as exc:      # pragma: no cover
            kc = float(grid[int(np.argmin(np.abs(vals)))])
            mode_note = f"（二分失败，改用网格近似：{exc}）"
    else:
        j = int(np.argmin(np.abs(vals)))
        kc = float(grid[j])
        mode_note = "（未找到穿越点，取最接近目标衰减比的网格增益）"

    m, res = sim_at(kc)
    pk = m.get("peak_times", [])
    if len(pk) >= 2:
        Ts = float(pk[1] - pk[0])
    elif len(pk) == 1:
        Ts = float(pk[0])
    else:
        Ts = float("nan")

    table = DECAY_TABLE_101 if abs(ratio - 0.1) < 1e-9 else DECAY_TABLE_41
    if mode == "P":
        kp, ti, td = kc, np.nan, np.nan
    elif mode == "PI":
        kp, ti, td = (1.25 * kc if ratio < 0.2 else 0.8 * kc,
                      (2.0 * Ts if ratio < 0.2 else 0.5 * Ts), np.nan)
    elif mode == "PD":
        kp, ti, td = (1.25 * kc if ratio < 0.2 else 0.8 * kc, np.nan,
                      (0.2 * Ts if ratio < 0.2 else 0.1 * Ts))
    else:
        kp, ti, td = (1.25 * kc if ratio < 0.2 else 0.8 * kc,
                      (0.8 * Ts if ratio < 0.2 else 0.3 * Ts),
                      (0.2 * Ts if ratio < 0.2 else 0.1 * Ts))
    ki = kp / ti if np.isfinite(ti) and ti > 0 else 0.0
    kd = kp * td if np.isfinite(td) else 0.0

    achieved = m.get("decay_ratio", np.nan)
    return TuningResult(label, mode, kp, ki, kd, ti, td,
                        kc=kc, ts_osc=Ts,
                        note=(f"搜索到 Kc={kc:.4g}，实测衰减比 δ={achieved:.3f}（目标 {ratio:g}），"
                              f"振荡周期 Ts={Ts:.4g} s；{table[mode]}{mode_note}"),
                        extra={"decay_ratio_achieved": achieved, "table": table,
                               "probe": res, "metrics": m})


# --------------------------------------------------------------------------- #
#  数值寻优自动整定（对照组）
# --------------------------------------------------------------------------- #
DEFAULT_WEIGHTS = {"overshoot": 1.0, "ts": 1.0, "ess": 6.0, "iae": 1.0, "control": 0.03}


def auto_optimize(plant: LTIPlant, mode: str = "PID",
                  weights: Optional[dict] = None, x0: Optional[tuple] = None,
                  t_end: Optional[float] = None, n_samples: int = 700,
                  maxiter: int = 70) -> TuningResult:
    """多目标数值寻优：J = w1·σ% + w2·ts + w3·|ess| + w4·IAE + w5·Σu²。

    以对数增益为优化变量保证正定性，采用 Nelder-Mead 单纯形法。
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    t_end = t_end or _default_horizon(plant)

    if x0 is None:
        zn = zn_closed_loop(plant, mode)
        if zn.valid:
            x0 = (zn.kp, zn.ki, zn.kd)
        else:
            x0 = (1.0, 1.0, 0.1)

    active = {"P": (0,), "PI": (0, 1), "PD": (0, 2), "PID": (0, 1, 2)}[mode]
    base = np.array([max(float(v), 1e-4) for v in x0], float)

    def unpack(x):
        g = base.copy()
        g[list(active)] = np.exp(x)
        return float(g[0]), float(g[1]), float(g[2])

    def cost(x):
        kp, ki, kd = unpack(x)
        if not (1e-4 < kp < 1e5) or ki < 0 or kd < 0 or ki > 1e5 or kd > 1e5:
            return 1e6
        pid = PID(kp=kp, ki=ki, kd=kd, mode=mode)
        try:
            res = simulate(plant, pid, t_end=t_end, n_samples=n_samples)
        except Exception:
            return 1e6
        m = compute_metrics(res.t, res.y, res.r)
        if not np.isfinite(m.get("overshoot", np.nan)):
            return 1e6
        ts = m.get("ts_2", np.nan)
        ts_pen = 10.0 if not np.isfinite(ts) else min(ts, 50.0)
        j = (w["overshoot"] * max(0.0, m["overshoot"]) / 100.0
             + w["ts"] * ts_pen / 10.0
             + w["ess"] * min(abs(m["ess"]), 5.0)
             + w["iae"] * min(m["iae"], 200.0) / 10.0
             + w["control"] * min(float(np.max(np.abs(res.u))), 1e3) / 10.0)
        if not np.isfinite(j):
            return 1e6
        return float(j)

    x_start = np.log(base[list(active)])
    try:
        opt = minimize(cost, x_start, method="Nelder-Mead",
                       options={"maxiter": maxiter, "xatol": 5e-3, "fatol": 5e-4})
        kp, ki, kd = unpack(opt.x)
        ok = np.isfinite(opt.fun)
    except Exception as exc:      # pragma: no cover
        kp, ki, kd = base
        ok = False
        opt = None

    pid = PID(kp=kp, ki=ki, kd=kd, mode=mode)
    res = simulate(plant, pid, t_end=t_end, n_samples=3000)
    m = compute_metrics(res.t, res.y, res.r)
    return TuningResult("数值寻优法", mode, kp, ki, kd,
                        ti=kp / ki if ki > 1e-12 else np.nan,
                        td=kd / kp if kp > 1e-12 else np.nan,
                        valid=bool(ok),
                        note=f"多目标加权寻优（权重 {w}），目标函数 J={opt.fun:.4f}" if opt is not None else "寻优失败",
                        extra={"metrics": m, "result": res,
                               "fun": float(opt.fun) if opt is not None else np.nan})


# --------------------------------------------------------------------------- #
#  统一入口：一次算齐所有方法（第 3 模块"自动对比"用）
# --------------------------------------------------------------------------- #
def tune_all(plant: LTIPlant, mode: str = "PID", ratio: float = 0.25,
             include_optimize: bool = True, kc_hi: float = 500.0) -> list:
    out = [
        zn_closed_loop(plant, mode),
        zn_open_loop(plant, mode),
        decay_curve(plant, mode, ratio=ratio, kc_hi=kc_hi),
    ]
    if include_optimize:
        try:
            out.append(auto_optimize(plant, mode))
        except Exception as exc:      # pragma: no cover
            out.append(TuningResult("数值寻优法", mode, 0, valid=False,
                                    note=f"寻优失败：{exc}"))
    return out


def evaluate_tuning(plant: LTIPlant, result: TuningResult,
                    t_end: Optional[float] = None, n_samples: int = 3000):
    """对某组整定参数做一次仿真并返回 (SimResult, 指标 dict)。"""
    t_end = t_end or _default_horizon(plant)
    pid = result.to_pid()
    res = simulate(plant, pid, t_end=t_end, n_samples=n_samples)
    m = compute_metrics(res.t, res.y, res.r, label=result.method)
    return res, m
