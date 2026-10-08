# -*- coding: utf-8 -*-
"""二维参数扫描：在 (Kp,Ki) / (Kp,Kd) / (Ki,Kd) 平面上网格化计算性能指标。

用途：把 PID 参数与性能指标的关系画成热力图，直观看出「参数往哪调」，
并自动定位最优格点。扫描用的仿真点数可单独指定（默认比正常仿真少），
以控制网格面积 × 单次仿真 的总耗时。
"""
from __future__ import annotations

import numpy as np

from .control import PID, Actuator, simulate
from .metrics import compute_metrics, control_quality
from .models import LTIPlant

# 指标定义：(键, 中文名, 是否「越小越好」, 单位)
SWEEP_METRICS = [
    ("overshoot", "超调量 σ", True, "%"),
    ("ts_2", "调节时间 ts(2%)", True, "s"),
    ("iae", "误差积分 IAE", True, ""),
    ("score", "综合评分", False, "分"),
]

AXIS_NAMES = {"kp": "Kp（比例增益）", "ki": "Ki（积分增益）", "kd": "Kd（微分增益）"}


def default_ranges(kp: float, ki: float, kd: float, x_key: str, y_key: str,
                   ku: float | None = None):
    """给出默认扫描范围：以当前参数为基准，参考临界增益 Ku 适当扩展。"""
    scale = {"kp": max(kp, 1e-3), "ki": max(ki, 1e-3), "kd": max(kd, 1e-3)}
    if ku and np.isfinite(ku) and ku > 0:
        scale["kp"] = max(scale["kp"], 0.4 * ku)

    def rng(key):
        top = 2.5 * scale[key] if key != "kp" else 2.0 * scale[key]
        lo = 0.0
        if key == "kd":                     # Kd 从 0 起扫更有意义
            lo = 0.0
        return float(lo), float(top)

    return rng(x_key), rng(y_key)


def sweep_2d(plant: LTIPlant, *, x_key: str, y_key: str, x_vals, y_vals,
             fixed: dict, mode: str = "PID", t_end: float = 30.0,
             n_samples: int = 800, actuator: Actuator | None = None,
             noise_std: float = 0.0, seed: int = 0) -> dict:
    """在 (x_key, y_key) 平面上逐格仿真，返回各性能指标的二维网格。

    fixed 给出未参与扫描的那个增益（例如扫 Kp-Ki 时 fixed={"kd": 0.2}）。
    """
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    X, Y = np.meshgrid(x_vals, y_vals, indexing="xy")

    keys = [m[0] for m in SWEEP_METRICS] + ["tv", "ess", "ess_rel", "responded"]
    Z = {k: np.full(X.shape, np.nan) for k in keys}

    for i, yv in enumerate(y_vals):
        for j, xv in enumerate(x_vals):
            g = dict(fixed)
            g[x_key] = float(xv)
            g[y_key] = float(yv)
            pid = PID(kp=float(g.get("kp", 0.0)), ki=float(g.get("ki", 0.0)),
                      kd=float(g.get("kd", 0.0)), mode=mode)
            try:
                r = simulate(plant, pid, t_end=float(t_end), n_samples=int(n_samples),
                             actuator=actuator, noise_std=float(noise_std), seed=int(seed))
            except Exception:
                continue
            m = compute_metrics(r.t, r.y, r.r)
            for k in keys:
                Z[k][i, j] = float(m.get(k, np.nan))
            Z["tv"][i, j] = control_quality(r.u, float(t_end) / int(n_samples))["tv"]

    return {"x_key": x_key, "y_key": y_key, "x_vals": x_vals, "y_vals": y_vals,
            "X": X, "Y": Y, "Z": Z, "fixed": dict(fixed), "mode": mode}


def best_point(sw: dict, metric: str = "score"):
    """在网格中定位最优点（按指标方向自动判断大小）。"""
    z = sw["Z"][metric]
    lower_is_better = dict((m[0], m[2]) for m in SWEEP_METRICS).get(metric, True)
    good = np.isfinite(z)
    if not good.any():
        return None
    zz = np.where(good, z, np.inf if lower_is_better else -np.inf)
    idx = np.unravel_index(np.argmin(zz) if lower_is_better else np.argmax(zz), z.shape)
    i, j = int(idx[0]), int(idx[1])
    return {"x": float(sw["x_vals"][j]), "y": float(sw["y_vals"][i]),
            "value": float(z[i, j]), "i": i, "j": j}


def feasible_region(sw: dict, sigma_max: float = 20.0, ts_max: float | None = None):
    """满足约束（超调 ≤ sigma_max，调节时间 ≤ ts_max）的参数区域掩码。"""
    z_ov = sw["Z"]["overshoot"]
    ok = np.isfinite(z_ov) & (z_ov <= float(sigma_max))
    if ts_max is not None:
        z_ts = sw["Z"]["ts_2"]
        ok &= np.isfinite(z_ts) & (z_ts <= float(ts_max))
    return ok
