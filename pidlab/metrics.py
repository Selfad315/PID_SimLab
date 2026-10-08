# -*- coding: utf-8 -*-
"""控制系统性能指标计算：超调量、上升/峰值/调节时间、稳态误差、误差积分指标等。"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.signal import find_peaks

# numpy 1.x 用 trapz，numpy 2.x 改名 trapezoid
try:
    _trapz = np.trapezoid
except AttributeError:  # numpy < 2.0
    _trapz = np.trapz


def _steady_value(t: np.ndarray, y: np.ndarray, frac: float = 0.02) -> float:
    n = max(3, int(frac * y.size))
    return float(np.mean(y[-n:]))


def _cross_time(t: np.ndarray, y: np.ndarray, level: float) -> float:
    """信号 y 首次到达 level 的时刻（线性插值提高分辨率）。"""
    idx = np.argmax(y >= level)
    if y[idx] < level:
        return float("nan")
    if idx == 0:
        return float(t[0])
    y0, y1 = float(y[idx - 1]), float(y[idx])
    if abs(y1 - y0) < 1e-15:
        return float(t[idx])
    frac = (level - y0) / (y1 - y0)
    return float(t[idx - 1] + frac * (t[idx] - t[idx - 1]))


def compute_metrics(t: np.ndarray, y: np.ndarray, r: Optional[np.ndarray] = None,
                    band: float = 0.02, label: str = "") -> dict:
    """计算一段阶跃响应的全部性能指标。

    band: 调节时间的误差带（0.02 -> 2%，0.05 -> 5%）
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    r = np.full_like(y, y[-1] if r is None else 1.0) if r is None else np.asarray(r, float)

    y_ss = _steady_value(t, y)
    r_ss = _steady_value(t, r)
    span = float(t[-1] - t[0])

    out = {"label": label, "y_ss": y_ss, "r_ss": r_ss,
           "ess": r_ss - y_ss, "ess_rel": (r_ss - y_ss) / abs(r_ss) * 100 if abs(r_ss) > 1e-12 else np.nan}

    # ---- 发散/不稳定判定 ----
    finite = np.isfinite(y)
    if (not finite.all()) or np.max(np.abs(y)) > 1e6:
        out.update({"stable": False, "overshoot": np.nan, "undershoot": np.nan,
                    "tp": np.nan, "tr": np.nan, "tr_0_100": np.nan, "tr_full": np.nan, "ts_2": np.nan,
                    "ts_5": np.nan, "peak": np.nan, "iae": np.nan, "ise": np.nan,
                    "itae": np.nan, "decay_ratio": np.nan, "n_osc": 0,
                    "band_2": np.nan, "band_5": np.nan, "settled": False,
                    "score": 0.0, "responded": True,   # 不稳定方案评分为 0，供排序比较时使用
                    "rise_note": ""})
        return out

    # ---- 无响应判定 ----
    # 输出几乎不动时（例如控制器增益全为 0），σ / ts 会算出「0% / 0s」这种
    # 漂亮但完全错误的结果，必须单独识别并判为无效方案。
    span_y = float(np.max(y) - np.min(y))
    if abs(r_ss) > 1e-9 and span_y < 1e-4 * abs(r_ss):
        out.update({"stable": False, "responded": False, "overshoot": np.nan,
                    "undershoot": np.nan, "tp": np.nan, "tr": np.nan, "tr_0_100": np.nan,
                    "tr_full": np.nan, "ts_2": np.nan, "ts_5": np.nan, "peak": float(np.max(y)),
                    "iae": float(_trapz(np.abs(r - y), t)), "ise": float(_trapz((r - y) ** 2, t)),
                    "itae": float(_trapz(t * np.abs(r - y), t)), "decay_ratio": np.nan,
                    "n_osc": 0, "band_2": np.nan, "band_5": np.nan, "settled": False,
                    "score": 0.0, "rise_note": "系统几乎无响应，性能指标不适用"})
        return out

    base = y_ss
    if abs(base) < 1e-9:
        base = r_ss if abs(r_ss) > 1e-9 else 1.0

    y_max = float(np.max(y))
    y_min = float(np.min(y))
    out["peak"] = y_max
    out["overshoot"] = (y_max - y_ss) / abs(base) * 100.0
    out["undershoot"] = max(0.0, (y_ss - y_min) / abs(base) * 100.0)
    out["tp"] = float(t[int(np.argmax(y))])

    # ---- 上升时间 ----
    lv10, lv90 = 0.1 * base, 0.9 * base
    t10 = _cross_time(t, y, lv10)
    t90 = _cross_time(t, y, lv90)
    out["tr"] = t90 - t10 if np.isfinite(t10) and np.isfinite(t90) else np.nan
    # 经典 0->100% 上升时间：从起始值上升到首次到达稳态值(100%)所经历的时间
    t100 = _cross_time(t, y, y_ss)
    out["tr_full"] = (t100 - float(t[0])) if np.isfinite(t100) else np.nan
    out["tr_0_100"] = out["tr_full"]   # 兼容别名

    # ---- 调节时间（最后一次离开误差带） ----
    for b, key in ((0.02, "ts_2"), (0.05, "ts_5")):
        tol = b * abs(base)
        outside = np.abs(y - y_ss) > tol
        if not outside.any():
            out[key] = float(t[0])
            out["settled"] = True
        else:
            last = int(np.nonzero(outside)[0][-1])
            if last >= y.size - 1:
                out[key] = float("nan")
                out["settled"] = False
            else:
                out[key] = float(t[last + 1])
                out["settled"] = True
    out["band_2"], out["band_5"] = out["ts_2"], out["ts_5"]

    # ---- 误差积分指标 ----
    e = r - y
    out["iae"] = float(_trapz(np.abs(e), t))
    out["ise"] = float(_trapz(e ** 2, t))
    out["itae"] = float(_trapz(t * np.abs(e), t))

    # ---- 衰减比 / 振荡次数（衰减曲线法整定需要） ----
    resid = y - y_ss
    pk, _ = find_peaks(resid)
    if pk.size:
        h = resid[pk]
        positive = h[h > 0]
        if positive.size >= 2:
            out["decay_ratio"] = float(positive[1] / positive[0])
        elif positive.size == 1:
            out["decay_ratio"] = 0.0
        else:
            out["decay_ratio"] = np.nan
        out["n_osc"] = int(positive.size)
        pos_idx = [int(i) for i in np.nonzero(h > 0)[0]]
        out["peak_times"] = [float(t[pk[i]]) for i in pos_idx]
    else:
        out["decay_ratio"] = np.nan
        out["n_osc"] = 0
        out["peak_times"] = []

    # ---- 综合性能评分（0-100，越大越好；用于自动比较） ----
    out["score"] = _score(out)
    out["responded"] = True
    out["stable"] = bool(out["settled"] or (np.isfinite(out["ts_5"])))
    out["rise_note"] = ""
    return out


def _score(m: dict) -> float:
    """一个可解释的综合评分：超调、调节时间、稳态误差、积分误差加权。"""
    if not np.isfinite(m.get("overshoot", np.nan)):
        return 0.0
    s_ov = np.clip(100.0 - 1.5 * max(0.0, m["overshoot"]), 0, 100)
    ts = m.get("ts_2", np.nan)
    s_ts = 0.0 if not np.isfinite(ts) else np.clip(100.0 - 5.0 * ts, 0, 100)
    ess = abs(m.get("ess_rel", np.nan))
    s_ess = 0.0 if not np.isfinite(ess) else np.clip(100.0 - 5.0 * ess, 0, 100)
    iae = m.get("iae", np.nan)
    s_iae = 0.0 if not np.isfinite(iae) else np.clip(100.0 - 20.0 * iae, 0, 100)
    return float(0.35 * s_ov + 0.30 * s_ts + 0.20 * s_ess + 0.15 * s_iae)


# --------------------------------------------------------------------------- #
#  抗干扰指标
# --------------------------------------------------------------------------- #
def disturbance_metrics(t: np.ndarray, y: np.ndarray, y_ref: float, t_dist: float,
                        band: float = 0.02) -> dict:
    """扰动作用后的动态偏差与恢复时间。

    y_ref : 扰动前稳态值（理想值）
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    mask = t >= t_dist
    if not mask.any():
        return {"max_dev": np.nan, "max_dev_rel": np.nan, "recovery": np.nan,
                "peak_time": np.nan, "iae_dist": np.nan, "overshoot_after": np.nan}
    tt, yy = t[mask], y[mask]
    dev = yy - y_ref
    i = int(np.argmax(np.abs(dev)))
    tol = band * abs(y_ref) if abs(y_ref) > 1e-9 else band
    outside = np.abs(dev) > tol
    if not outside.any():
        recovery = 0.0
    else:
        last = int(np.nonzero(outside)[0][-1])
        recovery = float(tt[last + 1] - t_dist) if last + 1 < tt.size else float("nan")
    return {"max_dev": float(dev[i]),
            "max_dev_rel": float(abs(dev[i]) / abs(y_ref) * 100) if abs(y_ref) > 1e-9 else np.nan,
            "recovery": recovery,
            "peak_time": float(tt[i] - t_dist),
            "iae_dist": float(_trapz(np.abs(dev), tt)),
            "overshoot_after": float(max(0.0, np.max(yy) - y_ref) / abs(y_ref) * 100) if abs(y_ref) > 1e-9 else np.nan}


# --------------------------------------------------------------------------- #
#  表格整理
# --------------------------------------------------------------------------- #
DISPLAY_KEYS = [
    ("label", "方案"),
    ("overshoot", "超调量 σ%"),
    ("tr", "上升时间 tr(10-90%)"),
    ("tr_full", "上升时间 tr(0-100%)"),
    ("tp", "峰值时间 tp (s)"),
    ("ts_5", "调节时间 ts(5%)"),
    ("ts_2", "调节时间 ts(2%)"),
    ("ess", "稳态误差 ess"),
    ("ess_rel", "相对稳态误差 %"),
    ("iae", "IAE"),
    ("ise", "ISE"),
    ("itae", "ITAE"),
    ("decay_ratio", "衰减比 δ"),
    ("score", "综合评分"),
]


def metrics_table(rows: list) -> "object":
    """把 [dict, ...] 指标列表转成便于 st.dataframe 显示的 DataFrame。"""
    import pandas as pd
    data = []
    for m in rows:
        rec = {}
        for k, title in DISPLAY_KEYS:
            v = m.get(k, np.nan)
            if k == "label":
                rec[title] = v
            elif isinstance(v, (int, float, np.floating)):
                rec[title] = None if (isinstance(v, float) and not np.isfinite(v)) else round(float(v), 4)
            else:
                rec[title] = v
        data.append(rec)
    return pd.DataFrame(data)


# --------------------------------------------------------------------------- #
#  控制量品质指标（评价微分滤波 / 抗饱和等改进措施）
# --------------------------------------------------------------------------- #
def control_quality(u, dt: float | None = None) -> dict:
    """控制量品质：总变差 TV 衡量抖动/平滑度，另给峰值与有效值。

    TV = Σ|u(k+1) − u(k)| —— 信号越抖，TV 越大；
    不完全微分的作用正是把被噪声放大的高频抖动压下去，使 TV 显著下降。
    """
    u = np.asarray(u, dtype=float)
    if u.size < 2:
        return {"tv": np.nan, "tv_rate": np.nan, "u_max": np.nan, "u_rms": np.nan}
    tv = float(np.sum(np.abs(np.diff(u))))
    return {"tv": tv,
            "tv_rate": (tv / float(dt)) if dt else np.nan,
            "u_max": float(np.max(np.abs(u))),
            "u_rms": float(np.sqrt(np.mean(u ** 2)))}
