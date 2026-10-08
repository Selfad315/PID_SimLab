# -*- coding: utf-8 -*-
"""PID 控制器、执行器非线性与闭环时域仿真引擎。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from .models import SecondOrderPlant, c2d_zoh, poly_add, poly_mul

MODES = ("P", "PI", "PD", "PID")

# 抗积分饱和方案
AW_MODES = {
    "none": "无抗饱和",
    "back": "反算法（back-calculation）",
    "conditional": "条件积分（conditional integration）",
    "clamping": "积分限幅（clamping）",
}
MODE_NAMES = {"P": "P 纯比例", "PI": "PI 比例积分",
              "PD": "PD 比例微分", "PID": "PID 比例积分微分"}


# --------------------------------------------------------------------------- #
#  PID 控制器
# --------------------------------------------------------------------------- #
@dataclass
class PID:
    r"""理想 PID（并联式）：

        u(t) = Kp*e(t) + Ki*∫e dt + Kd*de_f/dt

    微分项默认"对量测微分"(derivative-on-measurement)以避免给定值突变
    引起的微分冲击(derivative kick)；微分信号经过一阶低通滤波 Tf = Td/N。
    """

    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    mode: str = "PID"
    N: float = 10.0        # 微分滤波系数（越大越接近理想微分）
    name: str = "PID"

    # ---------------- 改进措施 ----------------
    integral_separation: bool = False   # 积分分离：大偏差时切除积分，抑制超调
    sep_threshold: float = 0.5          # 积分分离阈值 ε，|e| > ε 时切除积分
    derivative_filter: bool = True      # True=不完全微分（一阶滤波）；False=完全微分（理想微分）

    def gains(self):
        """按工作模式返回实际生效的 (Kp, Ki, Kd)。"""
        m = self.mode.upper()
        kp = float(self.kp)
        ki = float(self.ki) if m in ("PI", "PID") else 0.0
        kd = float(self.kd) if m in ("PD", "PID") else 0.0
        return kp, ki, kd

    def tf(self):
        """理想 PID 的传递函数 C(s) = (Kd s^2 + Kp s + Ki)/s （微分不滤波，仅作频域分析）。"""
        kp, ki, kd = self.gains()
        num = np.array([kd, kp, ki], dtype=float)
        den = np.array([1.0, 0.0], dtype=float)
        # 去掉分子首部无意义的零，保持严格真/齐次形式
        while num.size > 1 and abs(num[0]) < 1e-15:
            num = num[1:]
        return num, den

    def describe(self) -> str:
        kp, ki, kd = self.gains()
        extra = []
        if self.integral_separation:
            extra.append(f"积分分离(ε={self.sep_threshold:g})")
        extra.append("不完全微分" if self.derivative_filter else "完全微分")
        return (f"{MODE_NAMES.get(self.mode.upper(), self.mode)}: "
                f"Kp={kp:g}, Ki={ki:g}, Kd={kd:g}  [{'/'.join(extra)}]")


# --------------------------------------------------------------------------- #
#  执行器：饱和 / 死区 / 速率限幅
# --------------------------------------------------------------------------- #
@dataclass
class Actuator:
    r"""执行器非线性模型。

    dead_zone : 死区半宽 delta，|u|<=delta 时输出 0，否则 sign(u)*(|u|-delta)
    u_min/max : 饱和上下限（enabled=True 时生效）
    rate_limit: 速率限幅 (单位/秒)，0 表示不限速
    """

    enabled: bool = False
    u_min: float = -10.0
    u_max: float = 10.0
    dead_zone: float = 0.0
    rate_limit: float = 0.0
    enabled_dead_zone: bool = False
    enabled_rate_limit: bool = False

    def apply(self, u: float, u_prev: float, dt: float) -> float:
        v = float(u)
        # 1) 死区（不灵敏区）
        if self.enabled_dead_zone and self.dead_zone > 0:
            dz = float(self.dead_zone)
            if abs(v) <= dz:
                v = 0.0
            else:
                v = np.sign(v) * (abs(v) - dz)
        # 2) 速率限幅
        if self.enabled_rate_limit and self.rate_limit > 0:
            dv_max = float(self.rate_limit) * dt
            v = float(np.clip(v, u_prev - dv_max, u_prev + dv_max))
        # 3) 饱和
        if self.enabled:
            v = float(np.clip(v, self.u_min, self.u_max))
        return v

    def is_active(self) -> bool:
        return bool(self.enabled or self.enabled_dead_zone or self.enabled_rate_limit)

    def label(self) -> str:
        parts = []
        if self.enabled:
            parts.append(f"饱和[{self.u_min:g},{self.u_max:g}]")
        if self.enabled_dead_zone:
            parts.append(f"死区±{self.dead_zone:g}")
        if self.enabled_rate_limit:
            parts.append(f"速率限幅{self.rate_limit:g}/s")
        return " + ".join(parts) if parts else "线性"


# --------------------------------------------------------------------------- #
#  扰动信号
# --------------------------------------------------------------------------- #
def step_disturbance(t0: float, magnitude: float = 0.2) -> Callable:
    f = lambda t: np.where(np.asarray(t) >= t0, magnitude, 0.0) * np.ones_like(np.asarray(t, float))
    f.descriptor = ("step", float(t0), float(magnitude))     # 供上层缓存使用
    return f


def pulse_disturbance(t0: float, magnitude: float = 0.5, width: float = 1.0) -> Callable:
    def f(t):
        t = np.asarray(t, float)
        return np.where((t >= t0) & (t < t0 + width), magnitude, 0.0)
    f.descriptor = ("pulse", float(t0), float(magnitude), float(width))
    return f


def sine_disturbance(t0: float, magnitude: float = 0.2, freq: float = 1.0,
                     duration: float = 10.0) -> Callable:
    def f(t):
        t = np.asarray(t, float)
        s = np.sin(2 * np.pi * freq * (t - t0))
        return np.where((t >= t0) & (t < t0 + duration), magnitude * s, 0.0)
    f.descriptor = ("sine", float(t0), float(magnitude), float(freq), float(duration))
    return f


def ramp_disturbance(t0: float, slope: float = 0.1) -> Callable:
    def f(t):
        t = np.asarray(t, float)
        return np.where(t >= t0, slope * (t - t0), 0.0)
    f.descriptor = ("ramp", float(t0), float(slope))
    return f


DISTURBANCE_TYPES = {
    "阶跃负载扰动": "step",
    "脉冲扰动": "pulse",
    "正弦扰动": "sine",
    "斜坡扰动": "ramp",
}


# --------------------------------------------------------------------------- #
#  仿真结果容器
# --------------------------------------------------------------------------- #
@dataclass
class SimResult:
    t: np.ndarray
    r: np.ndarray
    y: np.ndarray
    u: np.ndarray            # 执行器输出（进对象前）
    e: np.ndarray            # 误差 e = r - y
    u_raw: np.ndarray        # 控制器未限幅输出
    dist: np.ndarray         # 扰动信号
    plant_input: np.ndarray  # 真正进入对象的状态输入
    y_meas: Optional[np.ndarray] = None   # 控制器实际看到的量测值（含噪声）
    pid: Optional[PID] = None
    plant: Optional[SecondOrderPlant] = None
    actuator: Optional[Actuator] = None
    label: str = ""

    def to_dataframe(self, prefix: str = "") -> pd.DataFrame:
        df = pd.DataFrame({
            "t": self.t, "r": self.r, "y": self.y, "e": self.e,
            "u": self.u, "u_raw": self.u_raw,
            "y_meas": self.y_meas if self.y_meas is not None else self.y,
            "dist": self.dist, "plant_input": self.plant_input,
        })
        if prefix:
            df = df.rename(columns={c: f"{prefix}_{c}" for c in df.columns if c != "t"})
        return df


# --------------------------------------------------------------------------- #
#  闭环仿真主循环
# --------------------------------------------------------------------------- #
def simulate(plant: SecondOrderPlant, pid: PID, t_end: float = 20.0, *,
             dt: Optional[float] = None, n_samples: int = 3000,
             ref: float = 1.0, disturbance: Optional[Callable] = None,
             actuator: Optional[Actuator] = None, anti_windup: bool = True,
             anti_windup_mode: str = "back",
             derivative_on_measurement: bool = True,
             noise_std: float = 0.0, seed: int = 0,
             x0: Optional[Sequence[float]] = None,
             label: str = "") -> SimResult:
    """闭环时域仿真。

    采用"离散化 + 状态推进"结构：
      e -> PID -> 执行器(饱和/死区/限速) -> [纯滞后] -> 对象 -> y
    纯滞后用历史缓冲 + 线性插值实现，精度与固定步长同阶。
    """
    t_end = float(t_end)
    dt = float(dt) if dt else t_end / int(n_samples)
    n_steps = int(round(t_end / dt))
    t = np.arange(n_steps + 1) * dt

    A, B, C, D = plant.ss()
    n = A.shape[0]
    Ad, Bd = c2d_zoh(A, B, dt)

    r = np.full(t.size, float(ref))
    dist = np.zeros(t.size) if disturbance is None else np.asarray(disturbance(t), float) * np.ones(t.size)

    y = np.zeros(t.size)
    e = np.zeros(t.size)
    u = np.zeros(t.size)
    u_raw_arr = np.zeros(t.size)
    p_in = np.zeros(t.size)
    v_hist = np.zeros(t.size)      # 控制器（执行器后）输出历史，用于滞后

    x = np.zeros(n) if x0 is None else np.asarray(x0, float).copy()

    kp, ki, kd = pid.gains()
    integral = 0.0
    d_filt = 0.0
    e_prev = 0.0
    y_prev = 0.0
    v_prev = 0.0
    p_prev = 0.0

    delay_steps = plant.delay / dt if dt > 0 else 0.0

    # ---- 微分形式：不完全微分 Tf=Td/N；完全微分 Tf=0（alpha=1，不做滤波）----
    if pid.derivative_filter and kd > 0 and kp > 1e-12 and pid.N > 0:
        Tf = (kd / kp) / pid.N
    else:
        Tf = 0.0
    alpha = 1.0 if Tf <= 1e-15 else dt / (Tf + dt)

    # ---- 量测噪声：控制器看到的是 y + noise，指标仍按真实 y 计算 ----
    rng = np.random.default_rng(int(seed))
    noise = rng.normal(0.0, float(noise_std), t.size) if noise_std > 0 else np.zeros(t.size)
    y_meas = np.zeros(t.size)

    # ---- 抗积分饱和模式 ----
    aw = "none" if not anti_windup else str(anti_windup_mode)
    sep_steps = 0        # 统计积分被切除的步数

    for k in range(t.size):
        # 1) 真实输出 + 含噪声的量测值
        yk = float((C @ x).item() + D[0, 0] * p_prev)
        y[k] = yk
        yk_m = yk + noise[k]
        y_meas[k] = yk_m
        ek = r[k] - yk_m
        e[k] = ek

        # 2) 积分分离：|e| > ε 时切除积分项，同时冻结积分器（避免恢复瞬间突跳）
        sep = bool(pid.integral_separation) and (abs(ek) > float(pid.sep_threshold))
        if sep:
            sep_steps += 1
        integral_prev = integral
        if not sep:
            integral += ek * dt

        # 3) 微分项（完全 / 不完全微分）
        if k == 0:
            raw_d = 0.0
        elif derivative_on_measurement:
            raw_d = -(yk_m - y_prev) / dt    # 对量测微分，避免给定值跳变的微分冲击
        else:
            raw_d = (ek - e_prev) / dt
        d_filt = raw_d if alpha >= 1.0 else (1.0 - alpha) * d_filt + alpha * raw_d

        # 4) 控制器输出（积分分离生效时积分项置零）
        u_raw = kp * ek + (0.0 if sep else ki * integral) + kd * d_filt

        # 5) 执行器非线性（死区 / 速率限幅 / 饱和）
        uk = u_raw if actuator is None else actuator.apply(u_raw, v_prev, dt)

        # 6) 抗积分饱和：三种方案都只修正"积分器状态"（供后续步使用），
        #    本步输出仍取执行器实际输出 uk。
        u_sat = u_raw
        if actuator is not None and actuator.enabled:
            u_sat = float(np.clip(u_raw, actuator.u_min, actuator.u_max))
        sat_diff = u_sat - u_raw

        if ki > 1e-12 and abs(sat_diff) > 1e-12 and not sep:
            if aw == "back":
                # 反算法：I ← I + (u_sat − u_raw)/Ki
                integral += sat_diff / ki
            elif aw == "conditional":
                # 条件积分：输出已饱和、且误差仍在把输出往同一方向推时，撤销本步积分累加。
                #   sat_diff < 0 → 正向饱和（u_raw > u_max），此时 ek > 0 会继续推高积分 → 停积分
                #   sat_diff > 0 → 反向饱和（u_raw < u_min），此时 ek < 0 会继续压低积分 → 停积分
                if (sat_diff < 0 and ek > 0) or (sat_diff > 0 and ek < 0):
                    integral = integral_prev
            elif aw == "clamping" and actuator is not None:
                # 积分限幅：把积分贡献限制在执行器量程内
                integral = float(np.clip(integral, actuator.u_min / ki, actuator.u_max / ki))

        u[k] = uk
        u_raw_arr[k] = u_raw
        v_hist[k] = uk

        # 7) 纯滞后：线性插值取 u(t - tau)
        if delay_steps <= 0:
            v_delayed = uk
        else:
            pos = k - delay_steps
            if pos <= 0:
                v_delayed = 0.0
            else:
                i0 = int(np.floor(pos))
                frac = pos - i0
                if i0 >= t.size - 1:
                    v_delayed = v_hist[-1]
                else:
                    v_delayed = v_hist[i0] * (1 - frac) + v_hist[i0 + 1] * frac

        # 8) 对象输入 = 控制器输出(滞后) + 负载扰动
        pk = float(v_delayed + dist[k])
        p_in[k] = pk

        # 9) 状态推进（ZOH 精确离散）
        x = (Ad @ x + (Bd[:, :1] @ np.array([[pk]]))[:, 0])

        e_prev = ek
        y_prev = yk_m          # 微分对量测微分，须用含噪的量测值
        v_prev = uk
        p_prev = pk

    # 末尾补一次输出
    y[-1] = float((C @ x).item() + D[0, 0] * p_prev)

    return SimResult(t=t, r=r, y=y, u=u, e=e, u_raw=u_raw_arr, dist=dist,
                     plant_input=p_in, y_meas=y_meas, pid=pid, plant=plant,
                     actuator=actuator, label=label or pid.describe())


def simulate_open_loop(plant: SecondOrderPlant, t_end: float = 20.0, *,
                       dt: Optional[float] = None, n_samples: int = 3000,
                       amplitude: float = 1.0) -> SimResult:
    """开环单位阶跃响应（第 1 模块"基础二阶系统"用）。"""
    dt = float(dt) if dt else t_end / int(n_samples)
    n_steps = int(round(t_end / dt))
    t = np.arange(n_steps + 1) * dt
    A, B, C, D = plant.ss()
    Ad, Bd = c2d_zoh(A, B, dt)
    x = np.zeros(A.shape[0])
    y = np.zeros(t.size)
    p = np.zeros(t.size)
    for k in range(t.size):
        y[k] = float((C @ x).item() + D[0, 0] * (p[k - 1] if k else 0.0))
        # 纯滞后：阶跃输入延迟 tau 施加，等价于响应曲线整体右移 tau
        pk = amplitude if t[k] >= plant.delay else 0.0
        p[k] = pk
        x = Ad @ x + (Bd[:, :1] @ np.array([[pk]]))[:, 0]
    y[-1] = float((C @ x).item() + D[0, 0] * p[-1])
    r = np.full(t.size, amplitude)
    return SimResult(t=t, r=r, y=y, u=p.copy(), e=r - y, u_raw=p.copy(),
                     dist=np.zeros(t.size), plant_input=p, pid=None, plant=plant,
                     actuator=None, label="开环阶跃响应")


# --------------------------------------------------------------------------- #
#  线性闭环传递函数（用于频域分析 / 稳定裕度）
# --------------------------------------------------------------------------- #
def open_loop_tf(plant: SecondOrderPlant, pid: PID):
    """开环传递函数 L(s) = C(s)G(s) 的 (num, den)。"""
    nc, dc = pid.tf()
    ng, dg = plant.num, plant.den
    return poly_mul(nc, ng), poly_mul(dc, dg)


def closed_loop_tf(plant: SecondOrderPlant, pid: PID):
    """单位负反馈闭环传递函数 T(s) = L/(1+L) 的 (num, den)。"""
    nl, dl = open_loop_tf(plant, pid)
    num = nl
    den = poly_add(dl, nl)
    return num, den
