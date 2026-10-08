# -*- coding: utf-8 -*-
"""PID_SimLab —— 传递函数 / 状态空间 / 二阶系统建模工具。

设计原则
--------
1. 只依赖 numpy + scipy，不依赖 python-control，答辩现场 Anaconda 开箱即用；
2. 传递函数统一转成"可控标准型"状态空间，再对矩阵指数做零阶保持离散化，
   这样时域仿真在数学上是精确的（对线性对象而言无离散化误差）；
3. 全部采用 SI 无关的归一化单位，便于任意调节时间尺度。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.linalg import expm


# --------------------------------------------------------------------------- #
#  传递函数 -> 状态空间
# --------------------------------------------------------------------------- #
def tf_to_ss(num: Sequence[float], den: Sequence[float]):
    """连续传递函数 G(s)=num(s)/den(s) -> 可控标准型 (A, B, C, D)。

    参数均为"降幂"系数，例如 G(s)=10/(s^2+2s+10) 写作 num=[10], den=[1,2,10]。
    """
    den = np.asarray(den, dtype=float).ravel()
    num = np.asarray(num, dtype=float).ravel()
    if den.size < 2:
        raise ValueError("分母阶次至少为 1（至少一阶系统）")
    if abs(den[0]) < 1e-300:
        raise ValueError("分母首项系数不能为 0")

    a0 = den[0]
    den = den / a0
    num = num / a0
    n = den.size - 1  # 系统阶次

    if num.size > n + 1:
        raise ValueError("分子阶次高于分母，属于非真系统，暂不支持")

    # 右对齐补零，使 num_full 长度 = n+1，与 s^n ... s^0 对齐
    num_full = np.zeros(n + 1)
    num_full[n + 1 - num.size:] = num

    d = float(num_full[0])       # 直接传递项 D
    num_sp = num_full.copy()
    num_sp[0] = 0.0
    if abs(d) > 0.0:
        num_sp = num_sp - d * den  # 分离出严格真部分

    beta = num_sp[1:]                 # s^{n-1} ... s^0 的系数（降幂）
    C = beta[::-1].reshape(1, -1)     # 升幂排列后即为输出矩阵

    A = np.zeros((n, n))
    if n > 1:
        A[:-1, 1:] = np.eye(n - 1)
    A[-1, :] = -den[1:][::-1]         # 末行 = [-a_n ... -a_1]

    B = np.zeros((n, 1))
    B[-1, 0] = 1.0

    return A, B, C, np.array([[d]])


def c2d_zoh(A: np.ndarray, B: np.ndarray, dt: float):
    """零阶保持（ZOH）精确离散化，返回 (Ad, Bd)。

    利用增广矩阵 [[A,B],[0,0]] 的矩阵指数一次算得，避免求逆：
        Ad = e^{A dt},  Bd = A^{-1}(Ad - I) B
    """
    n = A.shape[0]
    m = B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    Md = expm(M * dt)
    return Md[:n, :n], Md[:n, n:]


def freqresp(num: Sequence[float], den: Sequence[float], w) -> np.ndarray:
    """频率响应 G(jw)，w 单位 rad/s。"""
    w = np.asarray(w, dtype=float)
    jw = 1j * w
    return np.polyval(np.asarray(num, float), jw) / np.polyval(np.asarray(den, float), jw)


def poly_mul(a, b) -> np.ndarray:
    return np.convolve(np.asarray(a, float), np.asarray(b, float))


def poly_add(a, b) -> np.ndarray:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = max(a.size, b.size)
    out = np.zeros(n)
    out[n - a.size:] += a
    out[n - b.size:] += b
    return out


# --------------------------------------------------------------------------- #
#  二阶对象
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#  通用 LTI 对象
# --------------------------------------------------------------------------- #
def _poly_str(coeffs, var: str = "s") -> str:
    """把降幂系数格式化成可读的多项式字符串。"""
    c = np.asarray(coeffs, dtype=float)
    n = c.size - 1
    parts = []
    for i, v in enumerate(c):
        if abs(v) < 1e-12:
            continue
        pw = n - i
        av = abs(v)
        sgn = " - " if v < 0 else (" + " if parts else "")
        if pw == 0:
            parts.append(f"{sgn}{av:g}")
        elif pw == 1:
            parts.append(f"{sgn}{av:g}{var}")
        else:
            parts.append(f"{sgn}{av:g}{var}^{pw}")
    return "".join(parts) if parts else "0"


@dataclass(eq=False)
class LTIPlant:
    """通用线性定常对象：  G(s) = num(s) / den(s) · e^(-tau·s)

    num / den 均为 **降幂** 系数，例如 G(s)=10/(s²+2s+10) 写作 num=[10], den=[1,2,10]。
    该类统一提供状态空间、频率特性、零极点等接口，供仿真 / 整定 / 绘图模块调用，
    因此整个平台不再局限于二阶对象。
    """

    num: np.ndarray
    den: np.ndarray
    delay: float = 0.0
    label: str = "LTI 对象"

    def __post_init__(self):
        self.num = np.atleast_1d(np.asarray(self.num, dtype=float)).ravel()
        self.den = np.atleast_1d(np.asarray(self.den, dtype=float)).ravel()
        while self.num.size > 1 and abs(self.num[0]) < 1e-15:
            self.num = self.num[1:]          # 去掉分子前导零，便于判阶

    # ---------------- 基本属性 ----------------
    @property
    def order(self) -> int:
        return int(self.den.size - 1)

    @property
    def dc_gain(self) -> float:
        """直流增益 G(0)；含积分环节时返回 inf。"""
        if abs(self.den[-1]) < 1e-15:
            return float("inf")
        return float(self.num[-1] / self.den[-1])

    # ---------------- 状态空间 / 频域 ----------------
    def ss(self):
        return tf_to_ss(self.num, self.den)

    def freqresp(self, w) -> np.ndarray:
        g = freqresp(self.num, self.den, w)
        if self.delay:
            g = g * np.exp(-1j * np.asarray(w, float) * self.delay)
        return g

    def poles(self) -> np.ndarray:
        return np.roots(self.den)

    def zeros(self) -> np.ndarray:
        return np.roots(self.num) if self.num.size > 1 else np.array([])

    def is_stable(self, tol: float = 1e-9) -> bool:
        """严格稳定（全部极点位于左半开平面）。"""
        return bool(np.all(np.real(self.poles()) < -tol))

    def stability_label(self, tol: float = 1e-9) -> str:
        """稳定 / 临界稳定 / 不稳定 —— 便于界面显示。"""
        re = np.real(self.poles())
        if np.any(re > tol):
            return "不稳定"
        if np.any(np.abs(re) <= tol):
            return "临界稳定"
        return "稳定"

    def second_order_params(self):
        """若恰好是标准二阶形式 K·wn²/(s²+2ζwn·s+wn²)，返回 (K, wn, zeta)；否则 None。"""
        if self.num.size != 1 or self.den.size != 3:
            return None
        a0, a1, a2 = self.den
        if abs(a0) < 1e-15 or (a2 / a0) <= 0:
            return None
        wn = float(np.sqrt(a2 / a0))
        zeta = float((a1 / a0) / (2.0 * wn))
        K = float(self.num[0] / a2)
        return K, wn, zeta

    def describe(self) -> str:
        g = f"G(s) = ({_poly_str(self.num)}) / ({_poly_str(self.den)})"
        if self.delay:
            g += f" · e^(-{self.delay:g}s)"
        tag = self.stability_label()
        return f"{g}    [{self.label}，{self.order} 阶，{tag}]"

    def as_dict(self) -> dict:
        return {"对象模型": self.label, "阶次": self.order,
                "分子系数 num": str(list(np.round(self.num, 6))),
                "分母系数 den": str(list(np.round(self.den, 6))),
                "纯滞后 τ (s)": self.delay, "直流增益 G(0)": self.dc_gain}


class SecondOrderPlant(LTIPlant):
    r"""标准二阶对象（保留旧接口，内部即 LTIPlant）：

        G(s) = K · wn² / (s² + 2ζwn·s + wn²) · e^(-τs)
    """

    def __init__(self, K: float = 1.0, wn: float = 1.0, zeta: float = 0.5, delay: float = 0.0):
        self.K = float(K)
        self.wn = float(wn)
        self.zeta = float(zeta)
        super().__init__(num=np.array([self.K * self.wn ** 2], dtype=float),
                         den=np.array([1.0, 2.0 * self.zeta * self.wn, self.wn ** 2], dtype=float),
                         delay=float(delay), label="标准二阶对象")

    @property
    def wd(self) -> float:
        """阻尼振荡频率 (rad/s)；zeta>=1 时无物理意义，返回 0。"""
        if self.zeta < 1.0:
            return self.wn * np.sqrt(max(0.0, 1.0 - self.zeta ** 2))
        return 0.0

    def describe(self) -> str:
        kind = ("无阻尼" if abs(self.zeta) < 1e-9 else
                "欠阻尼" if self.zeta < 1 else
                "临界阻尼" if abs(self.zeta - 1) < 1e-9 else "过阻尼")
        return (f"G(s) = {self.K:g}·{self.wn:g}² / (s² + 2·{self.zeta:g}·{self.wn:g}·s "
                f"+ {self.wn:g}²)" + (f" · e^(-{self.delay:g}s)" if self.delay else "")
                + f"    [{kind}, zeta={self.zeta:g}, wn={self.wn:g} rad/s]")

    def as_dict(self) -> dict:
        return {"对象模型": "标准二阶对象", "K": self.K, "ωn (rad/s)": self.wn,
                "阻尼比 ζ": self.zeta, "纯滞后 τ (s)": self.delay}


# --------------------------------------------------------------------------- #
#  对象模型库
# --------------------------------------------------------------------------- #
def _b_second_order(p):
    return (np.array([p["K"] * p["wn"] ** 2]),
            np.array([1.0, 2.0 * p["zeta"] * p["wn"], p["wn"] ** 2]))


def _b_first_order(p):
    return np.array([p["K"]]), np.array([p["T"], 1.0])


def _b_integrator(p):
    # K / [s(Ts+1)]
    return np.array([p["K"]]), np.array([p["T"], 1.0, 0.0])


def _b_third_order(p):
    den = np.convolve([p["T1"], 1.0], np.convolve([p["T2"], 1.0], [p["T3"], 1.0]))
    return np.array([p["K"]]), den


def _b_nmp(p):
    # K(1 - Tz·s) / [(T1·s+1)(T2·s+1)]
    num = np.array([-p["K"] * p["Tz"], p["K"]])
    den = np.convolve([p["T1"], 1.0], [p["T2"], 1.0])
    return num, den


MODEL_SPECS = {
    "标准二阶对象": dict(
        formula="K·ωn² / (s² + 2ζωn·s + ωn²) · e^(−τs)",
        kind="std", build=_b_second_order,
        params=[("K", "直流增益 K", 1.0, 0.01, 100.0, 0.1),
                ("wn", "自然频率 ωn (rad/s)", 1.0, 0.01, 50.0, 0.05),
                ("zeta", "阻尼比 ζ", 0.5, 0.0, 5.0, 0.05)],
        note="自控课最经典的二阶形式，可与解析公式逐项对照"),

    "一阶惯性对象": dict(
        formula="K / (T·s + 1) · e^(−τs)",
        kind="first", build=_b_first_order,
        params=[("K", "直流增益 K", 1.0, 0.01, 100.0, 0.1),
                ("T", "时间常数 T (s)", 1.0, 0.01, 100.0, 0.1)],
        note="0 型一阶系统：对阶跃输入存在静差，可演示纯 P 控制为何消不掉差"),

    "带积分对象": dict(
        formula="K / [s·(T·s + 1)] · e^(−τs)",
        kind="integrator", build=_b_integrator,
        params=[("K", "增益 K", 1.0, 0.01, 100.0, 0.1),
                ("T", "时间常数 T (s)", 0.5, 0.01, 100.0, 0.1)],
        note="Ⅰ 型系统：自身含积分环节，对阶跃输入理论上无静差"),

    "三阶惯性对象": dict(
        formula="K / [(T₁s+1)(T₂s+1)(T₃s+1)] · e^(−τs)",
        kind="third", build=_b_third_order,
        params=[("K", "直流增益 K", 1.0, 0.01, 100.0, 0.1),
                ("T1", "时间常数 T₁ (s)", 1.0, 0.01, 100.0, 0.1),
                ("T2", "时间常数 T₂ (s)", 0.5, 0.01, 100.0, 0.1),
                ("T3", "时间常数 T₃ (s)", 0.2, 0.01, 100.0, 0.1)],
        note="高阶对象：相角更容易穿越 −180°，Z-N 临界比例度法天然可用"),

    "非最小相位对象": dict(
        formula="K·(1 − Tz·s) / [(T₁s+1)(T₂s+1)] · e^(−τs)",
        kind="nmp", build=_b_nmp,
        params=[("K", "直流增益 K", 1.0, 0.01, 100.0, 0.1),
                ("Tz", "右半平面零点常数 Tz (s)", 0.5, 0.01, 100.0, 0.05),
                ("T1", "时间常数 T₁ (s)", 1.0, 0.01, 100.0, 0.1),
                ("T2", "时间常数 T₂ (s)", 0.5, 0.01, 100.0, 0.1)],
        note="含右半平面零点：起始阶段反向响应，是经典的控制难点"),

    "自定义传递函数": dict(
        formula="num(s) / den(s) · e^(−τs)     自行输入降幂系数",
        kind="custom", build=None, params=[],
        note="直接输入分子/分母系数，例如 num = 1, 2      den = 1, 3, 2"),
}


def build_plant(model_key: str, params: dict, delay: float = 0.0,
                num=None, den=None) -> LTIPlant:
    """按模型库条目构造被控对象。"""
    if model_key not in MODEL_SPECS:
        raise KeyError(f"未知对象模型：{model_key}")
    spec = MODEL_SPECS[model_key]
    if spec["kind"] == "custom":
        if num is None or den is None:
            raise ValueError("自定义传递函数需要同时给出 num 与 den")
        return LTIPlant(num=num, den=den, delay=float(delay), label="自定义传递函数")
    n, d = spec["build"](params)
    return LTIPlant(num=n, den=d, delay=float(delay), label=model_key)


# --------------------------------------------------------------------------- #
#  二阶系统理论公式（用于第 1 模块"仿真 vs 理论"互验）
# --------------------------------------------------------------------------- #
def second_order_theory(zeta: float, wn: float, band: float = 0.02) -> dict:
    """二阶系统单位阶跃响应的经典理论指标。

    定义说明（答辩重点）
    -------------------
    * 上升时间 tr 采用经典 0→100% 定义  tr = (pi - arccos zeta)/wd；
      时域仿真里同时给出工程常用的 10%→90% 上升时间，两者定义不同、不可直接比较。
    * 调节时间 ts 给出"衰减包络上界" -ln(Delta)/(zeta*wn)，它是充分条件、结果偏保守；
      工程近似还常用 3/(zeta*wn) (Delta=2%)、4/(zeta*wn) (Delta=5%)。
    """
    out = {"overshoot": np.nan, "tp": np.nan, "tr": np.nan, "ts": np.nan,
           "ts_approx": np.nan, "wd": np.nan, "formula_note": ""}
    if zeta <= 0:
        out["formula_note"] = "zeta<=0：系统无阻尼或发散，理论公式不适用"
        return out

    if zeta < 1.0:
        wd = wn * np.sqrt(1.0 - zeta ** 2)
        out["wd"] = wd
        out["overshoot"] = float(np.exp(-np.pi * zeta / np.sqrt(1 - zeta ** 2)) * 100.0)
        out["tp"] = float(np.pi / wd)
        out["tr"] = float((np.pi - np.arccos(zeta)) / wd)
        out["ts"] = float(-np.log(band) / (zeta * wn))
        out["ts_approx"] = float((3.0 if band <= 0.02 else 4.0) / (zeta * wn))
        out["formula_note"] = ("sigma% = e^{-pi*zeta/sqrt(1-zeta^2)}; tp = pi/wd; "
                               "tr(0-100%) = (pi-arccos zeta)/wd; "
                               "ts(包络上界) = -ln(Delta)/(zeta*wn)")
    elif abs(zeta - 1.0) < 1e-9:
        out["overshoot"] = 0.0
        out["ts"] = float(-np.log(band) / wn)
        out["ts_approx"] = float((3.0 if band <= 0.02 else 4.0) / wn)
        out["formula_note"] = "临界阻尼：无超调，ts 按主导极点近似"
    else:
        out["overshoot"] = 0.0
        z = zeta
        slow = (z - np.sqrt(z * z - 1)) * wn
        out["ts"] = float(-np.log(band) / slow)
        out["ts_approx"] = float((3.0 if band <= 0.02 else 4.0) / slow)
        out["formula_note"] = "过阻尼：无超调，按主导极点近似 ts"
    return out
