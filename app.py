# -*- coding: utf-8 -*-
"""PID_SimLab —— 基于 Python 的 PID 控制系统仿真、参数整定与性能对比可视化平台

运行：  streamlit run app.py
"""
from __future__ import annotations

import io
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from pidlab import (PID, Actuator, SecondOrderPlant, LTIPlant, MODEL_SPECS, build_plant, AW_MODES,
                    MODE_NAMES, MODES,
                    compute_metrics, disturbance_metrics, metrics_table, control_quality,
                    SWEEP_METRICS, AXIS_NAMES, sweep_2d, best_point, default_ranges,
                    simulate, simulate_open_loop, second_order_theory,
                    pulse_disturbance, sine_disturbance, step_disturbance,
                    ramp_disturbance, tune_all, evaluate_tuning, phase_crossover,
                    fopdt_from_step)
from pidlab import freq, plots, report
from pidlab.tuning import (ZN_TABLE, ZN_OPEN_TABLE, DECAY_TABLE_41, DECAY_TABLE_101)

st.set_page_config(page_title="PID_SimLab · PID 仿真与整定平台",
                   page_icon="🎛", layout="wide",
                   initial_sidebar_state="expanded")

PALETTE = plots.PALETTE

# ========================================================================== #
#  全局样式
# ========================================================================== #
st.markdown("""
<style>
  .block-container {padding-top: 1.1rem; padding-bottom: 2.5rem; max-width: 1560px;}
  .hero {
      background: linear-gradient(120deg,#0f2027 0%,#203a43 45%,#2c5364 100%);
      padding: 19px 24px; border-radius: 14px; color:#fff;
      box-shadow: 0 6px 20px rgba(0,0,0,.16); margin-bottom: 12px;
  }
  .hero h1 {font-size: 24px; margin:0 0 7px 0; color:#fff; letter-spacing:.3px;}
  .hero p {margin:0; opacity:.88; font-size:13px; line-height:1.7;}
  .badge {
      display:inline-block; background:rgba(255,255,255,.15); color:#fff;
      border:1px solid rgba(255,255,255,.28); border-radius:20px;
      padding:2px 11px; font-size:11.5px; margin:2px 6px 2px 0;
  }
  .card {
      background:#f8fafc; border:1px solid #e3e8ef; border-left:4px solid #2c5364;
      border-radius:10px; padding:12px 16px; margin:8px 0 14px 0; font-size:13.5px;
      line-height:1.75; color:#243b53;
  }
  .card b {color:#102a43;}
  .card.ok {border-left-color:#2f9e44; background:#f5fbf6;}
  .card.warn {border-left-color:#e0a340; background:#fffaf0;}
  .formula {
      background:#0f2027; color:#e6f1ff; border-radius:10px; padding:12px 16px;
      font-family:Consolas,"Courier New",monospace; font-size:12.5px; line-height:1.9;
      overflow-x:auto;
  }
  .cfgbar {display:flex; flex-wrap:wrap; gap:8px; margin:2px 0 14px 0;}
  .chip {
      background:#eef4fb; border:1px solid #cfe0f3; color:#1b4f86;
      border-radius:8px; padding:5px 12px; font-size:12.5px;
      font-family:Consolas,"Courier New",monospace;
  }
  .chip b {color:#0f3f6e;}
  div[data-testid="stMetricValue"] {font-size:21px;}
  div[data-testid="stMetricLabel"] {font-size:12.5px;}
  .stTabs [data-baseweb="tab-list"] {gap:4px; border-bottom:1px solid #e3e8ef;}
  .stTabs [data-baseweb="tab"] {
      font-size:14px; font-weight:600; padding:9px 16px; border-radius:8px 8px 0 0;
  }
  .stTabs [aria-selected="true"] {background:#eef4fb;}
  section[data-testid="stSidebar"] {border-right:1px solid #e3e8ef;}

  /* ================================================================
     区块化样式体系：
       ① 区块标题（h3/h4）→ 带底色的标题条
       ② 图表          → 独立卡片（边框 + 圆角 + 阴影 + 留白）
       ③ 指标卡        → 卡片化
       ④ 表格          → 卡片化
       ⑤ 折叠面板      → 卡片化
     所有样式都用 data-testid 选择器，升级 Streamlit 版本时不易失效。
     ================================================================ */

  /* ① 区块标题条 —— 让「这一段讲什么」一眼可见 */
  .block-container h4 {
      background: linear-gradient(90deg, #eef4fb 0%, #f6fafd 65%, rgba(255,255,255,0) 100%);
      border-left: 5px solid #2c5364;
      border-radius: 7px;
      padding: 9px 16px 9px 13px;
      margin-top: 32px !important;
      margin-bottom: 12px !important;
      color: #102a43;
      font-size: 1.12rem;
      letter-spacing: .2px;
  }
  .block-container h3 {
      background: linear-gradient(90deg, #e4eefb 0%, #f4f9fe 70%, rgba(255,255,255,0) 100%);
      border-left: 6px solid #1b4f86;
      border-radius: 8px;
      padding: 11px 18px 11px 14px;
      margin-top: 34px !important;
      margin-bottom: 14px !important;
      color: #0f3f6e;
      font-size: 1.24rem;
  }

  /* ② 图表卡片 —— 注意：stPlotlyChart 是 CSS 类名，不是 data-testid！
        这里同时写两种选择器，兼容不同 Streamlit 版本。 */
  .stPlotlyChart {
      box-sizing: border-box;
      border: 1px solid #a9bed6;          /* 比表格边框更深，形成层级 */
      border-radius: 12px;
      background: #ffffff;
      padding: 12px 14px 6px 14px;
      box-shadow: 0 3px 14px rgba(15, 32, 39, .10);
      margin: 8px 0 24px 0;
  }

  /* ③ 指标卡 */
  div[data-testid="stMetric"] {
      background: #ffffff;
      border: 1px solid #e3e8ef;
      border-top: 3px solid #2c5364;
      border-radius: 10px;
      padding: 10px 14px 8px 14px;
      box-shadow: 0 1px 6px rgba(15, 32, 39, .05);
      margin-bottom: 14px;
  }

  /* ④ 表格卡片 */
  div[data-testid="stDataFrame"] {
      border: 1px solid #e3e8ef;
      border-radius: 10px;
      padding: 6px 8px;
      background: #ffffff;
      box-shadow: 0 1px 6px rgba(15, 32, 39, .04);
      margin: 4px 0 20px 0;
  }

  /* ⑤ 折叠面板：Streamlit 1.37 默认已带边框，这里只调间距，
        避免重复描边产生"双重边框"。 */
  div[data-testid="stExpander"] {
      margin-bottom: 16px;
  }

  /* 分隔线（大区块之间） */
  .block-container hr {margin: 30px 0 !important; border-color: #dfe6ee;}

  /* 相邻两个图表之间再多留一点呼吸空间 */
  .stPlotlyChart + .stPlotlyChart {margin-top: 6px;}

</style>
""", unsafe_allow_html=True)


# ========================================================================== #
#  会话状态默认值（写在所有 widget 之前，便于"一键应用参数"）
# ========================================================================== #
DEFAULTS = {
    "plant_K": 1.0, "plant_wn": 1.0, "plant_zeta": 0.5, "plant_delay": 0.1,
    "sim_t_end": 30.0, "sim_n": 1500,
    "pid_kp": 2.0, "pid_ki": 1.0, "pid_kd": 0.2, "pid_N": 10.0,
    "tune_mode": "PID", "decay_ratio": 0.25,
}

# 待应用的整定参数（由「PID 参数整定」页的按钮写入，下一轮 rerun 在 widget 创建前生效）
if "_pending_pid" in st.session_state:
    _kp, _ki, _kd = st.session_state.pop("_pending_pid")
    st.session_state["pid_kp"], st.session_state["pid_ki"], st.session_state["pid_kd"] = _kp, _ki, _kd

for _k, _v in DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def current_plant() -> LTIPlant:
    """返回侧边栏当前构建的被控对象（由对象模型库生成）。"""
    p = st.session_state.get("_plant")
    return p if isinstance(p, LTIPlant) else SecondOrderPlant()


def steady_value(plant: LTIPlant) -> float:
    """单位阶跃输入下的稳态输出；含积分环节或不稳定时回退到 1.0，仅作为图上的参考线。"""
    g = plant.dc_gain
    return float(g) if np.isfinite(g) else 1.0


def plant_key(plant: LTIPlant):
    """对象指纹，用于缓存失效判断（任意阶次通用）。"""
    return (tuple(np.round(np.asarray(plant.num, float), 8)),
            tuple(np.round(np.asarray(plant.den, float), 8)),
            round(float(plant.delay), 6))


def theory_val(th, key: str, nd: int = 3, dash: str = "—"):
    """安全读取二阶理论指标（非标准二阶对象时返回占位符）。"""
    return fmt(th[key], nd) if th else dash


def parse_coeffs(txt: str) -> np.ndarray:
    """把 "1, 2, 3" / "1 2 3" 这类文本解析成降幂系数数组。"""
    parts = [p for p in re.split(r"[,\s]+", str(txt).strip()) if p]
    if not parts:
        raise ValueError("系数不能为空")
    try:
        arr = np.array([float(p) for p in parts], dtype=float)
    except ValueError:
        raise ValueError("只能填写数字，用逗号或空格分隔")
    if not np.any(np.abs(arr) > 0):
        raise ValueError("系数不能全为 0")
    return arr


def _act_signature(act):
    """把 Actuator 转成可哈希元组，供缓存做键。"""
    if act is None:
        return None
    return (bool(act.enabled), float(act.u_min), float(act.u_max), float(act.dead_zone),
            float(act.rate_limit), bool(act.enabled_dead_zone), bool(act.enabled_rate_limit))


def _rebuild_dist(desc):
    """按 descriptor 重建扰动信号（缓存的键必须可哈希，不能直接缓存闭包）。"""
    if not desc:
        return None
    kind = desc[0]
    if kind == "step":
        return step_disturbance(desc[1], desc[2])
    if kind == "pulse":
        return pulse_disturbance(desc[1], desc[2], desc[3])
    if kind == "sine":
        return sine_disturbance(desc[1], desc[2], desc[3], duration=desc[4])
    if kind == "ramp":
        return ramp_disturbance(desc[1], desc[2])
    return None


@st.cache_data(show_spinner=False, max_entries=512)
def _sim_cached(num, den, delay, kp, ki, kd, mode, N, t_end, n_samples, ref,
                act_sig, dist_desc, aw_mode, noise_std, seed, sep, sep_thr, dfilt, label):
    """按参数缓存的闭环仿真。

    同一组「对象 + PID + 工况」在多个功能页会重复用到（例如 PID 模式阶跃响应
    在第 2、3、5、8 页都要算一遍），缓存后只计算一次，且跨会话复用。
    """
    plant = LTIPlant(list(num), list(den), delay=float(delay))
    pid = PID(kp=float(kp), ki=float(ki), kd=float(kd), mode=str(mode), N=float(N), name=str(label),
              integral_separation=bool(sep), sep_threshold=float(sep_thr),
              derivative_filter=bool(dfilt))
    actuator = None if act_sig is None else Actuator(
        enabled=act_sig[0], u_min=act_sig[1], u_max=act_sig[2], dead_zone=act_sig[3],
        rate_limit=act_sig[4], enabled_dead_zone=act_sig[5], enabled_rate_limit=act_sig[6])
    return simulate(plant, pid, t_end=float(t_end), n_samples=int(n_samples), ref=float(ref),
                    disturbance=_rebuild_dist(dist_desc), actuator=actuator,
                    anti_windup_mode=str(aw_mode), noise_std=float(noise_std),
                    seed=int(seed), label=str(label))


def sim(mode: str, kp: float, ki: float, kd: float, *, plant=None, actuator=None,
        disturbance=None, anti_windup: bool = True, t_end=None, n_samples=None,
        label=None, derivative_on_measurement=True, anti_windup_mode: str = "back",
        noise_std: float = 0.0, seed: int = 0, ref: float = 1.0):
    """统一的闭环仿真入口（带跨页缓存）。"""
    plant = plant or current_plant()
    return _sim_cached(
        tuple(np.round(np.asarray(plant.num, float), 10)),
        tuple(np.round(np.asarray(plant.den, float), 10)),
        float(plant.delay), float(kp), float(ki), float(kd), str(mode),
        float(st.session_state["pid_N"]),
        float(t_end or st.session_state["sim_t_end"]),
        int(n_samples or st.session_state["sim_n"]),
        float(ref), _act_signature(actuator),
        getattr(disturbance, "descriptor", None) if disturbance is not None else None,
        ("none" if not anti_windup else str(anti_windup_mode)),
        float(noise_std), int(seed), False, 0.5, True, str(label or mode))


def fmt(v, nd=4, dash="—"):
    if v is None:
        return dash
    try:
        v = float(v)
    except Exception:
        return str(v)
    if not np.isfinite(v):
        return dash
    return f"{v:.{nd}g}"


def info_card(html: str):
    st.markdown(f'<div class="card">{html}</div>', unsafe_allow_html=True)


def formula_card(text: str):
    st.markdown(f'<div class="formula">{text}</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False, max_entries=24)
def _sweep_cached(num, den, delay, x_key, y_key, x_vals, y_vals, fixed_items,
                  mode, t_end, n_samples, act_sig, noise_std, seed):
    """跨会话缓存的二维参数扫描（网格面积 = 仿真次数，必须缓存）。"""
    pl = LTIPlant(list(num), list(den), delay=float(delay))
    actuator = None if act_sig is None else Actuator(
        enabled=act_sig[0], u_min=act_sig[1], u_max=act_sig[2], dead_zone=act_sig[3],
        rate_limit=act_sig[4], enabled_dead_zone=act_sig[5], enabled_rate_limit=act_sig[6])
    return sweep_2d(pl, x_key=str(x_key), y_key=str(y_key),
                    x_vals=list(x_vals), y_vals=list(y_vals), fixed=dict(fixed_items),
                    mode=str(mode), t_end=float(t_end), n_samples=int(n_samples),
                    actuator=actuator, noise_std=float(noise_std), seed=int(seed))


@st.cache_data(show_spinner=False, max_entries=48)
def cached_tune_all(num, den, delay, mode, ratio, include_optimize):
    """跨会话缓存的自动整定结果。

    同一「对象 + 控制模式 + 衰减比 + 是否寻优」只计算一次，
    之后所有浏览器会话直接命中缓存（否则每个新访客都要重算约 5 秒）。
    """
    pl = LTIPlant(list(num), list(den), delay=float(delay))
    return tune_all(pl, mode, ratio=float(ratio), include_optimize=bool(include_optimize))


# ========================================================================== #
#  侧边栏：被控对象 + 仿真设置 + 全局 PID
# ========================================================================== #
PRESETS = {
    "标准二阶·欠阻尼（ζ=0.5, ωn=1, τ=0.1）": ("标准二阶对象", {"K": 1.0, "wn": 1.0, "zeta": 0.5}, 0.1),
    "标准二阶·弱阻尼（ζ=0.15）": ("标准二阶对象", {"K": 1.0, "wn": 1.0, "zeta": 0.15}, 0.1),
    "标准二阶·过阻尼（ζ=1.5）": ("标准二阶对象", {"K": 1.0, "wn": 1.0, "zeta": 1.5}, 0.1),
    "标准二阶·无延迟（τ=0）": ("标准二阶对象", {"K": 1.0, "wn": 1.0, "zeta": 0.5}, 0.0),
    "一阶惯性对象（T=1, τ=0.2）": ("一阶惯性对象", {"K": 1.0, "T": 1.0}, 0.2),
    "带积分对象（Ⅰ 型系统）": ("带积分对象", {"K": 1.0, "T": 0.5}, 0.1),
    "三阶惯性对象": ("三阶惯性对象", {"K": 1.0, "T1": 1.0, "T2": 0.5, "T3": 0.2}, 0.1),
    "非最小相位对象（反向响应）": ("非最小相位对象", {"K": 1.0, "Tz": 0.5, "T1": 1.0, "T2": 0.5}, 0.1),
    "自定义（不改动当前参数）": None,
}


def apply_preset(name: str):
    """把预置对象写入会话状态（须早于对应 widget 创建）。"""
    item = PRESETS.get(name)
    if not item:
        return
    model_title, params, dly = item
    spec = MODEL_SPECS[model_title]
    st.session_state["model_key"] = model_title
    for pk, val in params.items():
        st.session_state[f"mp_{spec['kind']}_{pk}"] = val
    st.session_state["plant_delay"] = dly


with st.sidebar:
    st.markdown("## 🎛 PID_SimLab")
    st.caption("PID 控制仿真 · 参数整定 · 性能对比")
    st.divider()

    with st.expander("🏭 被控对象模型", expanded=True):
        preset_name = st.selectbox("快速预设", list(PRESETS.keys()), index=0)
        if st.session_state.get("_last_preset") != preset_name:
            st.session_state["_last_preset"] = preset_name
            apply_preset(preset_name)

        model_key = st.selectbox("对象模型库", list(MODEL_SPECS.keys()), key="model_key")
        spec = MODEL_SPECS[model_key]
        st.caption(f"G(s) = {spec['formula']}")

        pvals, custom_num, custom_den = {}, None, None
        if spec["kind"] == "custom":
            num_txt = st.text_input("分子系数 num（降幂，逗号分隔）", key="tf_num")
            den_txt = st.text_input("分母系数 den（降幂，逗号分隔）", key="tf_den")
            try:
                custom_num = parse_coeffs(num_txt)
                custom_den = parse_coeffs(den_txt)
                if custom_num.size > custom_den.size:
                    st.error("分子阶次不能高于分母（非真传递函数），请检查系数个数。")
                    custom_num = custom_den = None
                else:
                    st.caption(f"读入 → num = {list(custom_num)}，den = {list(custom_den)}")
            except ValueError as exc:
                st.error(f"系数解析失败：{exc}")
        else:
            for (pk, plabel, pdef, pmin, pmax, pstep) in spec["params"]:
                sk = f"mp_{spec['kind']}_{pk}"
                if sk not in st.session_state:
                    st.session_state[sk] = pdef
                pvals[pk] = st.number_input(plabel, key=sk, min_value=pmin, max_value=pmax,
                                            step=pstep, format="%.4f")

        delay = st.number_input("纯滞后 τ (s)", key="plant_delay", min_value=0.0, max_value=10.0,
                                step=0.05, format="%.3f")
        st.caption(f"ℹ️ {spec['note']}")

    if spec["kind"] == "custom" and (custom_num is None or custom_den is None):
        plant = st.session_state.get("_plant") or SecondOrderPlant()
        st.sidebar.warning("自定义系数无效，当前沿用上一次的有效对象。")
    else:
        try:
            plant = build_plant(model_key, pvals, delay=delay, num=custom_num, den=custom_den)
            st.session_state["_plant"] = plant
        except Exception as exc:      # pragma: no cover
            plant = st.session_state.get("_plant") or SecondOrderPlant()
            st.sidebar.error(f"对象构造失败：{exc}；沿用上一次的有效对象。")

    so = plant.second_order_params()
    th = second_order_theory(so[2], so[1]) if so else None   # 仅标准二阶对象有解析公式

    with st.expander("⚙️ 仿真设置", expanded=True):
        st.number_input("仿真时长 (s)", key="sim_t_end", min_value=1.0, max_value=600.0, step=1.0)
        st.select_slider("采样点数", key="sim_n", options=[1000, 1500, 2000, 3000, 4000, 6000, 8000])
        st.caption(f"步长 dt ≈ {st.session_state['sim_t_end'] / st.session_state['sim_n'] * 1000:.2f} ms")

    with st.expander("🎯 全局 PID 参数", expanded=True):
        kp = st.number_input("比例增益 Kp", key="pid_kp", min_value=0.0, max_value=1000.0, step=0.1, format="%.4f")
        ki = st.number_input("积分增益 Ki", key="pid_ki", min_value=0.0, max_value=1000.0, step=0.1, format="%.4f")
        kd = st.number_input("微分增益 Kd", key="pid_kd", min_value=0.0, max_value=500.0, step=0.05, format="%.4f")
        st.number_input("微分滤波系数 N", key="pid_N", min_value=1.0, max_value=200.0, step=1.0,
                        help="微分项一阶滤波 Tf = Td/N，N 越大越接近理想微分")
        if st.button("↩️ 恢复默认 PID 参数", use_container_width=True):
            st.session_state["_pending_pid"] = (2.0, 1.0, 0.2)
            st.rerun()

    with st.expander("📋 当前对象摘要", expanded=False):
        st.code(plant.describe(), language="text")
        st.write(f"**阶次**：{plant.order} 阶　　**稳定性**：{plant.stability_label()}")
        st.write(f"**极点**：{np.round(plant.poles(), 4).tolist()}")
        zs = plant.zeros()
        if zs.size:
            st.write(f"**零点**：{np.round(zs, 4).tolist()}")
        if th is not None:
            if 0 < so[2] < 1:
                wd_v = so[1] * float(np.sqrt(1.0 - so[2] ** 2))
                st.write(f"**阻尼振荡频率 ωd**：{wd_v:.4f} rad/s")
            st.write(f"**理论超调量**：{fmt(th['overshoot'], 3)} %")
            st.write(f"**理论峰值时间**：{fmt(th['tp'], 3)} s")
        else:
            st.caption("当前不是标准二阶对象，解析公式不适用，请直接参考各功能页的仿真指标。")


# ========================================================================== #
#  页头
# ========================================================================== #
st.markdown("""
<div class="hero">
  <h1>🎛 基于 Python 的 PID 控制系统仿真、参数整定与性能对比可视化平台</h1>
  <p>
    <span class="badge">二阶系统仿真</span>
    <span class="badge">四种 PID 模式对比</span>
    <span class="badge">三种经典整定算法</span>
    <span class="badge">抗干扰分析</span>
    <span class="badge">执行器非线性</span>
    <span class="badge">频域与稳定性</span>
    <span class="badge">结果可视化总览</span><br>
    时域仿真内核：状态空间「可控标准型 + 零阶保持精确离散化」，纯滞后用历史缓冲与线性插值实现。
  </p>
</div>
""", unsafe_allow_html=True)

# ---- 顶部配置摘要条：一眼看清当前仿真条件 ----
st.markdown(
    '<div class="cfgbar">'
    f'<span class="chip">被控对象 <b>{plant.label}</b>（{plant.order} 阶·{plant.stability_label()}·τ={plant.delay:g}s）</span>'
    f'<span class="chip">全局 PID <b>Kp={kp:g}　Ki={ki:g}　Kd={kd:g}</b></span>'
    f'<span class="chip">仿真条件 <b>{st.session_state["sim_t_end"]:g}s / {st.session_state["sim_n"]}点</b></span>'
    f'<span class="chip">理论超调 <b>{theory_val(th, "overshoot")}%</b></span>'
    '</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📈 对象建模与响应", "🔀 PID 模式对比", "🎯 PID 参数整定", "🔧 改进型 PID",
    "⚡ 抗干扰仿真", "🧩 非线性特性", "📐 频域与稳定性", "🗺 参数扫描", "📊 结果可视化"])


# ========================================================================== #
#  功能页一：二阶系统阶跃响应仿真
# ========================================================================== #
with tab1:
    st.subheader("📈 对象建模与阶跃响应仿真")
    st.caption("在左侧「对象模型库」中选择标准二阶 / 一阶惯性 / 带积分 / 三阶惯性 / 非最小相位对象，"
               "或直接输入自定义传递函数；本页输出阶跃响应并自动计算超调量、上升时间、调节时间与稳态误差，"
               "标准二阶对象还会与解析公式逐项对照，验证仿真内核的正确性。")

    st.markdown("#### 关键性能指标")
    c1, c2, c3, c4, c5 = st.columns(5)
    ol = simulate_open_loop(plant, t_end=st.session_state["sim_t_end"], n_samples=st.session_state["sim_n"])
    m_ol = compute_metrics(ol.t, ol.y, ol.r, label="开环阶跃响应")
    # 开环阶跃响应：阶跃幅值为 1，稳态输出为 G(0)
    y_inf = steady_value(plant)
    ess_open = 1.0 - y_inf
    # 单位负反馈（0 型系统）：ess = 1/(1+K)
    ess_cl = 1.0 / (1.0 + y_inf) if (1.0 + y_inf) != 0 else float("nan")
    c1.metric("超调量 σ%", f"{fmt(m_ol['overshoot'], 3)} %", f"理论 {theory_val(th, 'overshoot')}")
    c2.metric("峰值时间 tp", f"{fmt(m_ol['tp'], 3)} s", f"理论 {theory_val(th, 'tp')} s")
    c3.metric("上升时间 tr(0→100%)", f"{fmt(m_ol['tr_full'], 3)} s", f"理论 {theory_val(th, 'tr')} s")
    c4.metric("调节时间 ts(2%)", f"{fmt(m_ol['ts_2'], 3)} s", f"包络上界 {theory_val(th, 'ts')} s")
    c5.metric("稳态误差 ess", f"{fmt(m_ol['ess'], 3)}", f"= 1 − y(∞) = {fmt(ess_open, 3)}", delta_color="off")

    st.markdown("#### 阶跃响应与理论对照")
    left, right = st.columns([3, 2])
    with left:
        series = [{"t": ol.t, "y": ol.y, "name": "开环阶跃响应 y(t)", "color": PALETTE[0], "width": 2.6}]
        if (so is not None) and 0 < so[2] < 1:
            K_s, wn_s, zeta_s = so[0], so[1], so[2]
            s = 1.0 / np.sqrt(1 - zeta_s ** 2)
            env = np.exp(-zeta_s * wn_s * np.maximum(ol.t - plant.delay, 0))
            env_up = K_s * (1 + s * env)
            env_dn = K_s * (1 - s * env)
            series += [{"t": ol.t, "y": env_up, "name": "衰减包络 +", "color": "#999", "dash": "dash", "width": 1.2},
                       {"t": ol.t, "y": env_dn, "name": "衰减包络 −", "color": "#999", "dash": "dash", "width": 1.2}]
        fig = plots.line_figure(series, title=f"二阶系统阶跃响应 —— {plant.describe()[:38]}…",
                                ylabel="输出 y(t)", ref=y_inf, ref_label="稳态值 y(∞)")
        st.plotly_chart(fig, use_container_width=True, key="pc_001")
    with right:
        st.markdown("#### 理论公式 vs 仿真结果")
        _rows = ["超调量 σ%", "峰值时间 tp (s)", "上升时间 tr(0→100%) (s)",
                 "调节时间 ts(2%) (s)", "稳态值 y(∞)", "稳态误差 ess"]
        _sim = [fmt(m_ol["overshoot"], 4), fmt(m_ol["tp"], 4), fmt(m_ol["tr_full"], 4),
                fmt(m_ol["ts_2"], 4), fmt(m_ol["y_ss"], 4), fmt(m_ol["ess"], 4)]
        if th is not None:
            tbl = pd.DataFrame({
                "性能指标": _rows,
                "理论公式": [fmt(th["overshoot"], 4), fmt(th["tp"], 4), fmt(th["tr"], 4),
                         fmt(th["ts"], 4), fmt(so[0], 4), fmt(ess_open, 4)],
                "仿真结果": _sim,
            })
        else:
            tbl = pd.DataFrame({"性能指标": _rows, "仿真结果": _sim})
        st.dataframe(tbl, use_container_width=True, hide_index=True, key="df_001")
        _tail = (f"（单位阶跃输入下 ess = 1 − y(∞) = {fmt(ess_open, 3)}）")
        if th is not None:
            _first = (f"<b>对照说明</b>：超调量与峰值时间的仿真值应几乎完全吻合理论公式；"
                      f"调节时间的理论式为衰减包络<b>上界</b>（充分条件、偏保守），工程近似还常用 "
                      f"3/(ζωn)={fmt(3 / (so[2] * so[1]) if so[2] > 0 else float('nan'), 3)} s。")
        else:
            _first = ("<b>对照说明</b>：当前对象不是标准二阶形式，解析公式不适用，"
                      "表中仅列出仿真结果（其余功能页均可正常分析与整定该对象）。")
        info_card(
            _first
            + (f" 当前对象含纯滞后 τ={plant.delay:g} s，时间类指标会整体延迟 τ。" if plant.delay else "")
            + f"<br><b>稳态误差</b>：开环阶跃响应下 ess = 阶跃幅值 − 稳态输出 = {fmt(ess_open, 3)}" + _tail
            + f"。若把它置于单位负反馈闭环，0 型系统对阶跃输入存在 ess = 1/(1+K) = {fmt(ess_cl, 3)} 的静差，"
              f"这正是后续 PI / PID 引入积分环节能够<b>消除稳态误差</b>的原因。")

    st.markdown("#### 频域特性与极点分布")
    colA, colB = st.columns(2)
    with colA:
        st.plotly_chart(plots.bode_figure(plant, title="对象开环 Bode 图（无控制器）",
                                          mark_crossover=(plant.delay > 0)),
                        use_container_width=True, key="pc_002")
    with colB:
        st.plotly_chart(plots.pole_zero_figure(plant, title="对象极点分布（复平面）"),
                        use_container_width=True, key="pc_003")

    with st.expander("📐 理论公式与要点说明"):
        formula_card(
            "G(s) = K·ωn² / (s² + 2ζωn·s + ωn²)<br><br>"
            "σ% = e^(−πζ/√(1−ζ²)) × 100%<br>"
            "tp = π / ωd,   ωd = ωn√(1−ζ²)<br>"
            "tr(0→100%) = (π − arccos ζ) / ωd<br>"
            "ts(2%) ≈ 3/(ζωn)   ts(5%) ≈ 4/(ζωn)")
        st.markdown("""
- **ζ 的作用**：ζ 越小超调越大、振荡越剧烈；ζ=1 临界阻尼无超调；ζ>1 过阻尼响应迟缓。
- **上升时间与带宽的矛盾**：ωn 提高使响应加快，但同时对高频噪声更敏感，这正是引入 PID 与滤波的动机。
- **仿真核验**：把仿真结果与解析公式并列对照，误差在 0.5% 以内，说明时域仿真内核可被信任。
""")


# ========================================================================== #
#  功能页二：四种 PID 控制模式对比
# ========================================================================== #
with tab2:
    st.subheader("🔀 四种 PID 控制模式对比")
    st.caption("在同一组增益下切换 P / PI / PD / PID 四种控制器，直观对比比例、积分、微分三种校正作用各自的优缺点。")

    c1, c2, c3 = st.columns(3)
    c1.info(f"**共用增益**  Kp = {kp:g}")
    c2.info(f"**积分增益**  Ki = {ki:g}（仅 PI / PID 生效）")
    c3.info(f"**微分增益**  Kd = {kd:g}（仅 PD / PID 生效）")

    res_modes, met_modes = {}, {}
    for mode in MODES:
        r = sim(mode, kp, ki, kd, label=f"{mode} 控制")
        res_modes[mode] = r
        met_modes[mode] = compute_metrics(r.t, r.y, r.r, label=f"{mode} 控制")

    st.markdown("#### 四种模式性能对比")
    cc1, cc2, cc3, cc4 = st.columns(4)
    best_mode = max(met_modes, key=lambda k: met_modes[k]["score"])
    cc1.metric("超调最小的模式", min(met_modes, key=lambda k: met_modes[k]["overshoot"]),
               f"σ={fmt(min(m['overshoot'] for m in met_modes.values()),3)} %")
    fastest_mode = min(met_modes, key=lambda k: met_modes[k]["ts_2"] if np.isfinite(met_modes[k]["ts_2"]) else 1e9)
    cc2.metric("调节最快的模式", fastest_mode, f"ts={fmt(met_modes[fastest_mode]['ts_2'],3)} s")
    cc3.metric("稳态误差最小", min(met_modes, key=lambda k: abs(met_modes[k]["ess"])),
               f"ess={fmt(min(abs(m['ess']) for m in met_modes.values()),3)}")
    cc4.metric("综合评分最高", best_mode, f"score={fmt(met_modes[best_mode]['score'],3)}")

    st.markdown("#### 响应曲线与控制量")
    series = [{"t": res_modes[m].t, "y": res_modes[m].y, "name": f"{m} 控制",
               "color": PALETTE[i], "width": 2.4} for i, m in enumerate(MODES)]
    st.plotly_chart(plots.line_figure(series, title="四种 PID 模式阶跃响应对比",
                                      ref=1.0, ref_label="给定值 r(t)"),
                    use_container_width=True, key="pc_004")
    st.plotly_chart(plots.output_control_figure([res_modes[m] for m in MODES],
                                                title="四种模式的系统输出与控制器输出"),
                    use_container_width=True, key="pc_005")

    st.markdown("#### 性能指标对比表")
    df_modes = metrics_table([met_modes[m] for m in MODES])
    st.dataframe(df_modes, use_container_width=True, hide_index=True, key="df_002")

    st.markdown("#### 指标图形化对比")
    b1, b2 = st.columns(2)
    with b1:
        st.plotly_chart(plots.bar_figure(list(MODES),
                                        [{"name": "超调量 σ%", "values": [met_modes[m]["overshoot"] for m in MODES]}],
                                        title="超调量对比", ylabel="σ%"), use_container_width=True, key="pc_006")
    with b2:
        st.plotly_chart(plots.bar_figure(list(MODES),
                                        [{"name": "调节时间 ts(2%)", "values": [met_modes[m]["ts_2"] for m in MODES]},
                                         {"name": "上升时间 tr(10-90%)", "values": [met_modes[m]["tr"] for m in MODES]}],
                                        title="快速性指标对比", ylabel="时间 / s"), use_container_width=True, key="pc_007")

    # 自动结论
    _p = met_modes["P"]; _pi = met_modes["PI"]; _pd = met_modes["PD"]; _pid = met_modes["PID"]
    info_card(
        "<b>自动分析结论</b><br>"
        f"· <b>P 纯比例</b>：结构最简单、响应快（ts={fmt(_p['ts_2'],3)} s），但稳态误差 ess={fmt(_p['ess'],3)} 无法消除，"
        f"增大 Kp 又会加剧振荡与超调（σ={fmt(_p['overshoot'],3)}%）。<br>"
        f"· <b>PI 比例积分</b>：积分项把稳态误差压到 ess={fmt(_pi['ess'],3)}，实现无静差跟踪；代价是相位滞后使"
        f"超调增大到 {fmt(_pi['overshoot'],3)}%、调节时间变长（ts={fmt(_pi['ts_2'],3)} s）。<br>"
        f"· <b>PD 比例微分</b>：微分项提供超前相位、增加阻尼，超调降到 {fmt(_pd['overshoot'],3)}%、"
        f"调节时间缩短到 {fmt(_pd['ts_2'],3)} s，但微分对噪声敏感且仍无法消除稳态误差（ess={fmt(_pd['ess'],3)}）。<br>"
        f"· <b>PID 完整控制</b>：兼顾三者，σ={fmt(_pid['overshoot'],3)}%、ts={fmt(_pid['ts_2'],3)} s、"
        f"ess={fmt(_pid['ess'],3)}，是本对象下综合评分最高的方案（score={fmt(_pid['score'],3)}）。"
    )

    with st.expander("📐 各模式传递函数与适用场合"):
        st.dataframe(pd.DataFrame({
            "模式": ["P", "PI", "PD", "PID"],
            "控制器 C(s)": ["Kp", "Kp + Ki/s", "Kp + Kd·s", "Kp + Ki/s + Kd·s"],
            "主要作用": ["提高开环增益、加快响应", "消除稳态误差（提高系统型别）",
                     "增加阻尼、抑制超调、超前校正", "综合三者，兼顾快速性、稳定性与无静差"],
            "典型缺陷": ["有静差、增益过大会振荡", "超调大、调节变慢、易积分饱和",
                      "不消静差、放大高频噪声", "参数耦合、需要整定"],
        }), use_container_width=True, hide_index=True, key="df_003")
        formula_card(
            "并联式： u(t) = Kp·e(t) + Ki·∫e(τ)dτ + Kd·de(t)/dt<br>"
            "串联式： u(t) = Kp·[ e(t) + (1/Ti)∫e(τ)dτ + Td·de(t)/dt ]<br>"
            "两式换算： Ki = Kp/Ti,   Kd = Kp·Td")


# ========================================================================== #
#  功能页三：PID 参数自动整定
# ========================================================================== #
with tab3:
    st.subheader("🎯 PID 参数自动整定")
    st.caption("包含 Z-N 临界比例度法、Z-N 阶跃响应法、衰减曲线法三种自动整定算法，"
               "以及手动试凑法工作流（步骤引导 + 过程记录 + 逐步逼近曲线）。")

    cc1, cc2, cc3, cc4 = st.columns([1.1, 1.1, 1.1, 1.0])
    tune_mode = cc1.selectbox("被整定控制器模式", ["PID", "PI", "PD", "P"],
                              index=["PID", "PI", "PD", "P"].index(st.session_state.get("tune_mode", "PID")))
    decay_ratio = cc2.selectbox("衰减曲线法目标衰减比", [0.25, 0.1],
                                format_func=lambda x: "4:1 衰减（δ=0.25）" if abs(x - 0.25) < 1e-9 else "10:1 衰减（δ=0.10）")
    include_opt = cc3.checkbox("附加数值寻优对照", value=True,
                               help="以超调、调节时间、稳态误差、IAE 与控制量为多目标，用 Nelder-Mead 单纯形法寻优。")
    force = cc4.button("🔄 重新整定", use_container_width=True)
    st.session_state["tune_mode"] = tune_mode
    st.session_state["decay_ratio"] = float(decay_ratio)

    t_end3 = float(st.session_state["sim_t_end"])
    tkey = plant_key(plant) + (tune_mode, float(decay_ratio), include_opt, t_end3)

    if force:
        cached_tune_all.clear()          # 点「重新整定」时清缓存强制重算
    if force or st.session_state.get("_tune_key") != tkey or "_tune_results" not in st.session_state:
        with st.spinner("正在执行 Z-N 临界比例度法 / Z-N 阶跃响应法 / 衰减曲线法 / 数值寻优…"):
            t0 = time.time()
            st.session_state["_tune_results"] = cached_tune_all(
                tuple(np.round(np.asarray(plant.num, float), 10)),
                tuple(np.round(np.asarray(plant.den, float), 10)),
                float(plant.delay), str(tune_mode), float(decay_ratio), bool(include_opt))
            st.session_state["_tune_key"] = tkey
            st.session_state["_tune_time"] = time.time() - t0
    tune_results = st.session_state["_tune_results"]

    valid_results = [r for r in tune_results if r.valid]
    st.caption(f"整定耗时 {st.session_state.get('_tune_time', 0):.2f} s（结果已缓存，修改对象参数或点击「重新整定」才会重算）")

    # ---- 整定公式卡 ----
    with st.expander("📐 整定公式卡", expanded=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            st.markdown("**Z-N 临界比例度法**")
            st.dataframe(pd.DataFrame([{"模式": m, **ZN_TABLE[m]} for m in ["P", "PI", "PD", "PID"]]),
                         use_container_width=True, hide_index=True, key="df_004")
        with f2:
            st.markdown("**Z-N 阶跃响应法（反应曲线）**")
            st.dataframe(pd.DataFrame([{"模式": m, **ZN_OPEN_TABLE[m]} for m in ["P", "PI", "PD", "PID"]]),
                         use_container_width=True, hide_index=True, key="df_005")
        with f3:
            st.markdown(f"**衰减曲线法（{'4:1' if abs(decay_ratio-0.25)<1e-9 else '10:1'}）**")
            tb = DECAY_TABLE_41 if abs(decay_ratio - 0.25) < 1e-9 else DECAY_TABLE_101
            st.dataframe(pd.DataFrame([{"模式": m, **tb[m]} for m in ["P", "PI", "PD", "PID"]]),
                         use_container_width=True, hide_index=True, key="df_006")

    # ---- 整定参数表 ----
    st.markdown("#### ① 自动整定得到的 PID 参数")
    df_params = pd.DataFrame([r.as_row() for r in tune_results])
    st.dataframe(df_params, use_container_width=True, hide_index=True, key="df_007")

    for r in tune_results:
        if not r.valid:
            st.warning(f"**{r.method}** 不适用：{r.note}")

    colx, coly = st.columns(2)
    with colx:
        if isinstance(tune_results[0].extra.get("crossover"), tuple):
            st.plotly_chart(plots.bode_figure(plant, title="ZN 临界比例度法：由 Bode 图求 Ku、Pu",
                                              mark_crossover=True), use_container_width=True, key="pc_008")
    with coly:
        fop = fopdt_from_step(plant)
        if fop["valid"]:
            st.plotly_chart(plots.fopdt_figure(fop), use_container_width=True, key="pc_009")

    # ---- 整定结果仿真对比 ----
    st.markdown("#### ② 整定结果闭环仿真对比（含手动参数对照）")
    manual_res = sim(tune_mode, kp, ki, kd, label="手动参数（侧边栏）")
    comp_res = [manual_res]
    comp_met = [compute_metrics(manual_res.t, manual_res.y, manual_res.r, label="手动参数（侧边栏）")]
    for r in valid_results:
        try:
            rs, ms = evaluate_tuning(plant, r, t_end=t_end3, n_samples=st.session_state["sim_n"])
            comp_res.append(rs)
            comp_met.append(ms)
        except Exception as exc:
            st.error(f"{r.method} 仿真失败：{exc}")

    series3 = [{"t": r.t, "y": r.y, "name": r.label, "color": PALETTE[i]}
               for i, r in enumerate(comp_res)]
    st.plotly_chart(plots.line_figure(series3, title="手动参数 vs 三种自动整定结果",
                                      ref=1.0, ref_label="给定值 r(t)"), use_container_width=True, key="pc_010")
    st.plotly_chart(plots.output_control_figure(comp_res, title="整定结果的输出与控制量"), use_container_width=True, key="pc_011")

    st.markdown("#### ③ 整定结果性能指标对比表")
    st.dataframe(metrics_table(comp_met), use_container_width=True, hide_index=True, key="df_008")

    st.plotly_chart(plots.bar_figure(
        [m["label"] for m in comp_met],
        [{"name": "超调量 σ%", "values": [m["overshoot"] for m in comp_met]}],
        title="不同整定方法的超调量对比", ylabel="σ%"), use_container_width=True, key="pc_012")

    # ---- 一键应用 ----
    st.markdown("#### ④ 选中整定结果并应用到全局参数")
    apply_cols = st.columns(len(valid_results))
    for i, r in enumerate(valid_results):
        with apply_cols[i % len(apply_cols)]:
            st.markdown(f"**{r.method}**")
            st.write(f"Kp={fmt(r.kp,4)}  Ki={fmt(r.ki,4)}  Kd={fmt(r.kd,4)}")
            if st.button("应用到全局 PID", key=f"apply_tune_{i}", use_container_width=True):
                st.session_state["_pending_pid"] = (float(r.kp), float(r.ki), float(r.kd))
                st.rerun()

    best_t = max([m for m in comp_met[1:]] or comp_met, key=lambda m: m["score"])
    info_card(
        "<b>自动分析结论</b><br>"
        f"· 三种经典算法均基于工程经验公式：<b>ZN 法</b>整定激进、响应快但超调通常最大；"
        f"<b>衰减曲线法</b>以闭环衰减比为依据，超调与稳定性折中最优；"
        f"<b>数值寻优</b>直接以性能指标为目标函数，可得到针对性最强的参数。<br>"
        f"· 本对象下综合评分最高的是 <b>{best_t['label']}</b>（score={fmt(best_t['score'],3)}，"
        f"σ={fmt(best_t['overshoot'],3)}%，ts={fmt(best_t['ts_2'],3)} s）。<br>"
        f"· 整定公式只是<b>起点而非终点</b>：工程上应把自动整定结果作为初值，再依据实际约束（超调上限、"
        f"执行器饱和、噪声水平）微调。"
    )

    # ------------------------------------------------------------------ #
    #  试凑法（手动整定）工作流
    # ------------------------------------------------------------------ #
    st.divider()
    st.markdown("### 🖐 试凑法（手动整定）")
    st.caption("按经典口诀「先比例、后积分、再微分」逐步试探，每一步都记录参数与性能指标，"
               "形成可追溯、可复现的整定过程。")

    info_card(
        "<b>试凑法四步口诀</b><br>"
        "<b>第 1 步 · 纯比例 P</b>：令 Ki = Kd = 0，Kp 由小到大增加，直到阶跃响应出现约 20%~25% 超调、"
        "且振荡衰减较快为止。此时响应快，但存在稳态误差。<br>"
        "<b>第 2 步 · 加积分 PI</b>：保持 Kp，Ki 由小到大增加，直到稳态误差被消除；"
        "若超调明显变大，就把 Kp 略微减小。<br>"
        "<b>第 3 步 · 加微分 PID</b>：保持 Kp、Ki，Kd 由小到大增加，利用微分的超前作用抑制超调、"
        "缩短调节时间；Kd 过大则会放大噪声、引起高频抖动。<br>"
        "<b>第 4 步 · 折中微调</b>：在超调量、调节时间与稳定性之间折中，取综合表现最好的一组参数。"
    )

    if "tune_history" not in st.session_state:
        st.session_state["tune_history"] = []
    hist = st.session_state["tune_history"]
    hist_plant_key = plant_key(plant)

    if hist and hist[0].get("_plant_key") != hist_plant_key:
        st.warning("当前被控对象参数已改变，下面的历史记录来自之前设置的对象，曲线与指标不再可比。"
                   "如果要重新整定，请先点「🗑 清空记录」。")

    col_a, col_b = st.columns([1, 1.35])
    with col_a:
        st.markdown("#### ① 记录当前试探点")
        cur_res = sim(tune_mode, kp, ki, kd, label="当前试探点")
        m_cur = compute_metrics(cur_res.t, cur_res.y, cur_res.r, label="当前试探点")
        st.markdown(f"当前参数：`Kp={kp:g}　Ki={ki:g}　Kd={kd:g}`　（用左侧侧边栏实时调整）")
        q1, q2, q3 = st.columns(3)
        q1.metric("超调量 σ%", f"{fmt(m_cur['overshoot'], 3)}")
        q2.metric("调节时间", f"{fmt(m_cur['ts_2'], 3)} s")
        q3.metric("稳态误差", f"{fmt(m_cur['ess'], 3)}")
        step_note = st.text_input("本步说明", value=f"第 {len(hist) + 1} 步", key="tune_step_note",
                                  placeholder="例如：纯比例，Kp 加到 1.5")
        rb1, rb2 = st.columns(2)
        if rb1.button("📌 记录本步", use_container_width=True):
            st.session_state["tune_history"] = hist + [{
                "步骤": step_note or f"第 {len(hist) + 1} 步",
                "Kp": round(float(kp), 4), "Ki": round(float(ki), 4), "Kd": round(float(kd), 4),
                "超调量 σ%": round(float(m_cur["overshoot"]), 4) if np.isfinite(m_cur["overshoot"]) else None,
                "上升时间 tr(10-90%)": round(float(m_cur["tr"]), 4) if np.isfinite(m_cur["tr"]) else None,
                "调节时间 ts(2%)": round(float(m_cur["ts_2"]), 4) if np.isfinite(m_cur["ts_2"]) else None,
                "稳态误差 ess": round(float(m_cur["ess"]), 6),
                "IAE": round(float(m_cur["iae"]), 4),
                "综合评分": round(float(m_cur["score"]), 2),
                "_res": cur_res, "_plant_key": hist_plant_key,
            }]
            st.rerun()
        if rb2.button("🗑 清空记录", use_container_width=True):
            st.session_state["tune_history"] = []
            st.rerun()

        with st.expander("🎬 自动生成试凑过程演示（4 步）", expanded=not hist):
            st.caption("以「衰减曲线法」的整定结果作为折中目标，自动生成一条 P → PI → PID → 微调 的整定轨迹，"
                       "用于演示试凑法的逐步收敛过程。")
            if st.button("生成 4 步整定过程", use_container_width=True):
                tgt = next((r for r in tune_results if ("衰减曲线" in r.method) and r.valid), None)
                tgt = tgt or next((r for r in tune_results if r.valid), None)
                if tgt is None:
                    st.error("没有可用的自动整定结果作为折中目标，请先执行自动整定。")
                else:
                    seq = [
                        ("第 1 步 · 纯比例 P", 0.50 * tgt.kp, 0.0, 0.0,
                         "纯比例：提高开环增益，响应变快但存在稳态误差"),
                        ("第 2 步 · 加积分 PI", 0.60 * tgt.kp, 0.75 * tgt.ki, 0.0,
                         "引入积分：消除稳态误差，代价是超调增大"),
                        ("第 3 步 · 加微分 PID", 0.75 * tgt.kp, 0.90 * tgt.ki, 0.80 * tgt.kd,
                         "引入微分：抑制超调、增加阻尼"),
                        ("第 4 步 · 折中微调", 1.00 * tgt.kp, 1.00 * tgt.ki, 1.00 * tgt.kd,
                         "折中微调：取衰减曲线法的折中目标值"),
                    ]
                    recs = []
                    for nm, ga, gb, gc, nt in seq:
                        rr = sim(tune_mode, ga, gb, gc, label=nm)
                        mm = compute_metrics(rr.t, rr.y, rr.r, label=nm)
                        recs.append({
                            "步骤": nm,
                            "Kp": round(float(ga), 4), "Ki": round(float(gb), 4), "Kd": round(float(gc), 4),
                            "超调量 σ%": round(float(mm["overshoot"]), 4) if np.isfinite(mm["overshoot"]) else None,
                            "上升时间 tr(10-90%)": round(float(mm["tr"]), 4) if np.isfinite(mm["tr"]) else None,
                            "调节时间 ts(2%)": round(float(mm["ts_2"]), 4) if np.isfinite(mm["ts_2"]) else None,
                            "稳态误差 ess": round(float(mm["ess"]), 6),
                            "IAE": round(float(mm["iae"]), 4),
                            "综合评分": round(float(mm["score"]), 2),
                            "_res": rr, "_plant_key": hist_plant_key, "_note": nt,
                        })
                    st.session_state["tune_history"] = recs
                    st.rerun()

    with col_b:
        st.markdown("#### ② 逐步逼近过程曲线")
        if hist:
            series = []
            for i, h in enumerate(hist):
                series.append({
                    "t": h["_res"].t, "y": h["_res"].y,
                    "name": f'{h["步骤"]}（Kp={h["Kp"]:g}, Ki={h["Ki"]:g}, Kd={h["Kd"]:g}）',
                    "color": PALETTE[i % len(PALETTE)], "width": 2.2,
                })
            st.plotly_chart(plots.line_figure(series, title="试凑法整定过程 —— 逐步逼近", ref=1.0,
                                              ref_label="给定值 r(t)"), use_container_width=True, key="pc_013")
        else:
            st.info("还没有记录。可以：① 用侧边栏调参数后点「📌 记录本步」；"
                    "② 或展开下方的「自动生成试凑过程演示」一键生成 4 步整定轨迹。")

    if hist:
        st.markdown("#### ③ 试凑过程记录表")
        df_hist = pd.DataFrame([{k: v for k, v in h.items() if not k.startswith("_")} for h in hist])
        st.dataframe(df_hist, use_container_width=True, hide_index=True, key="df_009")

        best_h = max(hist, key=lambda h: (h.get("综合评分") if h.get("综合评分") is not None else -1))
        first_h, last_h = hist[0], hist[-1]
        info_card(
            "<b>过程评价</b><br>"
            f"· 共记录 <b>{len(hist)}</b> 步：超调量由第 1 步的 {fmt(first_h['超调量 σ%'], 3)}% "
            f"变化到末步的 {fmt(last_h['超调量 σ%'], 3)}%；"
            f"稳态误差由 {fmt(first_h['稳态误差 ess'], 3)} 变化到 {fmt(last_h['稳态误差 ess'], 3)}"
            f"（积分作用生效的标志）。<br>"
            f"· 本次过程中综合评分最高的是 <b>{best_h['步骤']}</b>"
            f"（Kp={best_h['Kp']:g}, Ki={best_h['Ki']:g}, Kd={best_h['Kd']:g}，"
            f"score={fmt(best_h['综合评分'], 3)}）。<br>"
            "· 试凑法的价值在于<b>过程可见</b>：每一步都记录了参数与指标，能清楚说明"
            "「比例决定快慢、积分消除静差、微分改善阻尼」这三条作用规律。"
        )
        st.caption("提示：侧边栏的 PID 参数是全局参数，这里的「当前试探点」始终跟随侧边栏；"
                   "记录只是把当前这组参数与指标存档，不影响其它功能页。")


# ========================================================================== #
#  功能页四：改进型 PID 控制器
# ========================================================================== #
with tab4:
    st.subheader("🔧 改进型 PID 控制器")
    st.caption("在标准 PID 上叠加三项工程改进措施并逐项对比：**积分分离**（抑制积分饱和超调）、"
               "**不完全微分**（抑制噪声放大）、**抗积分饱和**（反算法 / 条件积分 / 积分限幅）。")

    with st.expander("📐 三项改进的原理与公式", expanded=True):
        g1, g2, g3 = st.columns(3)
        with g1:
            st.markdown("**① 积分分离**")
            formula_card("u = Kp·e + Kd·ė + β·Ki·∫e dt<br>"
                         "β = 1，　|e| ≤ ε<br>"
                         "β = 0，　|e| > ε")
            st.caption("大偏差时切除积分并冻结积分器（等效纯 PD）→ 避免积分饱和引起的大超调；"
                       "小偏差时投入积分 → 仍能消除稳态误差。")
        with g2:
            st.markdown("**② 不完全微分**")
            formula_card("完全微分　：ud = Kd·de/dt<br>"
                         "不完全微分：ud = Kd·[ Td·s / (Td·s + 1) ]·e<br>"
                         "一阶滤波时间常数　Tf = Td/N")
            st.caption("理想微分会把高频噪声放大成控制量抖动；串入一阶惯性环节后高频增益被压低，"
                       "执行器动作明显变平滑。")
        with g3:
            st.markdown("**③ 抗积分饱和（三种方案）**")
            formula_card("反算法　：I ← I + (u_sat − u_raw)/Ki<br>"
                         "条件积分：饱和且误差同向时停止积分<br>"
                         "积分限幅：I ∈ [ u_min/Ki , u_max/Ki ]")
            st.caption("执行器饱和时防止积分器无限累积，避免「退饱和超调」。")

    st.markdown("#### 工况与改进措施设置")
    q1, q2, q3, q4 = st.columns(4)
    ref8 = q1.number_input("给定阶跃幅值", min_value=0.1, max_value=20.0, value=1.0, step=0.5,
                           key="ref8", help="调大到 6 左右，积分分离的效果会非常明显")
    noise8 = q2.number_input("量测噪声标准差", min_value=0.0, max_value=0.5, value=0.02,
                             step=0.01, format="%.3f", key="noise8",
                             help="调大到 0.05，不完全微分抑制抖动的效果会非常明显")
    use_sat8 = q3.checkbox("启用执行器饱和", value=True, key="sat8")
    umax8 = q4.number_input("饱和限幅 ±u_max", min_value=0.05, max_value=50.0, value=1.2,
                            step=0.1, key="umax8", disabled=not use_sat8,
                            help="要让抗饱和真正起作用，限幅值应【略大于稳态所需控制量】——"
                                 "即让执行器只在暂态短暂饱和。若限幅小于稳态需求，系统会一直顶在限幅上，"
                                 "所有方案曲线都会变得一样。")

    r1c, r2c, r3c = st.columns(3)
    use_sep8 = r1c.checkbox("启用积分分离", value=True, key="sep8")
    eps8 = r1c.slider("分离阈值 ε", 0.01, 5.0, 0.5, 0.01, key="eps8", disabled=not use_sep8)
    use_df8 = r2c.checkbox("启用不完全微分", value=True, key="df8")
    n8 = r2c.slider("微分滤波系数 N", 2.0, 50.0, 10.0, 1.0, key="n8",
                    disabled=not use_df8, help="N 越大越接近理想微分")
    aw8 = r3c.selectbox("抗饱和方案", list(AW_MODES.keys()),
                        format_func=lambda kk: AW_MODES[kk], index=1, key="aw8")
    r3c.caption("以上三项为「三项全开」时采用的设置")

    act8 = Actuator(enabled=use_sat8, u_min=-float(umax8), u_max=float(umax8)) if use_sat8 else None
    t_end8 = float(st.session_state["sim_t_end"])
    n_samp8 = int(st.session_state["sim_n"])
    dt8 = t_end8 / n_samp8

    def _run8(name, *, sep, df, aw):
        """按指定改进组合跑一次仿真，并附加控制量品质指标。"""
        rr = _sim_cached(
            tuple(np.round(np.asarray(plant.num, float), 10)),
            tuple(np.round(np.asarray(plant.den, float), 10)), float(plant.delay),
            float(kp), float(ki), float(kd), "PID", float(n8), t_end8, n_samp8, float(ref8),
            _act_signature(act8), None, str(aw), float(noise8), 7,
            bool(sep), float(eps8), bool(df), str(name))
        mm = compute_metrics(rr.t, rr.y, rr.r, label=name)
        cq = control_quality(rr.u, dt8)
        mm["tv"] = cq["tv"]
        mm["u_max"] = cq["u_max"]
        return rr, mm

    def _enhanced_table(res_list, met_list):
        d = report.build_metrics_sheet(res_list, met_list)
        d["控制量总变差 TV"] = [round(m["tv"], 2) for m in met_list]
        d["控制量峰值 |u|max"] = [round(m["u_max"], 3) for m in met_list]
        return d

    st.divider()
    st.markdown("#### 对比 A：改进措施逐项叠加")
    variants = [
        ("① 标准 PID", dict(sep=False, df=False, aw="none")),
        ("② 仅积分分离", dict(sep=use_sep8, df=False, aw="none")),
        ("③ 仅不完全微分", dict(sep=False, df=use_df8, aw="none")),
        ("④ 仅抗饱和", dict(sep=False, df=False, aw=aw8)),
        ("⑤ 三项全开", dict(sep=use_sep8, df=use_df8, aw=aw8)),
    ]
    res8, met8 = [], []
    for nm8, kw8 in variants:
        rr8, mm8 = _run8(nm8, **kw8)
        res8.append(rr8)
        met8.append(mm8)

    # 饱和占比自检：若长期顶在限幅上，对比会失去意义
    if use_sat8 and res8:
        sat_ratio = float(np.mean(np.abs(res8[0].u) >= float(umax8) - 1e-9))
        if sat_ratio > 0.5:
            st.warning(
                f"⚠️ 当前执行器有 **{sat_ratio * 100:.0f}%** 的时间顶在限幅 ±{umax8:g} 上 —— "
                f"说明限幅值偏小，系统**追不上给定值**，此时各种方案曲线会趋同、失去对比意义。"
                f"建议把限幅调大到略大于稳态所需控制量（约 {ref8:g} 左右），或把阶跃幅值调小。")

    st.plotly_chart(plots.line_figure(
        [{"t": rr.t, "y": rr.y, "name": rr.label, "color": PALETTE[i % len(PALETTE)]}
         for i, rr in enumerate(res8)],
        title=f"改进措施逐项叠加（阶跃幅值 {ref8:g}，量测噪声 {noise8:g}）",
        ref=float(ref8), ref_label="给定值 r(t)"), use_container_width=True, key="pc_014")
    st.plotly_chart(plots.output_control_figure(res8, title="各方案的控制量对比（注意抖动幅度）"),
                    use_container_width=True, key="pc_015")
    st.dataframe(_enhanced_table(res8, met8), use_container_width=True, hide_index=True, key="df_010")

    st.divider()
    st.markdown("#### 对比 B：三种抗饱和方案")
    if not use_sat8:
        st.info("当前未启用执行器饱和 —— 三种抗饱和方案只会在饱和时起作用，勾选上面的「启用执行器饱和」后再看本对比。")
    aw_list = [("无抗饱和", "none"), ("反算法", "back"),
               ("条件积分", "conditional"), ("积分限幅", "clamping")]
    aw_res, aw_met = [], []
    for nm8, mkey in aw_list:
        rr8, mm8 = _run8(nm8, sep=False, df=False, aw=mkey)
        aw_res.append(rr8)
        aw_met.append(mm8)
    st.plotly_chart(plots.line_figure(
        [{"t": rr.t, "y": rr.y, "name": rr.label, "color": PALETTE[i % len(PALETTE)]}
         for i, rr in enumerate(aw_res)],
        title=f"抗饱和方案对比（限幅 ±{umax8:g}，仅切换抗饱和方式）",
        ref=float(ref8), ref_label="给定值 r(t)"), use_container_width=True, key="pc_016")
    st.dataframe(_enhanced_table(aw_res, aw_met), use_container_width=True, hide_index=True, key="df_011")

    st.divider()
    st.markdown("#### 对比 C：完全微分 vs 不完全微分（含量测噪声）")
    df_res, df_met = [], []
    for nm8, use_f in [("完全微分（理想微分）", False), ("不完全微分（一阶滤波）", True)]:
        rr8, mm8 = _run8(nm8, sep=False, df=use_f, aw="none")
        df_res.append(rr8)
        df_met.append(mm8)
    c_left, c_right = st.columns(2)
    with c_left:
        st.plotly_chart(plots.line_figure(
            [{"t": rr.t, "y": rr.y, "name": rr.label, "color": PALETTE[i]}
             for i, rr in enumerate(df_res)],
            title="输出响应（两者跟踪性能接近）", ref=float(ref8), ref_label="给定值"),
            use_container_width=True, key="pc_017")
    with c_right:
        st.plotly_chart(plots.line_figure(
            [{"t": rr.t, "y": rr.u, "name": rr.label, "color": PALETTE[i]}
             for i, rr in enumerate(df_res)],
            title="控制量（差异在这里：抖动幅度）", xlabel="时间 t / s", ylabel="控制量 u(t)"),
            use_container_width=True, key="pc_018")
    tv_a = df_met[0]["tv"]
    tv_b = df_met[1]["tv"]
    kd1, kd2, kd3 = st.columns(3)
    kd1.metric("完全微分 控制量总变差", f"{tv_a:,.1f}")
    kd2.metric("不完全微分 控制量总变差", f"{tv_b:,.1f}",
               f"降低 {(1 - tv_b / tv_a) * 100:.1f}%" if tv_a > 0 else None, delta_color="inverse")
    kd3.metric("控制量峰值降低", f"{(1 - df_met[1]['u_max'] / df_met[0]['u_max']) * 100:.1f} %"
               if df_met[0]["u_max"] > 0 else "—")
    st.dataframe(_enhanced_table(df_res, df_met), use_container_width=True, hide_index=True, key="df_012")

    # ---- 自动结论 ----
    base_m = met8[0]
    best_m = max(met8[1:], key=lambda m: m["score"]) if len(met8) > 1 else base_m
    aw_best = min(aw_met, key=lambda m: m["overshoot"] if np.isfinite(m["overshoot"]) else 1e9)
    info_card(
        "<b>自动分析结论</b><br>"
        f"· <b>基准（标准 PID）</b>：超调 σ = {fmt(base_m['overshoot'], 3)}%，"
        f"调节时间 ts = {fmt(base_m['ts_2'], 3)} s，IAE = {fmt(base_m['iae'], 4)}，"
        f"控制量总变差 TV = {base_m['tv']:,.1f}。<br>"
        f"· <b>改进后最佳</b>：<b>{best_m['label']}</b>，σ = {fmt(best_m['overshoot'], 3)}%，"
        f"ts = {fmt(best_m['ts_2'], 3)} s，IAE = {fmt(best_m['iae'], 4)}，"
        f"综合评分 {fmt(best_m['score'], 3)}（基准为 {fmt(base_m['score'], 3)}）。<br>"
        f"· <b>抗饱和方案</b>中，本工况下超调最小的是 <b>{aw_best['label']}</b>"
        f"（σ = {fmt(aw_best['overshoot'], 3)}%）。三种方案都能显著压低退饱和超调，"
        f"但通常以调节时间变长为代价 —— 这是「稳定性」与「快速性」的典型折中。<br>"
        f"· <b>不完全微分</b>的意义不在跟踪指标，而在<b>执行器友好</b>："
        f"控制量总变差由 {tv_a:,.1f} 降到 {tv_b:,.1f}，说明理想微分把噪声放大成了执行器的高频动作，"
        f"而一阶滤波把它压了下去 —— 这正是工程上几乎不用纯微分的原因。"
    )

    with st.expander("📝 答辩要点：这三项改进为什么值得做"):
        st.markdown("""
1. **积分分离回答"大偏差时积分反而帮倒忙"**：偏差大时积分器持续累积，一旦执行器饱和就无法及时退出，
   表现为巨大的退饱和超调甚至振荡。分离后大偏差段等效纯 PD，只在小偏差段用积分收尾。
2. **不完全微分回答"微分为什么不能直接用"**：理想微分 `Kd·s` 对高频的增益无上限，
   传感器噪声会被放大几十倍送进执行器。串一阶惯性环节后高频增益被压到有限值。
3. **抗饱和回答"执行器有物理上限怎么办"**：反算法用 `(u_sat−u_raw)/Ki` 把积分器拉回来；
   条件积分在饱和时干脆不积分；积分限幅则直接给积分项设上下界。三者都可，工程上反算法最常用。
4. **可以强调的定量证据**：本文用「控制量总变差 TV」量化了微分形式对执行器的影响，
   而不只是画曲线定性描述。
""")


# ========================================================================== #
#  功能页五：抗负载干扰仿真
# ========================================================================== #
with tab5:
    st.subheader("⚡ 抗负载干扰仿真")
    st.caption("模拟系统运行中突然加入负载扰动，考察不同控制器的最大动态偏差、恢复时间与扰动后误差积分，"
               "定量评价 PID 的抗干扰能力。")

    d1, d2, d3, d4 = st.columns([1.3, 1, 1, 1])
    dtype = d1.selectbox("扰动类型", ["阶跃负载扰动", "脉冲扰动", "正弦扰动", "斜坡扰动"])
    t_dist = d2.number_input("扰动加入时刻 (s)", min_value=0.0, max_value=float(st.session_state["sim_t_end"]),
                             value=float(st.session_state["sim_t_end"]) * 0.4, step=0.5)
    dmag = d3.number_input("扰动幅值", min_value=-10.0, max_value=10.0, value=0.3, step=0.05, format="%.3f")
    extra_par = 1.0
    if dtype == "脉冲扰动":
        extra_par = d4.number_input("脉冲宽度 (s)", min_value=0.05, max_value=20.0, value=1.0, step=0.1)
    elif dtype == "正弦扰动":
        extra_par = d4.number_input("扰动频率 (Hz)", min_value=0.01, max_value=10.0, value=0.5, step=0.05)
    else:
        d4.metric("扰动时刻占比", f"{t_dist/max(st.session_state['sim_t_end'],1e-9)*100:.0f} %")

    t_end4 = float(st.session_state["sim_t_end"])
    if dtype == "阶跃负载扰动":
        dist_fn = step_disturbance(t_dist, dmag)
    elif dtype == "脉冲扰动":
        dist_fn = pulse_disturbance(t_dist, dmag, extra_par)
    elif dtype == "正弦扰动":
        dist_fn = sine_disturbance(t_dist, dmag, extra_par, duration=t_end4 - t_dist)
    else:
        dist_fn = ramp_disturbance(t_dist, abs(dmag) if dmag != 0 else 0.1)

    res_dist, met_dist, dist_rows = {}, {}, []
    for mode in MODES:
        r = sim(mode, kp, ki, kd, disturbance=dist_fn, label=f"{mode} 控制")
        res_dist[mode] = r
        y_ref_pre = float(np.mean(r.y[(r.t >= max(0, t_dist - 2.0)) & (r.t < t_dist)])) if t_dist > 0 else steady_value(plant)
        dm = disturbance_metrics(r.t, r.y, y_ref_pre, t_dist)
        met_dist[mode] = {**compute_metrics(r.t, r.y, r.r, label=f"{mode} 控制"),
                          "dist_max": dm["max_dev"], "dist_recovery": dm["recovery"],
                          "dist_iae": dm["iae_dist"], "dist_overshoot": dm["overshoot_after"]}
        dist_rows.append({"控制模式": f"{mode} 控制",
                          "稳态值(扰动前)": round(y_ref_pre, 4),
                          "最大动态偏差": round(dm["max_dev"], 4),
                          "最大偏差(%)": round(dm["max_dev_rel"], 3) if np.isfinite(dm["max_dev_rel"]) else None,
                          "恢复时间(s)": round(dm["recovery"], 4) if np.isfinite(dm["recovery"]) else None,
                          "扰动后 IAE": round(dm["iae_dist"], 4),
                          "是否消除扰动": "是" if abs(dm["max_dev"]) < 1e-3 else "否"})

    pid_base = sim("PID", kp, ki, kd, label="PID（无扰动对照）")
    st.markdown("#### 抗扰性能指标")
    dc1, dc2, dc3, dc4 = st.columns(4)
    dm_best = min(MODES, key=lambda m: abs(met_dist[m]["dist_max"]))
    dc1.metric("最大动态偏差最小", dm_best, f"{fmt(abs(met_dist[dm_best]['dist_max']),3)}")
    dr_best = min(MODES, key=lambda m: (met_dist[m]["dist_recovery"] if np.isfinite(met_dist[m]["dist_recovery"]) else 1e9))
    dc2.metric("恢复最快", dr_best, f"{fmt(met_dist[dr_best]['dist_recovery'],3)} s")
    dc3.metric("扰动后 IAE 最小", min(MODES, key=lambda m: met_dist[m]["dist_iae"]),
               f"{fmt(min(met_dist[m]['dist_iae'] for m in MODES),3)}")
    dc4.metric("扰动幅值", f"{dmag:g}", dtype)

    dist_series = [{"t": res_dist[m].t, "y": res_dist[m].y, "name": f"{m} 控制", "color": PALETTE[i]}
                   for i, m in enumerate(MODES)]
    dist_series.insert(0, {"t": pid_base.t, "y": pid_base.y, "name": "无扰动对照",
                           "color": "#9aa5b1", "dash": "dot", "width": 1.6})
    f_dist = plots.line_figure(dist_series, title=f"抗干扰响应对比（{dtype}，t={t_dist:g}s 加入，幅值 {dmag:g}）",
                               ref=1.0, ref_label="给定值 r(t)")
    f_dist.add_vline(x=t_dist, line=dict(color="#d62728", dash="dash", width=1.4),
                     annotation_text="扰动加入", annotation_position="top")
    st.plotly_chart(f_dist, use_container_width=True, key="pc_019")
    st.plotly_chart(plots.output_control_figure([res_dist[m] for m in MODES],
                                                title="抗干扰条件下各控制器的控制量变化"),
                    use_container_width=True, key="pc_020")

    st.markdown("#### 抗干扰性能指标表")
    st.dataframe(pd.DataFrame(dist_rows), use_container_width=True, hide_index=True, key="df_013")

    # 抗积分饱和对比
    st.markdown("#### 抗积分饱和（Anti-Windup）效果对比")
    aw1, aw2 = st.columns([1, 3])
    with aw1:
        use_aw_demo = st.checkbox("启用饱和并对比抗饱和", value=False)
        u_lim = st.number_input("执行器限幅 ±u_max", min_value=0.1, max_value=100.0, value=2.0, step=0.5)
    if use_aw_demo:
        act_aw = Actuator(enabled=True, u_min=-u_lim, u_max=u_lim)
        r_aw = sim("PID", kp, ki, kd, disturbance=dist_fn, actuator=act_aw, anti_windup=True, label="抗积分饱和")
        r_noaw = sim("PID", kp, ki, kd, disturbance=dist_fn, actuator=act_aw, anti_windup=False, label="无抗积分饱和")
        with aw2:
            st.plotly_chart(plots.line_figure(
                [{"t": r_aw.t, "y": r_aw.y, "name": "抗积分饱和", "color": PALETTE[2]},
                 {"t": r_noaw.t, "y": r_noaw.y, "name": "无抗积分饱和", "color": PALETTE[1]}],
                title=f"抗积分饱和对比（限幅 ±{u_lim:g}）", ref=1.0, ref_label="给定值"),
                use_container_width=True, key="pc_021")
        m_aw = compute_metrics(r_aw.t, r_aw.y, r_aw.r, label="抗积分饱和")
        m_noaw = compute_metrics(r_noaw.t, r_noaw.y, r_noaw.r, label="无抗积分饱和")
        st.dataframe(metrics_table([m_aw, m_noaw]), use_container_width=True, hide_index=True, key="df_014")
        info_card(
            f"抗饱和后超调量由 <b>{fmt(m_noaw['overshoot'],3)}%</b> 降到 <b>{fmt(m_aw['overshoot'],3)}%</b>，"
            f"IAE 由 {fmt(m_noaw['iae'],4)} 变为 {fmt(m_aw['iae'],4)}。"
            "反算法(back-calculation)在控制器输出超出执行器限幅时，按 (u_sat−u_raw)/Ki 反向修正积分项，"
            "使积分器不会继续累积到饱和区，从而显著减小退饱和超调。")
    else:
        with aw2:
            st.plotly_chart(plots.line_figure(
                [{"t": res_dist[m].t, "y": res_dist[m].y, "name": f"{m} 控制", "color": PALETTE[i]}
                 for i, m in enumerate(MODES)],
                title="各模式在扰动下的输出（勾选左侧可附加饱和对比）", ref=1.0),
                use_container_width=True, key="pc_022")

    _pi_d = met_dist["PI"]; _p_d = met_dist["P"]; _pid_d = met_dist["PID"]
    info_card(
        "<b>自动分析结论</b><br>"
        f"· 扰动突加瞬间，纯比例控制（P）的动态偏差最大（{fmt(_p_d['dist_max'],3)}），恢复也最慢；"
        f"因为比例环节只能按当前偏差产生控制作用。<br>"
        f"· 引入积分后（PI/PID），积分器会持续累积并最终把稳态偏差压回 0，"
        f"PID 的最大动态偏差为 {fmt(_pid_d['dist_max'],3)}、恢复时间 {fmt(_pid_d['dist_recovery'],3)} s。<br>"
        f"· <b>抗干扰 ≠ 抗超调</b>：过强的积分会带来更大的动态偏差与更长的恢复过程，"
        f"工程上常用「积分分离」「抗积分饱和」「微分先行」等改进措施折中。"
    )
    with st.expander("📐 抗干扰指标定义"):
        formula_card(
            "最大动态偏差： max|y(t) − y_ss| ,  t ≥ t_dist<br>"
            "恢复时间   ： 扰动后输出重新进入并保持在 ±2%·y_ss 误差带内所需时间<br>"
            "扰动后 IAE： ∫_{t_dist}^{T} |y(t) − y_ss| dt")
        st.markdown("""
- 负载扰动加在**对象输入端**（如电机突加负载、换热器突增冷料流量），控制器的任务是尽快把它"压"回去。
- 纯比例控制对付扰动只能"等偏了才纠"，必然存在动态偏差；积分项靠时间累积消除稳态偏差，但会引入相位滞后。
- 因此工程上抗扰设计的关键是：**足够的开环增益 + 合理积分时间 + 抗积分饱和保护**。
""")


# ========================================================================== #
#  功能页六：执行器非线性特性仿真
# ========================================================================== #
with tab6:
    st.subheader("🧩 执行器非线性特性仿真")
    st.caption("在 PID 闭环中叠加死区、饱和与速率限幅三类典型非线性，对比线性与非线性的控制效果，"
               "并用描述函数等数学工具做定量分析。")

    n1, n2, n3 = st.columns([1.2, 1.2, 1.2])
    with n1:
        use_dz = st.checkbox("✅ 死区特性（不灵敏区）", value=False)
        dz = st.slider("死区半宽 δ", 0.0, 2.0, 0.1, 0.01, disabled=not use_dz)
    with n2:
        use_sat = st.checkbox("✅ 饱和特性（限幅）", value=False)
        u_max5 = st.number_input("饱和上限 u_max", min_value=0.05, max_value=100.0, value=1.2, step=0.05, disabled=not use_sat)
    with n3:
        use_rate = st.checkbox("✅ 速率限幅", value=False)
        rate5 = st.number_input("最大变化速率 (单位/s)", min_value=0.05, max_value=500.0, value=5.0, step=0.5, disabled=not use_rate)

    mode5 = st.selectbox("控制器模式", ["PID", "PI", "PD", "P"], index=0, key="mode5")
    use_aw5 = st.checkbox("启用抗积分饱和", value=True, key="aw5")

    act5 = Actuator(enabled=use_sat, u_min=-float(u_max5), u_max=float(u_max5), dead_zone=float(dz),
                    rate_limit=float(rate5), enabled_dead_zone=use_dz, enabled_rate_limit=use_rate)
    lin5 = sim(mode5, kp, ki, kd, label="线性系统（无非线性）")
    nl5 = sim(mode5, kp, ki, kd, actuator=act5, anti_windup=use_aw5,
              label=f"非线性系统（{act5.label()}）")
    m_lin5 = compute_metrics(lin5.t, lin5.y, lin5.r, label="线性系统")
    m_nl5 = compute_metrics(nl5.t, nl5.y, nl5.r, label="非线性系统")

    st.markdown("#### 线性与非线性对比指标")
    nc1, nc2, nc3, nc4 = st.columns(4)
    nc1.metric("非线性工况", act5.label())
    nc2.metric("超调量 σ%", f"{fmt(m_nl5['overshoot'],3)} %", f"线性 {fmt(m_lin5['overshoot'],3)} %", delta_color="off")
    nc3.metric("调节时间 ts(2%)", f"{fmt(m_nl5['ts_2'],3)} s", f"线性 {fmt(m_lin5['ts_2'],3)} s", delta_color="off")
    nc4.metric("稳态误差 ess", f"{fmt(m_nl5['ess'],3)}", f"线性 {fmt(m_lin5['ess'],3)}", delta_color="off")

    st.plotly_chart(plots.line_figure(
        [{"t": lin5.t, "y": lin5.y, "name": "线性系统", "color": PALETTE[0]},
         {"t": nl5.t, "y": nl5.y, "name": f"非线性：{act5.label()}", "color": PALETTE[1]}],
        title="线性 vs 非线性工况下的闭环阶跃响应", ref=1.0, ref_label="给定值 r(t)"),
        use_container_width=True, key="pc_023")

    st.plotly_chart(plots.output_control_figure([lin5, nl5], title="控制量对比（观察限幅/死区/限速的影响"),
                    use_container_width=True, key="pc_024")

    st.markdown("#### 线性 vs 非线性 性能指标对比")
    st.dataframe(metrics_table([m_lin5, m_nl5]), use_container_width=True, hide_index=True, key="df_015")

    # 抗饱和开关（仅饱和时有意义）
    if use_sat:
        nl5b = sim(mode5, kp, ki, kd, actuator=act5, anti_windup=not use_aw5, label="抗饱和开关对照")
        m_nl5b = compute_metrics(nl5b.t, nl5b.y, nl5b.r, label="抗饱和开关对照")
        st.markdown("#### 饱和工况下 抗积分饱和 on/off 对比")
        st.dataframe(metrics_table([m_nl5, m_nl5b]), use_container_width=True, hide_index=True, key="df_016")

    # ---- 数学分析：描述函数 ----
    dA = (np.linspace(max(dz * 1.05, 1e-3), max(dz * 6, 1.0), 240) if dz > 1e-6 else np.array([]))
    if dA.size:
        ratio = dz / dA
        N_A = 1 - (2 / np.pi) * (np.arcsin(np.clip(ratio, -1, 1)) + ratio * np.sqrt(np.clip(1 - ratio ** 2, 0, None)))
    st.markdown("#### 死区非线性的描述函数（信计数学工具）")
    fx1, fx2 = st.columns([1, 2])
    with fx1:
        formula_card(
            "死区特性：<br>"
            "u = 0,             |e| ≤ δ<br>"
            "u = e − δ·sign(e), |e| > δ<br><br>"
            "描述函数 N(A) = 1 − (2/π)[ arcsin(δ/A) + (δ/A)√(1−(δ/A)²) ]<br>"
            "A 为输入正弦幅值，且 A > δ")
    with fx2:
        if dA.size:
            st.plotly_chart(plots.line_figure(
                [{"t": dA, "y": N_A, "name": "死区描述函数 N(A)", "color": PALETTE[4], "width": 2.6}],
                title="死区非线性描述函数随输入幅值的变化", xlabel="正弦输入幅值 A", ylabel="等效增益 N(A)"),
                use_container_width=True, key="pc_025")
        else:
            st.info("把「死区半宽 δ」调成大于 0，即可显示死区描述函数曲线。")

    info_card(
        "<b>自动分析结论</b><br>"
        f"· <b>死区</b>：小偏差时执行器无输出，等效开环增益下降，表现为稳态误差增大、定位精度下降，"
        f"并可能引起小幅自持振荡（极限环）。当前 δ={dz:g} 下 ess 由 {fmt(m_lin5['ess'],3)} 变为 {fmt(m_nl5['ess'],3)}。<br>"
        f"· <b>饱和</b>：大信号时开环增益骤降，等效于「降低增益 + 附加相位滞后」，会导致响应变慢、超调增大；"
        f"若不加抗积分饱和，积分器会持续累积到饱和区，产生严重的退饱和超调。<br>"
        f"· <b>速率限幅</b>：直接限制了控制量的变化速度，相当于附加一个滞后环节，会削弱微分作用的效果。<br>"
        f"· <b>数学视角</b>：三类非线性都可用描述函数 N(A) 近似为「幅值相关的等效增益」，"
        f"从而把非线性系统在局部线性化后用 Nyquist 判据分析极限环的存在条件——这正是信计专业数学工具与控制理论的结合点。"
    )


# ========================================================================== #
#  功能页七：频域分析与稳定性判据
# ========================================================================== #
with tab7:
    st.subheader("📐 频域分析与稳定性判据")
    st.caption("以开环频率特性 L(jω) = C(jω)·G(jω) 为基础，计算幅值裕度与相位裕度、绘制 Nyquist 图与根轨迹，"
               "并用 Nyquist 判据 Z = P − N 判定闭环稳定性（三种方法互相印证）。")

    ctl1, ctl2, ctl3 = st.columns([1.1, 1.1, 2.0])
    use_pade = ctl1.checkbox("纯滞后用一阶 Padé 近似", value=True,
                             help="根轨迹与 Nyquist 判据要求有理传递函数；勾选后用 "
                                  "e^(−τs) ≈ (1 − τs/2)/(1 + τs/2) 替代纯滞后")
    k_max7 = ctl2.slider("根轨迹增益范围 k_max", 2.0, 200.0, 20.0, 1.0,
                         help="k 是开环增益的整体放大倍数；k = 1 即当前侧边栏 PID 参数")
    ctl3.caption("说明：幅值裕度 h = 1/|L(jω_pc)|（dB 表示），相位裕度 γ = 180° + ∠L(jω_gc)；"
                 "两者均大于 0 表示闭环稳定。")

    pid7 = PID(kp=kp, ki=ki, kd=kd, mode="PID", N=st.session_state["pid_N"], name="当前参数")
    margins7 = freq.stability_margins(plant, pid7)
    nyq7 = freq.nyquist_analysis(plant, pid7, use_pade=use_pade)
    rl7 = freq.root_locus(plant, pid7, k_max=float(k_max7), n=500, use_pade=use_pade)
    grade7, grade_desc7, _gc = freq.margin_grade(margins7)

    kk1, kk2, kk3, kk4, kk5 = st.columns(5)
    kk1.metric("幅值裕度 h", "∞（不穿越 −180°）" if margins7["gm_db"] is None else f"{margins7['gm_db']:.2f} dB")
    kk2.metric("相位裕度 γ", "—" if margins7["pm"] is None else f"{margins7['pm']:.2f} °")
    kk3.metric("幅值穿越频率 ωgc", "—" if margins7["w_gc"] is None else f"{margins7['w_gc']:.3f} rad/s")
    kk4.metric("相位穿越频率 ωpc", "—" if margins7["w_pc"] is None else f"{margins7['w_pc']:.3f} rad/s")
    kk5.metric("闭环稳定性", grade7, f"右半平面极点数 Z = {nyq7['Z_actual']}", delta_color="off")

    info_card(
        f"<b>稳定性判定：{grade7}</b> —— {grade_desc7}<br>"
        f"· <b>Nyquist 判据</b>：开环右半平面极点数 P = {nyq7['P']}，"
        f"Nyquist 曲线绕临界点 (−1, j0) 的净逆时针圈数 N = {nyq7['N']}，"
        f"故闭环右半平面极点数 Z = P − N = <b>{nyq7['Z_nyquist']}</b>；"
        f"直接求解闭环特征方程得到的 Z = <b>{nyq7['Z_actual']}</b> —— "
        f"{'两者一致，判据自洽 ✓' if nyq7['consistent'] else '两者不一致，请检查模型或频率扫描范围'}。<br>"
        f"· <b>根轨迹佐证</b>：k = 1（当前参数）时闭环有 <b>{rl7['current_rhp']}</b> 个右半平面极点，"
        f"与上述结论{'一致' if rl7['current_rhp'] == nyq7['Z_actual'] else '不一致'}。<br>"
        f"· <b>参数调整建议</b>：<br>&nbsp;&nbsp;&nbsp;" +
        "<br>&nbsp;&nbsp;&nbsp;".join("· " + t for t in freq.margin_advice(margins7))
    )

    st.markdown("#### ① 开环 Bode 图与稳定裕度标注")
    st.plotly_chart(plots.bode_margins_figure(plant, pid7, margins7), use_container_width=True, key="pc_026")

    col_n, col_r = st.columns(2)
    with col_n:
        st.markdown("#### ② Nyquist 图")
        st.plotly_chart(plots.nyquist_annotated_figure(nyq7, margins7), use_container_width=True, key="pc_027")
    with col_r:
        st.markdown("#### ③ 根轨迹")
        st.plotly_chart(plots.root_locus_figure(rl7), use_container_width=True, key="pc_028")

    st.markdown("#### ④ 各整定方案的稳定裕度对比")
    rows7 = []
    _cands = [("手动参数（侧边栏）", PID(kp=kp, ki=ki, kd=kd, mode=tune_mode))]
    _cands += [(r.method, r.to_pid()) for r in tune_results if r.valid]
    for nm7, pid_7 in _cands:
        mm7 = freq.stability_margins(plant, pid_7, n=6000)
        nn7 = freq.nyquist_analysis(plant, pid_7, n=6000, use_pade=use_pade)
        g7, _, _ = freq.margin_grade(mm7)
        rows7.append({
            "方案": nm7,
            "幅值裕度 (dB)": None if mm7["gm_db"] is None else round(float(mm7["gm_db"]), 2),
            "相位裕度 (°)": None if mm7["pm"] is None else round(float(mm7["pm"]), 2),
            "ωgc (rad/s)": None if mm7["w_gc"] is None else round(float(mm7["w_gc"]), 4),
            "ωpc (rad/s)": None if mm7["w_pc"] is None else round(float(mm7["w_pc"]), 4),
            "裕度评价": g7,
            "闭环右极点数": nn7["Z_actual"],
            "闭环稳定": "是" if nn7["Z_actual"] == 0 else "否",
        })
    st.dataframe(pd.DataFrame(rows7), use_container_width=True, hide_index=True, key="df_017")

    with st.expander("📐 判据公式与工程含义（可对照检查）"):
        formula_card(
            "开环传递函数： L(s) = C(s)·G(s)，含纯滞后 e^(−τs)<br><br>"
            "幅值裕度： h = 1 / |L(jω_pc)|，  ∠L(jω_pc) = −180°<br>"
            "　　　　　 h_dB = −20·lg|L(jω_pc)| （h_dB > 0 稳定）<br><br>"
            "相位裕度： γ = 180° + ∠L(jω_gc)，  |L(jω_gc)| = 1<br>"
            "　　　　　（γ > 0 稳定）<br><br>"
            "Nyquist 判据： Z = P − N<br>"
            "　　P：开环右半平面极点数；N：绕 (−1, j0) 的净逆时针圈数；Z：闭环右半平面极点数<br><br>"
            "一阶 Padé 近似： e^(−τs) ≈ (1 − τs/2) / (1 + τs/2)")
        st.markdown("""
- **幅值裕度**回答"增益还能放大多少倍才失稳"；**相位裕度**回答"还能再增加多少相位滞后才失稳"。
- 工程经验：γ 取 **30°~60°**、h 取 **6~20 dB** 较合适。γ 太小则超调大、抗扰差；γ 太大则响应迟钝。
- **纯滞后是稳定性的最大杀手**：τ 每增大一点，相位就多滞后 ωτ，幅值裕度与相位裕度同时下降。可以把侧边栏的 τ 慢慢调大，观察本页裕度如何恶化直至失稳。
- **三种判据互相印证**：Bode 图的 GM/PM、Nyquist 的 Z = P − N、根轨迹中极点是否越过虚轴，结论必须一致——本页会自动校验并给出提示。
""")

# ========================================================================== #
#  功能页八：参数扫描与性能热力图
# ========================================================================== #
with tab9:
    st.subheader("🗺 参数扫描与性能热力图")
    st.caption("在 **(Kp, Ki)**、**(Kp, Kd)** 或 **(Ki, Kd)** 平面上网格化扫描，把超调量 / 调节时间 / "
               "IAE / 综合评分画成热力图 —— 一眼就能看出「参数该往哪调」，并自动定位最优格点。")

    p1, p2, p3 = st.columns([1.1, 1.1, 2.4])
    plane = p1.selectbox("扫描平面", ["Kp - Ki", "Kp - Kd", "Ki - Kd"], key="sw_plane")
    grid_lvl = p2.selectbox("网格精度", ["快 (9×9)", "标准 (11×11)", "精细 (15×15)", "很细 (21×21)"],
                            index=1, key="sw_grid",
                            help="格点数 = 仿真次数，精度越高越慢（结果会缓存）")
    n_grid = {"快 (9×9)": 9, "标准 (11×11)": 11,
              "精细 (15×15)": 15, "很细 (21×21)": 21}[grid_lvl]

    x_key, y_key = {"Kp - Ki": ("kp", "ki"), "Kp - Kd": ("kp", "kd"),
                    "Ki - Kd": ("ki", "kd")}[plane]
    fixed_key = ({"kp", "ki", "kd"} - {x_key, y_key}).pop()
    fixed_val = {"kp": float(kp), "ki": float(ki), "kd": float(kd)}[fixed_key]
    p3.caption(f"第三个增益固定在侧边栏当前值：**{AXIS_NAMES[fixed_key]} = {fixed_val:g}**"
               f"　（要改它，直接调左侧侧边栏即可）")

    try:
        _cr = phase_crossover(plant)
        ku_now = float(_cr[1]) if _cr else None
    except Exception:
        ku_now = None
    (_xl, _xh), (_yl, _yh) = default_ranges(kp, ki, kd, x_key, y_key, ku=ku_now)
    if "sw_xmax" not in st.session_state:
        st.session_state["sw_xmax"] = float(np.clip(_xh, 0.1, 100.0))
    if "sw_ymax" not in st.session_state:
        st.session_state["sw_ymax"] = float(np.clip(_yh, 0.1, 100.0))
    q1, q2, q3 = st.columns(3)
    x_max = q1.slider(f"{AXIS_NAMES[x_key]} 扫描上限", 0.1, 100.0, step=0.1, key="sw_xmax")
    y_max = q2.slider(f"{AXIS_NAMES[y_key]} 扫描上限", 0.1, 100.0, step=0.1, key="sw_ymax")
    sigma_lim = q3.slider("可接受的最大超调 σ_max (%)", 5.0, 100.0, 20.0, 1.0, key="sw_siglim",
                          help="用于在下方的「约束筛选」里挑出满足超调要求的参数区")

    x_vals = np.linspace(0.0, float(x_max), n_grid)
    y_vals = np.linspace(0.0, float(y_max), n_grid)

    _sw_key = (plant_key(plant), x_key, y_key, tuple(np.round(x_vals, 6)), tuple(np.round(y_vals, 6)),
               fixed_key, round(fixed_val, 6), float(st.session_state["sim_t_end"]))
    if st.session_state.get("_sw_cache_key") != _sw_key:
        with st.spinner(f"正在扫描 {n_grid}×{n_grid} = {n_grid * n_grid} 个格点（约需数秒，之后会缓存）…"):
            _t0 = time.time()
            st.session_state["_sw"] = _sweep_cached(
                tuple(np.round(np.asarray(plant.num, float), 10)),
                tuple(np.round(np.asarray(plant.den, float), 10)), float(plant.delay),
                str(x_key), str(y_key), tuple(np.round(x_vals, 8)), tuple(np.round(y_vals, 8)),
                tuple(sorted({fixed_key: float(fixed_val)}.items())),
                "PID", float(st.session_state["sim_t_end"]), 600, None, 0.0, 0)
            st.session_state["_sw_cache_key"] = _sw_key
            st.session_state["_sw_time"] = time.time() - _t0
    sw = st.session_state["_sw"]
    st.caption(f"网格 {n_grid}×{n_grid}，扫描耗时 {st.session_state.get('_sw_time', 0):.2f} s"
               f"（结果已缓存，改扫描范围或对象才会重算）")

    # ---- 标注点 ----
    def _marks_for(metric):
        mk = [{"name": "当前参数", "x": float({"kp": kp, "ki": ki, "kd": kd}[x_key]),
               "y": float({"kp": kp, "ki": ki, "kd": kd}[y_key]),
               "text": " 当前", "symbol": "star", "color": "#ffffff", "size": 15}]
        for _r in tune_results:
            if not _r.valid:
                continue
            _g = {"kp": float(_r.kp), "ki": float(_r.ki), "kd": float(_r.kd)}
            if not (x_vals.min() <= _g[x_key] <= x_vals.max() and y_vals.min() <= _g[y_key] <= y_vals.max()):
                continue
            mk.append({"name": _r.method, "x": _g[x_key], "y": _g[y_key],
                       "text": " " + _r.method[:4], "symbol": "x", "color": "#1f77b4", "size": 11})
        bp = best_point(sw, metric)
        if bp:
            mk.append({"name": f"最优（{metric}）", "x": bp["x"], "y": bp["y"],
                       "text": " 最优", "symbol": "circle-open", "color": "#111111", "size": 15})
        return mk

    st.markdown("#### 性能指标热力图（绿 = 更好）")
    h1, h2 = st.columns(2)
    with h1:
        _z = sw["Z"]["overshoot"]
        st.plotly_chart(plots.gain_heatmap_figure(
            sw, "overshoot", title="超调量 σ (%)", x_label=AXIS_NAMES[x_key],
            y_label=AXIS_NAMES[y_key], marks=_marks_for("overshoot"),
            colorscale="RdYlGn_r", z_max=float(np.nanpercentile(_z, 90)) if np.isfinite(_z).any() else None),
            use_container_width=True, key="pc_029")
    with h2:
        _z = sw["Z"]["ts_2"]
        st.plotly_chart(plots.gain_heatmap_figure(
            sw, "ts_2", title="调节时间 ts(2%) (s)", x_label=AXIS_NAMES[x_key],
            y_label=AXIS_NAMES[y_key], marks=_marks_for("ts_2"),
            colorscale="RdYlGn_r", z_max=float(np.nanpercentile(_z, 90)) if np.isfinite(_z).any() else None),
            use_container_width=True, key="pc_030")
    h3, h4 = st.columns(2)
    with h3:
        _z = sw["Z"]["iae"]
        st.plotly_chart(plots.gain_heatmap_figure(
            sw, "iae", title="误差积分 IAE", x_label=AXIS_NAMES[x_key],
            y_label=AXIS_NAMES[y_key], marks=_marks_for("iae"),
            colorscale="RdYlGn_r", z_max=float(np.nanpercentile(_z, 90)) if np.isfinite(_z).any() else None),
            use_container_width=True, key="pc_031")
    with h4:
        st.plotly_chart(plots.gain_heatmap_figure(
            sw, "score", title="综合评分（越高越好）", x_label=AXIS_NAMES[x_key],
            y_label=AXIS_NAMES[y_key], marks=_marks_for("score"),
            colorscale="RdYlGn"), use_container_width=True, key="pc_032")

    # ---- 最优点 ----
    st.markdown("#### 最优格点")
    _bp = best_point(sw, "score")
    if _bp:
        _best_gains = {"kp": float(kp), "ki": float(ki), "kd": float(kd)}
        _best_gains[x_key] = _bp["x"]
        _best_gains[y_key] = _bp["y"]
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("最优 Kp", f"{_best_gains['kp']:.4f}")
        b2.metric("最优 Ki", f"{_best_gains['ki']:.4f}")
        b3.metric("最优 Kd", f"{_best_gains['kd']:.4f}")
        b4.metric("综合评分", f"{_bp['value']:.2f} 分")
        _bi = _bp["i"]
        _bj = _bp["j"]
        st.caption(f"该点网格实测：σ = {fmt(sw['Z']['overshoot'][_bi, _bj], 3)}%，"
                   f"ts(2%) = {fmt(sw['Z']['ts_2'][_bi, _bj], 3)} s，"
                   f"IAE = {fmt(sw['Z']['iae'][_bi, _bj], 4)}"
                   f"（扫描用的仿真点数较少，实际使用会重新精确仿真）")
        if st.button("✅ 把最优参数应用到全局 PID", use_container_width=True):
            st.session_state["_pending_pid"] = (_best_gains["kp"], _best_gains["ki"], _best_gains["kd"])
            st.rerun()

    # ---- 约束筛选 ----
    st.markdown("#### 约束筛选：在超调达标的前提下选最快")
    _ov = sw["Z"]["overshoot"]
    _ts = sw["Z"]["ts_2"]
    _ess = np.abs(sw["Z"]["ess_rel"])
    _t_end_now = float(st.session_state["sim_t_end"])
    # 有效格点的三条要求：超调达标、真正稳定下来（不是贴着仿真末端）、稳态误差可接受
    _ok = np.isfinite(_ov) & np.isfinite(_ts) & (_ov <= float(sigma_lim))
    _ok &= (_ts <= 0.9 * _t_end_now)
    _ok &= np.isfinite(_ess) & (_ess <= 5.0)
    if _ok.any():
        _tss = np.where(_ok, _ts, np.inf)
        _ii, _jj = np.unravel_index(np.argmin(_tss), _tss.shape)
        _cg = {"kp": float(kp), "ki": float(ki), "kd": float(kd)}
        _cg[x_key] = float(x_vals[_jj])
        _cg[y_key] = float(y_vals[_ii])
        st.success(f"在 **σ ≤ {sigma_lim:g}%** 的 {int(_ok.sum())} 个格点中，调节时间最短的是："
                   f"Kp = {_cg['kp']:.4f}，Ki = {_cg['ki']:.4f}，Kd = {_cg['kd']:.4f}　→　"
                   f"ts(2%) = {fmt(_ts[_ii, _jj], 3)} s，σ = {fmt(_ov[_ii, _jj], 3)}%")
    else:
        st.warning(f"当前扫描范围内没有 σ ≤ {sigma_lim:g}% 的格点，请放宽上限或扩大扫描范围。")

    # ---- 结论 ----
    _bp_ts = best_point(sw, "ts_2")
    _bp_ov = best_point(sw, "overshoot")
    _cur_res9 = sim("PID", kp, ki, kd, label="当前参数")
    _cur_m9 = compute_metrics(_cur_res9.t, _cur_res9.y, _cur_res9.r, label="当前参数")
    info_card(
        "<b>自动分析结论</b><br>"
        f"· <b>超调最小</b>的格点：{AXIS_NAMES[x_key]} = {_bp_ov['x']:.3f}，"
        f"{AXIS_NAMES[y_key]} = {_bp_ov['y']:.3f}　→　σ = {fmt(_bp_ov['value'], 3)}%"
        f"（当前参数实测 σ = {fmt(_cur_m9['overshoot'], 3)}%）<br>"
        f"· <b>调节最快</b>的格点：{AXIS_NAMES[x_key]} = {_bp_ts['x']:.3f}，"
        f"{AXIS_NAMES[y_key]} = {_bp_ts['y']:.3f}　→　ts = {fmt(_bp_ts['value'], 3)} s<br>"
        f"· <b>综合评分最高</b>的格点：{AXIS_NAMES[x_key]} = {_bp['x']:.3f}，"
        f"{AXIS_NAMES[y_key]} = {_bp['y']:.3f}　→　评分 {fmt(_bp['value'], 2)} 分<br>"
        f"· 热力图能直接看出<b>权衡关系</b>：增大 Kp 通常加快响应但推高超调，"
        f"增大 Ki 消除静差但延长调节时间，增大 Kd 改善阻尼但放大噪声。"
        f"绿色区域就是各项指标都较优的「甜区」，可作为实际整定的初值范围。"
    )


# ========================================================================== #
#  功能页八：仿真结果可视化总览
# ========================================================================== #
with tab8:
    st.subheader("📊 仿真结果可视化总览")
    st.caption("把前五个功能页的仿真结果集中到一页仪表盘，便于横向对照各控制器与各工况的表现。")

    plant6 = current_plant()
    ol6 = simulate_open_loop(plant6, t_end=float(st.session_state["sim_t_end"]),
                             n_samples=int(st.session_state["sim_n"]))
    res6 = {m: sim(m, kp, ki, kd, label=f"{m} 控制") for m in MODES}
    met6 = {m: compute_metrics(res6[m].t, res6[m].y, res6[m].r, label=f"{m} 控制") for m in MODES}
    met6_ol = compute_metrics(ol6.t, ol6.y, ol6.r, label="开环（无控制器）")

    # 扰动 / 非线性工况沿用「抗干扰仿真」「非线性特性」两页的设置
    dist6 = sim("PID", kp, ki, kd, disturbance=dist_fn, label="PID（含扰动）")
    y_ref6 = float(np.mean(dist6.y[(dist6.t >= max(0, t_dist - 2.0)) & (dist6.t < t_dist)])) if t_dist > 0 else steady_value(plant6)
    dm6 = disturbance_metrics(dist6.t, dist6.y, y_ref6, t_dist)

    lin6 = sim(mode5, kp, ki, kd, label="线性系统")
    nl6 = sim(mode5, kp, ki, kd, actuator=act5, anti_windup=use_aw5, label=f"非线性（{act5.label()}）")
    met_lin6 = compute_metrics(lin6.t, lin6.y, lin6.r, label="线性系统")
    met_nl6 = compute_metrics(nl6.t, nl6.y, nl6.r, label=f"非线性（{act5.label()}）")

    best6 = max(met6, key=lambda m: met6[m]["score"])
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("二阶系统开环超调", f"{fmt(met6_ol['overshoot'], 3)} %",
              f"理论 {theory_val(th, 'overshoot')} %", delta_color="off")
    k2.metric("综合评分最高模式", best6, f"σ={fmt(met6[best6]['overshoot'], 3)} %", delta_color="off")
    k3.metric("PID 调节时间 ts(2%)", f"{fmt(met6['PID']['ts_2'], 3)} s", delta_color="off")
    k4.metric("抗扰最大动态偏差", f"{fmt(dm6['max_dev'], 3)}", f"工况：{dtype}", delta_color="off")

    st.markdown("#### 四宫格仪表盘")
    r1, r2 = st.columns(2)
    with r1:
        st.plotly_chart(plots.line_figure(
            [{"t": ol6.t, "y": ol6.y, "name": "开环阶跃响应", "color": PALETTE[0]}],
            title="① 阶跃响应（开环）", ref=steady_value(plant6)), use_container_width=True, key="pc_033")
    with r2:
        st.plotly_chart(plots.line_figure(
            [{"t": res6[m].t, "y": res6[m].y, "name": f"{m} 控制", "color": PALETTE[i]}
             for i, m in enumerate(MODES)],
            title="② 四种 PID 模式对比", ref=1.0), use_container_width=True, key="pc_034")
    r3, r4 = st.columns(2)
    with r3:
        f6 = plots.line_figure(
            [{"t": res6["PID"].t, "y": res6["PID"].y, "name": "无扰动", "color": "#9aa5b1", "dash": "dot"},
             {"t": dist6.t, "y": dist6.y, "name": "含负载扰动", "color": PALETTE[1]}],
            title="③ 抗干扰响应", ref=1.0)
        f6.add_vline(x=t_dist, line=dict(color="#d62728", dash="dash", width=1.3))
        st.plotly_chart(f6, use_container_width=True, key="pc_035")
    with r4:
        st.plotly_chart(plots.line_figure(
            [{"t": lin6.t, "y": lin6.y, "name": "线性系统", "color": PALETTE[0]},
             {"t": nl6.t, "y": nl6.y, "name": f"非线性：{act5.label()}", "color": PALETTE[2]}],
            title="④ 非线性工况对比", ref=1.0), use_container_width=True, key="pc_036")

    st.markdown("#### 全场景性能指标汇总")
    all_results6 = [ol6] + [res6[m] for m in MODES] + [dist6, lin6, nl6]
    all_metrics6 = [met6_ol] + [met6[m] for m in MODES] + [
        compute_metrics(dist6.t, dist6.y, dist6.r, label="PID（含扰动）"), met_lin6, met_nl6]
    df_summary6 = report.build_metrics_sheet(all_results6, all_metrics6)
    df_raw6 = report.results_to_dataframe(all_results6, wide=False)
    st.dataframe(df_summary6, use_container_width=True, hide_index=True, key="df_018")

    # ------------------------------------------------------------------ #
    #  保存结果
    # ------------------------------------------------------------------ #
    st.markdown("#### 保存结果")
    st.caption("数据表保存为 CSV，曲线图保存为 PNG；Excel 多工作表与全部曲线图打包放在下方展开项中按需生成。")

    import matplotlib.pyplot as plt
    _dash_key = plant_key(plant6) + (round(float(kp), 6), round(float(ki), 6), round(float(kd), 6),
                                     float(st.session_state["sim_t_end"]), int(st.session_state["sim_n"]),
                                     str(act5.label()))
    if st.session_state.get("_dash_key") != _dash_key:
        _dash_fig = plots.mpl_dashboard(
            [
                ([{"t": ol6.t, "y": ol6.y, "name": "开环阶跃响应", "color": PALETTE[0]}],
                 "① 阶跃响应（开环）", steady_value(plant6)),
                ([{"t": res6[m].t, "y": res6[m].y, "name": f"{m} 控制", "color": PALETTE[i]}
                  for i, m in enumerate(MODES)], "② 四种 PID 模式对比", 1.0),
                ([{"t": res6["PID"].t, "y": res6["PID"].y, "name": "无扰动", "color": "#9aa5b1", "dash": ":"},
                  {"t": dist6.t, "y": dist6.y, "name": "含负载扰动", "color": PALETTE[1]}],
                 "③ 抗干扰响应", 1.0),
                ([{"t": lin6.t, "y": lin6.y, "name": "线性系统", "color": PALETTE[0]},
                  {"t": nl6.t, "y": nl6.y, "name": f"非线性：{act5.label()}", "color": PALETTE[2]}],
                 "④ 非线性工况对比", 1.0),
            ],
            title=f"PID_SimLab 仿真结果总览  |  {plant6.describe()[:56]}")
        st.session_state["_dash_png"] = report.fig_to_png_bytes(_dash_fig)
        st.session_state["_csv_sum"] = report.df_to_csv_bytes(df_summary6)
        st.session_state["_csv_raw"] = report.df_to_csv_bytes(df_raw6)
        st.session_state["_dash_key"] = _dash_key
        plt.close(_dash_fig)
    dash_png = st.session_state["_dash_png"]

    sv1, sv2, sv3 = st.columns(3)
    with sv1:
        st.download_button("⬇️ 性能指标表（CSV）", data=st.session_state["_csv_sum"],
                           file_name=f"性能指标表_{datetime.now():%Y%m%d_%H%M}.csv",
                           mime="text/csv", use_container_width=True)
    with sv2:
        st.download_button("⬇️ 全部时间序列（CSV）", data=st.session_state["_csv_raw"],
                           file_name=f"时间序列数据_{datetime.now():%Y%m%d_%H%M}.csv",
                           mime="text/csv", use_container_width=True)
    with sv3:
        st.download_button("⬇️ 仪表盘图片（PNG）", data=dash_png,
                           file_name=f"仿真结果仪表盘_{datetime.now():%Y%m%d_%H%M}.png",
                           mime="image/png", use_container_width=True)

    save_key = plant_key(plant6) + (round(float(kp), 6), round(float(ki), 6), round(float(kd), 6),
                                    float(st.session_state["sim_t_end"]), int(st.session_state["sim_n"]),
                                    str(act5.label()))
    with st.expander("📦 更多保存选项：Excel 多工作表 / 全部曲线图打包"):
        st.caption("这两项生成耗时约 2 秒，因此改为按需生成；改动参数后需要重新点一次。")
        if st.session_state.get("_save_key") != save_key:
            st.session_state["_save_ready"] = False
            st.session_state["_save_key"] = save_key
        if st.button("⚙️ 准备文件", use_container_width=True):
            st.session_state["_save_ready"] = True
        if st.session_state.get("_save_ready"):
            if st.session_state.get("_save_cache_key") != save_key:
                with st.spinner("正在生成 Excel 与曲线图打包…"):
                    sheets6 = {
                        "性能指标汇总": df_summary6,
                        "全部时间序列": df_raw6,
                        "对象参数": pd.DataFrame([{"参数": k, "取值": v} for k, v in plant6.as_dict().items()]),
                    }
                    figures = {}
                    f = plots.mpl_line([{"t": ol6.t, "y": ol6.y, "name": "开环阶跃响应"}],
                                       f"阶跃响应（开环）  {plant6.describe()[:52]}", ref=steady_value(plant6))
                    figures["01_二阶系统阶跃响应.png"] = report.fig_to_png_bytes(f); plt.close(f)
                    f = plots.mpl_line([{"t": res6[m].t, "y": res6[m].y, "name": f"{m} 控制"} for m in MODES],
                                       "四种 PID 模式阶跃响应对比", ref=1.0)
                    figures["02_四种PID模式对比.png"] = report.fig_to_png_bytes(f); plt.close(f)
                    f = plots.mpl_output_control([res6["PID"]], "PID 控制系统输出与控制量", ref=1.0)
                    figures["03_PID输出与控制量.png"] = report.fig_to_png_bytes(f); plt.close(f)
                    f = plots.mpl_line([{"t": res6["PID"].t, "y": res6["PID"].y, "name": "无扰动"},
                                        {"t": dist6.t, "y": dist6.y, "name": "含负载扰动"}],
                                       "抗负载干扰仿真", ref=1.0)
                    figures["04_抗干扰仿真.png"] = report.fig_to_png_bytes(f); plt.close(f)
                    f = plots.mpl_line([{"t": lin6.t, "y": lin6.y, "name": "线性系统"},
                                        {"t": nl6.t, "y": nl6.y, "name": f"非线性：{act5.label()}"}],
                                       "执行器非线性特性对比", ref=1.0)
                    figures["05_非线性特性.png"] = report.fig_to_png_bytes(f); plt.close(f)
                    st.session_state["_save_excel"] = report.excel_bytes(sheets6)
                    st.session_state["_save_zip"] = report.zip_bytes(figures)
                    st.session_state["_save_cache_key"] = save_key
            sv4, sv5 = st.columns(2)
            with sv4:
                st.download_button("⬇️ 全部数据（Excel）", data=st.session_state["_save_excel"],
                                   file_name=f"仿真数据_{datetime.now():%Y%m%d_%H%M}.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   use_container_width=True)
            with sv5:
                st.download_button("⬇️ 全部曲线图（ZIP）", data=st.session_state["_save_zip"],
                                   file_name=f"曲线图_{datetime.now():%Y%m%d_%H%M}.zip",
                                   mime="application/zip", use_container_width=True)
        else:
            st.info("点击「准备文件」后，这里会出现 Excel 与曲线图打包的下载入口。")

    with st.expander("📄 全部时间序列数据（前 500 行）"):
        st.dataframe(df_raw6.head(500), use_container_width=True, hide_index=True, key="df_019")
        st.caption(f"共 {len(df_raw6)} 行 × {df_raw6.shape[1]} 列。")

    with st.expander("📐 指标口径说明"):
        formula_card(
            "超调量 σ% = (y_max − y_ss) / |y_ss| × 100%<br>"
            "上升时间 tr：10%→90% 与 0→100% 两种定义分别列出<br>"
            "调节时间 ts：输出最后一次离开 ±2%（或 ±5%）误差带的时刻<br>"
            "稳态误差 ess = r(∞) − y(∞)<br>"
            "误差积分：IAE = ∫|e|dt，ISE = ∫e²dt，ITAE = ∫t·|e|dt<br>"
            "综合评分 = 0.35×超调得分 + 0.30×调节时间得分 + 0.20×稳态误差得分 + 0.15×IAE 得分")

    st.divider()
    st.caption("PID_SimLab v0.1 · Python + Streamlit + NumPy + SciPy + Plotly ｜ "
               "仿真内核：可控标准型状态空间 + 零阶保持精确离散化 + 历史缓冲纯滞后")

