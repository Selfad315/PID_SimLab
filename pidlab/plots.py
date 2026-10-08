# -*- coding: utf-8 -*-
"""绘图模块：Plotly（界面交互） + Matplotlib（论文/报告出图）。"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm

# ---- 中文字体配置：优先使用系统自带微软雅黑/黑体，避免 PNG 导出出现方框 ----
_AVAILABLE_FONTS = {f.name for f in _fm.fontManager.ttflist}
for _cjk in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
             "DengXian", "Microsoft JhengHei", "SimSun"):
    if _cjk in _AVAILABLE_FONTS:
        plt.rcParams["font.sans-serif"] = [_cjk, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.autolayout"] = False

import plotly.graph_objects as go
from plotly.subplots import make_subplots

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
           "#9467bd", "#8c564b", "#17becf", "#e377c2",
           "#7f7f7f", "#bcbd22"]

_MARKERS = ["circle", "square", "triangle-up", "diamond", "cross", "star", "pentagon"]


# ========================================================================== #
#  Plotly：交互曲线
# ========================================================================== #
def _layout(fig, title: str, xlabel: str, ylabel: str, height: int = 460,
            legend_y: float = 1.02):
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=17)),
        xaxis_title=xlabel, yaxis_title=ylabel, height=height,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=legend_y, x=0),
        margin=dict(l=70, r=30, t=80, b=55),
    )
    fig.update_xaxes(showspikes=True, spikemode="across", spikethickness=1,
                     spikecolor="#999", spikedash="dot")
    return fig


def line_figure(series: Sequence[dict], title: str = "", xlabel: str = "时间 t / s",
                ylabel: str = "输出 y(t)", height: int = 460,
                ref: Optional[float] = None, ref_label: str = "给定值 r(t)",
                ylim: Optional[tuple] = None) -> go.Figure:
    """通用多曲线对比图。

    series: [{"t":..., "y":..., "name":..., "color":..., "dash":..., "width":...}, ...]
    """
    fig = go.Figure()
    if ref is not None:
        fig.add_hline(y=ref, line=dict(color="#444", dash="dash", width=1.4),
                      annotation_text=ref_label, annotation_position="top left")
    for i, s in enumerate(series):
        fig.add_trace(go.Scatter(
            x=np.asarray(s["t"]), y=np.asarray(s["y"]), mode="lines",
            name=s.get("name", f"曲线 {i+1}"),
            line=dict(color=s.get("color", PALETTE[i % len(PALETTE)]),
                      dash=s.get("dash", "solid"), width=s.get("width", 2.2)),
        ))
    _layout(fig, title, xlabel, ylabel, height)
    if ylim:
        fig.update_yaxes(range=list(ylim))
    return fig


def output_control_figure(results: Sequence, title: str = "系统输出与控制器输出",
                          ref: float = 1.0, height: int = 620) -> go.Figure:
    """上下双图：系统输出 y(t) 与控制器输出 u(t)。"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        subplot_titles=("① 系统输出 y(t)", "② 控制器/执行器输出 u(t)"))
    fig.add_hline(y=ref, line=dict(color="#444", dash="dash", width=1.3), row=1, col=1)
    for i, r in enumerate(results):
        c = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Scatter(x=r.t, y=r.y, name=r.label or f"方案{i+1}",
                                 line=dict(color=c, width=2.3)), row=1, col=1)
    for i, r in enumerate(results):
        c = PALETTE[i % len(PALETTE)]
        fig.add_trace(go.Scatter(x=r.t, y=r.u, name=r.label or f"方案{i+1}",
                                 line=dict(color=c, width=1.7, dash="dot"),
                                 showlegend=False, opacity=0.85), row=2, col=1)
        fig.add_trace(go.Scatter(x=r.t, y=r.u_raw, name="未限幅", visible="legendonly",
                                 line=dict(color=c, width=1, dash="dash")), row=2, col=1)
    fig.update_xaxes(title_text="时间 t / s", row=2, col=1)
    fig.update_yaxes(title_text="y(t)", row=1, col=1)
    fig.update_yaxes(title_text="u(t)", row=2, col=1)
    fig.update_layout(template="plotly_white", height=height, hovermode="x unified",
                      margin=dict(l=70, r=30, t=70, b=50),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig


def bar_figure(categories: Sequence[str], series: Sequence[dict], title: str = "",
               ylabel: str = "", height: int = 420) -> go.Figure:
    """分组柱状图：series=[{"name":..., "values":[...]}, ...]"""
    fig = go.Figure()
    for i, s in enumerate(series):
        fig.add_trace(go.Bar(x=list(categories), y=s["values"], name=s["name"],
                             marker_color=PALETTE[i % len(PALETTE)],
                             text=[None if v is None or not np.isfinite(v) else round(float(v), 3)
                                   for v in s["values"]],
                             textposition="outside"))
    fig.update_layout(barmode="group", template="plotly_white", height=height,
                      title=dict(text=title, x=0.01, xanchor="left", font=dict(size=17)),
                      yaxis_title=ylabel, margin=dict(l=70, r=30, t=70, b=50),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig


def bode_figure(plant, pid=None, w=None, title: str = "开环频率特性 Bode 图",
                height: int = 640, mark_crossover: bool = True) -> go.Figure:
    """开环 Bode 图（含 -180 度线与临界点标记），用于讲解 ZN 临界比例度法。"""
    wn = max(getattr(plant, "wn", 1.0), 1e-6)
    w = np.logspace(np.log10(wn / 1e3), np.log10(wn * 1e3), 3000) if w is None else np.asarray(w)

    g = plant.freqresp(w)
    if pid is not None:
        nc, dc = pid.tf()
        c = np.polyval(nc, 1j * w) / np.polyval(dc, 1j * w)
        g = g * c

    mag_db = 20 * np.log10(np.maximum(np.abs(g), 1e-300))
    phase_deg = np.degrees(np.unwrap(np.angle(g)))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("幅频特性 20lg|L(jw)|", "相频特性 ∠L(jw)"))
    fig.add_trace(go.Scatter(x=w, y=mag_db, name="幅值", line=dict(color=PALETTE[0], width=2.2)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=w, y=phase_deg, name="相位", line=dict(color=PALETTE[1], width=2.2)),
                  row=2, col=1)
    fig.add_hline(y=-180, line=dict(color="#888", dash="dash", width=1.2), row=2, col=1)
    fig.add_hline(y=0, line=dict(color="#bbb", dash="dot", width=1.0), row=1, col=1)

    if mark_crossover:
        try:
            from .tuning import phase_crossover
            cr = phase_crossover(plant) if pid is None else None
            if cr is not None:
                wu, Ku, Pu = cr[0], cr[1], cr[2]
                fig.add_trace(go.Scatter(x=[wu], y=[0.0], mode="markers+text",
                                         name=f"临界点 Ku={Ku:.3g}",
                                         text=[f" Ku={Ku:.3g}<br> Pu={Pu:.3g}s"],
                                         textposition="bottom right",
                                         marker=dict(color="#d62728", size=11, symbol="x")),
                              row=1, col=1)
                fig.add_vline(x=wu, line=dict(color="#d62728", dash="dot", width=1.2), row=2, col=1)
        except Exception:
            pass

    fig.update_xaxes(type="log", title_text="角频率 w / (rad/s)", row=2, col=1)
    fig.update_yaxes(title_text="幅值 / dB", row=1, col=1)
    fig.update_yaxes(title_text="相位 / 度", row=2, col=1)
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=70, r=30, t=70, b=50),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig


def pole_zero_figure(plant, pid=None, title: str = "闭环极点分布", height: int = 430) -> go.Figure:
    """极点—零点图（复平面）。"""
    if pid is None:
        num, den = plant.num, plant.den
    else:
        from .control import closed_loop_tf
        num, den = closed_loop_tf(plant, pid)
    poles = np.roots(den)
    zeros = np.roots(num) if len(num) > 1 else np.array([])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=poles.real, y=poles.imag, mode="markers", name="极点 ×",
                             marker=dict(symbol="x", size=14, color="#d62728")))
    if zeros.size:
        fig.add_trace(go.Scatter(x=zeros.real, y=zeros.imag, mode="markers", name="零点 ○",
                                 marker=dict(symbol="circle-open", size=13, color="#1f77b4",
                                             line=dict(width=2))))
    fig.add_hline(y=0, line=dict(color="#bbb", width=1))
    fig.add_vline(x=0, line=dict(color="#bbb", width=1))
    fig.update_yaxes(title_text="虚部 jω", scaleanchor="x", scaleratio=1)
    fig.update_xaxes(title_text="实部 σ")
    fig.update_layout(template="plotly_white", height=height, title=title,
                      margin=dict(l=70, r=30, t=70, b=50))
    return fig


def nyquist_figure(plant, pid=None, title: str = "Nyquist 图", height: int = 460) -> go.Figure:
    wn = max(getattr(plant, "wn", 1.0), 1e-6)
    w = np.logspace(np.log10(wn / 1e3), np.log10(wn * 1e3), 4000)
    g = plant.freqresp(w)
    if pid is not None:
        nc, dc = pid.tf()
        g = g * (np.polyval(nc, 1j * w) / np.polyval(dc, 1j * w))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g.real, y=g.imag, mode="lines", name="L(jw)",
                             line=dict(color=PALETTE[0], width=2.2)))
    fig.add_trace(go.Scatter(x=g[::-1].real, y=-g[::-1].imag, mode="lines", name="镜像",
                             line=dict(color=PALETTE[0], width=1.2, dash="dot"), showlegend=False))
    fig.add_trace(go.Scatter(x=[-1], y=[0], mode="markers+text", name="临界点 (-1, j0)",
                             marker=dict(symbol="x", size=14, color="#d62728"),
                             text=["(-1, j0)"], textposition="top right"))
    fig.add_hline(y=0, line=dict(color="#ccc", width=1))
    fig.add_vline(x=0, line=dict(color="#ccc", width=1))
    fig.update_xaxes(title_text="实部 Re", scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title_text="虚部 Im")
    fig.update_layout(template="plotly_white", height=height, title=title,
                      margin=dict(l=70, r=30, t=70, b=50))
    return fig


def fopdt_figure(fopdt: dict, title: str = "阶跃响应切线法辨识 FOPDT 模型", height: int = 460) -> go.Figure:
    """展示反应曲线法：阶跃响应 + 最大斜率切线 + L / T 标注。"""
    t, y = fopdt["t"], fopdt["y"]
    S, t_i, y_i = fopdt["S"], fopdt["t_i"], fopdt["y_i"]
    K, T, L = fopdt["K"], fopdt["T"], fopdt["L"]
    t_line = np.array([max(0.0, t_i - y_i / S - 0.05), t_i + 1.2 * T])
    y_line = S * (t_line - (t_i - y_i / S))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, name="阶跃响应 y(t)", line=dict(color=PALETTE[0], width=2.4)))
    fig.add_trace(go.Scatter(x=t_line, y=y_line, name="最大斜率切线", line=dict(color=PALETTE[1], dash="dash", width=1.8)))
    fig.add_hline(y=K, line=dict(color="#888", dash="dot"), annotation_text="稳态值 K")
    fig.add_vline(x=L, line=dict(color=PALETTE[2], dash="dot"), annotation_text=f"L={L:.3g}")
    fig.add_vline(x=L + T, line=dict(color=PALETTE[3], dash="dot"), annotation_text=f"T={T:.3g}")
    fig.add_trace(go.Scatter(x=[t_i], y=[y_i], mode="markers", name="最大斜率点(拐点)",
                             marker=dict(color=PALETTE[1], size=11)))
    fig.update_xaxes(title_text="时间 t / s")
    fig.update_yaxes(title_text="输出 y(t)")
    fig.update_layout(template="plotly_white", height=height, title=title,
                      margin=dict(l=70, r=30, t=70, b=50),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig


# ========================================================================== #
#  Matplotlib：用于导出 PNG（无需 kaleido）
# ========================================================================== #
def mpl_line(series: Sequence[dict], title: str = "", xlabel: str = "时间 t / s",
             ylabel: str = "输出 y(t)", ref: Optional[float] = None,
             figsize=(8.2, 4.8), dpi: int = 150):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    if ref is not None:
        ax.axhline(ref, color="#444", ls="--", lw=1.2, label="给定值 r(t)")
    for i, s in enumerate(series):
        ax.plot(s["t"], s["y"], label=s.get("name", f"曲线{i+1}"),
                color=s.get("color", PALETTE[i % len(PALETTE)]),
                ls=s.get("dash", "-"), lw=s.get("width", 1.9))
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def mpl_output_control(results: Sequence, title: str = "系统输出与控制器输出",
                       ref: float = 1.0, figsize=(8.6, 7.0), dpi: int = 150):
    fig, axes = plt.subplots(2, 1, figsize=figsize, dpi=dpi, sharex=True)
    axes[0].axhline(ref, color="#444", ls="--", lw=1.1)
    for i, r in enumerate(results):
        c = PALETTE[i % len(PALETTE)]
        axes[0].plot(r.t, r.y, color=c, lw=1.9, label=r.label or f"方案{i+1}")
        axes[1].plot(r.t, r.u, color=c, lw=1.5, ls="-")
    axes[0].set_ylabel("输出 y(t)")
    axes[1].set_ylabel("控制量 u(t)")
    axes[1].set_xlabel("时间 t / s")
    axes[0].set_title(title, fontsize=12)
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def mpl_bar(categories, values, title: str = "", ylabel: str = "",
            figsize=(8.2, 4.4), dpi: int = 150):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    x = np.arange(len(categories))
    ax.bar(x, values, color=PALETTE[0], width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def mpl_dashboard(panels, title: str = "", figsize=(13.2, 8.2), dpi: int = 150):
    """2×2 仪表盘出图（用于把四张主曲线保存为一张 PNG）。

    panels: [(series_list, 子图标题, 参考线值 or None), ...]，最多取 4 个。
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    for ax, item in zip(axes.ravel(), panels):
        series, ttl, ref = item
        if ref is not None:
            ax.axhline(ref, color="#444", ls="--", lw=1.2)
        for i, sd in enumerate(series):
            ax.plot(sd["t"], sd["y"], label=sd.get("name", f"曲线{i+1}"),
                    color=sd.get("color", PALETTE[i % len(PALETTE)]),
                    ls=sd.get("dash", "-"), lw=1.8)
        ax.set_title(ttl, fontsize=11)
        ax.set_xlabel("时间 t / s")
        ax.set_ylabel("输出 y(t)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, loc="best")
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96] if title else None)
    return fig


# ========================================================================== #
#  频域与稳定性分析专用图
# ========================================================================== #
def bode_margins_figure(plant, pid, margins: dict, title: str = "开环 Bode 图与稳定裕度",
                        height: int = 660):
    """带幅值/相位裕度标注的 Bode 图。"""
    w = np.asarray(margins["w"])
    mag_db = np.asarray(margins["mag_db"])
    phase = np.asarray(margins["phase_deg"])
    w_gc, w_pc = margins.get("w_gc"), margins.get("w_pc")
    pm, gm_db = margins.get("pm"), margins.get("gm_db")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        subplot_titles=("幅频特性 20lg|L(jω)|", "相频特性 ∠L(jω)"))
    fig.add_trace(go.Scatter(x=w, y=mag_db, name="幅值", line=dict(color=PALETTE[0], width=2.3)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=w, y=phase, name="相位", line=dict(color=PALETTE[1], width=2.3)),
                  row=2, col=1)
    fig.add_hline(y=0, line=dict(color="#888", dash="dash", width=1.2), row=1, col=1)
    fig.add_hline(y=-180, line=dict(color="#888", dash="dash", width=1.2), row=2, col=1)

    if w_gc is not None:
        fig.add_vline(x=w_gc, line=dict(color=PALETTE[2], dash="dot", width=1.3), row=1, col=1)
        fig.add_vline(x=w_gc, line=dict(color=PALETTE[2], dash="dot", width=1.3), row=2, col=1)
        fig.add_trace(go.Scatter(x=[w_gc], y=[0.0], mode="markers",
                                 name=f"幅值穿越 ωgc={w_gc:.3g}",
                                 marker=dict(color=PALETTE[2], size=11, symbol="circle")),
                      row=1, col=1)
        if pm is not None:
            fig.add_trace(go.Scatter(x=[w_gc], y=[-180.0 + pm], mode="markers+text",
                                     text=[f" γ={pm:.1f}°"], textposition="middle right",
                                     marker=dict(color=PALETTE[2], size=10, symbol="diamond"),
                                     name="相位裕度"), row=2, col=1)
    if w_pc is not None:
        fig.add_vline(x=w_pc, line=dict(color=PALETTE[3], dash="dot", width=1.3), row=1, col=1)
        fig.add_vline(x=w_pc, line=dict(color=PALETTE[3], dash="dot", width=1.3), row=2, col=1)
        fig.add_trace(go.Scatter(x=[w_pc], y=[-180.0], mode="markers+text",
                                 text=[f" 相位穿越 ωpc={w_pc:.3g}"], textposition="middle right",
                                 marker=dict(color=PALETTE[3], size=11, symbol="x"),
                                 name="相位穿越"), row=2, col=1)
        if gm_db is not None and np.isfinite(gm_db):
            fig.add_trace(go.Scatter(x=[w_pc], y=[-gm_db], mode="markers+text",
                                     text=[f" h={gm_db:.1f} dB"], textposition="middle right",
                                     marker=dict(color=PALETTE[3], size=10, symbol="square"),
                                     name="幅值裕度"), row=1, col=1)

    fig.update_xaxes(type="log", title_text="角频率 ω / (rad/s)", row=2, col=1)
    fig.update_yaxes(title_text="幅值 / dB", row=1, col=1)
    fig.update_yaxes(title_text="相位 / 度", row=2, col=1)
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=70, r=40, t=70, b=50),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig


def nyquist_annotated_figure(nyq: dict, margins: dict | None = None,
                             title: str = "Nyquist 图与临界点", height: int = 520):
    """Nyquist 图（正频段 + 镜像），标注 (-1, j0) 与绕数结论。"""
    L = np.asarray(nyq["L"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=L.real, y=L.imag, mode="lines", name="L(jω)，ω: 0 → +∞",
                             line=dict(color=PALETTE[0], width=2.4)))
    fig.add_trace(go.Scatter(x=L[::-1].real, y=-L[::-1].imag, mode="lines", name="镜像（ω < 0）",
                             line=dict(color=PALETTE[0], width=1.3, dash="dot"), opacity=0.75))
    fig.add_trace(go.Scatter(x=[-1.0], y=[0.0], mode="markers+text", name="临界点 (-1, j0)",
                             text=["  (-1, j0)"], textposition="middle right",
                             marker=dict(symbol="x", size=15, color="#d62728",
                                         line=dict(width=3, color="#d62728"))))
    fig.add_trace(go.Scatter(x=[L[0].real], y=[L[0].imag], mode="markers+text",
                             text=[" ω→0⁺"], textposition="top right", name="起点 ω→0⁺",
                             marker=dict(symbol="circle", size=9, color="#2f9e44")))
    fig.add_hline(y=0, line=dict(color="#ccc", width=1))
    fig.add_vline(x=0, line=dict(color="#ccc", width=1))
    fig.update_xaxes(title_text="实部 Re", scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title_text="虚部 Im")
    fig.update_layout(template="plotly_white", height=height, title=title,
                      margin=dict(l=70, r=30, t=70, b=50),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig


def root_locus_figure(rl: dict, title: str = "根轨迹（k = 0 → ∞）", height: int = 520):
    """根轨迹：闭环极点随开环标量增益 k 的移动轨迹。"""
    xs, ys, cs = [], [], []
    for k, pole_set in zip(rl["k"], rl["poles"]):
        for pp in pole_set:
            xs.append(float(np.real(pp)))
            ys.append(float(np.imag(pp)))
            cs.append(float(k))

    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color="#bbb", width=1))
    fig.add_vline(x=0, line=dict(color="#bbb", width=1))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name="闭环极点轨迹",
                             marker=dict(size=3.2, color=cs, colorscale="Viridis", opacity=0.7,
                                         colorbar=dict(title="增益 k", thickness=12, len=0.75))))
    fig.add_trace(go.Scatter(x=rl["ol_poles"].real, y=rl["ol_poles"].imag, mode="markers",
                             name="开环极点 ×",
                             marker=dict(symbol="x", size=13, color="#d62728",
                                         line=dict(width=2, color="#d62728"))))
    if np.size(rl["ol_zeros"]):
        fig.add_trace(go.Scatter(x=rl["ol_zeros"].real, y=rl["ol_zeros"].imag, mode="markers",
                                 name="开环零点 ○",
                                 marker=dict(symbol="circle-open", size=13, color="#1f77b4",
                                             line=dict(width=2.4, color="#1f77b4"))))
    fig.add_trace(go.Scatter(x=rl["current"].real, y=rl["current"].imag, mode="markers",
                             name="当前参数（k = 1）",
                             marker=dict(symbol="star", size=16, color="#ff7f0e",
                                         line=dict(width=1.2, color="#333"))))
    fig.update_xaxes(title_text="实部 σ")
    fig.update_yaxes(title_text="虚部 jω", scaleanchor="x", scaleratio=1)
    fig.update_layout(template="plotly_white", height=height, title=title,
                      margin=dict(l=70, r=30, t=70, b=50),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig
