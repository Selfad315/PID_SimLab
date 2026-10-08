# -*- coding: utf-8 -*-
"""数据与报告导出：CSV / Excel / PNG / Markdown / 一键 ZIP 打包。"""
from __future__ import annotations

import io
import re
import zipfile
from datetime import datetime
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
#  基础导出
# --------------------------------------------------------------------------- #
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def _safe_sheet_name(name: str) -> str:
    name = re.sub(r"[\[\]\*\?/\\:]", "_", str(name))
    return name[:31] or "Sheet"


def excel_bytes(sheets: Mapping[str, pd.DataFrame]) -> bytes:
    """多个 DataFrame -> 单文件多工作表 xlsx（字节流，供下载）。"""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        used = set()
        for name, df in sheets.items():
            sname = _safe_sheet_name(name)
            k = 1
            while sname in used:
                suffix = f"_{k}"
                sname = sname[:31 - len(suffix)] + suffix
                k += 1
            used.add(sname)
            (df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)).to_excel(
                writer, sheet_name=sname, index=False)
    return buf.getvalue()


def fig_to_png_bytes(fig, dpi: Optional[int] = None) -> bytes:
    """Matplotlib Figure -> PNG 字节流。"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi or fig.dpi)
    buf.seek(0)
    return buf.read()


def zip_bytes(files: Mapping[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def results_to_dataframe(results: Sequence, wide: bool = False) -> pd.DataFrame:
    """仿真结果列表 -> 便于导出的 DataFrame。"""
    if not results:
        return pd.DataFrame()
    if wide:
        frames = []
        for i, r in enumerate(results):
            df = r.to_dataframe(prefix=r.label or f"方案{i+1}")
            # 以时间轴对齐（各曲线采样点可能不同）
            frames.append(df.set_index(df.columns[0]))
        return pd.concat(frames, axis=1).reset_index().rename(columns={"index": "t"})
    frames = []
    for i, r in enumerate(results):
        df = r.to_dataframe()
        df.insert(0, "方案", r.label or f"方案{i+1}")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
#  自动实验报告
# --------------------------------------------------------------------------- #
def build_markdown_report(title: str, plant_desc: str, sections: Sequence[dict],
                          meta: Optional[dict] = None) -> str:
    """把各模块结果自动拼装成 Markdown 实验报告。

    sections: [{"heading": "3.1 四种 PID 模式对比",
                "text": "...", "table": DataFrame/None, "notes": [...]}, ...]
    """
    lines = [f"# {title}", ""]
    lines.append(f"**生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    if meta:
        lines.append("| 项目 | 取值 |")
        lines.append("| --- | --- |")
        for k, v in meta.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")
    lines.append("## 一、被控对象")
    lines.append("")
    lines.append("```")
    lines.append(plant_desc)
    lines.append("```")
    lines.append("")

    for i, sec in enumerate(sections, start=2):
        lines.append(f"## {i}、{sec.get('heading', '分析结果')}")
        lines.append("")
        if sec.get("text"):
            lines.append(str(sec["text"]))
            lines.append("")
        table = sec.get("table")
        if isinstance(table, pd.DataFrame) and not table.empty:
            lines.append(_df_to_markdown(table))
            lines.append("")
        for note in sec.get("notes", []) or []:
            lines.append(f"- {note}")
        if sec.get("notes"):
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("> 本报告由 **PID_SimLab 基于 Python 的 PID 控制系统仿真、参数整定与性能对比可视化平台** 自动生成。")
    return "\n".join(lines)


def _df_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:      # 未安装 tabulate 时的兜底
        head = "| " + " | ".join(str(c) for c in df.columns) + " |"
        sep = "| " + " | ".join("---" for _ in df.columns) + " |"
        rows = ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
        return "\n".join([head, sep] + rows)


def _fmt(v) -> str:
    if isinstance(v, (float, np.floating)):
        return "—" if not np.isfinite(v) else f"{v:.4g}"
    return "—" if v is None else str(v)


def build_metrics_sheet(results: Sequence, metrics: Sequence[dict]) -> pd.DataFrame:
    """把参数与性能指标合并成一张总表（模块 6 的主表）。"""
    from .metrics import DISPLAY_KEYS
    rows = []
    for i, m in enumerate(metrics):
        pid = getattr(results[i], "pid", None) if i < len(results) else None
        rec = {"方案": m.get("label", f"方案{i+1}")}
        if pid is not None:
            kp, ki, kd = pid.gains()
            rec.update({"模式": pid.mode, "Kp": round(kp, 4),
                        "Ki": round(ki, 4), "Kd": round(kd, 4)})
        for k, title in DISPLAY_KEYS:
            if k == "label":
                continue
            v = m.get(k, np.nan)
            rec[title] = None if isinstance(v, float) and not np.isfinite(v) else (
                round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v)
        rows.append(rec)
    return pd.DataFrame(rows)
