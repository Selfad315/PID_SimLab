# -*- coding: utf-8 -*-
"""PID_SimLab 核心算法包。

模块划分
--------
models  : 传递函数 / 状态空间 / 二阶对象 / 离散化 / 理论公式
control : PID 控制器、执行器非线性、扰动信号、闭环时域仿真
metrics : 超调量、上升/峰值/调节时间、稳态误差、误差积分、抗扰指标
tuning  : ZN 临界比例度法 / ZN 阶跃响应法 / 衰减曲线法 / 数值寻优
plots   : Plotly 交互图 + Matplotlib 出图
report  : Excel / CSV / PNG / Markdown 实验报告导出
"""
from .models import (LTIPlant, SecondOrderPlant, MODEL_SPECS, build_plant,
                     second_order_theory, tf_to_ss, c2d_zoh, freqresp, poly_add, poly_mul)
from .control import (PID, Actuator, SimResult, MODES, MODE_NAMES, AW_MODES, simulate,
                      simulate_open_loop, step_disturbance, pulse_disturbance,
                      sine_disturbance, ramp_disturbance, open_loop_tf, closed_loop_tf)
from .metrics import (compute_metrics, disturbance_metrics, metrics_table,
                      control_quality)
from .tuning import (TuningResult, phase_crossover, zn_closed_loop, zn_open_loop,
                     fopdt_from_step, decay_curve, auto_optimize, tune_all,
                     evaluate_tuning, ZN_TABLE, ZN_OPEN_TABLE,
                     DECAY_TABLE_41, DECAY_TABLE_101)

__version__ = "1.0.0"
__all__ = [n for n in dir() if not n.startswith("_")]
