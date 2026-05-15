#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
居民夜间多车聚合场景下电动汽车最优充电调度系统 —— Streamlit 演示应用
基于论文 MILP 模型，调用 CBC 求解器
运行方式：streamlit run app.py

★ 已与主仿真代码统一口径（参数/公式/策略逻辑/返回字段/显示格式）
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as fm
import pandas as pd
import time
import warnings
import io
import os
import tempfile
from pathlib import Path

warnings.filterwarnings('ignore')

# ── 高清导出 + 中文字体配置 ───────────────────────────────────────────────────
# 说明：只在 rcParams 里写 SimHei / Microsoft YaHei 不一定生效。
# 如果当前系统没装中文字体，Matplotlib 会退回到不支持中文的字体，于是中文会显示成“□□”。
# 下面会优先主动加载 Windows/macOS/Linux 常见中文字体；如果还是找不到，
# 可以把字体文件放到本脚本同级 fonts 文件夹，或在左侧上传 .ttf/.ttc/.otf 字体文件。
_FONT_STATUS = {"ok": False, "name": None, "path": None, "message": "尚未检测字体"}
_TEMP_FONT_FILES = []


def _candidate_font_paths():
    """返回常见中文字体路径；同时扫描脚本目录、当前目录、fonts 目录。"""
    script_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    common = [
        # Windows
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyh.ttf",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\simkai.ttf",
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/Library/Fonts/Songti.ttc",
        # Linux / Streamlit Cloud / Docker
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
    ]
    local_dirs = [script_dir, Path.cwd(), script_dir / "fonts", Path.cwd() / "fonts"]
    local = []
    for d in local_dirs:
        if d.exists():
            for ext in ("*.ttf", "*.ttc", "*.otf"):
                local.extend(str(x) for x in d.glob(ext))
    # 去重并保持顺序
    seen, out = set(), []
    for fp in common + local:
        if fp not in seen:
            seen.add(fp)
            out.append(fp)
    return out


def _font_supports_chinese(font_path):
    """用 Matplotlib 自带 FT2Font 检查字体是否含中文字符。"""
    try:
        ft = matplotlib.ft2font.FT2Font(str(font_path))
        cmap = ft.get_charmap()
        return ord("中") in cmap or ord("电") in cmap or ord("车") in cmap
    except Exception:
        return False


def _activate_font(font_path):
    """主动注册并启用指定字体。"""
    global _FONT_STATUS
    try:
        font_path = str(font_path)
        fm.fontManager.addfont(font_path)
        font_name = fm.FontProperties(fname=font_path).get_name()
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = [font_name]
        matplotlib.rcParams["axes.unicode_minus"] = False
        matplotlib.rcParams["figure.dpi"] = 180
        matplotlib.rcParams["savefig.dpi"] = 600
        # 插入 Word/PPT 推荐用 SVG；文字转路径后，文档端不再依赖本机字体
        matplotlib.rcParams["svg.fonttype"] = "path"
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        _FONT_STATUS = {
            "ok": True,
            "name": font_name,
            "path": font_path,
            "message": f"已启用中文字体：{font_name}",
        }
        return True
    except Exception as exc:
        _FONT_STATUS = {"ok": False, "name": None, "path": None, "message": f"字体加载失败：{exc}"}
        return False


def setup_chinese_font(uploaded_font=None):
    """设置中文字体。uploaded_font 是 Streamlit file_uploader 返回的对象，可为空。"""
    global _TEMP_FONT_FILES, _FONT_STATUS

    # 1) 用户上传字体时优先使用上传字体
    if uploaded_font is not None:
        suffix = Path(uploaded_font.name).suffix.lower() or ".ttf"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded_font.getbuffer())
        tmp.close()
        _TEMP_FONT_FILES.append(tmp.name)
        if _font_supports_chinese(tmp.name) and _activate_font(tmp.name):
            return _FONT_STATUS
        _FONT_STATUS = {
            "ok": False, "name": None, "path": tmp.name,
            "message": "上传的字体不含常用中文字符，请换 SimHei、微软雅黑、思源黑体或 Noto Sans CJK。"
        }
        return _FONT_STATUS

    # 2) 常见路径 + 本地 fonts 目录
    for fp in _candidate_font_paths():
        if os.path.exists(fp) and _font_supports_chinese(fp):
            if _activate_font(fp):
                return _FONT_STATUS

    # 3) 扫描系统字体目录，找到真正支持中文的字体
    for fp in fm.findSystemFonts(fontpaths=None, fontext="ttf") + fm.findSystemFonts(fontpaths=None, fontext="otf"):
        if _font_supports_chinese(fp):
            if _activate_font(fp):
                return _FONT_STATUS

    # 4) 没找到时保底配置；此时中文仍可能显示方框，因此会在界面提示
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["figure.dpi"] = 180
    matplotlib.rcParams["savefig.dpi"] = 600
    matplotlib.rcParams["svg.fonttype"] = "path"
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    _FONT_STATUS = {
        "ok": False,
        "name": None,
        "path": None,
        "message": "当前运行环境没有检测到中文字体，所以图中的中文会显示成方框。"
    }
    return _FONT_STATUS


def show_font_status_in_sidebar():
    """在侧边栏显示字体状态，并允许上传字体。"""
    with st.sidebar.expander("🈶 中文字体/高清导出设置", expanded=not _FONT_STATUS.get("ok", False)):
        uploaded_font = st.file_uploader(
            "中文仍是方框时，上传一个字体文件",
            type=["ttf", "ttc", "otf"],
            help="推荐：msyh.ttc、simhei.ttf、NotoSansCJK-Regular.ttc、SourceHanSansSC-Regular.otf"
        )
        if uploaded_font is not None:
            setup_chinese_font(uploaded_font)
        if _FONT_STATUS.get("ok"):
            st.success(_FONT_STATUS["message"])
            st.caption(f"字体文件：{_FONT_STATUS.get('path')}")
        else:
            st.error(_FONT_STATUS["message"])
            st.markdown(
                "把中文字体文件放到代码同级的 `fonts/` 文件夹，或在这里上传字体文件后再生成图。"
            )
        st.caption("导出图片建议优先下载 SVG，插入 Word/PPT 后缩放不容易模糊。")


def fig_to_bytes(fig, fmt="png", dpi=600):
    """把 Matplotlib 图保存成字节，用于下载按钮。"""
    buf = io.BytesIO()
    if fmt.lower() == "png":
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    else:
        fig.savefig(buf, format=fmt, bbox_inches="tight", facecolor="white")
    return buf.getvalue()


def show_fig_with_downloads(fig, base_name="figure"):
    """Streamlit 显示图，并提供 SVG/PDF/600DPI PNG 下载。"""
    st.pyplot(fig, use_container_width=True)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in base_name)[:80] or "figure"
    key = f"{safe_name}_{id(fig)}"
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "📥 下载 SVG 矢量图",
            fig_to_bytes(fig, "svg"),
            file_name=f"{safe_name}.svg",
            mime="image/svg+xml",
            key=f"{key}_svg",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "📥 下载 PDF 矢量图",
            fig_to_bytes(fig, "pdf"),
            file_name=f"{safe_name}.pdf",
            mime="application/pdf",
            key=f"{key}_pdf",
            use_container_width=True,
        )
    with col3:
        st.download_button(
            "📥 下载 600DPI PNG",
            fig_to_bytes(fig, "png", dpi=600),
            file_name=f"{safe_name}.png",
            mime="image/png",
            key=f"{key}_png",
            use_container_width=True,
        )


# 先尝试加载系统中文字体；Streamlit 启动后还可以在侧边栏上传字体重新加载
setup_chinese_font()

plt.rcParams.update({
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.linewidth':     1.2,
    'axes.grid':          True,
    'grid.alpha':         0.35,
    'grid.linestyle':     '--',
    'grid.linewidth':     0.8,
    'xtick.direction':    'out',
    'ytick.direction':    'out',
    'font.size':          11,
    'axes.titlesize':     13,
    'axes.labelsize':     12,
    'legend.fontsize':    10,
    'legend.framealpha':  0.9,
    'legend.edgecolor':   '#cccccc',
    'figure.facecolor':   'white',
    'axes.facecolor':     '#f9f9f9',
})

try:
    import pulp
except ImportError:
    raise ImportError("请先安装 PuLP：pip install pulp")

try:
    import streamlit as st
except ImportError:
    raise ImportError("请先安装 Streamlit：pip install streamlit")


# =============================================================================
# 1. 全局常量与配色（与主代码§2完全一致）
# =============================================================================
PRICE_PEAK    = 0.5609   # 峰段电价 元/kWh（论文式3.13）
PRICE_VALLEY  = 0.3109   # 谷段电价 元/kWh
ETA_DEFAULT   = 0.92     # 充电效率
KAPPA_DEFAULT = 0.045    # 退化惩罚系数（论文式2.2）
S_SEG_DEFAULT = 5        # 分段线性化段数（论文§4.3.7验证）
EPS = 1e-6               # 极小正数

# ── 策略名称与配色（论文§3.3.3定义的四种策略）──
STRATEGY_NAMES   = ['MILP有序充电', '全时段均匀充电', '仅谷段充电', '最大功率优先充电']
STRATEGY_COLORS  = ['#C0392B', '#2980B9', '#27AE60', '#F39C12']
STRATEGY_MARKERS = ['o', 's', '^', 'D']
STRATEGY_LS      = ['-', '--', '-.', ':']

# 默认 12 辆样本车（论文附录 A，与主代码 DEFAULT_VEHICLES_BASE 完全一致）
VEHICLES_DEFAULT = [
    {'编号': 'EV01', '容量': 44.90, '最大功率': 7,  '初始SOC': 0.25, '目标SOC': 0.90},
    {'编号': 'EV02', '容量': 51.80, '最大功率': 7,  '初始SOC': 0.32, '目标SOC': 0.90},
    {'编号': 'EV03', '容量': 68.20, '最大功率': 7,  '初始SOC': 0.28, '目标SOC': 0.90},
    {'编号': 'EV04', '容量': 49.52, '最大功率': 7,  '初始SOC': 0.40, '目标SOC': 0.90},
    {'编号': 'EV05', '容量': 52.80, '最大功率': 7,  '初始SOC': 0.35, '目标SOC': 0.90},
    {'编号': 'EV06', '容量': 68.82, '最大功率': 7,  '初始SOC': 0.22, '目标SOC': 0.90},
    {'编号': 'EV07', '容量': 75.00, '最大功率': 7,  '初始SOC': 0.45, '目标SOC': 0.90},
    {'编号': 'EV08', '容量': 66.00, '最大功率': 7,  '初始SOC': 0.30, '目标SOC': 0.90},
    {'编号': 'EV09', '容量': 45.99, '最大功率': 7,  '初始SOC': 0.55, '目标SOC': 0.90},
    {'编号': 'EV10', '容量': 70.20, '最大功率': 7,  '初始SOC': 0.38, '目标SOC': 0.90},
    {'编号': 'EV11', '容量': 62.50, '最大功率': 11, '初始SOC': 0.27, '目标SOC': 0.90},
    {'编号': 'EV12', '容量': 94.50, '最大功率': 11, '初始SOC': 0.50, '目标SOC': 0.90},
]


# =============================================================================
# 2. 辅助函数（与主代码§2a完全一致）
# =============================================================================
def _clamp_soc(soc):
    return max(0.0, min(1.0, float(soc)))


def _target_met(soc, target, eps=EPS):
    return soc >= target - eps


def _next_soc(soc, power, eta, dt, capacity):
    return _clamp_soc(soc + eta * power * dt / capacity)


def _physical_power_limit(soc, capacity, eta, dt):
    room = max(0.0, 1.0 - soc)
    return room * capacity / (eta * dt) if eta * dt > 0 else 0.0


def _rule_feasible(completion_rate, eps=EPS):
    return completion_rate >= 1.0 - eps


# ★ 修正9：从主代码移植缺失的辅助函数
def _find_valley_slots(prices, eps=EPS):
    """找出最低电价对应的时段（与主代码一致）"""
    min_price = min(prices)
    return [t for t, p in enumerate(prices) if abs(p - min_price) <= eps]


def _dispatch_one_slot_greedy(vehicles, soc, remain_power, eta=ETA_DEFAULT,
                              dt=0.5, order_indices=None):
    """
    论文§3.3.3："按车辆编号顺序优先满足靠前的车辆"。
    与主代码逻辑完全一致，增加 dt 参数以支持灵活时段长度。
    """
    N = len(vehicles)
    p_slot = np.zeros(N)
    if order_indices is None:
        order_indices = list(range(N))
    remain = float(remain_power)
    for i in order_indices:
        v = vehicles[i]
        if remain <= EPS or _target_met(soc[i], v['目标SOC']):
            continue
        need_grid = max(0.0, (v['目标SOC'] - soc[i]) * v['容量']) / max(eta, EPS)
        power = max(0.0, min(v['最大功率'], remain, need_grid / dt,
                             _physical_power_limit(soc[i], v['容量'], eta, dt)))
        p_slot[i] = power
        soc[i] = _next_soc(soc[i], power, eta, dt, v['容量'])
        remain -= power
    return p_slot


def slot_labels(n_slots, start_hour=22, dt=0.5):
    labels = []
    h, m = start_hour, 0
    for _ in range(n_slots):
        labels.append(f'{h:02d}:{m:02d}')
        m += int(dt * 60)
        if m >= 60:
            h = (h + 1) % 24
            m = 0
    return labels


def compute_ref_values(vehicles, n_slots, dt=0.5, kappa=KAPPA_DEFAULT,
                       eta=ETA_DEFAULT, price_peak=PRICE_PEAK):
    """
    论文式(3.8): T_ref = N × T × Δt
    论文式(3.9): C_ref = (Σ需求电量 / η) × π_peak
    论文式(3.10): D_ref = N × T × κ × Δt
    与主代码完全一致。
    """
    N = len(vehicles)
    T_ref = N * n_slots * dt
    bat_energy = sum((v['目标SOC'] - v['初始SOC']) * v['容量'] for v in vehicles)
    C_ref = (bat_energy / eta) * price_peak
    D_ref = N * n_slots * kappa * dt
    return T_ref, C_ref, D_ref


def compute_obj(T_total, C_total, D_total, T_ref, C_ref, D_ref,
                w_T=0.30, w_C=0.50, w_D=0.20):
    """论文式(3.4): J = w_T*(T/T_ref) + w_C*(C/C_ref) + w_D*(D/D_ref)"""
    return w_T * (T_total / T_ref) + w_C * (C_total / C_ref) + w_D * (D_total / D_ref)


# =============================================================================
# 3. MILP 求解器（与主代码§4完全对齐）
# =============================================================================
def solve_milp(vehicles, prices, P_agg_max,
               T_ref=None, C_ref=None, D_ref=None,
               w_T=0.30, w_C=0.50, w_D=0.20,
               kappa=KAPPA_DEFAULT, eta=ETA_DEFAULT,
               S=S_SEG_DEFAULT, time_limit=180,
               dt=0.5, solver_msg=False):
    N = len(vehicles)
    T = len(prices)

    # ★ 修正10：参考值回退时使用 PRICE_PEAK，与主代码一致
    if T_ref is None or C_ref is None or D_ref is None:
        T_ref, C_ref, D_ref = compute_ref_values(
            vehicles, n_slots=T, dt=dt, kappa=kappa, eta=eta,
            price_peak=PRICE_PEAK
        )

    prob = pulp.LpProblem("EV_MILP", pulp.LpMinimize)

    # 决策变量（论文式3.1–3.3）
    y = [[pulp.LpVariable(f"y_{i}_{t}", cat='Binary')
          for t in range(T)] for i in range(N)]
    delta = [[[pulp.LpVariable(f"d_{i}_{t}_{s}", lowBound=0)
               for s in range(S)] for t in range(T)] for i in range(N)]
    P_var = [[pulp.LpVariable(f"P_{i}_{t}", lowBound=0)
              for t in range(T)] for i in range(N)]
    soc_var = [[pulp.LpVariable(f"soc_{i}_{t}", lowBound=0.0, upBound=1.0)
                for t in range(T + 1)] for i in range(N)]

    for i, v in enumerate(vehicles):
        cap = v['容量']
        P_max = v['最大功率']
        seg_w = P_max / S  # 论文式(2.10)

        prob += soc_var[i][0] == v['初始SOC']  # 论文式(3.19)

        for t in range(T):
            for s in range(S):
                prob += delta[i][t][s] <= seg_w * y[i][t]  # 式(3.15)
            prob += P_var[i][t] == pulp.lpSum(delta[i][t][s] for s in range(S))  # 式(3.14)
            prob += P_var[i][t] <= P_max * y[i][t]   # 式(3.21)上
            prob += P_var[i][t] >= EPS * y[i][t]     # 式(3.21)下
            prob += (soc_var[i][t + 1] ==
                     soc_var[i][t] + eta * P_var[i][t] * dt / cap)  # 式(3.18)

        prob += soc_var[i][T] >= v['目标SOC']  # 式(3.20)

    # 聚合功率约束（论文式3.17）
    for t in range(T):
        prob += pulp.lpSum(P_var[i][t] for i in range(N)) <= P_agg_max

    # 目标函数（论文式3.4–3.12）
    T_expr = pulp.lpSum(y[i][t] * dt for i in range(N) for t in range(T))       # 式(3.6)
    C_expr = pulp.lpSum(prices[t] * P_var[i][t] * dt
                        for i in range(N) for t in range(T))                      # 式(3.11)
    D_expr = pulp.lpSum(
        (kappa * dt * (2 * s + 1) / (S * vehicles[i]['最大功率'])) * delta[i][t][s]
        for i in range(N) for t in range(T) for s in range(S))                    # 式(2.15)

    prob += (w_T / T_ref) * T_expr + (w_C / C_ref) * C_expr + (w_D / D_ref) * D_expr

    # ★ 修正8：求解器选择与主代码统一，只用 PULP_CBC_CMD
    t0 = time.time()
    try:
        solver = pulp.PULP_CBC_CMD(msg=bool(solver_msg), timeLimit=time_limit)
    except Exception:
        solver = pulp.getSolver('PULP_CBC_CMD', msg=bool(solver_msg), timeLimit=time_limit)

    prob.solve(solver)
    solve_time = time.time() - t0

    status_code = prob.status
    status_str = str(pulp.LpStatus.get(status_code, str(status_code))).upper().replace(' ', '_')
    near_tl = (solve_time >= time_limit * 0.95)
    is_optimal = (status_str == 'OPTIMAL')

    if near_tl and not is_optimal and status_code == 1:
        print(
            f"\n    ⚠ 求解时间({solve_time:.1f}s)接近时限({time_limit}s)，"
            f"求解器状态为 {status_str}，返回当前可行解"
        )

    # ★ 修正2：不可行判定与主代码完全一致 —— 直接检查 prob.status != 1
    #   主代码在 status != 1 时直接返回 infeasible，completion_rate = np.nan
    if prob.status != 1:
        return {
            'feasible': False, 'obj': None,
            'T_total': None, 'C_total': None, 'D_total': None,
            'completion_rate': np.nan,
            'P_schedule': None, 'soc_schedule': None,
            'solve_time': round(solve_time, 2),
            'milp_status_code': status_code, 'milp_status_str': status_str,
            'near_time_limit': near_tl,
            'infeasible_reason': status_str, 'unmet_energy': np.nan,
            'delivered_kWh': np.nan,
        }

    # ★ 修正2：status==1 时直接读取求解器输出（与主代码一致，不再额外重模拟）
    P_sched = np.array([[max(0.0, pulp.value(P_var[i][t]) or 0.0)
                         for t in range(T)] for i in range(N)])
    soc_sched = np.array([[pulp.value(soc_var[i][t]) or 0.0
                           for t in range(T + 1)] for i in range(N)])

    # ★ 修正1：T_total 使用二元变量 y 计算，与主代码完全一致
    T_total = sum(round(pulp.value(y[i][t]) or 0) * dt
                  for i in range(N) for t in range(T))
    C_total = float(sum(prices[t] * P_sched[i][t] * dt
                        for i in range(N) for t in range(T)))
    # D_total 使用精确二次公式回算（与主代码一致）
    D_total = float(sum(kappa * (P_sched[i][t] / vehicles[i]['最大功率']) ** 2 * dt
                        for i in range(N) for t in range(T)))
    obj = compute_obj(T_total, C_total, D_total, T_ref, C_ref, D_ref, w_T, w_C, w_D)
    n_done = sum(1 for i, v in enumerate(vehicles)
                 if _target_met(soc_sched[i][-1], v['目标SOC']))
    delivered = float(sum(eta * P_sched[i][t] * dt
                          for i in range(N) for t in range(T)))

    return {
        'feasible': True, 'obj': round(obj, 6),
        'T_total': round(T_total, 4), 'C_total': round(C_total, 4),
        'D_total': round(D_total, 6),
        'completion_rate': n_done / N,
        'P_schedule': P_sched, 'soc_schedule': soc_sched,
        'solve_time': round(solve_time, 2),
        'milp_status_code': status_code, 'milp_status_str': status_str,
        'near_time_limit': near_tl,
        'infeasible_reason': None, 'unmet_energy': 0.0,
        'delivered_kWh': round(delivered, 2),
    }


# =============================================================================
# 4. 三种规则策略（与主代码§5完全对齐）
# =============================================================================

# ★ 修正4：_metrics_from_schedule 返回7项，与主代码一致（增加 unmet、delivered）
def _metrics_from_schedule(P_sched, vehicles, prices,
                           kappa=KAPPA_DEFAULT, eta=ETA_DEFAULT, dt=0.5):
    N, T = P_sched.shape
    soc_sched = np.zeros((N, T + 1))
    eff_sched = np.zeros((N, T))

    for i, v in enumerate(vehicles):
        soc_sched[i][0] = _clamp_soc(v['初始SOC'])
        for t in range(T):
            p_req = max(0.0, float(P_sched[i][t]))
            p_phy = _physical_power_limit(soc_sched[i][t], v['容量'], eta, dt)
            p_eff = min(p_req, p_phy)
            eff_sched[i][t] = p_eff
            soc_sched[i][t + 1] = _next_soc(
                soc_sched[i][t], p_eff, eta, dt, v['容量'])

    T_total = float(sum(dt for i in range(N) for t in range(T)
                        if eff_sched[i][t] > 0.01))
    C_total = float(sum(prices[t] * eff_sched[i][t] * dt
                        for i in range(N) for t in range(T)))
    D_total = float(sum(kappa * (eff_sched[i][t] / vehicles[i]['最大功率']) ** 2 * dt
                        for i in range(N) for t in range(T)))
    n_done = sum(1 for i, v in enumerate(vehicles)
                 if _target_met(soc_sched[i][-1], v['目标SOC']))
    # ★ 新增：unmet 和 delivered，与主代码一致
    unmet = sum(max(0.0, (v['目标SOC'] - soc_sched[i][-1]) * v['容量'])
                for i, v in enumerate(vehicles))
    delivered = float(sum(eta * eff_sched[i][t] * dt
                          for i in range(N) for t in range(T)))
    return T_total, C_total, D_total, n_done / N, unmet, soc_sched, delivered


# ★ 修正5：strategy_valley_only 使用 _find_valley_slots + _dispatch_one_slot_greedy
#   与主代码逻辑完全一致
def strategy_valley_only(vehicles, prices, P_agg_max, eta=ETA_DEFAULT, dt=0.5,
                         valley_slots=None):
    """对比策略1（论文§3.3.3）：仅在谷段充电。"""
    N = len(vehicles)
    T = len(prices)
    P = np.zeros((N, T))
    soc = np.array([_clamp_soc(v['初始SOC']) for v in vehicles], dtype=float)
    if valley_slots is None:
        valley_slots = _find_valley_slots(prices)
    for t in valley_slots:
        P[:, t] = _dispatch_one_slot_greedy(vehicles, soc, P_agg_max, eta, dt,
                                            list(range(N)))
    return P


def strategy_uniform(vehicles, prices, P_agg_max, eta=ETA_DEFAULT, dt=0.5):
    """对比策略2（论文§3.3.3）：全时段均匀充电。与主代码完全一致。"""
    N = len(vehicles)
    T = len(prices)
    P = np.zeros((N, T))

    for i, v in enumerate(vehicles):
        grid_need = max(0.0, (v['目标SOC'] - v['初始SOC']) * v['容量']) / eta
        P[i, :] = min(grid_need / (T * dt), v['最大功率'])

    for t in range(T):
        total = P[:, t].sum()
        if total > P_agg_max + EPS:
            P[:, t] *= P_agg_max / total
    return P


# ★ 修正5：strategy_max_power 使用 _dispatch_one_slot_greedy，与主代码一致
def strategy_max_power(vehicles, prices, P_agg_max, eta=ETA_DEFAULT, dt=0.5):
    """对比策略3（论文§3.3.3）：最大功率优先充电，按车辆编号顺序分配。"""
    N = len(vehicles)
    T = len(prices)
    P = np.zeros((N, T))
    soc = np.array([_clamp_soc(v['初始SOC']) for v in vehicles], dtype=float)
    for t in range(T):
        P[:, t] = _dispatch_one_slot_greedy(vehicles, soc, P_agg_max, eta, dt,
                                            list(range(N)))
    return P


# =============================================================================
# 5. 一键运行所有策略（与主代码§6对齐，返回字段完整）
# =============================================================================
def run_all_strategies(vehicles, prices, P_agg_max, T_ref, C_ref, D_ref,
                       w_T=0.30, w_C=0.50, w_D=0.20,
                       kappa=KAPPA_DEFAULT, eta=ETA_DEFAULT, dt=0.5,
                       S=S_SEG_DEFAULT, time_limit=180):
    """
    返回 dict{策略名: 结果字典}，每个结果字典字段与主代码完全一致。
    """
    results = {}

    # MILP有序充电
    # ★ 修正3：将 S 参数传入 solve_milp
    r = solve_milp(vehicles, prices, P_agg_max, T_ref, C_ref, D_ref,
                   w_T=w_T, w_C=w_C, w_D=w_D, kappa=kappa, eta=eta,
                   S=S, dt=dt, time_limit=time_limit)
    results['MILP有序充电'] = r

    # ★ 修正6：规则策略结果包含与主代码一致的完整字段
    rule_funcs = {
        '全时段均匀充电': strategy_uniform,
        '仅谷段充电':     strategy_valley_only,
        '最大功率优先充电': strategy_max_power,
    }
    for name, func in rule_funcs.items():
        P_sched = func(vehicles, prices, P_agg_max, eta=eta, dt=dt)
        Tt, Ct, Dt, cr, unmet, soc_sched, dlv = _metrics_from_schedule(
            P_sched, vehicles, prices, kappa=kappa, eta=eta, dt=dt)
        obj = compute_obj(Tt, Ct, Dt, T_ref, C_ref, D_ref, w_T, w_C, w_D)
        feasible = _rule_feasible(cr)
        results[name] = {
            'feasible': feasible, 'obj': round(obj, 6),
            'T_total': round(Tt, 4), 'C_total': round(Ct, 4),
            'D_total': round(Dt, 6), 'completion_rate': cr,
            'P_schedule': P_sched, 'soc_schedule': soc_sched,
            'solve_time': 0.0, 'unmet_energy': unmet,
            'delivered_kWh': round(dlv, 2),
            'milp_status_code': None, 'milp_status_str': None,
            'near_time_limit': False,
            'infeasible_reason': '未全部车辆达标' if not feasible else None,
        }
    return results


# =============================================================================
# 6. 显示辅助（与主代码显示逻辑对齐）
# =============================================================================
def _format_cr_display(strategy_name, r):
    """
    达标率显示格式与论文表4.3一致：
    - MILP不可行 → '—(无可行解)'
    - 规则策略部分达标 → 'XX.X%(部分)'
    - 可行 → 'XX.X%'
    """
    cr = r.get('completion_rate', np.nan)
    feasible = bool(r.get('feasible', False)) and r.get('obj') is not None
    if strategy_name == 'MILP有序充电' and not feasible:
        return '—(无可行解)'
    if isinstance(cr, float) and np.isnan(cr):
        return '—'
    cr_pct = cr * 100
    if feasible:
        return f"{cr_pct:.1f}%"
    else:
        return f"{cr_pct:.1f}%(部分)"


def _fmt_value(value, digits=4, dash='—'):
    """安全格式化数值。None/NaN 显示为破折号。与主代码一致。"""
    if value is None:
        return dash
    try:
        if pd.isna(value):
            return dash
    except Exception:
        pass
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _milp_optimality_note(result):
    """MILP最优性说明，与主代码一致。"""
    status_str = result.get('milp_status_str')
    feasible = bool(result.get('feasible', False)) and result.get('obj') is not None
    near_tl = bool(result.get('near_time_limit', False))

    if not feasible:
        return f"不可行({status_str})" if status_str else '不可行'

    if str(status_str).upper() == 'OPTIMAL':
        return '最优(接近时限)' if near_tl else '最优'

    if near_tl:
        return f"达到时限，返回当前可行解({status_str})" if status_str else '达到时限，返回当前可行解'

    return f"可行({status_str})" if status_str else '可行'


def _result_status_text(strategy_name, r):
    """生成展示用状态文字。与主代码一致。"""
    feasible = bool(r.get('feasible', False)) and r.get('obj') is not None
    has_metrics = r.get('obj') is not None

    if strategy_name == 'MILP有序充电':
        return _milp_optimality_note(r)

    if feasible:
        return '规则可行'
    if has_metrics:
        return '规则部分达标'
    return '规则不可行'


# =============================================================================
# 7. 个人用户单车充电规划模式
# =============================================================================
def personal_mode():
    """个人用户单车充电规划模式"""
    st.title("🚗 我的充电规划助手")
    st.caption("输入你的车辆信息，一键获取最省钱的充电时段安排")

    # ── 简洁输入 ──────────────────────────────────────────────────────
    st.header("📝 填写你的车辆信息")

    col1, col2 = st.columns(2)
    with col1:
        car_name = st.text_input("你的车型（备注用）", "我的电动车")
        battery_cap = st.number_input("电池总容量 (kWh)", 20.0, 200.0, 60.0, 1.0,
                                       help="可在车辆说明书或App中查看")
        max_power = st.selectbox("家用充电桩功率 (kW)",
                                  [3.5, 7.0, 11.0], index=1,
                                  format_func=lambda x: f"{x} kW")
    with col2:
        current_soc = st.slider("当前电量 (%)", 5, 90, 30, 1,
                                 help="当前仪表盘显示的电量百分比")
        target_soc = st.slider("期望充到 (%)", 50, 100, 90, 5,
                                help="明早出门前希望达到的电量")
        depart_time = st.selectbox("明早几点出门？",
                                    [5, 6, 7, 8], index=2,
                                    format_func=lambda x: f"{x:02d}:00")

    if current_soc >= target_soc:
        st.success("🎉 当前电量已经满足你的需求，不需要充电！")
        return

    # 计算参数（与主代码参数口径一致）
    dt = 0.5
    start_hour = 22
    n_hours = (24 - start_hour) + depart_time
    n_slots = int(n_hours / dt)
    n_peak_slots = int((23 - start_hour) / dt)   # 22:00-23:00 为峰段
    price_peak = PRICE_PEAK
    price_valley = PRICE_VALLEY
    prices = [price_peak if t < n_peak_slots else price_valley for t in range(n_slots)]
    eta = ETA_DEFAULT
    kappa = KAPPA_DEFAULT

    vehicle = {
        '编号': 'MY_EV',
        '容量': battery_cap,
        '最大功率': max_power,
        '初始SOC': current_soc / 100.0,
        '目标SOC': min(1.0, target_soc / 100.0 + 0.005),
    }

    vehicles = [vehicle]

    # 需求预估
    energy_need = (target_soc - current_soc) / 100.0 * battery_cap
    min_hours = energy_need / (eta * max_power)
    naive_cost = energy_need / eta * price_peak
    valley_cost = energy_need / eta * price_valley

    st.divider()
    st.header("⚡ 充电需求预估")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("需补充电量", f"{energy_need:.1f} kWh")
    c2.metric("最少充电时间", f"{min_hours:.1f} 小时")
    c3.metric("全峰段电费", f"¥{naive_cost:.2f}")
    c4.metric("全谷段电费", f"¥{valley_cost:.2f}",
              delta=f"省 ¥{naive_cost - valley_cost:.2f}",
              delta_color="inverse")

    # ── 求解 ──────────────────────────────────────────────────────────
    if st.button("🔍 帮我规划最优充电方案", type="primary", use_container_width=True):
        T_ref, C_ref, D_ref = compute_ref_values(vehicles, n_slots, dt=dt,
                                                   kappa=kappa, eta=eta,
                                                   price_peak=price_peak)

        with st.spinner("正在计算最优方案..."):
            # MILP 最优
            milp_r = solve_milp(vehicles, prices, max_power * 1.1,
                                T_ref, C_ref, D_ref,
                                w_T=0.30, w_C=0.50, w_D=0.20,
                                kappa=kappa, eta=eta, S=S_SEG_DEFAULT,
                                dt=dt, time_limit=60)
            # 对比：最大功率直充
            P_max_sched = strategy_max_power(vehicles, prices, max_power * 1.1,
                                              eta=eta, dt=dt)
            T_m, C_m, D_m, cr_m, unmet_m, soc_m, dlv_m = _metrics_from_schedule(
                P_max_sched, vehicles, prices, kappa=kappa, eta=eta, dt=dt)

        if milp_r['feasible']:
            st.success("✅ 规划完成！")

            # ── 核心结果 ──────────────────────────────────────────
            st.header("📋 你的最优充电方案")

            saved = C_m - milp_r['C_total']
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("优化后总电费", f"¥{milp_r['C_total']:.2f}")
            cc2.metric("直充电费", f"¥{C_m:.2f}")
            cc3.metric("帮你省了", f"¥{max(0, saved):.2f}",
                       delta=f"{max(0, saved) / max(C_m, 0.01) * 100:.1f}%" if saved > 0 else "—")

            slabels = slot_labels(n_slots, start_hour, dt)
            P_sched = milp_r['P_schedule'][0]

            # ── 充电时段安排（用户友好版）──────────────────────
            st.subheader("🕐 充电时段安排")

            schedule_data = []
            for t in range(n_slots):
                p = P_sched[t]
                h_start = int((start_hour + t * dt) % 24)
                m_start = int((t * dt % 1) * 60)
                h_end = int((start_hour + (t + 1) * dt) % 24)
                m_end = int(((t + 1) * dt % 1) * 60)
                time_str = f"{h_start:02d}:{m_start:02d}—{h_end:02d}:{m_end:02d}"

                if p > 0.1:
                    price_now = prices[t]
                    cost_now = p * dt * price_now
                    schedule_data.append({
                        '时段': time_str,
                        '充电功率': f"{p:.2f} kW",
                        '电价': f"{'⚡峰段' if price_now > 0.4 else '🌙谷段'} ¥{price_now:.4f}",
                        '该时段电费': f"¥{cost_now:.3f}",
                        '状态': '🔌 充电中',
                    })
                else:
                    schedule_data.append({
                        '时段': time_str,
                        '充电功率': '—',
                        '电价': f"{'⚡峰段' if prices[t] > 0.4 else '🌙谷段'}",
                        '该时段电费': '—',
                        '状态': '😴 等待',
                    })

            st.dataframe(pd.DataFrame(schedule_data), use_container_width=True,
                         hide_index=True)

            # ── 图：功率 + SOC 双轴图 ─────────────────────────
            st.subheader("📈 充电过程可视化")

            fig, ax1 = plt.subplots(figsize=(12, 5))

            colors_bar = ['#E74C3C' if prices[t] > 0.4 else '#3498DB'
                          for t in range(n_slots)]
            ax1.bar(range(n_slots), P_sched, color=colors_bar, alpha=0.7,
                    label='充电功率 (kW)')
            ax1.set_ylabel('充电功率 (kW)', color='#2C3E50')
            ax1.set_xlabel('时段')
            ax1.set_xticks(range(0, n_slots, 2))
            ax1.set_xticklabels(slabels[::2], rotation=45, fontsize=8)

            ax2 = ax1.twinx()
            soc_vals = milp_r['soc_schedule'][0] * 100
            ax2.plot(range(n_slots + 1), soc_vals, color='#27AE60',
                     linewidth=2.5, marker='o', markersize=5,
                     markerfacecolor='white', markeredgewidth=2,
                     label='电量 SOC (%)', zorder=5)
            ax2.axhline(y=target_soc, color='#E67E22', linestyle='--',
                        linewidth=1.5, label=f'目标电量 {target_soc}%')
            ax2.set_ylabel('电量 SOC (%)', color='#27AE60')
            ax2.set_ylim(0, 105)

            peak_patch = mpatches.Patch(color='#E74C3C', alpha=0.7, label='峰段时段')
            valley_patch = mpatches.Patch(color='#3498DB', alpha=0.7, label='谷段时段')
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(handles=[peak_patch, valley_patch] + lines2,
                       loc='upper left', frameon=True, fontsize=9)

            # 不在导出的图片中添加标题
            plt.tight_layout()
            show_fig_with_downloads(fig, "ev_charging_figure")
            plt.close()

            # ── 简明建议 ─────────────────────────────────────
            st.divider()
            st.header("💡 充电建议")

            charging_slots = [t for t in range(n_slots) if P_sched[t] > 0.1]
            if charging_slots:
                first_t = charging_slots[0]
                last_t = charging_slots[-1]
                h_first = int((start_hour + first_t * dt) % 24)
                m_first = int((first_t * dt % 1) * 60)
                h_last_end = int((start_hour + (last_t + 1) * dt) % 24)
                m_last_end = int(((last_t + 1) * dt % 1) * 60)

                valley_count = sum(1 for t in charging_slots if prices[t] < 0.4)
                peak_count = len(charging_slots) - valley_count

                st.markdown(f"""
### 📌 核心建议

1. **最佳插枪时间**：`{h_first:02d}:{m_first:02d}` 开始充电
2. **预计充满时间**：`{h_last_end:02d}:{m_last_end:02d}` 达到 {target_soc}% 电量
3. **充电总时长**：约 `{len(charging_slots) * dt:.1f}` 小时
4. **预计电费**：`¥{milp_r['C_total']:.2f}`

### 💰 省钱提示
- 共有 **{valley_count}** 个时段在谷段充电（电价 ¥{price_valley}/kWh）
- {'⚠️ 有 ' + str(peak_count) + ' 个时段在峰段充电（电价 ¥' + str(price_peak) + '/kWh），因为时间窗较紧' if peak_count > 0 else '✅ 全部在谷段充电，已是最省方案！'}
- 相比从22:00直接满功率充电，**节省约 ¥{max(0, saved):.2f}**
                """)

            # ── 下载充电方案 ─────────────────────────────────
            st.divider()
            csv_data = pd.DataFrame(schedule_data).to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                "📥 下载我的充电方案 (CSV)",
                csv_data.encode('utf-8-sig'),
                "my_charging_plan.csv",
                "text/csv",
                use_container_width=True
            )

        else:
            st.error("❌ 在当前时间窗内无法充到目标电量！")
            st.markdown(f"""
**可能的原因和建议：**
- 充电桩功率 {max_power} kW × {n_hours} 小时 = 最多充 {max_power * n_hours * eta:.1f} kWh
- 但你需要补充 {energy_need:.1f} kWh
- 建议：**提早插枪** 或 **降低目标电量** 或 **使用更大功率充电桩**
            """)


# =============================================================================
# 8. 多车聚合调度模式（论文演示）
# =============================================================================
def multi_vehicle_mode():
    """原有的多车聚合调度模式"""
    st.title("🔋 居民夜间多车聚合充电最优调度系统")
    st.caption("基于混合整数线性规划（MILP）模型 · 宝鸡文理学院 · 毕业设计演示")

    # ─────────────────────────────────────────────────────────────────────
    # 侧边栏：系统参数
    # ─────────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ 系统参数")

        st.subheader("时间设置")
        start_hour = 22
        end_hour = st.selectbox("充电结束时间", [5, 6, 7], index=2,
                                format_func=lambda x: f"次日 {x:02d}:00")
        dt = 0.5
        n_hours = (24 - start_hour) + end_hour
        n_slots = int(n_hours / dt)
        st.info(f"充电时间窗：22:00 — 次日{end_hour:02d}:00（{n_hours}小时，{n_slots}个时段）")

        st.subheader("电价参数")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            price_peak = st.number_input("峰段电价(元/kWh)", 0.30, 1.50, PRICE_PEAK, 0.01,
                                         format="%.4f")
        with col_p2:
            price_valley = st.number_input("谷段电价(元/kWh)", 0.10, 1.00, PRICE_VALLEY, 0.01,
                                           format="%.4f")

        # 22:00-23:00为峰段（前2个时段），之后为谷段（与主代码一致）
        n_peak_slots = int((23 - start_hour) / dt)
        prices = [price_peak if t < n_peak_slots else price_valley for t in range(n_slots)]

        st.subheader("系统约束")
        P_agg_max = st.number_input("聚合功率上限 (kW)", 20.0, 200.0, 49.0, 1.0)
        eta = st.number_input("充电效率 η", 0.80, 1.00, ETA_DEFAULT, 0.01, format="%.2f")
        kappa = st.number_input("退化惩罚系数 κ", 0.01, 0.20, KAPPA_DEFAULT, 0.005,
                                format="%.3f")

        st.subheader("📊 权重设置")
        w_T = st.slider("时长权重 w₁", 0.0, 1.0, 0.30, 0.05)
        w_C = st.slider("电费权重 w₂", 0.0, 1.0, 0.50, 0.05)
        w_D = st.slider("退化权重 w₃", 0.0, 1.0, 0.20, 0.05)
        w_sum = w_T + w_C + w_D
        if abs(w_sum - 1.0) > 0.01:
            st.error(f"⚠️ 权重之和 = {w_sum:.2f}，应为 1.00，请调整！")

        st.subheader("求解设置")
        S_seg = st.selectbox("分段线性化段数 S", [3, 5, 8, 10], index=1)
        time_limit = st.number_input("求解时间限制 (秒)", 30, 600, 180, 30)

    # ─────────────────────────────────────────────────────────────────────
    # 主区域：车辆参数表
    # ─────────────────────────────────────────────────────────────────────
    st.header("🚗 车辆参数设置")

    col_mode1, col_mode2 = st.columns([1, 3])
    with col_mode1:
        n_vehicles = st.number_input("车辆数量", 1, 20, 12, 1)

    default_vehicles = []
    for k in range(n_vehicles):
        v = VEHICLES_DEFAULT[k % len(VEHICLES_DEFAULT)].copy()
        v['编号'] = f'EV{k+1:02d}'
        default_vehicles.append(v)

    df_default = pd.DataFrame(default_vehicles)
    df_default.columns = ['车辆编号', '电池容量(kWh)', '最大充电功率(kW)', '初始SOC', '目标SOC']

    st.write("可直接在下表中修改参数（修改后点击下方按钮重新求解）：")
    edited_df = st.data_editor(
        df_default,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "车辆编号": st.column_config.TextColumn("车辆编号", disabled=True),
            "电池容量(kWh)": st.column_config.NumberColumn("电池容量(kWh)", min_value=10.0,
                                                           max_value=200.0, step=0.1,
                                                           format="%.1f"),
            "最大充电功率(kW)": st.column_config.NumberColumn("最大功率(kW)", min_value=1.0,
                                                              max_value=22.0, step=0.5,
                                                              format="%.1f"),
            "初始SOC": st.column_config.NumberColumn("初始SOC", min_value=0.05,
                                                     max_value=0.95, step=0.01,
                                                     format="%.2f"),
            "目标SOC": st.column_config.NumberColumn("目标SOC", min_value=0.50,
                                                     max_value=1.00, step=0.01,
                                                     format="%.2f"),
        }
    )

    vehicles = []
    for _, row in edited_df.iterrows():
        vehicles.append({
            '编号':     row['车辆编号'],
            '容量':     float(row['电池容量(kWh)']),
            '最大功率': float(row['最大充电功率(kW)']),
            '初始SOC':  float(row['初始SOC']),
            '目标SOC':  float(row['目标SOC']),
        })

    total_demand = sum((v['目标SOC'] - v['初始SOC']) * v['容量'] for v in vehicles)
    max_delivery = P_agg_max * n_slots * dt
    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("总补能需求", f"{total_demand:.1f} kWh")
    col_info2.metric("理论最大交付", f"{max_delivery:.1f} kWh")
    col_info3.metric("需求/容量比", f"{total_demand/eta/max_delivery*100:.1f}%")

    if total_demand / eta > max_delivery:
        st.warning("⚠️ 总充电需求超过系统理论最大交付能力，MILP可能不可行！")

    # ─────────────────────────────────────────────────────────────────────
    # 求解按钮
    # ─────────────────────────────────────────────────────────────────────
    st.divider()

    if st.button("🚀 开始优化求解", type="primary", use_container_width=True):
        if abs(w_sum - 1.0) > 0.01:
            st.error("请先将权重之和调整为 1.00！")
            return

        T_ref, C_ref, D_ref = compute_ref_values(vehicles, n_slots, dt=dt,
                                                   kappa=kappa, eta=eta,
                                                   price_peak=price_peak)

        with st.spinner("正在求解，请稍候……（MILP + 3种规则策略）"):
            # ★ 修正3：将侧边栏的 S_seg 传入 run_all_strategies → solve_milp
            all_results = run_all_strategies(
                vehicles, prices, P_agg_max, T_ref, C_ref, D_ref,
                w_T=w_T, w_C=w_C, w_D=w_D,
                kappa=kappa, eta=eta, dt=dt,
                S=S_seg, time_limit=time_limit
            )

        milp_r = all_results['MILP有序充电']

        if milp_r['feasible']:
            st.success(f"✅ MILP 求解成功！用时 {milp_r['solve_time']:.1f} 秒  "
                       f"（{_milp_optimality_note(milp_r)}）")
        else:
            st.error("❌ MILP 求解不可行！请检查参数（降低目标SOC / 增加功率上限 / 延长时间窗）。")

        # ── 核心指标卡片 ──────────────────────────────────────────────
        st.header("📋 求解结果总览")

        if milp_r['feasible']:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("综合目标值", f"{milp_r['obj']:.4f}")
            c2.metric("总充电时长", f"{milp_r['T_total']:.1f} h")
            c3.metric("总电费成本", f"{milp_r['C_total']:.2f} 元")
            c4.metric("总退化惩罚", f"{milp_r['D_total']:.4f}")
            c5.metric("求解时间", f"{milp_r['solve_time']:.1f} s")

        # ── 四策略对比表 ──────────────────────────────────────────────
        # ★ 修正7：使用 _format_cr_display 和 _fmt_value 安全处理 NaN
        st.subheader("📊 四种策略综合对比")

        compare_rows = []
        for name in STRATEGY_NAMES:
            r = all_results[name]
            has_metrics = r.get('obj') is not None

            if name == 'MILP有序充电':
                status_icon = '✅ ' + _milp_optimality_note(r) if r['feasible'] else '❌ ' + _milp_optimality_note(r)
            else:
                status_icon = '✅ 可行' if r['feasible'] else ('⚠️ 部分达标' if has_metrics else '❌ 不可行')

            compare_rows.append({
                '策略': name,
                '状态': status_icon,
                '综合目标值': _fmt_value(r.get('obj'), 6) if has_metrics else '—',
                '充电时长(h)': _fmt_value(r.get('T_total'), 1) if has_metrics else '—',
                '电费成本(元)': _fmt_value(r.get('C_total'), 2) if has_metrics else '—',
                '退化惩罚': _fmt_value(r.get('D_total'), 6) if has_metrics else '—',
                '达标率': _format_cr_display(name, r),
                '交付电量(kWh)': _fmt_value(r.get('delivered_kWh'), 2),
            })
        st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)

        # ── 可视化 Tab ────────────────────────────────────────────────
        st.header("📈 可视化分析")
        slabels = slot_labels(n_slots, start_hour, dt)
        n_peak = n_peak_slots

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 聚合负荷曲线", "🔥 MILP功率热力图", "📈 SOC演化曲线",
            "📦 成本构成分析", "📋 详细功率数据"
        ])

        # ── Tab1：聚合负荷 ────────────────────────────────────────────
        with tab1:
            fig, ax = plt.subplots(figsize=(12, 5))
            x = np.arange(n_slots)

            for name, color, marker, ls in zip(STRATEGY_NAMES, STRATEGY_COLORS,
                                                STRATEGY_MARKERS, STRATEGY_LS):
                r = all_results[name]
                if r['P_schedule'] is not None:
                    agg = r['P_schedule'].sum(axis=0)
                else:
                    agg = np.zeros(n_slots)
                ax.plot(x, agg, color=color, marker=marker, linestyle=ls,
                        markevery=2, linewidth=2.2, markersize=7, label=name,
                        markerfacecolor='white', markeredgewidth=1.8)

            ax.axhline(y=P_agg_max, color='#2C3E50', linestyle='--',
                       linewidth=2, label=f'聚合上限 {P_agg_max} kW')
            ax.axvspan(-0.5, n_peak - 0.5, alpha=0.07, color='#E74C3C')
            ax.axvspan(n_peak - 0.5, n_slots - 0.5, alpha=0.05, color='#3498DB')
            if n_peak > 0:
                ax.axvline(x=n_peak - 0.5, color='gray', linestyle=':', linewidth=1.2)
                ax.text(n_peak / 2 - 0.5, P_agg_max * 0.92, '峰段',
                        fontsize=10, color='#C0392B', ha='center', fontweight='bold')
                ax.text(n_peak + 2, P_agg_max * 0.92, '谷段',
                        fontsize=10, color='#2980B9', ha='center', fontweight='bold')

            ax.set_xticks(x[::2])
            ax.set_xticklabels(slabels[::2], rotation=45, fontsize=9)
            ax.set_xlabel('时段')
            ax.set_ylabel('聚合充电功率 (kW)')
            # 不在导出的图片中添加标题
            ax.legend(frameon=True, loc='upper right', fontsize=9)
            plt.tight_layout()
            show_fig_with_downloads(fig, "ev_charging_figure")
            plt.close()

        # ── Tab2：MILP 功率热力图 ────────────────────────────────────
        with tab2:
            if milp_r['feasible'] and milp_r['P_schedule'] is not None:
                P_sched_milp = milp_r['P_schedule']
                N_v = P_sched_milp.shape[0]
                veh_ids = [v['编号'] for v in vehicles]
                p_max_all = max(v['最大功率'] for v in vehicles)

                fig, ax = plt.subplots(figsize=(14, max(4, N_v * 0.5)))
                # 使用 pcolormesh，导出 SVG/PDF 时比 imshow 更接近矢量效果
                x_edges = np.arange(n_slots + 1) - 0.5
                y_edges = np.arange(N_v + 1) - 0.5
                im = ax.pcolormesh(x_edges, y_edges, P_sched_milp, cmap='YlOrRd',
                                    vmin=0, vmax=p_max_all, shading='flat')
                ax.set_xlim(-0.5, n_slots - 0.5)
                ax.set_ylim(N_v - 0.5, -0.5)
                cbar = plt.colorbar(im, ax=ax, label='充电功率 (kW)',
                                    fraction=0.03, pad=0.02)
                cbar.ax.tick_params(labelsize=9)

                ax.set_xticks(range(n_slots))
                ax.set_xticklabels(slabels, rotation=45, fontsize=8)
                ax.set_yticks(range(N_v))
                ax.set_yticklabels(veh_ids, fontsize=9)
                ax.set_xlabel('时段')
                ax.set_ylabel('车辆编号')
                # 不在导出的图片中添加标题

                for i in range(N_v):
                    for t in range(n_slots):
                        val = P_sched_milp[i][t]
                        if val > 0.3:
                            ax.text(t, i, f'{val:.1f}', ha='center', va='center',
                                    fontsize=6 if N_v > 8 else 7,
                                    color='black', fontweight='bold')
                plt.tight_layout()
                show_fig_with_downloads(fig, "ev_charging_figure")
                plt.close()
            else:
                st.warning("MILP 不可行，无法生成热力图。")

        # ── Tab3：SOC 演化 ───────────────────────────────────────────
        with tab3:
            N_v = len(vehicles)
            if N_v <= 4:
                rep_ids = list(range(N_v))
            else:
                rep_ids = [0, N_v // 3, 2 * N_v // 3, N_v - 1]
                rep_ids = sorted(set(rep_ids))

            n_rep = len(rep_ids)
            fig, axes = plt.subplots(1, n_rep, figsize=(5 * n_rep, 5), sharey=True)
            if n_rep == 1:
                axes = [axes]
            x_soc = np.arange(n_slots + 1)
            slabels_soc = slabels + [f'{end_hour:02d}:00']

            for ax_idx, ev_idx in enumerate(rep_ids):
                ax = axes[ax_idx]
                for name, color, ls in zip(STRATEGY_NAMES, STRATEGY_COLORS, STRATEGY_LS):
                    r = all_results[name]
                    if r['soc_schedule'] is not None:
                        ax.plot(x_soc, r['soc_schedule'][ev_idx] * 100,
                                color=color, linewidth=2, linestyle=ls, label=name)

                target_pct = vehicles[ev_idx]['目标SOC'] * 100
                ax.axhline(y=target_pct, color='#2C3E50', linestyle='--',
                           linewidth=1.5, label=f'目标SOC={target_pct:.0f}%')
                ax.set_xticks(x_soc[::3])
                ax.set_xticklabels(slabels_soc[::3], rotation=45, fontsize=8)
                ax.set_xlabel('时段')
                if ax_idx == 0:
                    ax.set_ylabel('SOC (%)')
                v = vehicles[ev_idx]
                # 不在导出的图片中添加子图标题
                ax.set_ylim(0, 108)
                if ax_idx == n_rep - 1:
                    ax.legend(fontsize=7, loc='lower right', frameon=True)

            # 不在导出的图片中添加总标题
            plt.tight_layout()
            show_fig_with_downloads(fig, "ev_charging_figure")
            plt.close()

        # ── Tab4：成本构成堆叠柱图 ──────────────────────────────────
        with tab4:
            fig, ax = plt.subplots(figsize=(11, 5.5))
            names_plot = []
            t_parts, c_parts, d_parts = [], [], []
            T_ref_v, C_ref_v, D_ref_v = T_ref, C_ref, D_ref

            # ★ 与主代码一致：规则策略即使部分达标，也显示其实际成本构成
            for name in STRATEGY_NAMES:
                r = all_results[name]
                if r.get('obj') is None:
                    continue
                names_plot.append(name)
                t_parts.append(w_T * r['T_total'] / T_ref_v)
                c_parts.append(w_C * r['C_total'] / C_ref_v)
                d_parts.append(w_D * r['D_total'] / D_ref_v)

            if names_plot:
                x_bar = np.arange(len(names_plot))
                ax.bar(x_bar, t_parts, label=f'时长项 (w={w_T})',
                       color='#2980B9', alpha=0.9, edgecolor='white')
                ax.bar(x_bar, c_parts, bottom=t_parts,
                       label=f'电费项 (w={w_C})', color='#F39C12', alpha=0.9,
                       edgecolor='white')
                d_bottoms = [a + b for a, b in zip(t_parts, c_parts)]
                ax.bar(x_bar, d_parts, bottom=d_bottoms,
                       label=f'退化项 (w={w_D})', color='#27AE60', alpha=0.9,
                       edgecolor='white')

                for xi, (tp, cp, dp) in enumerate(zip(t_parts, c_parts, d_parts)):
                    total = tp + cp + dp
                    ax.text(xi, tp / 2, f'{tp:.4f}', ha='center', va='center',
                            fontsize=8.5, color='white', fontweight='bold')
                    ax.text(xi, tp + cp / 2, f'{cp:.4f}', ha='center', va='center',
                            fontsize=8.5, color='white', fontweight='bold')
                    ax.text(xi, tp + cp + dp / 2, f'{dp:.4f}', ha='center', va='center',
                            fontsize=8.5, color='white', fontweight='bold')
                    ax.text(xi, total + 0.004, f'合计 {total:.4f}', ha='center',
                            va='bottom', fontsize=9, fontweight='bold', color='#2C3E50')

                ax.set_xticks(x_bar)
                ax.set_xticklabels(names_plot, fontsize=9)
                ax.set_ylabel('归一化加权目标值')
                # 不在导出的图片中添加标题
                ax.legend(frameon=True)
                ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
                plt.tight_layout()
                show_fig_with_downloads(fig, "ev_charging_figure")
                plt.close()
            else:
                st.warning("没有可行策略，无法绘制成本构成图。")

        # ── Tab5：详细功率数据 & 下载 ────────────────────────────────
        with tab5:
            if milp_r['feasible'] and milp_r['P_schedule'] is not None:
                P_sched_milp = milp_r['P_schedule']
                veh_ids = [v['编号'] for v in vehicles]
                power_df = pd.DataFrame(
                    np.round(P_sched_milp, 2),
                    index=veh_ids,
                    columns=slabels
                )
                st.write("**MILP 各车辆各时段充电功率 (kW)：**")
                st.dataframe(power_df, use_container_width=True)

                soc_df = pd.DataFrame(
                    np.round(milp_r['soc_schedule'] * 100, 2),
                    index=veh_ids,
                    columns=slabels + [f'{end_hour:02d}:00']
                )
                st.write("**MILP 各车辆各时段 SOC (%)：**")
                st.dataframe(soc_df, use_container_width=True)

                st.divider()
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    csv_power = power_df.to_csv(encoding='utf-8-sig')
                    st.download_button(
                        "📥 下载充电功率方案 (CSV)",
                        csv_power.encode('utf-8-sig'),
                        "MILP_charging_power_schedule.csv",
                        "text/csv",
                        use_container_width=True
                    )
                with col_dl2:
                    csv_soc = soc_df.to_csv(encoding='utf-8-sig')
                    st.download_button(
                        "📥 下载SOC演化数据 (CSV)",
                        csv_soc.encode('utf-8-sig'),
                        "MILP_soc_evolution.csv",
                        "text/csv",
                        use_container_width=True
                    )

                st.divider()
                st.write("**各车辆充电汇总：**")
                summary_rows = []
                for i, v in enumerate(vehicles):
                    e_charged = sum(P_sched_milp[i][t] * dt * eta for t in range(n_slots))
                    t_active = sum(dt for t in range(n_slots) if P_sched_milp[i][t] > 0.01)
                    avg_p = np.mean([P_sched_milp[i][t] for t in range(n_slots)
                                     if P_sched_milp[i][t] > 0.01]) if t_active > 0 else 0
                    final_soc = milp_r['soc_schedule'][i][-1]
                    summary_rows.append({
                        '车辆': v['编号'],
                        '电池容量(kWh)': v['容量'],
                        '初始SOC': f"{v['初始SOC']:.2f}",
                        '最终SOC': f"{final_soc:.4f}",
                        '目标SOC': f"{v['目标SOC']:.2f}",
                        '是否达标': '✅' if _target_met(final_soc, v['目标SOC']) else '❌',
                        '实际充入(kWh)': f"{e_charged:.2f}",
                        '充电时段数': f"{int(t_active / dt)}",
                        '平均功率(kW)': f"{avg_p:.2f}",
                    })
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True,
                             hide_index=True)

            else:
                st.warning("MILP 不可行，无详细数据可展示。")

        # ── 底部信息 ──────────────────────────────────────────────────
        st.divider()
        with st.expander("ℹ️ 模型说明"):
            st.markdown("""
            **模型概述**

            本系统基于混合整数线性规划（MILP）模型，在居民夜间固定时间窗内，
            为多辆电动汽车协调分配各时段的充电功率。

            **综合目标函数（论文式3.4）**

            $$Z = w_1 \\frac{T_{total}}{T_{ref}} + w_2 \\frac{C_{total}}{C_{ref}} + w_3 \\frac{D_{total}}{D_{ref}}$$

            其中 $T_{total}$ 为总充电时长（式3.6），$C_{total}$ 为总电费成本（式3.11），$D_{total}$ 为总退化惩罚（式2.15分段线性近似）。

            **归一化参考值（论文式3.8–3.10）**
            - $T_{ref} = N \\times T \\times \\Delta t$
            - $C_{ref} = (\\sum \\text{需求电量} / \\eta) \\times \\pi_{peak}$
            - $D_{ref} = N \\times T \\times \\kappa \\times \\Delta t$

            **约束条件**
            - 单车充电功率约束（式3.21）：$\\varepsilon \\cdot y_{i,t} \\leq P_{i,t} \\leq P_{max,i} \\cdot y_{i,t}$
            - SOC 演化约束（式3.18）：$SOC_{i,t+1} = SOC_{i,t} + \\eta \\cdot P_{i,t} \\cdot \\Delta t / C_i$
            - 初始SOC约束（式3.19）：$SOC_{i,0} = SOC_{init,i}$
            - 目标 SOC 约束（式3.20）：$SOC_{i,T} \\geq SOC_{target,i}$
            - 聚合功率约束（式3.17）：$\\sum_i P_{i,t} \\leq P_{agg,max}$

            **电池退化模型（论文§2.2）**
            - 退化惩罚采用分段线性化近似（式2.10–2.15），段数 S 可调
            - 退化系数 $\\kappa$ 反映充电功率对电池寿命的影响

            **对比策略（论文§3.3.3）**
            1. **全时段均匀充电**：将充电需求均匀分摊到所有时段
            2. **仅谷段充电**：仅在最低电价时段充电，按编号顺序贪心分配
            3. **最大功率优先充电**：从第一个时段开始以最大功率充电，按编号顺序分配
            """)

        with st.expander("ℹ️ 当前运行参数汇总"):
            param_col1, param_col2, param_col3 = st.columns(3)
            with param_col1:
                st.write(f"- 车辆数量：{len(vehicles)}")
                st.write(f"- 时间窗：22:00—{end_hour:02d}:00（{n_hours}h）")
                st.write(f"- 时段数：{n_slots}")
                st.write(f"- 时段长度：{dt}h")
            with param_col2:
                st.write(f"- 聚合功率上限：{P_agg_max} kW")
                st.write(f"- 峰段电价：{price_peak} 元/kWh")
                st.write(f"- 谷段电价：{price_valley} 元/kWh")
                st.write(f"- 充电效率：{eta}")
            with param_col3:
                st.write(f"- 退化系数 κ：{kappa}")
                st.write(f"- 权重 (w₁,w₂,w₃)：({w_T},{w_C},{w_D})")
                st.write(f"- 分段数 S：{S_seg}")
                st.write(f"- 归一化参考值：T={T_ref:.1f}, C={C_ref:.2f}, D={D_ref:.6f}")


# =============================================================================
# 9. 主入口：模式选择
# =============================================================================
def main():
    st.set_page_config(
        page_title="居民夜间多车聚合充电调度系统",
        page_icon="🔋",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    show_font_status_in_sidebar()

    mode = st.sidebar.radio(
        "🔀 选择模式",
        ["🏠 个人充电规划", "🏢 多车聚合调度（论文演示）"],
        index=1
    )

    if mode == "🏠 个人充电规划":
        personal_mode()
    else:
        multi_vehicle_mode()


if __name__ == '__main__':
    main()
