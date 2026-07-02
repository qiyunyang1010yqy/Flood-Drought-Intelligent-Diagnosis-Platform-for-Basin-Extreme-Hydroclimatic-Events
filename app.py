# -*- coding: utf-8 -*-
"""
洪旱智诊·流域极端水文事件融合分析平台

Flood-Drought Intelligent Diagnosis Platform for Basin Extreme Hydroclimatic Events

本脚本仅负责 Streamlit 前端展示与交互组织：
- 不下载数据
- 不重新识别事件
- 不重新训练模型
- 不重新计算生态响应

运行：
streamlit run streamlit_flood_drought_intelligent_diagnosis.py
"""

from __future__ import annotations

from pathlib import Path
import json
import html

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False
    px = None
    go = None

import matplotlib.pyplot as plt


PLATFORM_CN = "洪旱智诊·流域极端水文事件融合分析平台"
PLATFORM_EN = "Flood-Drought Intelligent Diagnosis Platform for Basin Extreme Hydroclimatic Events"

SCRIPT_DIR = Path(__file__).resolve().parent
# Streamlit Cloud 部署时，app.py 位于仓库根目录；07/08/09 阶段结果表也放在仓库根目录下。
VIS_DIR = SCRIPT_DIR / "10_visualization_platform"
BASE_DIR = SCRIPT_DIR
DIR_07 = BASE_DIR / "07_event_identification"
DIR_08 = BASE_DIR / "08_driver_analysis"
DIR_09 = BASE_DIR / "09_ecological_response"
REPORT_DIR = VIS_DIR / "report"
BOUNDARY_GEOJSON = (
    BASE_DIR
    / "01_boundary"
    / "derived_four_basins"
    / "four_yangtze_representative_basins_hybas_lev07_simplified_for_gee.geojson"
)

PAGES = [
    "首页",
    "研究配置与数据输入",
    "事件识别结果",
    "年际变化",
    "驱动因子分析",
    "生态响应分析",
    "结论总结与发展建议",
]

AVAILABLE_BASINS = {
    "长江流域": ["嘉陵江流域", "汉江流域", "洞庭湖流域", "鄱阳湖流域"],
}

BASIN_LABELS = {
    "嘉陵江流域": "Jialing River Basin",
    "汉江流域": "Hanjiang River Basin",
    "洞庭湖流域": "Dongting Lake Basin",
    "鄱阳湖流域": "Poyang Lake Basin",
}

BASIN_TYPES = {
    "嘉陵江流域": "上游山地支流型，代表地形起伏与快速水分转换情景。",
    "汉江流域": "调控型支流流域，代表支流水文过程与人类调度影响情景。",
    "洞庭湖流域": "湖泊—河网调蓄型，代表湖泊调蓄与湿涝转换情景。",
    "鄱阳湖流域": "季节性湖泊水位波动型，代表湖泊水位涨落与生态响应情景。",
}

BASIN_POINTS = {
    "长江流域": {"lon": 112.0, "lat": 30.0, "desc": "长江流域示范中心"},
    "嘉陵江流域": {"lon": 106.55, "lat": 29.70, "desc": BASIN_TYPES["嘉陵江流域"]},
    "汉江流域": {"lon": 114.05, "lat": 30.62, "desc": BASIN_TYPES["汉江流域"]},
    "洞庭湖流域": {"lon": 112.80, "lat": 28.90, "desc": BASIN_TYPES["洞庭湖流域"]},
    "鄱阳湖流域": {"lon": 116.20, "lat": 29.20, "desc": BASIN_TYPES["鄱阳湖流域"]},
}

DATA_CATALOG = {
    "维度 1：气象驱动": [
        ("逐日/月降水量（含极端降水指数 Rx1day/Rx5day）", "极端事件识别、SPI/SPEI 计算、DTF 触发判定", "必填"),
        ("逐日/月平均气温", "PET 估算、蒸散耗热背景", "必填"),
        ("逐日/月潜在蒸散发 PET", "水分盈亏计算，识别气象干旱向土壤干旱传播", "必填"),
    ],
    "维度 2：陆面水文状态": [
        ("逐日/月土壤湿度，多层如 0-10cm、10-40cm", "SSMI 计算、前期干旱背景判定、骤旱识别", "必填"),
        ("逐日/月径流或河川流量", "水文干旱指数 RSI、干旱传播滞后分析", "必填"),
        ("湖泊/水库水位与水体面积", "湖泊调蓄效应评估、FTD 的湖库缓冲判定", "选填"),
    ],
    "维度 3：下垫面与地形": [
        ("DEM 数字高程，分辨率≥1km", "子流域划分、坡度提取、地形对降水再分配影响", "必填"),
        ("河网水系，流向/流量累积", "关键传播路径识别、上游来水对下游急转的贡献", "必填"),
        ("土壤属性，砂粒、黏粒、有机碳、容重", "下渗能力、有效含水率计算、物理约束模型输入", "必填"),
    ],
    "维度 4：生态与植被": [
        ("月 NDVI / FVC / NPP，至少其一", "滞后 1-12 月生态响应分析、植被脆弱性分区", "必填"),
        ("SIF / MNDWI", "区分光合抑制型和枯黄型干旱生态响应", "选填"),
    ],
    "维度 5：土地利用": [
        ("土地利用/覆被，CLCD/ESA", "区分耕地、林地、湿地响应差异，产流系数分区", "必填"),
    ],
    "维度 6：社会经济，风险暴露": [
        ("人口密度格网", "灾害暴露度评估、综合风险制图", "选填"),
        ("GDP 密度格网", "经济损失潜在评估", "选填"),
    ],
    "维度 7：人类活动调控": [
        ("水库/大坝位置与库容", "量化人类调度对天然径流的干扰系数，输入 SHAP 作为调节变量", "选填"),
    ],
    "维度 8：未来预估": [
        ("CMIP6 降尺度数据，SSP245/SSP585 情景", "未来 DFAA 概率推演、气候变化适应性策略生成", "选填"),
    ],
}

UPLOAD_GROUPS = [
    ("研究区边界文件上传", ["geojson", "zip", "gpkg"]),
    ("气象驱动数据上传", ["csv", "xlsx", "tif", "nc", "zip"]),
    ("陆面水文状态数据上传", ["csv", "xlsx", "tif", "nc", "zip"]),
    ("下垫面与地形数据上传", ["csv", "xlsx", "tif", "nc", "geojson", "zip", "gpkg"]),
    ("生态与植被数据上传", ["csv", "xlsx", "tif", "nc", "zip"]),
    ("土地利用数据上传", ["csv", "xlsx", "tif", "nc", "zip"]),
    ("社会经济数据上传", ["csv", "xlsx", "tif", "nc", "zip"]),
    ("人类活动调控数据上传", ["csv", "xlsx", "geojson", "zip", "gpkg"]),
    ("未来情景数据上传", ["csv", "xlsx", "tif", "nc", "zip"]),
]

DEVELOPMENT_SUGGESTIONS = {
    "水文学": "加强降水—土壤湿度—径流—湖泊水位全过程监测，建立日尺度洪旱事件链识别模型，重点揭示气象干旱向水文干旱传播、湿涝向干旱转化的滞后机制。",
    "气象学与气候变化": "加强季风异常、副热带高压、极端降水过程和高温蒸散背景对洪旱急转的影响研究，结合 CMIP6 情景数据开展未来洪旱复合事件概率推演。",
    "地理信息科学与遥感": "构建多源遥感数据融合监测体系，集成 CHIRPS、ERA5-Land、MODIS、SMAP、Sentinel、Landsat 等数据，实现流域干湿状态、植被响应、水体面积变化和土地利用变化的快速监测。",
    "生态学": "关注洪旱急转对湿地、森林、农田和湖泊生态系统的差异化影响，利用 NDVI、FVC、NPP、SIF 等指标识别生态系统滞后响应、恢复能力与脆弱性分区。",
    "农业科学": "围绕关键作物生育期建立农业旱涝急转风险指标，结合土壤湿度、灌溉条件和作物物候，优化抗旱排涝协同管理和农业生产适应策略。",
    "水利工程与湖库调度": "将水库库容、调度规则、堤防系统、河湖连通性和湖泊水位变化纳入模型，评估水利工程对天然洪旱过程的削弱、放大或延迟效应，优化联合调度方案。",
    "城市规划与应急管理": "针对下游湖区和城市群，建立洪涝/湿涝异常与人口、GDP、基础设施暴露度的叠加评估机制，完善城市排涝、应急避险和灾害链防控方案。",
    "数据科学与人工智能": "引入 XGBoost、Random Forest、SHAP、时序深度学习和因果推断方法，构建可解释的洪旱急转智能诊断模型，提升事件识别、风险预测和驱动归因能力。",
    "公共管理与政策治理": "推动跨部门数据共享和流域协同治理，建立气象、水利、生态环境、农业、自然资源和应急管理部门之间的联合预警与响应机制。",
}


st.set_page_config(page_title=PLATFORM_CN, layout="wide", initial_sidebar_state="expanded")


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --deep-blue: #0b2f5b;
            --river-blue: #0f6fae;
            --cyan: #12a6b4;
            --eco-green: #1d9a6c;
            --drought-orange: #e6972f;
            --bg: #f5f8fb;
            --text: #22313f;
            --muted: #66788a;
            --border: #d9e3ec;
        }
        .stApp {
            background: linear-gradient(180deg, #f5f8fb 0%, #ffffff 55%);
            color: var(--text);
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1380px;
        }
        .hero {
            border-radius: 18px;
            padding: 28px 30px;
            color: #fff;
            background: linear-gradient(135deg, #082548 0%, #0f6fae 58%, #12a6b4 100%);
            box-shadow: 0 14px 34px rgba(5, 38, 76, 0.22);
            margin-bottom: 18px;
        }
        .hero-title {
            font-size: clamp(1.7rem, 3.2vw, 3rem);
            font-weight: 800;
            letter-spacing: 0;
            line-height: 1.16;
            margin-bottom: 8px;
        }
        .hero-subtitle {
            font-size: 1.05rem;
            opacity: 0.92;
            line-height: 1.55;
            max-width: 1040px;
        }
        .mode-card, .metric-card, .map-card, .config-card, .data-card,
        .basin-info-card, .conclusion-card, .recommendation-card,
        .disabled-page-card, .warning-card, .success-card {
            border-radius: 14px;
            padding: 18px 20px;
            background: #fff;
            box-shadow: 0 8px 24px rgba(15, 49, 89, 0.09);
            border: 1px solid rgba(217, 227, 236, 0.95);
            margin-bottom: 14px;
        }
        .mode-card {
            min-height: 150px;
            border-top: 4px solid var(--river-blue);
        }
        .metric-card {
            border-left: 5px solid var(--cyan);
            min-height: 116px;
        }
        .metric-card .metric-value {
            font-size: 1.85rem;
            font-weight: 800;
            color: var(--deep-blue);
        }
        .metric-card .metric-label {
            color: var(--muted);
            font-size: 0.92rem;
            margin-top: 4px;
        }
        .map-card {
            padding: 16px;
            overflow: hidden;
        }
        .config-card {
            border-left: 5px solid var(--deep-blue);
        }
        .data-card {
            background: #fbfdff;
            margin-bottom: 10px;
        }
        .section-title {
            font-size: 1.35rem;
            font-weight: 800;
            color: var(--deep-blue);
            margin: 10px 0 14px;
        }
        .small-muted {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.55;
        }
        .required-tag, .optional-tag {
            display: inline-block;
            border-radius: 999px;
            padding: 3px 10px;
            font-size: 0.78rem;
            font-weight: 700;
            margin-top: 8px;
        }
        .required-tag {
            background: #fff2df;
            color: #a86100;
            border: 1px solid #ffd39b;
        }
        .optional-tag {
            background: #eaf6f2;
            color: #1d7e5d;
            border: 1px solid #c7e7dd;
        }
        .warning-card {
            background: #fff8eb;
            border-left: 5px solid var(--drought-orange);
        }
        .success-card {
            background: #eefbf6;
            border-left: 5px solid var(--eco-green);
        }
        .disabled-page-card {
            background: #f8fafc;
            text-align: center;
            padding: 34px 28px;
        }
        .conclusion-card {
            border-left: 5px solid var(--river-blue);
        }
        .recommendation-card {
            border-left: 5px solid var(--eco-green);
            background: #fbfffd;
        }
        .basin-info-card {
            border-left: 5px solid var(--cyan);
        }
        .step-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 28px;
            height: 28px;
            border-radius: 50%;
            background: var(--deep-blue);
            color: #fff;
            font-weight: 800;
            margin-right: 8px;
        }
        .flow-card {
            border-radius: 12px;
            background: #ffffff;
            border: 1px solid var(--border);
            padding: 14px 16px;
            text-align: center;
            min-height: 76px;
            box-shadow: 0 6px 18px rgba(15, 49, 89, 0.06);
        }
        .stButton > button {
            border-radius: 10px;
            font-weight: 700;
            border: 1px solid #c9d8e5;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #0b2f5b, #0f6fae);
            border: 0;
        }
        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b2f5b 0%, #113f73 100%);
        }
        div[data-testid="stSidebar"] * {
            color: #ffffff;
        }
        div[data-testid="stSidebar"] .stRadio label {
            color: #ffffff !important;
        }
        @media (max-width: 760px) {
            .hero { padding: 20px 18px; }
            .mode-card, .metric-card, .map-card, .config-card, .data-card {
                padding: 14px 15px;
            }
            .hero-title { font-size: 1.75rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    defaults = {
        "current_page": "首页",
        "nav_page": "首页",
        "pending_nav_page": None,
        "analysis_ready": False,
        "use_library_data": False,
        "analysis_mode": None,
        "selected_basin": "长江流域",
        "select_subbasin": False,
        "selected_subbasin": "全部典型子流域",
        "year_start": 2000,
        "year_end": 2020,
        "uploader_reset_token": 0,
        "custom_boundary_uploaded": False,
        "custom_data_uploaded": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def go_to_page(page: str) -> None:
    """统一切换页面，避免侧边栏需要点击两次。"""
    st.session_state.current_page = page
    st.session_state.pending_nav_page = page


def clear_library_data_selection() -> None:
    """清除“使用本库已有数据”的选择，并关闭结果页访问。"""
    st.session_state.use_library_data = False
    st.session_state.analysis_ready = False
    st.session_state.custom_boundary_uploaded = False
    st.session_state.custom_data_uploaded = False
    st.session_state.uploader_reset_token = st.session_state.get("uploader_reset_token", 0) + 1


def clear_current_analysis() -> None:
    """清除当前区域分析任务，让平台回到可重新配置状态。"""
    st.session_state.analysis_ready = False
    st.session_state.selected_basin = "长江流域"
    st.session_state.select_subbasin = False
    st.session_state.selected_subbasin = "全部典型子流域"
    st.session_state.year_start = 2000
    st.session_state.year_end = 2020
    go_to_page("研究配置与数据输入")


def html_card(class_name: str, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="{class_name}">
            <div style="font-weight:800;color:#0b2f5b;font-size:1.05rem;margin-bottom:8px;">{html.escape(title)}</div>
            <div style="color:#33485d;line-height:1.6;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def read_table(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def read_geojson(path_text: str) -> dict | None:
    path = Path(path_text)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_data() -> dict[str, pd.DataFrame]:
    return {
        "monthly": read_table(str(DIR_07 / "monthly_event_flags.csv")),
        "event_table": read_table(str(DIR_07 / "event_table.csv")),
        "event_summary": read_table(str(DIR_07 / "basin_event_summary.csv")),
        "yearly": read_table(str(DIR_07 / "basin_year_event_frequency.csv")),
        "threshold": read_table(str(DIR_07 / "threshold_sensitivity.csv")),
        "driver_summary": read_table(str(DIR_08 / "basin_driver_summary.csv")),
        "lag_corr": read_table(str(DIR_08 / "lag_correlation_analysis.csv")),
        "model_metrics": read_table(str(DIR_08 / "transition_model_metrics.csv")),
        "feature_importance": read_table(str(DIR_08 / "transition_feature_importance.csv")),
        "ndvi_window": read_table(str(DIR_09 / "ndvi_event_window_composites.csv")),
        "ndvi_response": read_table(str(DIR_09 / "basin_ndvi_response_summary.csv")),
        "ndvi_lag": read_table(str(DIR_09 / "ndvi_response_lag_summary.csv")),
        "ndvi_sig": read_table(str(DIR_09 / "ndvi_response_significance_tests.csv")),
        "eco_summary": read_table(str(DIR_09 / "ecological_response_summary.csv")),
    }


def scan_library_files() -> list[str]:
    files: list[str] = []
    for folder in [DIR_07, DIR_08, DIR_09]:
        if folder.exists():
            files.extend([f"{folder.name}/{p.name}" for p in sorted(folder.glob("*.csv"))])
    if files:
        return files
    return [
        "07_event_identification/monthly_event_flags.csv",
        "07_event_identification/event_table.csv",
        "07_event_identification/basin_year_event_frequency.csv",
        "07_event_identification/basin_event_summary.csv",
        "07_event_identification/threshold_sensitivity.csv",
        "08_driver_analysis/driver_feature_panel.csv",
        "08_driver_analysis/lag_correlation_analysis.csv",
        "08_driver_analysis/transition_model_metrics.csv",
        "08_driver_analysis/transition_feature_importance.csv",
        "08_driver_analysis/basin_driver_summary.csv",
        "09_ecological_response/ecological_feature_panel.csv",
        "09_ecological_response/ndvi_event_window_composites.csv",
        "09_ecological_response/basin_ndvi_response_summary.csv",
        "09_ecological_response/ecological_response_summary.csv",
    ]


def active_basin_names() -> list[str]:
    if st.session_state.get("selected_basin") != "长江流域":
        return list(AVAILABLE_BASINS["长江流域"])
    if not st.session_state.get("select_subbasin"):
        return list(AVAILABLE_BASINS["长江流域"])
    selected = st.session_state.get("selected_subbasin", "全部典型子流域")
    if selected == "全部典型子流域":
        return list(AVAILABLE_BASINS["长江流域"])
    return [selected]


def filter_active_basin(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "basin_name" not in df.columns:
        return df
    return df[df["basin_name"].isin(active_basin_names())].copy()


def render_metric_card(label: str, value: str | int | float, hint: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{html.escape(str(value))}</div>
            <div class="metric-label">{html.escape(label)}</div>
            <div class="small-muted">{html.escape(hint)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_map(selected_area: str | None = None):
    selected_area = selected_area or st.session_state.get("selected_subbasin") or "长江流域"
    selected_areas = active_basin_names() if selected_area in ["长江流域", "全部典型子流域"] else [selected_area]

    points = []
    for name, item in BASIN_POINTS.items():
        if name == "长江流域":
            continue
        is_selected = name in selected_areas
        points.append(
            {
                "name": name,
                "label": BASIN_LABELS.get(name, name),
                "lon": item["lon"],
                "lat": item["lat"],
                "desc": item["desc"],
                "status": "Selected" if is_selected else "Reference",
                "size": 18 if is_selected else 11,
            }
        )
    point_df = pd.DataFrame(points)

    if not HAS_PLOTLY:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.scatter(point_df["lon"], point_df["lat"], s=point_df["size"] * 12)
        for _, row in point_df.iterrows():
            ax.text(row["lon"], row["lat"], row["label"], fontsize=8)
        ax.set_title("Yangtze demonstration map")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        return fig

    geojson = read_geojson(str(BOUNDARY_GEOJSON))
    if geojson is not None:
        locations = [f["properties"].get("basin_cn") for f in geojson.get("features", [])]
        z_values = [1 if loc in selected_areas else 0.25 for loc in locations]
        colorscale = [[0, "#c8d7e6"], [0.5, "#58b7c7"], [1, "#e6972f"]]
        fig = go.Figure()
        fig.add_trace(
            go.Choroplethmapbox(
                geojson=geojson,
                locations=locations,
                z=z_values,
                featureidkey="properties.basin_cn",
                colorscale=colorscale,
                showscale=False,
                marker_opacity=0.42,
                marker_line_width=1.4,
                marker_line_color="#0b2f5b",
                text=locations,
                hovertemplate="%{location}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scattermapbox(
                lon=point_df["lon"],
                lat=point_df["lat"],
                text=point_df["name"],
                customdata=point_df[["desc", "status"]],
                mode="markers+text",
                textposition="top center",
                marker={
                    "size": point_df["size"],
                    "color": ["#e6972f" if s == "Selected" else "#0f6fae" for s in point_df["status"]],
                    "opacity": 0.88,
                },
                hovertemplate="<b>%{text}</b><br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>",
            )
        )
        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox_center={"lat": 30.0, "lon": 111.5},
            mapbox_zoom=4.1,
            height=470,
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
        )
        return fig

    fig = go.Figure(
        go.Scattergeo(
            lon=point_df["lon"],
            lat=point_df["lat"],
            text=point_df["name"],
            customdata=point_df[["desc", "status"]],
            mode="markers+text",
            textposition="top center",
            marker={
                "size": point_df["size"],
                "color": ["#e6972f" if s == "Selected" else "#0f6fae" for s in point_df["status"]],
            },
            hovertemplate="<b>%{text}</b><br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>",
        )
    )
    fig.update_geos(
        scope="asia",
        center={"lat": 30.0, "lon": 111.5},
        projection_scale=4.1,
        showland=True,
        landcolor="#eef3f7",
        showcountries=True,
        countrycolor="#a7b7c6",
        showcoastlines=True,
        coastlinecolor="#a7b7c6",
    )
    fig.update_layout(height=470, margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig


def render_interactive_map(selected_area: str | None = None) -> None:
    fig = build_map(selected_area)
    if HAS_PLOTLY:
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True})
    else:
        st.pyplot(fig)
    if BOUNDARY_GEOJSON.exists():
        st.caption("已检测并加载项目内四个典型子流域 GeoJSON 边界；地图支持平移、放大和缩小。")
    else:
        st.caption("当前地图为示范定位，正式部署可接入权威矢量边界；本页不伪造精确边界。")


def plot_bar(df: pd.DataFrame, x: str, y: list[str], title: str):
    if HAS_PLOTLY:
        plot_df = df.copy()
        fig = px.bar(plot_df, x=x, y=y, barmode="group", title=title, height=440)
        fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(df.set_index(x)[y])


def plot_line(df: pd.DataFrame, x: str, y: list[str], color: str | None, title: str):
    if HAS_PLOTLY:
        if color:
            fig = px.line(df, x=x, y=y, color=color, markers=True, title=title, height=440)
        else:
            fig = px.line(df, x=x, y=y, markers=True, title=title, height=440)
        fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(df.set_index(x)[y])


def page_home() -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-title">{PLATFORM_CN}</div>
            <div class="hero-subtitle">{PLATFORM_EN}</div>
            <div class="hero-subtitle" style="margin-top:12px;">
            本平台面向流域极端洪旱、旱涝急转及其生态响应过程，构建数据接入、事件识别、驱动诊断、生态响应、风险研判与发展建议一体化分析框架。当前版本已接入长江流域及四个典型子流域示范数据，用于展示平台核心功能流程。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">流域空间认知与示范定位</div>', unsafe_allow_html=True)
    st.markdown('<div class="map-card">', unsafe_allow_html=True)
    render_interactive_map("全部典型子流域")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">请选择分析模式</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        html_card(
            "mode-card",
            "典型流域分析",
            "使用平台内置的典型流域示范数据，快速开展极端洪旱、旱涝急转、驱动因子和生态响应分析。",
        )
        if st.button("进入典型流域分析", use_container_width=True, type="primary"):
            st.session_state.analysis_mode = "典型流域分析"
            go_to_page("研究配置与数据输入")
            st.rerun()
    with right:
        html_card(
            "mode-card",
            "自定义研究区模式",
            "用户可上传自定义研究区边界与多源水文气象数据，构建面向特定流域的个性化分析流程。当前版本为接口预留模式，完整计算服务将在后续版本开放。",
        )
        if st.button("进入自定义研究区模式", use_container_width=True):
            st.session_state.analysis_mode = "自定义研究区模式"
            go_to_page("研究配置与数据输入")
            st.rerun()

    st.markdown(
        """
        <div class="warning-card">
        当前版本为典型流域示范版，采用月尺度流域面平均数据开展原型分析；洪涝结果更准确表述为洪涝/湿涝异常；NDVI 响应代表流域平均植被状态变化，不能直接等同于作物减产或灾损。
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_basin_selector() -> None:
    st.markdown('<div class="section-title"><span class="step-pill">A</span>研究区选择</div>', unsafe_allow_html=True)
    mode = st.session_state.get("analysis_mode") or "典型流域分析"

    if mode == "典型流域分析":
        c1, c2 = st.columns([1.1, 0.9])
        with c1:
            selected_basin = st.selectbox("研究流域", list(AVAILABLE_BASINS.keys()), index=0)
            st.session_state.selected_basin = selected_basin
            st.session_state.select_subbasin = st.checkbox("是否选择其子流域", value=st.session_state.select_subbasin)
            if st.session_state.select_subbasin:
                options = ["全部典型子流域"] + AVAILABLE_BASINS[selected_basin]
                st.session_state.selected_subbasin = st.selectbox("请选择子流域", options)
            else:
                st.session_state.selected_subbasin = "全部典型子流域"

            y1, y2 = st.columns(2)
            with y1:
                st.session_state.year_start = st.selectbox("开始年份", [2000], index=0)
            with y2:
                st.session_state.year_end = st.selectbox("结束年份", [2020], index=0)
            st.info("当前示范数据库可用时间范围：2000—2020 年。当前结果文件的实际可用年份以平台结果数据表为准。")

        with c2:
            selected = st.session_state.selected_subbasin if st.session_state.select_subbasin else "长江流域"
            if selected == "全部典型子流域":
                text = "当前选择全部典型子流域，用于对比不同流域类型下的洪旱急转与生态响应差异。"
            elif selected in BASIN_TYPES:
                text = BASIN_TYPES[selected]
            else:
                text = "当前分析对象为长江流域，平台将展示四个典型子流域的示范结果。"
            html_card("basin-info-card", selected, text)
            render_interactive_map(selected)

    else:
        html_card(
            "warning-card",
            "自定义研究区模式",
            "当前为接口预留版本。用户可上传研究区边界与多源水文气象数据，查看数据需求清单；自动化计算与结果生成将在后续版本开放。",
        )
        token = st.session_state.get("uploader_reset_token", 0)
        boundary_file = st.file_uploader(
            "上传研究区边界，支持 geojson、zip shp、gpkg",
            type=["geojson", "zip", "gpkg"],
            key=f"custom_boundary_{token}",
        )
        data_files = st.file_uploader(
            "上传多源输入数据，支持 csv、xlsx、tif、nc、zip",
            type=["csv", "xlsx", "tif", "nc", "zip"],
            accept_multiple_files=True,
            key=f"custom_data_{token}",
        )
        st.session_state.custom_boundary_uploaded = boundary_file is not None
        st.session_state.custom_data_uploaded = bool(data_files)


def render_data_catalog() -> None:
    st.markdown('<div class="section-title"><span class="step-pill">B</span>数据输入</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="config-card">
        平台支持多源水文气象、陆面状态、生态遥感、土地利用和社会经济数据接入，用于构建极端洪旱与旱涝急转事件识别、机制诊断和风险研判模型。
        </div>
        """,
        unsafe_allow_html=True,
    )

    for dimension, items in DATA_CATALOG.items():
        with st.expander(dimension, expanded=False):
            cols = st.columns(2)
            for idx, (name, purpose, need) in enumerate(items):
                tag_class = "required-tag" if need == "必填" else "optional-tag"
                with cols[idx % 2]:
                    st.markdown(
                        f"""
                        <div class="data-card">
                            <div style="font-weight:800;color:#0b2f5b;">{html.escape(name)}</div>
                            <div class="small-muted" style="margin-top:6px;">对应算法模块：{html.escape(purpose)}</div>
                            <span class="{tag_class}">{html.escape(need)}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


def render_upload_area() -> None:
    st.markdown('<div class="section-title">上传数据或调用平台示范数据</div>', unsafe_allow_html=True)

    left, right = st.columns([1.2, 1])
    with left:
        with st.expander("上传区", expanded=False):
            token = st.session_state.get("uploader_reset_token", 0)
            for label, extensions in UPLOAD_GROUPS:
                st.file_uploader(label, type=extensions, accept_multiple_files=True, key=f"upload_{label}_{token}")

    with right:
        html_card(
            "config-card",
            "平台内置示范数据",
            "典型流域分析模式可以直接调用本库已有结果数据，快速查看事件识别、年际变化、驱动诊断与生态响应模块。",
        )
        use_col, clear_col = st.columns([1, 1])
        with use_col:
            if st.button("使用本库已有数据", use_container_width=True, type="primary"):
                st.session_state.use_library_data = True
                st.session_state.analysis_mode = "典型流域分析"
        with clear_col:
            if st.button("清除已有数据", use_container_width=True):
                clear_library_data_selection()
                st.rerun()
        if st.session_state.get("use_library_data"):
            st.markdown(
                '<div class="success-card">已选择平台内置典型流域示范数据，可开始分析。</div>',
                unsafe_allow_html=True,
            )
            files = scan_library_files()
            st.caption("当前已接入的本库数据文件：")
            st.dataframe(pd.DataFrame({"file": files}), use_container_width=True, hide_index=True)


def page_config() -> None:
    st.markdown('<div class="section-title">研究配置与数据输入</div>', unsafe_allow_html=True)
    mode = st.radio(
        "分析模式",
        ["典型流域分析", "自定义研究区模式"],
        index=0 if st.session_state.get("analysis_mode") != "自定义研究区模式" else 1,
        horizontal=True,
    )
    if mode != st.session_state.get("analysis_mode"):
        st.session_state.analysis_mode = mode
        st.session_state.analysis_ready = False
        if mode == "自定义研究区模式":
            st.session_state.use_library_data = False

    render_basin_selector()
    render_data_catalog()
    render_upload_area()

    st.markdown("---")
    start_col, reset_col = st.columns([1.4, 1])
    with start_col:
        start_clicked = st.button("开始分析", type="primary", use_container_width=True)
    with reset_col:
        reset_clicked = st.button("清除当前区域分析", use_container_width=True)

    if reset_clicked:
        clear_current_analysis()
        st.info("已清除当前区域分析，可重新选择流域和数据。")
        st.rerun()

    if start_clicked:
        if st.session_state.analysis_mode == "典型流域分析":
            if st.session_state.use_library_data:
                st.session_state.analysis_ready = True
                st.session_state.year_start = 2000
                st.session_state.year_end = 2020
                st.success("分析任务已创建，当前使用平台内置典型流域示范数据，可在左侧功能栏查看事件识别、年际变化、驱动诊断与生态响应结果。")
            else:
                st.session_state.analysis_ready = False
                st.error("请上传必要输入数据，或点击“使用本库已有数据”调用平台内置示范数据。")
        else:
            st.session_state.analysis_ready = False
            uploaded_boundary = st.session_state.get("custom_boundary_uploaded")
            uploaded_data = st.session_state.get("custom_data_uploaded")
            if uploaded_boundary or uploaded_data:
                st.warning("已接收上传数据。当前版本暂未开放自定义研究区自动计算，后续版本将支持在线计算与结果生成。")
            else:
                st.error("自定义研究区模式当前为接口预留版本。请上传完整数据后再提交，或切换至典型流域分析模式。")


def render_disabled_page() -> None:
    st.markdown(
        """
        <div class="disabled-page-card">
            <h3 style="color:#0b2f5b;margin-bottom:8px;">请先完成研究配置与数据输入</h3>
            <p style="color:#66788a;">结果模块需要先创建分析任务。请选择典型流域分析并点击“使用本库已有数据”，或在后续版本中接入自定义研究区数据。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("前往研究配置", type="primary"):
        go_to_page("研究配置与数据输入")
        st.rerun()


def require_ready() -> bool:
    if st.session_state.get("analysis_ready") and st.session_state.get("use_library_data"):
        return True
    render_disabled_page()
    return False


def page_event_results(data: dict[str, pd.DataFrame]) -> None:
    if not require_ready():
        return
    st.markdown('<div class="section-title">事件识别结果</div>', unsafe_allow_html=True)
    html_card(
        "config-card",
        "模块说明",
        "该模块基于平台内置典型流域示范数据，识别干旱、洪涝/湿涝异常、旱转涝 DTF 与涝转旱 FTD 事件，用于揭示不同流域类型下极端水文气象事件的区域差异。",
    )
    summary = filter_active_basin(data["event_summary"])
    yearly = filter_active_basin(data["yearly"])
    event_table = filter_active_basin(data["event_table"])
    if summary.empty:
        st.warning("事件识别结果表为空。")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("干旱月份", int(summary["total_drought_months"].sum()))
    with c2:
        render_metric_card("洪涝/湿涝月份", int(summary["total_flood_months"].sum()))
    with c3:
        render_metric_card("DTF 次数", int(summary["total_dtf_events"].sum()))
    with c4:
        render_metric_card("FTD 次数", int(summary["total_ftd_events"].sum()))

    plot_bar(
        summary,
        "basin_name",
        ["total_drought_months", "total_flood_months", "total_dtf_events", "total_ftd_events"],
        "四流域事件频次对比",
    )
    plot_bar(summary, "basin_name", ["total_dtf_events", "total_ftd_events"], "DTF 与 FTD 对比")

    with st.expander("查看原始结果表", expanded=False):
        st.dataframe(summary, use_container_width=True)
        st.dataframe(yearly, use_container_width=True)
        st.dataframe(event_table, use_container_width=True)


def page_annual(data: dict[str, pd.DataFrame]) -> None:
    if not require_ready():
        return
    st.markdown('<div class="section-title">年际变化</div>', unsafe_allow_html=True)
    html_card(
        "config-card",
        "模块说明",
        "该模块展示 2000—2020 年间典型流域干旱、洪涝/湿涝及旱涝急转事件的年际波动特征，用于识别极端事件的阶段性增强、减弱与转换风险变化。",
    )
    yearly = filter_active_basin(data["yearly"])
    if yearly.empty:
        st.warning("年际变化表为空。")
        return
    actual_min = int(yearly["year"].min())
    actual_max = int(yearly["year"].max())
    st.caption(f"当前示范数据库界面时间范围：2000—2020 年；结果表实际年份：{actual_min}—{actual_max} 年。")
    basin = st.selectbox("选择流域", list(yearly["basin_name"].drop_duplicates()))
    sub = yearly[yearly["basin_name"] == basin].sort_values("year")
    plot_line(sub, "year", ["drought_months", "flood_months", "dtf_count", "ftd_count"], None, f"{basin} 年际变化")
    with st.expander("查看年际变化数据表", expanded=False):
        st.dataframe(sub, use_container_width=True)


def page_driver(data: dict[str, pd.DataFrame]) -> None:
    if not require_ready():
        return
    st.markdown('<div class="section-title">驱动因子分析</div>', unsafe_allow_html=True)
    html_card(
        "config-card",
        "模块说明",
        "该模块从前期降水、土壤湿度、气温、NDVI 与地形条件等角度，分析极端洪旱及旱涝急转事件发生前的水热背景和致灾因子差异。",
    )
    st.markdown(
        '<div class="warning-card">DTF/FTD 属于相对稀有事件，当前特征重要性结果主要用于探索性归因，不宜解释为强预测模型结论。</div>',
        unsafe_allow_html=True,
    )
    metrics = data["model_metrics"]
    importance = data["feature_importance"]
    lag_corr = filter_active_basin(data["lag_corr"])
    if metrics.empty:
        st.warning("驱动模型指标表为空。")
        return

    m1, m2 = st.columns(2)
    for col, target in zip([m1, m2], ["dtf_flag", "ftd_flag"]):
        row = metrics[metrics["target"] == target]
        if not row.empty:
            with col:
                render_metric_card(f"{target} ROC-AUC", f"{float(row['roc_auc'].iloc[0]):.3f}", "探索性转换风险模型")

    target = st.selectbox("选择转换目标", list(importance["target"].drop_duplicates()))
    sub_imp = importance[importance["target"] == target].sort_values("importance", ascending=False).head(10)
    if HAS_PLOTLY and not sub_imp.empty:
        fig = px.bar(sub_imp[::-1], x="importance", y="feature", orientation="h", title=f"{target} 特征重要性")
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(sub_imp, use_container_width=True)

    target_event = st.selectbox("选择事件类型查看滞后相关", list(lag_corr["target_event"].drop_duplicates()))
    sub_corr = lag_corr[lag_corr["target_event"] == target_event].sort_values("abs_correlation", ascending=False).head(12)
    if HAS_PLOTLY and not sub_corr.empty:
        fig = px.bar(sub_corr[::-1], x="abs_correlation", y="feature", color="basin_name", orientation="h", title=f"{target_event} 滞后相关")
        fig.update_layout(height=460, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("查看驱动因子原始结果表", expanded=False):
        st.dataframe(metrics, use_container_width=True)
        st.dataframe(importance, use_container_width=True)
        st.dataframe(lag_corr, use_container_width=True)
        st.dataframe(filter_active_basin(data["driver_summary"]), use_container_width=True)


def page_ecology(data: dict[str, pd.DataFrame]) -> None:
    if not require_ready():
        return
    st.markdown('<div class="section-title">生态响应分析</div>', unsafe_allow_html=True)
    html_card(
        "config-card",
        "模块说明",
        "该模块基于 NDVI 标准化异常，分析干旱、洪涝/湿涝、DTF 与 FTD 事件前后植被状态的变化及滞后响应特征。",
    )
    st.markdown(
        '<div class="warning-card">NDVI 代表流域平均植被活力变化，不能直接等同于作物产量损失、局地农田灾损或城市绿地破坏。</div>',
        unsafe_allow_html=True,
    )
    response = filter_active_basin(data["ndvi_response"])
    window = filter_active_basin(data["ndvi_window"])
    lag = filter_active_basin(data["ndvi_lag"])
    sig = filter_active_basin(data["ndvi_sig"])
    if response.empty:
        st.warning("生态响应汇总表为空。")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card("响应记录类型", response["event_type"].nunique())
    with c2:
        render_metric_card("平均 1-3 月响应", f"{response['mean_response_1_3'].mean():.2f}")
    with c3:
        render_metric_card("平均 4-6 月响应", f"{response['mean_response_4_6'].mean():.2f}")

    basin = st.selectbox("选择流域", list(window["basin_name"].drop_duplicates()))
    event_type = st.selectbox("选择事件类型", list(window["event_type"].drop_duplicates()))
    sub_window = window[(window["basin_name"] == basin) & (window["event_type"] == event_type)].sort_values("relative_month")
    if HAS_PLOTLY and not sub_window.empty:
        fig = px.line(sub_window, x="relative_month", y="mean_ndvi_z", markers=True, title=f"{basin} / {event_type} NDVI 响应窗口")
        fig.add_hline(y=0, line_dash="dash", line_color="#66788a")
        fig.add_vline(x=0, line_dash="dot", line_color="#e6972f")
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(sub_window, use_container_width=True)

    with st.expander("查看生态响应原始结果表", expanded=False):
        st.dataframe(response, use_container_width=True)
        st.dataframe(lag, use_container_width=True)
        st.dataframe(sig, use_container_width=True)


def render_basin_conclusions() -> None:
    conclusions = {
        "嘉陵江流域": "上游山地支流型流域，干旱与湿涝频次接近，DTF 相对活跃，说明其水分状态转换较快。该类流域可能受地形降水、坡面汇流与前期土壤干湿状态共同影响，干旱背景下的降水恢复容易触发由干向湿的快速转换。",
        "汉江流域": "支流型与调控型流域特征明显，干湿异常频次接近，DTF 较活跃，体现较强的干湿波动。该类流域除受降水和土壤湿度控制外，还可能受到水库调度、跨流域调水和人类活动影响，正式研究中需要进一步纳入水利工程调控变量。",
        "洞庭湖流域": "湖泊—河网调蓄型流域，洪涝/湿涝异常较突出，FTD 更活跃，说明湿涝后转干过程值得重点关注。该类流域的风险形成可能与季风降水、入湖径流、湖泊调蓄能力和退水过程有关。",
        "鄱阳湖流域": "季节性湖泊水位波动型流域，干湿频次接近，DTF 与 FTD 次数接近，湖泊调蓄、季节性水位变化和生态响应可能更重要。该类流域需重点关注枯水期提前、湿涝后生态恢复和湖泊湿地植被响应。",
    }
    cols = st.columns(2)
    for idx, (name, text) in enumerate(conclusions.items()):
        with cols[idx % 2]:
            html_card("conclusion-card", name, html.escape(text))


def page_conclusions() -> None:
    if not require_ready():
        return
    st.markdown('<div class="section-title">结论总结与发展建议</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="warning-card"><b>本分析基于大数据模型与 AI 智能分析获得，仅供参考。</b></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">第一部分：综合结论</div>', unsafe_allow_html=True)
    render_basin_conclusions()

    st.markdown('<div class="section-title">第二部分：综合发展建议</div>', unsafe_allow_html=True)
    html_card(
        "recommendation-card",
        "多学科融合治理体系",
        "建议构建“气象预警—水文过程—生态响应—社会风险—工程调控”一体化治理体系。通过融合遥感监测、流域水文模拟、人工智能诊断、湖库调度优化和社会经济暴露评估，提升极端洪旱和旱涝急转事件的识别、预警、响应和适应能力。",
    )

    st.markdown('<div class="section-title">第三部分：按学科领域查看发展建议</div>', unsafe_allow_html=True)
    field = st.selectbox("请选择发展建议领域", list(DEVELOPMENT_SUGGESTIONS.keys()))
    html_card("recommendation-card", field, html.escape(DEVELOPMENT_SUGGESTIONS[field]))

    st.markdown('<div class="section-title">第四部分：正式研究升级方向</div>', unsafe_allow_html=True)
    steps = [
        "典型流域示范数据",
        "全长江流域多尺度数据接入",
        "日尺度事件识别",
        "径流/水位/湖泊调蓄过程建模",
        "人类活动归因",
        "非平稳风险评估",
        "智能预警与决策支持",
    ]
    cols = st.columns(4)
    for idx, step in enumerate(steps):
        with cols[idx % 4]:
            st.markdown(f'<div class="flow-card">{html.escape(step)}</div>', unsafe_allow_html=True)


def render_sidebar() -> str:
    st.sidebar.markdown(f"### {PLATFORM_CN}")
    st.sidebar.caption(PLATFORM_EN)
    pending_page = st.session_state.get("pending_nav_page")
    if pending_page in PAGES:
        st.session_state.current_page = pending_page
        st.session_state.nav_page = pending_page
        st.session_state.pending_nav_page = None

    current = st.session_state.get("current_page", "首页")
    if current not in PAGES:
        current = "首页"
    if st.session_state.get("nav_page") not in PAGES:
        st.session_state.nav_page = current
    page = st.sidebar.radio("功能导航", PAGES, key="nav_page")
    st.session_state.current_page = page

    st.sidebar.markdown("---")
    if st.session_state.get("analysis_ready") and st.session_state.get("use_library_data"):
        st.sidebar.success("示范分析任务已创建")
        st.sidebar.write(f"模式：{st.session_state.get('analysis_mode')}")
        selected = st.session_state.get("selected_subbasin", "全部典型子流域")
        st.sidebar.write(f"研究区：{selected}")
        st.sidebar.write("年份：2000—2020")
        if st.sidebar.button("清除当前区域分析", use_container_width=True):
            clear_current_analysis()
            st.rerun()
    else:
        st.sidebar.warning("尚未创建分析任务")
    return page


def main() -> None:
    inject_css()
    init_state()
    data = load_data()
    page = render_sidebar()

    if page == "首页":
        page_home()
    elif page == "研究配置与数据输入":
        page_config()
    elif page == "事件识别结果":
        page_event_results(data)
    elif page == "年际变化":
        page_annual(data)
    elif page == "驱动因子分析":
        page_driver(data)
    elif page == "生态响应分析":
        page_ecology(data)
    elif page == "结论总结与发展建议":
        page_conclusions()


if __name__ == "__main__":
    main()
