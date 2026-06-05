"""
지역 축제 효과성 분석 대시보드
사용법: streamlit run app.py
DB 파일: festival_analysis.db (분석 결과) + festival_raw.db (원본 데이터)
"""

import sqlite3
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ────────────────────────────────────────────────────────
# 0. 기본 설정
# ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="지역 축제 효과성 분석",
    page_icon="🎪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 색상 팔레트
TYPE_COLORS = {
    "Type A — 전방위 우수형":   "#534AB7",
    "Type B — 경제·방문 균형형": "#185FA5",
    "Type C — SNS 주도형":      "#0F6E56",
    "Type D — 효과 미흡형":     "#993C1D",
}
TYPE_SHORT = {
    "Type A — 전방위 우수형":   "Type A",
    "Type B — 경제·방문 균형형": "Type B",
    "Type C — SNS 주도형":      "Type C",
    "Type D — 효과 미흡형":     "Type D",
}
PERIOD_ORDER = [
    "BEFORE_3M", "BEFORE_2M", "BEFORE_1M",
    "FESTIVAL",
    "AFTER_1M", "AFTER_2M", "AFTER_3M", "AFTER_6M",
]
PERIOD_LABEL = {
    "BEFORE_3M": "Before 3M", "BEFORE_2M": "Before 2M", "BEFORE_1M": "Before 1M",
    "FESTIVAL":  "축제 기간",
    "AFTER_1M":  "After 1M",  "AFTER_2M":  "After 2M",
    "AFTER_3M":  "After 3M",  "AFTER_6M":  "After 6M",
}


# ────────────────────────────────────────────────────────
# 1. 데이터 로드
# ────────────────────────────────────────────────────────
@st.cache_data
def load_analysis_db():
    conn = sqlite3.connect("festival_analysis.db")

    score = pd.read_sql("""
        SELECT festival_name, composite_score, cluster_label, tier,
               avg_buzz_lift, avg_buzz_retention, avg_spend_lift,
               avg_retention_rate, avg_outsider_ratio,
               avg_visitor_per_pop, avg_visitor_lift
        FROM fact_composite_score
        ORDER BY composite_score DESC
    """, conn)

    retention = pd.read_sql("""
        SELECT * FROM fact_retention_v2
    """, conn)

    visitor = pd.read_sql("""
        SELECT festival_name, festival_year, festival_days,
               외지인방문자수, 현지인방문자수, 전체방문자수, 일평균방문자수
        FROM fact_visitor
    """, conn)

    conn.close()
    return score, retention, visitor


@st.cache_data
def load_raw_db():
    conn = sqlite3.connect("festival_raw.db")

    spending = pd.read_sql("""
        SELECT festival_name, festival_year, period, spending_million
        FROM fact_spending
        ORDER BY festival_name, festival_year, period
    """, conn)

    sns = pd.read_sql("""
        SELECT festival_name, festival_year, period, search_volume
        FROM fact_sns
        ORDER BY festival_name, festival_year, period
    """, conn)

    visitor_ts = pd.read_sql("""
        SELECT festival_name, festival_year, period,
               전체방문자수, 외지인방문자수, 현지인방문자수
        FROM fact_visitor_ts
        ORDER BY festival_name, festival_year, period
    """, conn)

    conn.close()
    return spending, sns, visitor_ts


@st.cache_data
def build_classifier(score_df):
    """로지스틱 회귀 모델 학습"""
    features = [
        "avg_buzz_lift", "avg_buzz_retention", "avg_spend_lift",
        "avg_retention_rate", "avg_outsider_ratio",
        "avg_visitor_per_pop", "avg_visitor_lift",
    ]
    type_map = {
        "Type A — 전방위 우수형":   0,
        "Type B — 경제·방문 균형형": 1,
        "Type C — SNS 주도형":      2,
        "Type D — 효과 미흡형":     3,
    }
    X = score_df[features].values
    y = score_df["cluster_label"].map(type_map).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000, random_state=42)
    clf.fit(Xs, y)
    return clf, scaler, features


# ────────────────────────────────────────────────────────
# 2. 공통 유틸
# ────────────────────────────────────────────────────────
def hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """hex 색상을 rgba() 문자열로 변환 (plotly 호환)"""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def type_badge(label: str) -> str:
    colors = {
        "Type A — 전방위 우수형":   ("#EEEDFE", "#534AB7"),
        "Type B — 경제·방문 균형형": ("#E6F1FB", "#185FA5"),
        "Type C — SNS 주도형":      ("#E1F5EE", "#0F6E56"),
        "Type D — 효과 미흡형":     ("#FAECE7", "#993C1D"),
    }
    bg, fg = colors.get(label, ("#F0F0F0", "#333"))
    short = TYPE_SHORT.get(label, label)
    return f'<span style="background:{bg};color:{fg};padding:2px 10px;border-radius:4px;font-size:12px;font-weight:500">{short}</span>'


def period_ratio(df, val_col):
    """period별 값을 BEFORE 평균 대비 비율로 변환"""
    before = df[df["period"].str.startswith("BEFORE")][val_col].mean()
    if before == 0 or pd.isna(before):
        return df.assign(ratio=np.nan)
    df = df.copy()
    df["ratio"] = df[val_col] / before * 100
    return df


# ────────────────────────────────────────────────────────
# 3. 사이드바
# ────────────────────────────────────────────────────────
score_df, retention_df, visitor_df = load_analysis_db()
spending_df, sns_df, visitor_ts_df = load_raw_db()
clf, scaler, features = build_classifier(score_df)

with st.sidebar:
    st.markdown("## 🎪 축제 효과성 분석")
    st.caption("인구감소지역 27개 축제 · 2022–2025")
    st.divider()

    tab_sel = st.radio(
        "분석 탭 선택",
        ["📊 전체 현황", "🔍 축제 상세", "📈 지속성 분석", "🤖 타입 분류기"],
        label_visibility="collapsed",
    )
    st.divider()

    # 공통 필터
    all_types = sorted(score_df["cluster_label"].unique())
    sel_types = st.multiselect(
        "축제 유형 필터",
        options=all_types,
        default=all_types,
        format_func=lambda x: TYPE_SHORT[x],
    )

    st.caption(f"선택된 축제: {len(score_df[score_df['cluster_label'].isin(sel_types)])}개 / 27개")


filtered_score = score_df[score_df["cluster_label"].isin(sel_types)].copy()


# ════════════════════════════════════════════════════════
# TAB 1: 전체 현황
# ════════════════════════════════════════════════════════
if tab_sel == "📊 전체 현황":
    st.markdown("## 📊 전체 현황")
    st.caption("27개 인구감소지역 축제의 효과성 종합 평가")

    # KPI 4개
    c1, c2, c3, c4 = st.columns(4)
    type_counts = score_df["cluster_label"].value_counts()
    with c1:
        n_a = int(type_counts.get("Type A — 전방위 우수형", 0))
        n_b = int(type_counts.get("Type B — 경제·방문 균형형", 0))
        st.metric("효과성 우수 (Type A·B)", f"{n_a + n_b}개",
                  help="종합 점수 상위 그룹")
    with c2:
        st.metric("평균 소비 증폭률",
                  f"{score_df['avg_spend_lift'].mean():.1f}%",
                  help="BEFORE 기준선 대비")
    with c3:
        st.metric("평균 외지인 비율",
                  f"{score_df['avg_outsider_ratio'].mean():.1f}%")
    with c4:
        n_d = int(type_counts.get("Type D — 효과 미흡형", 0))
        st.metric("효과 미흡 (Type D)", f"{n_d}개",
                  help="전체의 52%")

    st.divider()

    # 종합 점수 막대 차트
    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.markdown("#### 축제별 종합 효과성 점수")
        fig = px.bar(
            filtered_score.sort_values("composite_score"),
            x="composite_score",
            y="festival_name",
            color="cluster_label",
            color_discrete_map=TYPE_COLORS,
            orientation="h",
            labels={"composite_score": "종합 점수", "festival_name": "",
                    "cluster_label": "유형"},
            height=max(400, len(filtered_score) * 22),
        )
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
            margin=dict(l=10, r=20, t=10, b=30),
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(range=[0, 100], gridcolor="rgba(200,200,200,0.3)"),
        )
        fig.add_vline(x=60, line_dash="dot", line_color="gray", opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("#### 유형별 분포")
        type_summary = (
            score_df.groupby("cluster_label")
            .agg(n=("festival_name", "count"),
                 avg_score=("composite_score", "mean"),
                 avg_spend=("avg_spend_lift", "mean"),
                 avg_retention=("avg_retention_rate", "mean"))
            .reset_index()
        )
        fig_pie = px.pie(
            type_summary,
            values="n",
            names="cluster_label",
            color="cluster_label",
            color_discrete_map=TYPE_COLORS,
            hole=0.45,
        )
        fig_pie.update_traces(textinfo="label+percent", showlegend=False)
        fig_pie.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=280)
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("#### 유형별 평균 지표")
        for _, row in type_summary.sort_values("avg_score", ascending=False).iterrows():
            color = TYPE_COLORS[row["cluster_label"]]
            st.markdown(
                f'<div style="border-left:3px solid {color};padding:6px 10px;margin:4px 0;'
                f'background:rgba(0,0,0,0.02);border-radius:0 6px 6px 0">'
                f'<b style="color:{color}">{TYPE_SHORT[row["cluster_label"]]}</b> '
                f'n={int(row["n"])} · 평균 {row["avg_score"]:.1f}점<br>'
                f'<span style="font-size:11px;color:#666">'
                f'소비증폭 {row["avg_spend"]:.1f}% · 소비유지 {row["avg_retention"]:.1f}%'
                f'</span></div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # 산점도: 소비 증폭 vs 유지율
    st.markdown("#### 소비 증폭률 vs 소비 유지율")
    fig_sc = px.scatter(
        filtered_score,
        x="avg_spend_lift",
        y="avg_retention_rate",
        color="cluster_label",
        color_discrete_map=TYPE_COLORS,
        size="composite_score",
        size_max=30,
        text="festival_name",
        labels={"avg_spend_lift": "소비 증폭률 (%)", "avg_retention_rate": "소비 유지율 (%)",
                "cluster_label": "유형"},
        height=480,
    )
    fig_sc.update_traces(textposition="top center", textfont_size=10)
    fig_sc.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.5,
                     annotation_text="기준선 100%")
    fig_sc.add_vline(x=100, line_dash="dot", line_color="gray", opacity=0.5)
    fig_sc.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        margin=dict(t=40),
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    # 축제 기간별 효과성 (작업 2 분석)
    st.divider()
    st.markdown("#### 축제 기간 구간별 효과성")
    st.caption("ANOVA 결과: 소비 p=0.22 / SNS p=0.25 — 기간 자체보다 콘텐츠 전략이 효과성의 주요인")

    # raw DB에서 기간 구간 분석
    dur_df = visitor_df.merge(
        score_df[["festival_name", "avg_spend_lift", "avg_sns_lift" if "avg_sns_lift" in score_df.columns
                  else "avg_buzz_lift", "avg_visitor_lift", "cluster_label"]],
        on="festival_name", how="left"
    )
    dur_df["days_group"] = pd.cut(
        dur_df["festival_days"],
        bins=[0, 4, 9, 15, 100],
        labels=["① 단기 1~4일", "② 중기 5~9일", "③ 장기 10~15일", "④ 초장기 16일+"],
    )

    dur_agg = dur_df.groupby("days_group", observed=True).agg(
        n=("festival_name", "count"),
        avg_spend=("avg_spend_lift", "mean"),
        avg_visitor=("avg_visitor_lift", "mean"),
    ).reset_index()

    fig_dur = make_subplots(specs=[[{"secondary_y": True}]])
    colors_dur = ["#AFA9EC", "#85B7EB", "#0F6E56", "#D4C9A8"]
    fig_dur.add_trace(go.Bar(
        x=dur_agg["days_group"], y=dur_agg["avg_spend"],
        name="소비 증폭률",
        marker_color=colors_dur,
        text=dur_agg["avg_spend"].round(1).astype(str) + "%",
        textposition="outside",
    ))
    fig_dur.add_trace(go.Scatter(
        x=dur_agg["days_group"], y=dur_agg["avg_visitor"],
        name="방문자 증폭률",
        mode="lines+markers",
        line=dict(color="#993C1D", width=2),
        marker=dict(size=8),
    ), secondary_y=True)
    fig_dur.update_layout(
        height=350,
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=20, b=20),
    )
    fig_dur.update_yaxes(title_text="소비 증폭률 (%)", secondary_y=False)
    fig_dur.update_yaxes(title_text="방문자 증폭률 (%)", secondary_y=True)
    st.plotly_chart(fig_dur, use_container_width=True)


# ════════════════════════════════════════════════════════
# TAB 2: 축제 상세
# ════════════════════════════════════════════════════════
elif tab_sel == "🔍 축제 상세":
    st.markdown("## 🔍 축제 상세 분석")

    festival_list = filtered_score["festival_name"].tolist()
    if not festival_list:
        st.warning("선택된 유형에 해당하는 축제가 없습니다.")
        st.stop()

    sel_festival = st.selectbox(
        "축제 선택",
        options=festival_list,
        format_func=lambda x: f"{x}  ({filtered_score.loc[filtered_score['festival_name']==x,'cluster_label'].values[0].split('—')[0].strip()})",
    )

    fest_info = score_df[score_df["festival_name"] == sel_festival].iloc[0]

    # 기본 정보
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(type_badge(fest_info["cluster_label"]), unsafe_allow_html=True)
        st.metric("종합 점수", f"{fest_info['composite_score']:.1f}점")
    with col2:
        st.metric("소비 증폭률", f"{fest_info['avg_spend_lift']:.1f}%")
        st.metric("소비 유지율", f"{fest_info['avg_retention_rate']:.1f}%")
    with col3:
        st.metric("SNS 증폭률", f"{fest_info['avg_buzz_lift']:.1f}%")
        st.metric("SNS 유지율", f"{fest_info['avg_buzz_retention']:.1f}%")
    with col4:
        st.metric("외지인 비율", f"{fest_info['avg_outsider_ratio']:.1f}%")
        st.metric("방문자 증폭률", f"{fest_info['avg_visitor_lift']:.1f}%")

    st.divider()

    # 시계열 차트 (3개 차원)
    years = sorted(spending_df[spending_df["festival_name"] == sel_festival]["festival_year"].unique())
    sel_year = st.select_slider("연도 선택", options=["전체(평균)"] + [str(y) for y in years])

    def get_ts_data(df, val_col, festival, year):
        sub = df[df["festival_name"] == festival].copy()
        if year != "전체(평균)":
            sub = sub[sub["festival_year"] == int(year)]
        sub = sub.groupby("period")[val_col].mean().reset_index()
        sub = period_ratio(sub, val_col)
        sub["period_label"] = sub["period"].map(PERIOD_LABEL)
        sub["order"] = sub["period"].map({p: i for i, p in enumerate(PERIOD_ORDER)})
        return sub.sort_values("order")

    sp_ts  = get_ts_data(spending_df,    "spending_million", sel_festival, sel_year)
    sns_ts = get_ts_data(sns_df,         "search_volume",   sel_festival, sel_year)
    vt_ts  = get_ts_data(visitor_ts_df,  "전체방문자수",     sel_festival, sel_year)

    fig_ts = make_subplots(
        rows=1, cols=3,
        subplot_titles=["소비액 변화율", "SNS 언급량 변화율", "방문자 변화율"],
        shared_yaxes=False,
    )
    color = TYPE_COLORS[fest_info["cluster_label"]]

    for i, (ts_df, name) in enumerate([(sp_ts, "소비"), (sns_ts, "SNS"), (vt_ts, "방문자")], 1):
        fig_ts.add_trace(go.Scatter(
            x=ts_df["period_label"], y=ts_df["ratio"],
            mode="lines+markers",
            name=name,
            line=dict(color=color, width=2.5),
            marker=dict(size=8, color=color),
            showlegend=False,
        ), row=1, col=i)
        fig_ts.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.4, row=1, col=i)

    fig_ts.add_vrect(
        x0="Before 1M", x1="After 1M",
        fillcolor="rgba(200,200,200,0.15)",
        layer="below", line_width=0,
    )
    fig_ts.update_layout(
        height=320,
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=20),
    )
    fig_ts.update_yaxes(title_text="BEFORE 대비 비율 (%)")
    st.plotly_chart(fig_ts, use_container_width=True)

    # 레이더 차트: 해당 축제 vs 유형 평균
    st.markdown("#### 지표 프로파일 비교")
    feats_radar = [
        "avg_buzz_lift", "avg_spend_lift", "avg_retention_rate",
        "avg_outsider_ratio", "avg_visitor_lift", "avg_buzz_retention",
    ]
    feat_labels = ["SNS증폭", "소비증폭", "소비유지율", "외지인비율", "방문증폭", "SNS유지"]

    type_avg = score_df[score_df["cluster_label"] == fest_info["cluster_label"]][feats_radar].mean()
    all_avg  = score_df[feats_radar].mean()

    # Min-Max 정규화 (전체 기준)
    mn = score_df[feats_radar].min()
    mx = score_df[feats_radar].max()
    norm = lambda s: ((s - mn) / (mx - mn + 1e-9) * 100).clip(0, 100)

    fest_norm = norm(fest_info[feats_radar])
    type_norm = norm(type_avg)
    all_norm  = norm(all_avg)

    fig_radar = go.Figure()
    for vals, name, clr, dash in [
        (all_norm,  "전체 평균",      "gray",  "dot"),
        (type_norm, f"{TYPE_SHORT[fest_info['cluster_label']]} 평균", color, "dash"),
        (fest_norm, sel_festival,    color,  "solid"),
    ]:
        fig_radar.add_trace(go.Scatterpolar(
            r=list(vals) + [vals.iloc[0]],
            theta=feat_labels + [feat_labels[0]],
            fill="toself" if dash == "solid" else "none",
            fillcolor=hex_to_rgba(color, 0.15) if dash == "solid" else "rgba(0,0,0,0)",
            name=name,
            line=dict(color=clr, width=2 if dash == "solid" else 1.5, dash=dash),
            opacity=0.9 if dash == "solid" else 0.7,
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(range=[0, 100], visible=True, tickfont_size=9)),
        showlegend=True,
        height=380,
        margin=dict(l=60, r=60, t=20, b=20),
        legend=dict(orientation="h", y=-0.05),
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# ════════════════════════════════════════════════════════
# TAB 3: 지속성 분석
# ════════════════════════════════════════════════════════
elif tab_sel == "📈 지속성 분석":
    st.markdown("## 📈 단기·장기 지속성 분석")
    st.caption("단기: AFTER 1~3개월 평균 / 장기: AFTER 6개월")

    # 유형별 지속성 집계
    ret_with_type = retention_df.merge(
        score_df[["festival_name", "cluster_label"]], on="festival_name", how="left"
    )
    ret_agg = ret_with_type.groupby("cluster_label").agg(
        sp_fest=("sp_lift", "mean"),
        sp_short=("sp_short_rate", "mean"),
        sp_long=("sp_long_rate", "mean"),
        sns_fest=("sns_lift", "mean"),
        sns_short=("sns_short_rate", "mean"),
        sns_long=("sns_long_rate", "mean"),
        vis_fest=("vis_lift", "mean"),
        vis_short=("vis_short_rate", "mean"),
        vis_long=("vis_long_rate", "mean"),
    ).reset_index()

    # 차원 선택
    dim = st.radio(
        "분석 차원",
        ["소비액", "SNS 언급량", "방문자 수"],
        horizontal=True,
    )
    dim_map = {
        "소비액":      ("sp_fest", "sp_short", "sp_long"),
        "SNS 언급량":  ("sns_fest", "sns_short", "sns_long"),
        "방문자 수":   ("vis_fest", "vis_short", "vis_long"),
    }
    c_fest, c_short, c_long = dim_map[dim]

    # 유형별 꺾은선 차트
    fig_pers = go.Figure()
    x_labels = ["축제 기간", "단기 (1~3M)", "장기 (6M)"]

    for _, row in ret_agg.iterrows():
        if row["cluster_label"] not in sel_types:
            continue
        color = TYPE_COLORS[row["cluster_label"]]
        fig_pers.add_trace(go.Scatter(
            x=x_labels,
            y=[row[c_fest], row[c_short], row[c_long]],
            mode="lines+markers",
            name=TYPE_SHORT[row["cluster_label"]],
            line=dict(color=color, width=3),
            marker=dict(size=10, color=color),
        ))

    fig_pers.add_hline(
        y=100, line_dash="dot", line_color="gray", opacity=0.5,
        annotation_text="기준선 (100%)",
        annotation_position="right",
    )
    fig_pers.update_layout(
        height=380,
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="BEFORE 기준 대비 비율 (%)", gridcolor="rgba(200,200,200,0.3)"),
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_pers, use_container_width=True)

    # 단기 vs 장기 비교 산점도
    st.markdown("#### 단기 vs 장기 소비 유지율 — 축제별")
    ret_fest = ret_with_type.groupby(["festival_name", "cluster_label"]).agg(
        sp_short=("sp_short_rate", "mean"),
        sp_long=("sp_long_rate", "mean"),
    ).reset_index()

    if not ret_fest[ret_fest["cluster_label"].isin(sel_types)].empty:
        fig_sl = px.scatter(
            ret_fest[ret_fest["cluster_label"].isin(sel_types)],
            x="sp_short", y="sp_long",
            color="cluster_label",
            color_discrete_map=TYPE_COLORS,
            text="festival_name",
            labels={"sp_short": "단기 소비 유지율 (%)", "sp_long": "장기(6M) 소비 유지율 (%)",
                    "cluster_label": "유형"},
            height=450,
        )
        fig_sl.add_shape(type="line", x0=70, y0=70, x1=150, y1=150,
                         line=dict(color="gray", dash="dot"), opacity=0.4)
        fig_sl.add_annotation(x=145, y=148, text="단기=장기", showarrow=False,
                               font=dict(size=10, color="gray"))
        fig_sl.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.3)
        fig_sl.add_vline(x=100, line_dash="dot", line_color="gray", opacity=0.3)
        fig_sl.update_traces(textposition="top center", textfont_size=9)
        fig_sl.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.05),
            margin=dict(t=30),
        )
        st.plotly_chart(fig_sl, use_container_width=True)

    # 소비 지속성 패턴 분류
    st.markdown("#### 소비 지속성 패턴 분류")
    ret_pattern = ret_fest.copy()
    ret_pattern["pattern"] = ret_pattern.apply(
        lambda r: "📈 지속상승형" if r["sp_long"] > r["sp_short"] > 100
        else ("🟦 단기유지형" if r["sp_short"] >= 100 and r["sp_long"] >= 100
              else ("🔄 조기회복형" if r["sp_short"] < 100 and r["sp_long"] >= 100
                    else "📉 지속하락형")),
        axis=1,
    )
    pattern_counts = ret_pattern[ret_pattern["cluster_label"].isin(sel_types)].groupby("pattern")["festival_name"].apply(list).reset_index()
    for _, row in pattern_counts.iterrows():
        with st.expander(f"{row['pattern']} ({len(row['festival_name'])}개)"):
            st.write(" · ".join(row["festival_name"]))


# ════════════════════════════════════════════════════════
# TAB 4: 타입 분류기
# ════════════════════════════════════════════════════════
elif tab_sel == "🤖 타입 분류기":
    st.markdown("## 🤖 신규 축제 타입 분류기")
    st.caption("다항 로지스틱 회귀 모델 · LOO-CV 정확도 88.9%")

    st.info(
        "새로운 축제의 3개년 평균 지표를 입력하면 Type A~D 중 어느 유형에 속하는지 예측합니다.\n\n"
        "**입력값은 모두 3개년 평균값(%)** 기준입니다.",
        icon="ℹ️",
    )

    # 표준화 파라미터 안내
    with st.expander("📐 모델 표준화 파라미터 (참고)"):
        param_df = pd.DataFrame({
            "변수": ["SNS 증폭률", "SNS 유지율", "소비 증폭률", "소비 유지율",
                     "외지인 비율", "인구대비 방문자", "방문자 증폭률"],
            "평균 (μ)": [116.8, 98.5, 112.4, 100.3, 58.9, 537.4, 110.2],
            "표준편차 (σ)": [24.0, 18.8, 15.6, 13.9, 14.2, 473.4, 13.3],
        })
        st.dataframe(param_df, hide_index=True, use_container_width=True)

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown("#### 지표 입력")
        inp = {}
        inp["avg_buzz_lift"]       = st.slider("SNS 증폭률 (%)",        60, 250, 120, 1)
        inp["avg_buzz_retention"]  = st.slider("SNS 유지율 (%)",         60, 200, 100, 1)
        inp["avg_spend_lift"]      = st.slider("소비 증폭률 (%)",        60, 200, 112, 1)
        inp["avg_retention_rate"]  = st.slider("소비 유지율 (%)",        60, 160, 100, 1)
        inp["avg_outsider_ratio"]  = st.slider("외지인 비율 (%)",         10, 100,  58, 1)
        inp["avg_visitor_per_pop"] = st.slider("인구대비 방문자 (%)",     50, 3000, 500, 10)
        inp["avg_visitor_lift"]    = st.slider("방문자 증폭률 (%)",       60, 200, 110, 1)

    with col_r:
        st.markdown("#### 분류 결과")

        # 예측
        x_new = np.array([[inp[f] for f in features]])
        x_scaled = scaler.transform(x_new)
        proba = clf.predict_proba(x_scaled)[0]
        pred_idx = int(np.argmax(proba))
        type_names = [
            "Type A — 전방위 우수형",
            "Type B — 경제·방문 균형형",
            "Type C — SNS 주도형",
            "Type D — 효과 미흡형",
        ]
        pred_type = type_names[pred_idx]
        pred_prob = proba[pred_idx] * 100
        pred_color = TYPE_COLORS[pred_type]

        # 신뢰도 평가
        conf_level = "높음 ✓" if pred_prob >= 90 else ("보통 △" if pred_prob >= 50 else "낮음 ⚠")

        st.markdown(
            f'<div style="background:{pred_color}18;border:1.5px solid {pred_color};'
            f'border-radius:10px;padding:16px 18px;margin-bottom:12px">'
            f'<div style="font-size:11px;color:{pred_color};font-weight:500;margin-bottom:4px">예측 결과</div>'
            f'<div style="font-size:22px;font-weight:700;color:{pred_color}">{TYPE_SHORT[pred_type]}</div>'
            f'<div style="font-size:13px;color:{pred_color};margin-top:2px">{pred_type.split("—")[1].strip()}</div>'
            f'<div style="margin-top:10px;font-size:12px;color:#555">'
            f'확률 <b>{pred_prob:.1f}%</b> · 신뢰도 <b>{conf_level}</b></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # 확률 바
        st.markdown("**타입별 소속 확률**")
        for i, (tn, p) in enumerate(zip(type_names, proba)):
            clr = TYPE_COLORS[tn]
            short = TYPE_SHORT[tn]
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0">'
                f'<span style="width:56px;font-size:11px;color:{clr};font-weight:500">{short}</span>'
                f'<div style="flex:1;height:10px;background:#eee;border-radius:4px;overflow:hidden">'
                f'<div style="width:{p*100:.1f}%;height:100%;background:{clr};border-radius:4px"></div>'
                f'</div>'
                f'<span style="font-size:11px;width:38px;text-align:right">{p*100:.1f}%</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # 전략 방향
        strategies = {
            "Type A — 전방위 우수형": [
                "성공 패턴 백서화 및 벤치마킹 모델화",
                "SNS 자생적 확산 구조 분석·타 축제 적용",
                "장기 소비 지속 콘텐츠 IP화",
            ],
            "Type B — 경제·방문 균형형": [
                "축제 직후 소규모 후속 이벤트로 단기 공백 해소",
                "재방문 쿠폰·지역화폐 연계",
                "지역 특산물 온라인 재구매 채널 연계",
            ],
            "Type C — SNS 주도형": [
                "당일치기 → 1박 이상 체류 전환 숙박 패키지",
                "축제 후 3개월 SNS 콘텐츠 지속 생산 지원",
                "지역 특산물 사후 온라인 구매 채널 연계",
            ],
            "Type D — 효과 미흡형": [
                "소비 유출 경로 진단 (외부 체인 비중 조사)",
                "외지인 접근성·홍보 채널 재검토",
                "3년 내 개선 미달 시 형식 전환 검토",
            ],
        }

        st.markdown("---")
        st.markdown(f"**{TYPE_SHORT[pred_type]} 권장 전략**")
        for s in strategies[pred_type]:
            st.markdown(f"- {s}")

        # 경계선 케이스 경고
        sorted_proba = sorted(proba, reverse=True)
        if sorted_proba[0] - sorted_proba[1] < 0.15:
            st.warning(
                f"⚠️ 1·2위 확률 차이 {(sorted_proba[0]-sorted_proba[1])*100:.1f}%p — "
                "경계선 케이스입니다. 추가 연도 데이터 확보 후 재분류 권장.",
                icon="⚠️",
            )


# ────────────────────────────────────────────────────────
# 푸터
# ────────────────────────────────────────────────────────
st.divider()
st.caption(
    "분석 기준: 인구감소지역 27개 축제 · 2022–2025 · 3개년 평균 지표 기반 "
    "| 방법론: Min-Max 정규화 + 가중합 → K-Means(K=4) + 다항 로지스틱 회귀(LOO-CV 88.9%)"
)
