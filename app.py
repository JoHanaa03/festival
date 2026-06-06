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
        ["📊 전체 현황", "🔍 축제 상세", "📈 지속성 분석", "🤖 타입 분류기", "💡 인사이트", "🗄️ 핵심 SQL"],
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



# ════════════════════════════════════════════════════════
# TAB 5: 인사이트
# ════════════════════════════════════════════════════════
elif tab_sel == "💡 인사이트":
    st.markdown("## 💡 핵심 인사이트")
    st.caption("분석 전체를 관통하는 6개 발견 — 지역 활성화를 위한 축제 개최 전략 방향")

    # ── 공통 데이터 준비 ──────────────────────────────────
    ret_with_type = retention_df.merge(
        score_df[["festival_name", "cluster_label"]], on="festival_name", how="left"
    )
    ret_fest = ret_with_type.groupby(["festival_name", "cluster_label"]).agg(
        sns_long=("sns_long_rate", "mean"),
        sp_long=("sp_long_rate", "mean"),
        sp_short=("sp_short_rate", "mean"),
    ).reset_index()

    score_df["outsider_group"] = pd.cut(
        score_df["avg_outsider_ratio"],
        bins=[0, 40, 60, 75, 100],
        labels=["~40%", "40~60%", "60~75%", "75%+"],
    )

    visitor_df["days_group"] = pd.cut(
        visitor_df["festival_days"],
        bins=[0, 4, 9, 15, 100],
        labels=["① 1~4일", "② 5~9일", "③ 10~15일", "④ 16일+"],
    )
    vis_merged = visitor_df.merge(
        score_df[["festival_name", "avg_spend_lift", "avg_buzz_lift", "avg_visitor_lift"]],
        on="festival_name", how="left",
    )

    TYPE_ORDER = [
        "Type A — 전방위 우수형",
        "Type B — 경제·방문 균형형",
        "Type C — SNS 주도형",
        "Type D — 효과 미흡형",
    ]
    TYPE_SHORT_LIST = ["Type A", "Type B", "Type C", "Type D"]
    TYPE_COLOR_LIST = ["#534AB7", "#185FA5", "#0F6E56", "#993C1D"]

    def ins_header(num, title, tag, color, body):
        col_n, col_b = st.columns([1, 11])
        with col_n:
            st.markdown(
                f'<div style="width:32px;height:32px;border-radius:50%;background:{color};'
                f'color:#fff;font-weight:700;font-size:14px;display:flex;align-items:center;'
                f'justify-content:center;margin-top:4px">{num}</div>',
                unsafe_allow_html=True,
            )
        with col_b:
            st.markdown(
                f'<span style="background:{color}18;color:{color};padding:2px 8px;'
                f'border-radius:4px;font-size:11px;font-weight:500">{tag}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**{title}**")
            st.markdown(body)

    # ══════════════════════════════════════════
    # 인사이트 1 — 방문자 유입 vs 소비 상관 산점도
    # ══════════════════════════════════════════
    ins_header("1", "방문자 유입이 소비를 만든다 — r = 0.84의 강한 상관",
               "방문·소비", "#534AB7",
               "방문자 증폭률과 소비 증폭률의 상관계수는 **r = 0.84**로 모든 지표 중 가장 높습니다. "
               "SNS 증폭과 소비의 상관(r=0.71)보다 높아, 온라인 마케팅보다 실제 방문 유도가 더 중요합니다.")

    fig1 = go.Figure()
    for lbl, clr in zip(TYPE_ORDER, TYPE_COLOR_LIST):
        sub = score_df[score_df["cluster_label"] == lbl]
        fig1.add_trace(go.Scatter(
            x=sub["avg_visitor_lift"], y=sub["avg_spend_lift"],
            mode="markers+text", name=TYPE_SHORT[lbl],
            marker=dict(color=clr, size=11, opacity=0.85,
                        line=dict(color="white", width=1)),
            text=sub["festival_name"], textposition="top center",
            textfont=dict(size=9),
        ))
    # 추세선
    x_all = score_df["avg_visitor_lift"].values
    y_all = score_df["avg_spend_lift"].values
    m, b = np.polyfit(x_all, y_all, 1)
    x_line = np.linspace(x_all.min(), x_all.max(), 50)
    fig1.add_trace(go.Scatter(
        x=x_line, y=m * x_line + b,
        mode="lines", name=f"추세선 (r=0.84)",
        line=dict(color="gray", width=1.5, dash="dot"),
        showlegend=True,
    ))
    fig1.update_layout(
        height=400,
        xaxis=dict(title="방문자 증폭률 (%)", gridcolor="rgba(200,200,200,0.3)"),
        yaxis=dict(title="소비 증폭률 (%)",   gridcolor="rgba(200,200,200,0.3)"),
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(t=20, b=30),
    )
    st.plotly_chart(fig1, use_container_width=True)
    with st.expander('🗄️ 관련 SQL — 방문자·소비 증폭률 산출'):
        st.code("""\
-- 축제별 3개년 평균 방문자·소비 증폭률 (상관 분석 기반 지표)
WITH visitor_ts_base AS (
    SELECT festival_name, festival_year,
           AVG(CASE WHEN period LIKE 'BEFORE%' THEN 전체방문자수 END)  AS before_avg,
           AVG(CASE WHEN period = 'FESTIVAL'   THEN 전체방문자수 END)  AS festival_avg
    FROM fact_visitor_ts
    GROUP BY festival_name, festival_year
),
spend_base AS (
    SELECT festival_name, festival_year,
           AVG(CASE WHEN period LIKE 'BEFORE%' THEN spending_million END) AS before_avg,
           AVG(CASE WHEN period = 'FESTIVAL'   THEN spending_million END) AS festival_avg
    FROM fact_spending
    GROUP BY festival_name, festival_year
)
SELECT
    v.festival_name,
    ROUND(AVG(v.festival_avg / NULLIF(v.before_avg,0) * 100), 1) AS avg_visitor_lift,
    ROUND(AVG(s.festival_avg / NULLIF(s.before_avg,0) * 100), 1) AS avg_spend_lift
FROM visitor_ts_base v
JOIN spend_base s USING (festival_name, festival_year)
GROUP BY v.festival_name
ORDER BY avg_visitor_lift DESC;
-- → r=0.84: 방문자 증폭이 클수록 소비 증폭도 크다
""", language='sql')
    st.divider()

    # ══════════════════════════════════════════
    # 인사이트 2 — 외지인 비율 구간별 SNS·소비 막대
    # ══════════════════════════════════════════
    ins_header("2", "외지인 비율 75% 이상이 분기점 — SNS 효과도 1.5배",
               "외지인·SNS", "#185FA5",
               "외지인 비율 **75% 이상** 구간의 SNS 증폭률은 155.8%로 40% 미만(103.5%)의 1.5배입니다. "
               "외지인이 자발적 콘텐츠 생산자가 되어 SNS 파급력을 높이기 때문입니다.")

    g2 = score_df.groupby("outsider_group", observed=True)[
        ["avg_buzz_lift", "avg_spend_lift", "composite_score"]
    ].mean().round(1).reset_index()
    groups2 = g2["outsider_group"].astype(str).tolist()
    bar_colors2 = ["#D0CDE8", "#A8C4E0", "#6A9CC8", "#185FA5"]

    fig2 = make_subplots(rows=1, cols=2,
                         subplot_titles=["외지인 비율 구간별 SNS 증폭률 (%)",
                                         "외지인 비율 구간별 종합 효과성 점수"],
                         horizontal_spacing=0.12)
    fig2.add_trace(go.Bar(
        x=groups2, y=g2["avg_buzz_lift"],
        marker_color=bar_colors2,
        text=g2["avg_buzz_lift"].astype(str) + "%",
        textposition="outside", showlegend=False,
    ), row=1, col=1)
    fig2.add_hline(y=100, line_dash="dot", line_color="gray",
                   opacity=0.5, row=1, col=1)
    fig2.add_trace(go.Bar(
        x=groups2, y=g2["composite_score"],
        marker_color=bar_colors2,
        text=g2["composite_score"].astype(str) + "점",
        textposition="outside", showlegend=False,
    ), row=1, col=2)
    # 75%+ 강조 박스
    for col_n in [1, 2]:
        fig2.add_vrect(x0=2.5, x1=3.5, fillcolor="#185FA520",
                       layer="below", line_width=0, row=1, col=col_n)
    fig2.update_layout(
        height=340, plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=20),
    )
    fig2.update_yaxes(gridcolor="rgba(200,200,200,0.3)")
    st.plotly_chart(fig2, use_container_width=True)
    with st.expander('🗄️ 관련 SQL — 외지인 비율 구간별 SNS·효과성 집계'):
        st.code("""\
-- 외지인 비율을 4개 구간으로 나눠 SNS 증폭률·종합 점수 비교
SELECT
    CASE
        WHEN outsider_ratio < 40  THEN '~40%'
        WHEN outsider_ratio < 60  THEN '40~60%'
        WHEN outsider_ratio < 75  THEN '60~75%'
        ELSE                           '75%+'
    END AS outsider_group,
    COUNT(*)                                               AS n,
    ROUND(AVG(avg_buzz_lift),     1)                      AS avg_sns_lift,
    ROUND(AVG(avg_spend_lift),    1)                      AS avg_spend_lift,
    ROUND(AVG(composite_score),   1)                      AS avg_score
FROM (
    -- 축제별 3개년 평균 외지인 비율
    SELECT festival_name,
           AVG(CAST(외지인방문자수 AS FLOAT) / NULLIF(전체방문자수,0) * 100) AS outsider_ratio
    FROM fact_visitor
    GROUP BY festival_name
) v
JOIN fact_composite_score s USING (festival_name)
GROUP BY outsider_group
ORDER BY outsider_group;
-- → 75%+ 구간이 SNS 155.8% — 다른 구간 대비 1.5배
""", language='sql')
    st.divider()

    # ══════════════════════════════════════════
    # 인사이트 3 — 유형별 소비 증폭·유지율 그룹 막대
    # ══════════════════════════════════════════
    ins_header("3", "소비 유지율이 진짜 지역 경제 활성화의 지표다",
               "소비 지속성", "#0F6E56",
               "Type B의 소비 유지율 **114.7%**로 기준선 이상. 반면 Type D는 **90.9%**로 오히려 감소합니다. "
               "방문자 수가 많아도 유지율이 낮으면 소비 유출(leakage)이 발생하는 신호입니다.")

    g3 = score_df.groupby("cluster_label").agg(
        avg_spend_lift=("avg_spend_lift", "mean"),
        avg_retention_rate=("avg_retention_rate", "mean"),
    ).reindex(TYPE_ORDER).round(1).reset_index()

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        name="소비 증폭률 (축제 기간)",
        x=[TYPE_SHORT[t] for t in g3["cluster_label"]],
        y=g3["avg_spend_lift"],
        marker_color=TYPE_COLOR_LIST,
        opacity=0.55,
        text=g3["avg_spend_lift"].round(1).astype(str) + "%",
        textposition="inside", textfont=dict(color="white", size=11),
    ))
    fig3.add_trace(go.Bar(
        name="소비 유지율 (After 평균)",
        x=[TYPE_SHORT[t] for t in g3["cluster_label"]],
        y=g3["avg_retention_rate"],
        marker_color=TYPE_COLOR_LIST,
        opacity=1.0,
        text=g3["avg_retention_rate"].round(1).astype(str) + "%",
        textposition="inside", textfont=dict(color="white", size=11),
    ))
    fig3.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.5,
                   annotation_text="기준선 100%", annotation_position="right")
    fig3.update_layout(
        barmode="group",
        height=360,
        yaxis=dict(title="BEFORE 기준 대비 비율 (%)", range=[70, 165],
                   gridcolor="rgba(200,200,200,0.3)"),
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.08),
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig3, use_container_width=True)
    with st.expander('🗄️ 관련 SQL — 유형별 소비 증폭률 vs 유지율'):
        st.code("""\
-- 유형별 소비 증폭률(축제 기간)과 유지율(사후) 비교
WITH yearly AS (
    SELECT festival_name, festival_year,
           AVG(CASE WHEN period LIKE 'BEFORE%' THEN spending_million END) AS before_avg,
           AVG(CASE WHEN period = 'FESTIVAL'   THEN spending_million END) AS festival_avg,
           AVG(CASE WHEN period LIKE 'AFTER%'  THEN spending_million END) AS after_avg
    FROM fact_spending
    GROUP BY festival_name, festival_year
),
fest_score AS (
    SELECT festival_name, cluster_label,
           ROUND(AVG(festival_avg / NULLIF(before_avg,0) * 100), 1) AS avg_spend_lift,
           ROUND(AVG(after_avg    / NULLIF(before_avg,0) * 100), 1) AS avg_retention_rate
    FROM yearly GROUP BY festival_name
)
SELECT
    s.cluster_label,
    ROUND(AVG(f.avg_spend_lift),     1) AS avg_spend_lift,
    ROUND(AVG(f.avg_retention_rate), 1) AS avg_retention_rate
FROM fest_score f
JOIN fact_composite_score s USING (festival_name)
GROUP BY s.cluster_label
ORDER BY avg_retention_rate DESC;
-- → Type B 유지율 114.7% vs Type D 90.9%
""", language='sql')
    st.divider()

    # ══════════════════════════════════════════
    # 인사이트 4 — SNS 장기 vs 소비 장기 산점도
    # ══════════════════════════════════════════
    ins_header("4", "SNS 장기 유지율 하락이 소비 급락을 예고 — r = 0.59",
               "SNS 지속성", "#0F6E56",
               "SNS 6개월 유지율과 소비 6개월 유지율의 상관계수 **r = 0.59**. "
               "Type D의 SNS 장기율 84.9%, 소비 87.4%가 동반 하락합니다. "
               "SNS 콘텐츠 지속 생산 지원이 장기 경제 효과의 보조 수단입니다.")

    fig4 = go.Figure()
    for lbl, clr in zip(TYPE_ORDER, TYPE_COLOR_LIST):
        sub = ret_fest[ret_fest["cluster_label"] == lbl]
        fig4.add_trace(go.Scatter(
            x=sub["sns_long"], y=sub["sp_long"],
            mode="markers+text", name=TYPE_SHORT[lbl],
            marker=dict(color=clr, size=10, opacity=0.85,
                        line=dict(color="white", width=1)),
            text=sub["festival_name"], textposition="top center",
            textfont=dict(size=9),
        ))
    # 추세선
    x4 = ret_fest["sns_long"].values
    y4 = ret_fest["sp_long"].values
    m4, b4 = np.polyfit(x4, y4, 1)
    x4_line = np.linspace(x4.min(), x4.max(), 50)
    fig4.add_trace(go.Scatter(
        x=x4_line, y=m4 * x4_line + b4,
        mode="lines", name="추세선 (r=0.59)",
        line=dict(color="gray", width=1.5, dash="dot"),
    ))
    fig4.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.4)
    fig4.add_vline(x=100, line_dash="dot", line_color="gray", opacity=0.4)
    fig4.update_layout(
        height=400,
        xaxis=dict(title="SNS 6개월 유지율 (%)", gridcolor="rgba(200,200,200,0.3)"),
        yaxis=dict(title="소비 6개월 유지율 (%)", gridcolor="rgba(200,200,200,0.3)"),
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(t=20, b=30),
    )
    st.plotly_chart(fig4, use_container_width=True)
    with st.expander('🗄️ 관련 SQL — SNS·소비 6개월 유지율 (장기 지속성)'):
        st.code("""\
-- AFTER_6M 기준 SNS·소비 장기 유지율 산출 (fact_retention_v2 기반)
SELECT
    r.festival_name,
    f.cluster_label,
    ROUND(AVG(r.sns_long_rate), 1) AS sns_6m_rate,   -- SNS 6개월 유지율
    ROUND(AVG(r.sp_long_rate),  1) AS spend_6m_rate  -- 소비 6개월 유지율
FROM fact_retention_v2 r
JOIN fact_composite_score f USING (festival_name)
GROUP BY r.festival_name, f.cluster_label
ORDER BY sns_6m_rate DESC;
-- fact_retention_v2 생성 핵심 로직:
-- short_avg = AVG(AFTER_1M ~ AFTER_3M)
-- long_avg  = AVG(AFTER_6M)            ← 여기서 사용
-- 상관계수 r=0.59: SNS 장기율이 낮을수록 소비 장기율도 낮다
""", language='sql')
    st.divider()

    # ══════════════════════════════════════════
    # 인사이트 5 — 축제 기간 구간별 꺾은선+막대
    # ══════════════════════════════════════════
    ins_header("5", "축제 기간 길이는 효과성과 통계적으로 무관하다",
               "축제 기간", "#BA7517",
               "ANOVA 결과 소비(p=0.22), SNS(p=0.25) 모두 구간 간 차이가 유의하지 않습니다. "
               "③ 장기(10~15일) 수치가 높아 보이지만 곡성세계장미축제 단독 효과가 크게 작용합니다. "
               "**기간보다 외지인 유입 전략과 콘텐츠 특성이 효과성의 실질 결정 요인입니다.**")

    g5 = vis_merged.groupby("days_group", observed=True)[
        ["avg_spend_lift", "avg_buzz_lift", "avg_visitor_lift"]
    ].mean().round(1).reset_index()
    g5_labels = g5["days_group"].astype(str).tolist()
    bar_colors5 = ["#D4C9A8", "#A8C4E0", "#0F6E56", "#D4C9A8"]

    fig5 = make_subplots(specs=[[{"secondary_y": True}]])
    fig5.add_trace(go.Bar(
        x=g5_labels, y=g5["avg_spend_lift"],
        name="소비 증폭률",
        marker_color=bar_colors5,
        text=g5["avg_spend_lift"].astype(str) + "%",
        textposition="outside",
    ), secondary_y=False)
    fig5.add_trace(go.Scatter(
        x=g5_labels, y=g5["avg_buzz_lift"],
        name="SNS 증폭률",
        mode="lines+markers",
        line=dict(color="#185FA5", width=2, dash="dash"),
        marker=dict(size=8),
    ), secondary_y=True)
    fig5.add_trace(go.Scatter(
        x=g5_labels, y=g5["avg_visitor_lift"],
        name="방문자 증폭률",
        mode="lines+markers",
        line=dict(color="#993C1D", width=2),
        marker=dict(size=8),
    ), secondary_y=True)
    # p-value 주석
    fig5.add_annotation(
        text="ANOVA: 소비 p=0.22 / SNS p=0.25<br>→ 통계적으로 유의하지 않음",
        xref="paper", yref="paper", x=0.98, y=0.98,
        showarrow=False, align="right",
        font=dict(size=10, color="#BA7517"),
        bgcolor="rgba(250,238,218,0.85)",
        bordercolor="#BA7517", borderwidth=0.5,
    )
    fig5.update_layout(
        height=380, plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.08),
        margin=dict(t=20, b=20),
    )
    fig5.update_yaxes(title_text="소비 증폭률 (%)",   secondary_y=False,
                      gridcolor="rgba(200,200,200,0.3)", range=[90, 145])
    fig5.update_yaxes(title_text="SNS·방문 증폭률 (%)", secondary_y=True, range=[90, 145])
    st.plotly_chart(fig5, use_container_width=True)
    with st.expander('🗄️ 관련 SQL — 축제 기간 구간별 효과성'):
        st.code("""\
-- festival_days를 4개 구간으로 나눠 소비·SNS·방문자 증폭률 비교
WITH before_avg AS (
    SELECT sp.festival_name, sp.festival_year,
           AVG(sp.spending_million) AS spend_base,
           AVG(sn.search_volume)    AS sns_base,
           AVG(vt.전체방문자수)      AS visit_base
    FROM fact_spending   sp
    JOIN fact_sns        sn ON sp.festival_name=sn.festival_name
                            AND sp.festival_year=sn.festival_year
                            AND sp.period=sn.period
    JOIN fact_visitor_ts vt ON sp.festival_name=vt.festival_name
                            AND sp.festival_year=vt.festival_year
                            AND sp.period=vt.period
    WHERE sp.period LIKE 'BEFORE%'
    GROUP BY sp.festival_name, sp.festival_year
)
SELECT
    CASE
        WHEN v.festival_days <= 4  THEN '① 단기 (1~4일)'
        WHEN v.festival_days <= 9  THEN '② 중기 (5~9일)'
        WHEN v.festival_days <= 15 THEN '③ 장기 (10~15일)'
        ELSE                            '④ 초장기 (16일+)'
    END AS days_group,
    COUNT(*)                                                            AS n,
    ROUND(AVG((fv.spend_fest-b.spend_base)/NULLIF(b.spend_base,0)*100),1) AS avg_spend_lift,
    ROUND(AVG((fv.sns_fest  -b.sns_base)  /NULLIF(b.sns_base,0)*100),  1) AS avg_sns_lift
FROM fact_visitor v
JOIN before_avg b   ON v.festival_name=b.festival_name AND v.festival_year=b.festival_year
JOIN festival_val fv ON v.festival_name=fv.festival_name AND v.festival_year=fv.festival_year
GROUP BY days_group;
-- ANOVA: 소비 F=1.506 p=0.220 / SNS F=1.408 p=0.247 → 모두 비유의
""", language='sql')
    st.divider()

    # ══════════════════════════════════════════
    # 인사이트 6 — 유형별 현황 스택 막대 + 점수 꺾은선
    # ══════════════════════════════════════════
    ins_header("6", "27개 중 14개(52%)가 효과 미흡 — 평가 체계 전환 필요",
               "정책 제언", "#993C1D",
               "현행 방문자 수 단일 평가로는 이 문제가 보이지 않습니다. "
               "Type D는 소비 유지율 평균 **90.9%**, SNS 기준선 미달 축제 6/14개입니다. "
               "**소비 유지율·외지인 비율을 공식 평가 지표에 포함하는 체계 전환이 필요합니다.**")

    g6 = score_df.groupby("cluster_label").agg(
        n=("festival_name", "count"),
        avg_retention=("avg_retention_rate", "mean"),
        avg_score=("composite_score", "mean"),
    ).reindex(TYPE_ORDER).round(1).reset_index()
    type_labels6 = [TYPE_SHORT[t] for t in g6["cluster_label"]]

    fig6 = make_subplots(rows=1, cols=2,
                         subplot_titles=["유형별 축제 수 및 소비 유지율",
                                         "유형별 평균 종합 효과성 점수"],
                         horizontal_spacing=0.14)
    # 왼쪽: 축제 수 막대 + 소비유지율 꺾은선
    fig6.add_trace(go.Bar(
        x=type_labels6, y=g6["n"],
        marker_color=TYPE_COLOR_LIST, opacity=0.7,
        name="축제 수",
        text=g6["n"].astype(str) + "개", textposition="inside",
        textfont=dict(color="white", size=12),
    ), row=1, col=1)
    # 오른쪽: 종합 점수 막대
    fig6.add_trace(go.Bar(
        x=type_labels6, y=g6["avg_score"],
        marker_color=TYPE_COLOR_LIST,
        name="종합 점수",
        text=g6["avg_score"].astype(str) + "점",
        textposition="outside",
        showlegend=False,
    ), row=1, col=2)
    # 소비 유지율 꺾은선 (오른쪽 y축 역할로 annotation 사용)
    for i, (lbl, ret_val) in enumerate(zip(type_labels6, g6["avg_retention"])):
        fig6.add_annotation(
            x=lbl, y=g6["n"].iloc[i] + 0.4,
            text=f"유지율<br>{ret_val:.0f}%",
            showarrow=False,
            font=dict(size=10, color=TYPE_COLOR_LIST[i]),
            row=1, col=1,
        )
    # Type D 강조
    fig6.add_vrect(x0=2.5, x1=3.5, fillcolor="#99321D15",
                   layer="below", line_width=0, row=1, col=1)
    fig6.add_vrect(x0=2.5, x1=3.5, fillcolor="#99321D15",
                   layer="below", line_width=0, row=1, col=2)
    fig6.update_layout(
        height=360, plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, margin=dict(t=40, b=20),
    )
    fig6.update_yaxes(gridcolor="rgba(200,200,200,0.3)")
    fig6.update_yaxes(title_text="축제 수 (개)", row=1, col=1)
    fig6.update_yaxes(title_text="종합 점수", row=1, col=2)
    st.plotly_chart(fig6, use_container_width=True)
    with st.expander('🗄️ 관련 SQL — 유형별 축제 수·소비 유지율·종합 점수'):
        st.code("""\
-- 유형별 현황 집계 — 효과 미흡 규모와 점수 격차 확인
SELECT
    cluster_label,
    COUNT(*)                                    AS festival_count,
    ROUND(AVG(composite_score),      1)         AS avg_score,
    ROUND(AVG(avg_retention_rate),   1)         AS avg_retention_rate,
    SUM(CASE WHEN avg_buzz_lift < 100 THEN 1
             ELSE 0 END)                        AS sns_below_baseline,
    ROUND(AVG(avg_outsider_ratio),   1)         AS avg_outsider_ratio
FROM fact_composite_score
GROUP BY cluster_label
ORDER BY avg_score DESC;
-- → Type D: 14개(52%), 유지율 90.9%, SNS 미달 6개
-- 방문자 수 단일 지표로는 이 차이가 보이지 않음
""", language='sql')
    st.divider()

    # ── 유형별 전략 방향 ──────────────────────────────────
    st.markdown("## 유형별 정책 전략 방향")
    STRATEGIES = [
        ("Type A — 전방위 우수형",   "#534AB7", "#EEEDFE", 1,  "86.9점",
         "3개 차원 모두 압도적. 소비·SNS·방문자 기준선 대비 전부 110% 이상.",
         ["성공 패턴 백서화 및 타 축제 벤치마킹 지원",
          "축제 전후 연계 체험 프로그램 확대로 체류 기간 연장",
          "브랜드 IP화 — 지역 상품·콘텐츠 라이선싱 연계",
          "6개월 후 방문자 추적 인터뷰 정례화"]),
        ("Type B — 경제·방문 균형형","#185FA5", "#E6F1FB", 7,  "55.0점",
         "소비 유지율 114%, SNS 유지율 126%. 축제 후에도 사람·돈이 머문다.",
         ["축제 직후 소규모 후속 이벤트로 단기 공백 해소",
          "재방문 쿠폰·지역화폐 연계로 재방문 전환",
          "지역 특산물 온라인 재구매 채널 연계",
          "계절 체험 프로그램 다변화"]),
        ("Type C — SNS 주도형",      "#0F6E56", "#E1F5EE", 5,  "54.1점",
         "SNS 증폭 136%로 강하나 소비 유지율 104%. 화제성이 소비로 연결되지 않는 구조.",
         ["당일치기 방문객 → 1박 이상 체류 전환 숙박 패키지",
          "축제 후 3개월 SNS 콘텐츠 지속 생산 지원",
          "지역 특산물 직거래 부스 및 온라인 사후 구매 연동",
          "재방문 유인 쿠폰·지역화폐 연계"]),
        ("Type D — 효과 미흡형",     "#993C1D", "#FAECE7", 14, "29.7점",
         "SNS 기준선 미달 12/14개. 소비 유지율 91%. 일회성 이벤트 패턴.",
         ["외지인 유치 전략 부재 점검 — 접근성·홍보 채널 분석",
          "소비 유출 경로 진단 (외부 체인 비중 조사)",
          "규모 축소 + 콘텐츠 특화로 비용 효율 개선",
          "3년 이내 성과 미달 시 형식 전환 또는 통폐합 검토"]),
    ]
    cols_s = st.columns(2)
    for idx, (label, color, bg, n, score, desc, items) in enumerate(STRATEGIES):
        with cols_s[idx % 2]:
            bullet_html = "".join(
                [f'<div style="font-size:12px;color:#444;padding:2px 0">— {it}</div>' for it in items]
            )
            st.markdown(
                f'<div style="background:{bg};border:0.5px solid {color}40;border-radius:10px;'
                f'padding:16px 18px;margin-bottom:12px">'
                f'<div style="font-size:10px;color:{color};font-weight:500;text-transform:uppercase;'
                f'letter-spacing:.06em;margin-bottom:4px">{TYPE_SHORT[label]} · n={n} · 평균 {score}</div>'
                f'<div style="font-size:14px;font-weight:700;color:{color};margin-bottom:6px">'
                f'{label.split("—")[1].strip()}</div>'
                f'<div style="font-size:11px;color:#555;margin-bottom:10px">{desc}</div>'
                f'{bullet_html}</div>',
                unsafe_allow_html=True,
            )



# ════════════════════════════════════════════════════════
# TAB 6: 핵심 SQL
# ════════════════════════════════════════════════════════
elif tab_sel == "🗄️ 핵심 SQL":
    st.markdown("## 🗄️ 핵심 SQL 쿼리")
    st.caption("festival_raw.db 기반 · 분석에 사용된 핵심 SQL문 정리")

    st.markdown(
        '<div style="background:#F5F3EE;border:0.5px solid #E2DFD8;border-radius:8px;'
        'padding:12px 16px;margin-bottom:20px"><b>공통 설계 원칙</b><br>'
        '<span style="font-size:12px;color:#555;line-height:2">'
        "① <b>BEFORE% 패턴 매칭</b> — LIKE 'BEFORE%'로 3M·2M·1M을 한 번에 묶어 기준선 산출<br>"
        '② <b>NULLIF(x, 0)</b> — 0 나누기 방지, 데이터 없는 축제 점수 왜곡 차단<br>'
        '③ <b>AVG not SUM</b> — 기간·규모가 다른 축제 간 비교 왜곡 방지'
        '</span></div>',
        unsafe_allow_html=True,
    )

    SQL_SECTIONS = [
        ("SQL 1 — SNS 언급량 구간 평균 및 증폭률",
         "축제×연도별 BEFORE 기준선 대비 FESTIVAL 증폭률과 AFTER 유지율 산출",
         """\
WITH base AS (
    SELECT
        festival_name,
        festival_year,
        AVG(CASE WHEN period LIKE 'BEFORE%' THEN search_volume END) AS before_avg,
        AVG(CASE WHEN period = 'FESTIVAL'   THEN search_volume END) AS festival_avg,
        AVG(CASE WHEN period LIKE 'AFTER%'  THEN search_volume END) AS after_avg
    FROM fact_sns
    GROUP BY festival_name, festival_year
)
SELECT
    festival_name,
    festival_year,
    ROUND(festival_avg / NULLIF(before_avg, 0) * 100, 1) AS buzz_lift_pct,
    ROUND(after_avg    / NULLIF(before_avg, 0) * 100, 1) AS buzz_retention_pct
FROM base
ORDER BY buzz_lift_pct DESC;"""),
        ("SQL 2 — 축제별 3개년 평균 SNS 증폭률 순위",
         "연도 노이즈를 제거한 축제 단위 대표 SNS 지표",
         """\
WITH base AS (
    SELECT festival_name, festival_year,
           AVG(CASE WHEN period LIKE 'BEFORE%' THEN search_volume END) AS before_avg,
           AVG(CASE WHEN period = 'FESTIVAL'   THEN search_volume END) AS festival_avg,
           AVG(CASE WHEN period LIKE 'AFTER%'  THEN search_volume END) AS after_avg
    FROM fact_sns
    GROUP BY festival_name, festival_year
)
SELECT
    festival_name,
    COUNT(festival_year)                                        AS years,
    ROUND(AVG(festival_avg / NULLIF(before_avg, 0) * 100), 1) AS avg_buzz_lift_pct,
    ROUND(AVG(after_avg    / NULLIF(before_avg, 0) * 100), 1) AS avg_buzz_retention_pct,
    ROUND(AVG(festival_avg), 0)                                AS avg_festival_volume
FROM base
GROUP BY festival_name
ORDER BY avg_buzz_lift_pct DESC;"""),
        ("SQL 3 — 소비 증폭률 및 유지율 (3개년 평균)",
         "축제 기간 소비 증폭과 사후 소비 유지율 — 경제적 효과의 핵심 지표",
         """\
WITH yearly AS (
    SELECT festival_name, festival_year,
           AVG(CASE WHEN period LIKE 'BEFORE%' THEN spending_million END) AS before_avg,
           AVG(CASE WHEN period = 'FESTIVAL'   THEN spending_million END) AS festival_avg,
           AVG(CASE WHEN period LIKE 'AFTER%'  THEN spending_million END) AS after_avg
    FROM fact_spending
    GROUP BY festival_name, festival_year
)
SELECT
    festival_name,
    ROUND(AVG(festival_avg / NULLIF(before_avg, 0) * 100), 1) AS avg_spend_lift_pct,
    ROUND(AVG(after_avg    / NULLIF(before_avg, 0) * 100), 1) AS avg_retention_rate,
    ROUND(AVG(festival_avg - before_avg), 0)                   AS avg_incremental_spend
FROM yearly
GROUP BY festival_name
ORDER BY avg_spend_lift_pct DESC;"""),
        ("SQL 4 — 방문자 유입 지표 (외지인 비율·인구대비·증폭률)",
         "외지인 비율, 인구 대비 방문자 규모, 방문자 증폭률 — 방문 차원 3개 지표",
         """\
WITH visitor_ts_base AS (
    SELECT festival_name, festival_year,
           AVG(CASE WHEN period LIKE 'BEFORE%' THEN 전체방문자수 END) AS before_visitor_avg,
           AVG(CASE WHEN period = 'FESTIVAL'   THEN 전체방문자수 END) AS festival_visitor
    FROM fact_visitor_ts
    GROUP BY festival_name, festival_year
)
SELECT
    v.festival_name,
    v.festival_year,
    ROUND(CAST(v.외지인방문자수 AS FLOAT)
          / NULLIF(v.전체방문자수, 0) * 100, 1)         AS outsider_ratio,
    ROUND(CAST(v.전체방문자수 AS FLOAT)
          / NULLIF(p.population, 0) * 100, 1)           AS visitor_per_pop_pct,
    ROUND(vt.festival_visitor
          / NULLIF(vt.before_visitor_avg, 0) * 100, 1)  AS visitor_lift_pct
FROM fact_visitor v
LEFT JOIN dim_population   p  ON v.sigungu = p.sigungu
LEFT JOIN visitor_ts_base vt  ON v.festival_name = vt.festival_name
                              AND v.festival_year  = vt.festival_year
ORDER BY visitor_lift_pct DESC;"""),
        ("SQL 5 — 3개 차원 통합 종합 점수 산출",
         "SNS·소비·방문자 3개 차원 지표 통합 (Min-Max 정규화 및 가중합은 Python에서 처리)",
         """\
WITH sns_agg AS (
    SELECT festival_name,
           AVG(festival_avg / NULLIF(before_avg,0) * 100) AS avg_buzz_lift,
           AVG(after_avg    / NULLIF(before_avg,0) * 100) AS avg_buzz_retention
    FROM (
        SELECT festival_name, festival_year,
               AVG(CASE WHEN period LIKE 'BEFORE%' THEN search_volume END)  AS before_avg,
               AVG(CASE WHEN period = 'FESTIVAL'   THEN search_volume END)  AS festival_avg,
               AVG(CASE WHEN period LIKE 'AFTER%'  THEN search_volume END)  AS after_avg
        FROM fact_sns GROUP BY festival_name, festival_year
    ) GROUP BY festival_name
),
spend_agg AS (
    SELECT festival_name,
           AVG(festival_avg / NULLIF(before_avg,0) * 100) AS avg_spend_lift,
           AVG(after_avg    / NULLIF(before_avg,0) * 100) AS avg_retention_rate
    FROM (
        SELECT festival_name, festival_year,
               AVG(CASE WHEN period LIKE 'BEFORE%' THEN spending_million END) AS before_avg,
               AVG(CASE WHEN period = 'FESTIVAL'   THEN spending_million END) AS festival_avg,
               AVG(CASE WHEN period LIKE 'AFTER%'  THEN spending_million END) AS after_avg
        FROM fact_spending GROUP BY festival_name, festival_year
    ) GROUP BY festival_name
)
-- 가중치: 소비(0.45) · SNS(0.30) · 방문자(0.25)
-- Min-Max 정규화 후 가중합 × 100 = 0~100점
SELECT sp.festival_name,
       sn.avg_buzz_lift,       -- w=0.20
       sn.avg_buzz_retention,  -- w=0.10
       sp.avg_spend_lift,      -- w=0.25 (핵심)
       sp.avg_retention_rate   -- w=0.20 (핵심)
FROM spend_agg sp
JOIN sns_agg sn ON sp.festival_name = sn.festival_name;"""),
        ("SQL 6 — 단기·장기 지속성 분리 (fact_retention_v2)",
         "AFTER 1~3M = 단기 평균, AFTER_6M = 장기 — 소비·SNS·방문자 3개 차원 모두 적용",
         """\
CREATE TABLE fact_retention_v2 AS
WITH spend_base AS (
    SELECT festival_name, festival_year,
        AVG(CASE WHEN period LIKE 'BEFORE%'                       THEN spending_million END) AS before_avg,
        AVG(CASE WHEN period = 'FESTIVAL'                         THEN spending_million END) AS festival_avg,
        AVG(CASE WHEN period IN ('AFTER_1M','AFTER_2M','AFTER_3M')THEN spending_million END) AS short_avg,
        AVG(CASE WHEN period = 'AFTER_6M'                        THEN spending_million END) AS long_avg
    FROM fact_spending GROUP BY festival_name, festival_year
)
-- sns_base, visit_base: 동일 구조 (컬럼명만 변경)
SELECT
    sp.festival_name, sp.festival_year,
    ROUND(sp.festival_avg / NULLIF(sp.before_avg,0)*100, 1) AS sp_lift,
    ROUND(sp.short_avg    / NULLIF(sp.before_avg,0)*100, 1) AS sp_short_rate,  -- 단기
    ROUND(sp.long_avg     / NULLIF(sp.before_avg,0)*100, 1) AS sp_long_rate    -- 장기(6M)
FROM spend_base sp;"""),
        ("SQL 7 — 축제 기간 구간별 효과성 (festival_raw.db)",
         "festival_days를 4개 구간으로 나눠 증폭률 비교. ANOVA p>0.05 — 기간보다 전략이 결정적",
         """\
WITH before_avg AS (
    SELECT sp.festival_name, sp.festival_year,
           AVG(sp.spending_million) AS spend_base,
           AVG(sn.search_volume)    AS sns_base,
           AVG(vt.전체방문자수)      AS visit_base
    FROM fact_spending   sp
    JOIN fact_sns        sn ON sp.festival_name=sn.festival_name
                            AND sp.festival_year=sn.festival_year
                            AND sp.period=sn.period
    JOIN fact_visitor_ts vt ON sp.festival_name=vt.festival_name
                            AND sp.festival_year=vt.festival_year
                            AND sp.period=vt.period
    WHERE sp.period LIKE 'BEFORE%'
    GROUP BY sp.festival_name, sp.festival_year
)
-- festival_val: period='FESTIVAL' / after_avg: period IN ('AFTER_1M','AFTER_2M','AFTER_3M')
SELECT
    v.festival_name,
    v.festival_days,
    CASE
        WHEN v.festival_days <= 4  THEN '① 단기 (1~4일)'
        WHEN v.festival_days <= 9  THEN '② 중기 (5~9일)'
        WHEN v.festival_days <= 15 THEN '③ 장기 (10~15일)'
        ELSE                            '④ 초장기 (16일+)'
    END AS days_group,
    ROUND((fv.spend_fest - b.spend_base)/NULLIF(b.spend_base,0)*100, 1) AS spend_lift,
    ROUND((fv.sns_fest   - b.sns_base)  /NULLIF(b.sns_base,0)*100,   1) AS sns_lift,
    ROUND((fv.visit_fest - b.visit_base)/NULLIF(b.visit_base,0)*100, 1) AS visitor_lift
FROM fact_visitor v
JOIN before_avg  b   ON v.festival_name=b.festival_name AND v.festival_year=b.festival_year
JOIN festival_val fv  ON v.festival_name=fv.festival_name AND v.festival_year=fv.festival_year
JOIN after_avg   aa  ON v.festival_name=aa.festival_name AND v.festival_year=aa.festival_year
ORDER BY v.festival_days;
-- ANOVA: 소비 F=1.506 p=0.220 / SNS F=1.408 p=0.247 → 모두 비유의"""),
    ]

    for title, desc, sql in SQL_SECTIONS:
        with st.expander(f"**{title}**", expanded=False):
            st.caption(desc)
            st.code(sql, language="sql")



# ────────────────────────────────────────────────────────
# 푸터
# ────────────────────────────────────────────────────────
st.divider()
st.caption(
    "분석 기준: 인구감소지역 27개 축제 · 2022–2025 · 3개년 평균 지표 기반 "
    "| 방법론: Min-Max 정규화 + 가중합 → K-Means(K=4) + 다항 로지스틱 회귀(LOO-CV 88.9%)"
)
