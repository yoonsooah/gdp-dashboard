# app.py
import streamlit as st
import pandas as pd
import numpy as np
import math
from pathlib import Path
import altair as alt
import plotly.express as px
import pycountry
import re

st.set_page_config(
    page_title="🌍 GDP Dashboard (완전판)",
    page_icon="🌍",
    layout="wide",
)

# --- Pretendard Bold 적용 CSS ---
st.markdown(
    """
    <style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

    html, body, .stApp {
        font-family: 'Pretendard', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Pretendard', sans-serif;
        font-weight: 800;
    }
    .stMetric {
        font-family: 'Pretendard', sans-serif;
        font-weight: 700;
    }

    /* 다크모드 지원 */
    @media (prefers-color-scheme: dark) {
        html, body, .stApp { color: #f3f4f6; background-color: #0b1220; }
        h1, h2, h3 { color: #f3f4f6; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌍 완전판 GDP 대시보드")
st.markdown("World Bank 데이터를 활용해 `GDP`, `Population`, `GDP per capita`를 보여줍니다. (1960–2022)")

# -------------------------
# 유틸: 연도 컬럼 자동 감지 및 melt 함수
# -------------------------
def detect_year_columns(df):
    return sorted([c for c in df.columns if re.match(r"^\d{4}$", str(c))], key=int)

def melt_wb_wide(df, value_name):
    if 'Country Name' not in df.columns or 'Country Code' not in df.columns:
        raise ValueError("CSV 파일에 'Country Name' 및 'Country Code' 컬럼 필요")
    year_cols = detect_year_columns(df)
    melted = df.melt(id_vars=['Country Name', 'Country Code'], value_vars=year_cols,
                     var_name='Year', value_name=value_name)
    melted['Year'] = pd.to_numeric(melted['Year'], errors='coerce')
    melted[value_name] = pd.to_numeric(melted[value_name], errors='coerce')
    return melted

@st.cache_data
def load_wb_csv(path: Path, value_name: str):
    if not path.exists():
        return None, f"파일을 찾을 수 없습니다: {path}"
    try:
        raw = pd.read_csv(path)
        df = melt_wb_wide(raw, value_name)
    except Exception as e:
        return None, str(e)
    return df, None

DATA_DIR = Path(__file__).parent / "data"
GDP_FILE = DATA_DIR / "gdp_data.csv"
POP_FILE = DATA_DIR / "population_data.csv"

gdp_df, err1 = load_wb_csv(GDP_FILE, "GDP")
pop_df, err2 = load_wb_csv(POP_FILE, "Population")

if gdp_df is None or pop_df is None:
    st.error("데이터 로딩 오류")
    if err1: st.write(f"- GDP 파일 오류: {err1}")
    if err2: st.write(f"- Population 파일 오류: {err2}")
    st.stop()

# -------------------------
# 데이터 병합
# -------------------------
merged = pd.merge(
    gdp_df, pop_df, on=['Country Name','Country Code','Year'], how='outer'
)
merged['GDP per capita'] = merged['GDP'] / merged['Population']
merged.replace([np.inf, -np.inf], np.nan, inplace=True)
merged['Year'] = merged['Year'].astype('Int64')

# -------------------------
# 국가 리스트 & ISO3 ↔ 이름
# -------------------------
country_df = merged[['Country Name','Country Code']].drop_duplicates().sort_values('Country Name')
country_name_to_iso3 = dict(zip(country_df['Country Name'], country_df['Country Code']))

def iso3_to_flag(iso3):
    try:
        if pd.isna(iso3): return ""
        c = pycountry.countries.get(alpha_3=iso3.upper())
        if not c: return ""
        return ''.join(chr(ord(ch)+127397) for ch in c.alpha_2.upper())
    except: return ""

# -------------------------
# 사이드바 필터
# -------------------------
st.sidebar.header("🔎 필터")
min_year, max_year = int(merged['Year'].min()), int(merged['Year'].max())
year_range = st.sidebar.slider("연도 범위 선택", min_value=min_year, max_value=max_year, value=(2000, max_year))

all_country_names = country_df['Country Name'].tolist()
default_selection = all_country_names[:6] if len(all_country_names)>=6 else all_country_names

def format_country(name):
    iso3 = country_name_to_iso3.get(name,"")
    flag = iso3_to_flag(iso3)
    return f"{flag} {name}" if flag else name

selected_names = st.sidebar.multiselect("국가 선택", options=all_country_names, default=default_selection, format_func=format_country)
if not selected_names: st.stop()
from_year, to_year = year_range
selected_iso3 = [country_name_to_iso3[n] for n in selected_names]

df_filtered = merged[(merged['Country Code'].isin(selected_iso3)) & (merged['Year'].between(from_year,to_year))]

# -------------------------
# GDP 시계열
# -------------------------
st.subheader("📈 GDP (Total) - 연도별 변화")
gdp_chart = alt.Chart(df_filtered).mark_line(point=True).encode(
    x='Year:Q',
    y='GDP:Q',
    color='Country Name:N',
    tooltip=['Country Name:N','Year:Q','GDP:Q']
).interactive().properties(height=420)
st.altair_chart(gdp_chart, use_container_width=True)

# -------------------------
# GDP per capita 시계열
# -------------------------
st.subheader("👤 GDP per Capita - 연도별 변화")
percap_chart = alt.Chart(df_filtered).mark_line(point=True).encode(
    x='Year:Q',
    y='GDP per capita:Q',
    color='Country Name:N',
    tooltip=['Country Name:N','Year:Q','GDP per capita:Q']
).interactive().properties(height=420)
st.altair_chart(percap_chart, use_container_width=True)

# -------------------------
# 연도별 Metric
# -------------------------
st.subheader(f"📊 {to_year}년 국가별 지표 비교")
year_df = merged[(merged['Year']==to_year)&(merged['Country Code'].isin(selected_iso3))].sort_values('GDP',ascending=False)
metrics_cols = st.columns(min(4,len(year_df)))
for i,(_,row) in enumerate(year_df.iterrows()):
    col = metrics_cols[i%len(metrics_cols)]
    with col:
        iso3 = row['Country Code']
        flag = iso3_to_flag(iso3)
        gdp_val = row['GDP']
        percap = row['GDP per capita']
        st.metric(label=f"{flag} {row['Country Name']}",
                  value=f"{gdp_val/1e9:,.1f} B USD" if not pd.isna(gdp_val) else "n/a",
                  delta=f"${percap:,.0f}" if not pd.isna(percap) else "n/a")

# -------------------------
# GDP per capita 막대차트
# -------------------------
st.subheader(f"📊 {to_year}년 1인당 GDP 비교")
bar = alt.Chart(year_df).mark_bar().encode(
    x=alt.X('Country Name:N', sort='-y'),
    y='GDP per capita:Q',
    color='Country Name:N',
    tooltip=['Country Name:N','GDP:Q','Population:Q','GDP per capita:Q']
).properties(height=420)
st.altair_chart(bar,use_container_width=True)

# -------------------------
# Plotly 지도
# -------------------------
st.subheader(f"🗺️ 전세계 1인당 GDP (choropleth) — {to_year}")
world_year = merged[merged['Year']==to_year]
fig = px.choropleth(
    world_year,
    locations='Country Code',
    color='GDP per capita',
    hover_name='Country Name',
    hover_data={'GDP':True,'Population':True,'GDP per capita':':.2f'},
    color_continuous_scale='Viridis',
    labels={'GDP per capita':'GDP per capita (US$)'},
    title=f"GDP per capita (US$) — {to_year}"
)
fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
st.plotly_chart(fig,use_container_width=True,height=600)

# -------------------------
# 데이터 다운로드
# -------------------------
st.markdown("---")
st.subheader("📥 데이터 다운로드")
csv = df_filtered.sort_values(['Country Name','Year']).to_csv(index=False).encode('utf-8')
st.download_button("CSV 다운로드", csv, file_name=f"gdp_selected_{from_year}_{to_year}.csv", mime="text/csv")

# -------------------------
# 푸터
# -------------------------
st.markdown("""
---
데이터 출처: World Bank Open Data (https://data.worldbank.org/)  
필요 파일: `data/gdp_data.csv` (GDP), `data/population_data.csv` (Population).  
필요 패키지: `streamlit`, `pandas`, `numpy`, `altair`, `plotly`, `pycountry`.
""")
