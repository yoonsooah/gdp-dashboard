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

st.markdown(
    """
    <style>
    /* 기본 스타일 */
    .stApp { background-color: #f7fafc; }
    h1, h2, h3 { color: #0f172a; }
    @media (prefers-color-scheme: dark) {
      .stApp { background-color: #0b1220; color: #e6eef8; }
      h1, h2, h3 { color: #e6eef8; }
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
    # 연도(4자리 숫자) 컬럼만 추출 (예: '1960','1961',...)
    year_cols = [c for c in df.columns if re.match(r"^\d{4}$", str(c))]
    # 정렬
    year_cols = sorted(year_cols, key=lambda x: int(x))
    return year_cols

def melt_wb_wide(df, value_name):
    """
    World Bank 형식(각 연도가 컬럼)에 맞춰 melt 처리.
    반드시 'Country Name'과 'Country Code' 컬럼을 포함해야 함.
    """
    # 표준 컬럼 확인
    if 'Country Name' not in df.columns or 'Country Code' not in df.columns:
        raise ValueError("CSV 파일에 'Country Name' 및 'Country Code' 컬럼이 필요합니다.")
    year_cols = detect_year_columns(df)
    if not year_cols:
        raise ValueError("연도(YYYY) 형식의 컬럼을 찾지 못했습니다. CSV 형식을 확인하세요.")
    melted = df.melt(id_vars=['Country Name', 'Country Code'], value_vars=year_cols,
                     var_name='Year', value_name=value_name)
    # 숫자형으로 변환
    melted['Year'] = pd.to_numeric(melted['Year'], errors='coerce')
    melted[value_name] = pd.to_numeric(melted[value_name], errors='coerce')
    return melted

# -------------------------
# 데이터 로딩 (캐시)
# -------------------------
@st.cache_data
def load_wb_csv(path: Path, value_name: str):
    if not path.exists():
        return None, f"파일을 찾을 수 없습니다: {path}"
    try:
        raw = pd.read_csv(path)
    except Exception as e:
        return None, f"CSV를 읽는 중 오류가 발생했습니다: {e}"
    try:
        df = melt_wb_wide(raw, value_name)
    except Exception as e:
        return None, f"데이터 전처리(melt) 중 오류가 발생했습니다: {e}"
    return df, None

# 데이터 파일 경로
DATA_DIR = Path(__file__).parent / "data"
GDP_FILE = DATA_DIR / "gdp_data.csv"
POP_FILE = DATA_DIR / "population_data.csv"

gdp_df, err1 = load_wb_csv(GDP_FILE, "GDP")
pop_df, err2 = load_wb_csv(POP_FILE, "Population")

if gdp_df is None or pop_df is None:
    st.error("데이터 파일 로딩 오류:")
    if err1:
        st.write(f"- GDP 파일 오류: {err1}")
    if err2:
        st.write(f"- Population 파일 오류: {err2}")
    st.info("해결 방법: World Bank에서 'GDP (current US$)' 및 'Population, total' 지표로 CSV를 내려받아\n"
            "프로젝트 폴더 `data/gdp_data.csv`, `data/population_data.csv`로 저장하세요.")
    st.stop()

# -------------------------
# 병합 및 per-capita 계산
# -------------------------
# 병합은 Country Code + Year 기준으로 (Country Name은 병합 후 선택)
merged = pd.merge(
    gdp_df,
    pop_df,
    on=['Country Name', 'Country Code', 'Year'],
    how='outer',
    suffixes=('_gdpfile', '_popfile')
)

# 결측치 정리: 같은 Country Name 열이 양쪽에 있을 경우 하나로 정리 (보통 동일하므로 첫 번째 사용)
# (이미 같은 이름으로 병합했으므로 중복 컬럼은 없음 — 안전 차원에서 처리)
# GDP per capita 계산 (Population == 0 또는 NaN 처리)
merged['GDP per capita'] = merged['GDP'] / merged['Population']
# GDP/Population이 무한대 등일 경우 NaN 처리
merged.replace([np.inf, -np.inf], np.nan, inplace=True)

# 연도 정수형
merged['Year'] = merged['Year'].astype('Int64')

# -------------------------
# 국가 리스트 & 맵핑
# -------------------------
# 사용 가능한 최근 연도 범위
min_year = int(merged['Year'].min())
max_year = int(merged['Year'].max())

# country name ↔ iso3 (Country Code) 매핑 (unique)
country_df = merged[['Country Name', 'Country Code']].drop_duplicates().sort_values('Country Name')
country_name_to_iso3 = dict(zip(country_df['Country Name'], country_df['Country Code']))
iso3_to_country_name = dict(zip(country_df['Country Code'], country_df['Country Name']))

# ISO3 -> 국기 이모지 함수
def iso3_to_flag(iso3):
    try:
        if pd.isna(iso3):
            return ""
        c = pycountry.countries.get(alpha_3=iso3.upper())
        if not c:
            return ""
        a2 = c.alpha_2
        # A->regional indicator symbol '🇦' 계산
        return ''.join(chr(ord(ch) + 127397) for ch in a2.upper())
    except Exception:
        return ""

# -------------------------
# 사이드바 (필터)
# -------------------------
st.sidebar.header("🔎 필터")
year_range = st.sidebar.slider("연도 범위 선택", min_value=min_year, max_value=max_year, value=(2000, max_year), step=1)

all_country_names = country_df['Country Name'].tolist()
# 기본 선택: 상위 몇개 (데이터 프레임의 처음 6개)
default_selection = all_country_names[:6] if len(all_country_names) >= 6 else all_country_names

def format_country(name):
    iso3 = country_name_to_iso3.get(name, "")
    flag = iso3_to_flag(iso3)
    return f"{flag} {name}" if flag else name

selected_names = st.sidebar.multiselect("국가 선택", options=all_country_names, default=default_selection, format_func=format_country)

if not selected_names:
    st.sidebar.warning("최소 한 개 이상의 국가를 선택하세요.")
    st.stop()

from_year, to_year = year_range

# -------------------------
# 필터링된 데이터
# -------------------------
selected_iso3 = [country_name_to_iso3[n] for n in selected_names]
df_filtered = merged[
    (merged['Country Code'].isin(selected_iso3)) &
    (merged['Year'].between(from_year, to_year))
].copy()

if df_filtered.empty:
    st.warning("선택한 조건에 맞는 데이터가 없습니다. 연도/국가 설정을 확인하세요.")
    st.stop()

# -------------------------
# 시각화: GDP 시계열 (Altair)
# -------------------------
st.subheader("📈 GDP (Total) - 연도별 변화")
gdp_chart = (
    alt.Chart(df_filtered)
    .mark_line(point=True)
    .encode(
        x=alt.X('Year:Q', title='Year'),
        y=alt.Y('GDP:Q', title='GDP (current US$)'),
        color=alt.Color('Country Name:N', legend=alt.Legend(title="Country")),
        tooltip=[alt.Tooltip('Country Name:N'), alt.Tooltip('Year:Q'), alt.Tooltip('GDP:Q', format=",.0f")]
    )
    .interactive()
    .properties(height=420)
)
st.altair_chart(gdp_chart, use_container_width=True)

# -------------------------
# 시각화: GDP per capita 시계열
# -------------------------
st.subheader("👤 GDP per Capita - 연도별 변화")
percap_chart = (
    alt.Chart(df_filtered)
    .mark_line(point=True)
    .encode(
        x=alt.X('Year:Q', title='Year'),
        y=alt.Y('GDP per capita:Q', title='GDP per capita (US$)'),
        color=alt.Color('Country Name:N', legend=None),
        tooltip=[alt.Tooltip('Country Name:N'), alt.Tooltip('Year:Q'), alt.Tooltip('GDP per capita:Q', format=",.2f")]
    )
    .interactive()
    .properties(height=420)
)
st.altair_chart(percap_chart, use_container_width=True)

# -------------------------
# 연도별 비교: 선택한 to_year
# -------------------------
st.subheader(f"📊 {to_year}년 국가별 지표 비교")

year_df = merged[(merged['Year'] == to_year) & (merged['Country Code'].isin(selected_iso3))].copy()
# 정렬
year_df = year_df.sort_values('GDP', ascending=False)

# Metric 카드: GDP(억 단위), 성장배율( from_year -> to_year )
metrics_cols = st.columns(min(4, len(year_df)))
for i, (_, row) in enumerate(year_df.iterrows()):
    col = metrics_cols[i % len(metrics_cols)]
    with col:
        country_name = row['Country Name']
        iso3 = row['Country Code']
        flag = iso3_to_flag(iso3)
        gdp_val = row['GDP']
        pop_val = row['Population']
        percap = row['GDP per capita']

        # from_year 값 찾기
        from_row = merged[(merged['Country Code'] == iso3) & (merged['Year'] == from_year)]
        if not from_row.empty and not pd.isna(from_row['GDP'].values[0]) and from_row['GDP'].values[0] != 0:
            growth = f"{(gdp_val / from_row['GDP'].values[0]):.2f}x"
        else:
            growth = "n/a"

        gdp_display = f"{(gdp_val / 1e9):,.1f} B USD" if not pd.isna(gdp_val) else "n/a"
        percap_display = f"${percap:,.0f}" if not pd.isna(percap) else "n/a"

        st.metric(label=f"{flag} {country_name}", value=gdp_display, delta=f"{percap_display} per person ({growth})")

# -------------------------
# 막대차트: GDP per capita (to_year)
# -------------------------
st.subheader(f"📊 {to_year}년 1인당 GDP 비교")
bar = (
    alt.Chart(year_df)
    .mark_bar()
    .encode(
        x=alt.X('Country Name:N', sort='-y', title='Country'),
        y=alt.Y('GDP per capita:Q', title='GDP per capita (US$)'),
        color=alt.Color('Country Name:N', legend=None),
        tooltip=[alt.Tooltip('Country Name:N'), alt.Tooltip('GDP:Q', format=",.0f"), alt.Tooltip('Population:Q', format=",.0f"), alt.Tooltip('GDP per capita:Q', format=",.2f")]
    )
    .properties(height=420)
)
st.altair_chart(bar, use_container_width=True)

# -------------------------
# 지도: 전세계 GDP per capita (to_year) - Plotly Choropleth
# -------------------------
st.subheader(f"🗺️ 전세계 1인당 GDP (choropleth) — {to_year}")

world_year = merged[merged['Year'] == to_year].copy()
# plotly는 ISO-3 코드 사용
fig = px.choropleth(
    world_year,
    locations='Country Code',
    color='GDP per capita',
    hover_name='Country Name',
    color_continuous_scale='Viridis',
    labels={'GDP per capita': 'GDP per capita (US$)'},
    title=f"GDP per capita (US$) — {to_year}",
)
fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
st.plotly_chart(fig, use_container_width=True, height=600)

# -------------------------
# 데이터 다운로드(사용자용)
# -------------------------
st.markdown("---")
st.subheader("📥 데이터 다운로드")
st.markdown("선택한 국가 및 연도 범위에 해당하는 데이터를 CSV로 다운로드할 수 있습니다.")
download_df = df_filtered.sort_values(['Country Name', 'Year'])
csv = download_df.to_csv(index=False).encode('utf-8')
st.download_button("CSV 다운로드 (선택한 데이터)", csv, file_name=f"gdp_selected_{from_year}_{to_year}.csv", mime="text/csv")

# -------------------------
# 푸터 & 출처
# -------------------------
st.markdown("""
---
데이터 출처: World Bank Open Data (https://data.worldbank.org/)  
필요 파일: `data/gdp_data.csv` (GDP, current US$), `data/population_data.csv` (Population, total).  
필요 패키지: `streamlit`, `pandas`, `numpy`, `altair`, `plotly`, `pycountry`.
""")
