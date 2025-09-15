import streamlit as st
import pandas as pd
import math
from pathlib import Path
import pycountry

# -----------------------------------------------------------------------------
# 페이지 기본 설정
st.set_page_config(
    page_title='GDP Dashboard',
    page_icon='🌍',
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CSS 스타일 커스터마이즈
st.markdown(
    """
    <style>
    body {
        font-family: "Segoe UI", sans-serif;
    }
    .main {
        background-color: #f9fafb;
    }
    h1, h2, h3 {
        color: #1e3a8a;
    }
    .stMetric {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    /* 다크모드 지원 */
    @media (prefers-color-scheme: dark) {
        .main {
            background-color: #111827;
        }
        h1, h2, h3, p, div {
            color: #f3f4f6 !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# 데이터 로딩 함수
@st.cache_data
def get_gdp_data():
    DATA_FILENAME = Path(__file__).parent / "data/gdp_data.csv"
    raw_gdp_df = pd.read_csv(DATA_FILENAME)

    MIN_YEAR, MAX_YEAR = 1960, 2022

    gdp_df = raw_gdp_df.melt(
        ["Country Name", "Country Code"],
        [str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
        "Year",
        "GDP",
    )

    gdp_df["Year"] = pd.to_numeric(gdp_df["Year"])
    return gdp_df

gdp_df = get_gdp_data()

# -----------------------------------------------------------------------------
# 국기 아이콘 함수
def get_flag(country_code):
    try:
        country = pycountry.countries.get(alpha_3=country_code)
        if country:
            return f":flag-{country.alpha_2.lower()}:"
    except:
        return ""
    return ""

# -----------------------------------------------------------------------------
# 헤더
st.markdown(
    """
    <div style="text-align:center;">
        <h1>🌍 Global GDP Dashboard</h1>
        <p style="font-size:18px; color:#374151;">
        세계은행(World Bank) 데이터를 활용한 글로벌 GDP 시각화 플랫폼입니다.<br>
        국가별 GDP 추세, 비교, 지도 시각화를 한눈에 확인하세요.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# -----------------------------------------------------------------------------
# 사이드바 설정
st.sidebar.header("⚙️ Settings")

min_value = gdp_df["Year"].min()
max_value = gdp_df["Year"].max()

from_year, to_year = st.sidebar.slider(
    "Select year range",
    min_value=min_value,
    max_value=max_value,
    value=[1990, max_value],
    step=1,
)

country_options = dict(zip(gdp_df["Country Code"], gdp_df["Country Name"]))
selected_countries = st.sidebar.multiselect(
    "Choose countries",
    options=country_options.keys(),
    default=["USA", "CHN", "JPN", "DEU", "FRA", "KOR"],
    format_func=lambda code: f"{get_flag(code)} {country_options.get(code, code)}",
)

# -----------------------------------------------------------------------------
# 필터링
filtered_df = gdp_df[
    (gdp_df["Country Code"].isin(selected_countries))
    & (gdp_df["Year"].between(from_year, to_year))
]

# -----------------------------------------------------------------------------
# GDP 타임라인
st.subheader("📈 GDP Over Time")
st.line_chart(filtered_df, x="Year", y="GDP", color="Country Code")

st.divider()

# -----------------------------------------------------------------------------
# 특정 연도 GDP 메트릭
st.subheader(f"💰 GDP in {to_year}")

first_year = gdp_df[gdp_df["Year"] == from_year]
last_year = gdp_df[gdp_df["Year"] == to_year]

cols = st.columns(4)
for i, country in enumerate(selected_countries):
    col = cols[i % 4]
    with col:
        first_val = first_year[first_year["Country Code"] == country]["GDP"]
        last_val = last_year[last_year["Country Code"] == country]["GDP"]

        if first_val.empty or last_val.empty:
            continue

        first_gdp = first_val.iloc[0] / 1e9
        last_gdp = last_val.iloc[0] / 1e9

        if math.isnan(first_gdp) or first_gdp == 0:
            growth = "n/a"
        else:
            growth = f"{last_gdp / first_gdp:,.2f}x"

        st.metric(
            label=f"{get_flag(country)} {country_options[country]} ({country})",
            value=f"{last_gdp:,.0f}B USD",
            delta=growth,
        )

st.divider()

# -----------------------------------------------------------------------------
# 바 차트 (비교)
st.subheader(f"🏆 GDP Comparison ({to_year})")
bar_data = last_year[last_year["Country Code"].isin(selected_countries)]
st.bar_chart(bar_data, x="Country Code", y="GDP", color="Country Code")

st.divider()

# -----------------------------------------------------------------------------
# 지도 시각화
st.subheader(f"🗺️ GDP Map ({to_year})")

# 좌표 데이터 (streamlit map은 lat/lon 필요 → 여기서는 단순히 placeholder)
# 실제로는 국가별 좌표/shape 데이터 필요 (추가 CSV나 geopandas 이용 권장)
# 간단 버전: 중심 좌표만 보여주기
world_coords = {
    "USA": [38, -97],
    "CHN": [35, 105],
    "JPN": [36, 138],
    "DEU": [51, 10],
    "FRA": [46, 2],
    "KOR": [36, 128],
}

map_data = pd.DataFrame([
    {"lat": world_coords[c][0], "lon": world_coords[c][1], "GDP": float(last_year[last_year["Country Code"] == c]["GDP"])}
    for c in selected_countries if c in world_coords
])

st.map(map_data, size="GDP")

st.divider()

# -----------------------------------------------------------------------------
# 푸터
st.markdown(
    """
    <div style="text-align:center; margin-top:50px; font-size:14px; color:#6b7280;">
        📊 Data Source: <a href="https://data.worldbank.org/" target="_blank">World Bank Open Data</a> 🌐 <br>
        Built with ❤️ using Streamlit <br>
        Supports 🌙 Dark Mode & 🚩 Country Flags
    </div>
    """,
    unsafe_allow_html=True
)