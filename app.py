import streamlit as st
from datetime import datetime, timedelta
import pytz
import data_loader as dl
import analysis as al
import ui_components as ui

# 기본 설정
st.set_page_config(layout="wide", page_title="Twoziq 투자 가이드")
ui.inject_custom_css()
KST = pytz.timezone('Asia/Seoul')
TODAY = datetime.now(KST).date()

# 세션 상태 초기화
if 'active_tab' not in st.session_state: st.session_state.active_tab = "빅테크 PER"

# 사이드바
with st.sidebar:
    st.header("⚙️ 설정")
    period_map = {"1년": 365, "3년": 1095, "5년": 1825}
    sel_p = st.selectbox("기간", list(period_map.keys()), index=0)
    start_d = TODAY - timedelta(days=period_map[sel_p])
    start_date = st.date_input("시작일", value=start_d).strftime('%Y-%m-%d')
    end_date = TODAY.strftime('%Y-%m-%d')

# 메뉴 탭
menu = ["빅테크 PER", "적립식 투자", "다중 티커 비교"]
cols = st.columns(len(menu))
for i, m in enumerate(menu):
    if cols[i].button(m, use_container_width=True, type="primary" if st.session_state.active_tab == m else "secondary"):
        st.session_state.active_tab = m
        st.rerun()

# 탭별 로직
if st.session_state.active_tab == "빅테크 PER":
    st.subheader("미국 빅테크 Top8 평균 PER")
    tickers = ['NVDA', 'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'AVGO', 'META', 'TSLA']
    df_tech = dl.load_big_tech_data(tickers)
    
    m_cap_sum = df_tech['MarketCap'].sum()
    income_sum = df_tech['NetIncome'].sum()
    
    c1, c2, c3 = st.columns(3)
    c1.metric("평균 PER", f"{m_cap_sum/income_sum:.2f}")
    c2.metric("총 시가총액", ui.format_value(m_cap_sum))
    c3.metric("총 순이익", ui.format_value(income_sum))
    
    st.dataframe(df_tech, use_container_width=True)

elif st.session_state.active_tab == "적립식 투자":
    ticker = st.text_input("티커 입력", value="QQQ").upper()
    df_hist, err = dl.load_historical_data(ticker, start_date, end_date)
    if not err:
        dca_res = al.calculate_dca_simulation(df_hist, 100, "매주")
        st.plotly_chart(ui.plot_dca_chart(dca_res, ticker), use_container_width=True)
