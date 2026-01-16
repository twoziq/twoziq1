import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import linregress
from datetime import date, timedelta, datetime
import time
import pytz

# ==============================================================================
# 0. 전역 설정 및 상수 정의 (수정: PER 기준 삭제)
# ==============================================================================
DEFAULT_BIG_TECH_TICKERS = ['NVDA', 'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'AVGO', 'META', 'TSLA']
DCA_DEFAULT_TICKER = "QQQ"  # DCA 탭 기본 티커
MULTI_DEFAULT_TICKERS = "DIA SPY QQQ SCHD"  # 다중 티커 탭 기본값
DEFAULT_RISK_FREE_RATE = 3.75 / 100  # 기준금리 3.75%

KST = pytz.timezone('Asia/Seoul')
NOW_KST = datetime.now(KST)
TODAY = NOW_KST.date()


# PER 기준 상수 (제거됨)

# PER 기준선 Plotly 스타일 (제거됨)


# ==============================================================================
# 1. 데이터 로드 및 캐싱 함수 (유지)
# ==============================================================================
@st.cache_data(ttl=3600)
def load_ticker_info(ticker, max_retries=3):
    """티커 정보를 로드합니다 (EPS, 회사 이름)."""
    for attempt in range(max_retries):
        try:
            data = yf.Ticker(ticker)
            info = data.info
            eps = info.get('trailingEps')
            if eps is None or eps == 0:
                eps = info.get('forwardEps')
            per_info = {
                'EPS': eps if eps else 0,
                'CompanyName': info.get('longName', ticker),
            }
            return per_info, None
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                time.sleep(wait_time)
            else:
                return None, f"Ticker information could not be loaded after {max_retries} attempts: {e}"
    return None, "Unexpected failure in Ticker Info loading."


@st.cache_data(ttl=3600)
def load_historical_data(ticker_or_list, start_date=None, end_date=None, max_retries=3):
    """yfinance에서 주가 데이터를 로드합니다. (단일/복수 티커 지원)"""

    start_date_arg = start_date
    end_date_arg = end_date
    period_arg = None  # 강제 None 처리

    for attempt in range(max_retries):
        try:
            hist = yf.download(ticker_or_list, start=start_date_arg, end=end_date_arg, period=period_arg,
                               progress=False)
            if hist.empty:
                return None, "해당 기간의 주가 데이터를 가져올 수 없습니다."
            return hist, None
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                time.sleep(wait_time)
            else:
                return None, f"데이터 로드 중 오류가 발생했습니다: {e}"
    return None, "Unexpected failure in Historical Data loading."


@st.cache_data(ttl=3600)
def load_big_tech_data(tickers):
    """요청된 빅테크 종목의 최신 재무 정보를 로드합니다 (현재 PER 계산용)."""
    data_list = []
    tickers_obj = yf.Tickers(tickers)

    for ticker in tickers:
        try:
            info = tickers_obj.tickers[ticker].info
            market_cap = info.get('marketCap', np.nan)
            trailing_pe = info.get('trailingPE', np.nan)
            net_income = market_cap / trailing_pe if market_cap and trailing_pe and trailing_pe > 0 else np.nan

            data_list.append({
                'Ticker': ticker,
                'MarketCap': market_cap,
                'TrailingPE': trailing_pe,
                'NetIncome': net_income,
            })
        except Exception:
            data_list.append({'Ticker': ticker, 'MarketCap': np.nan, 'TrailingPE': np.nan, 'NetIncome': np.nan})

    return pd.DataFrame(data_list)


@st.cache_data(ttl=3600)
def calculate_accurate_group_per_history(ticker_list, start_date, end_date):
    """빅테크 그룹의 시가총액 가중 평균 PER의 정확한 역사적 시계열을 계산합니다."""

    start_date_yf = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    end_date_yf = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    period_arg = None

    combined_market_cap = pd.DataFrame()
    combined_net_income = pd.DataFrame()
    valid_tickers = []

    with st.spinner("📊 PER 추이 계산 중..."):
        try:
            hist_all, hist_error = load_historical_data(
                ticker_list, start_date=start_date_yf,
                end_date=end_date_yf
            )
            if hist_all is None:
                return None, hist_error

            hist_closes = hist_all['Close'].dropna(axis=1, how='all')

        except Exception as e:
            return None, f"주가 데이터 병렬 로드 중 오류 발생: {e}"

        for ticker in ticker_list:
            if ticker not in hist_closes.columns: continue

            try:
                stock = yf.Ticker(ticker)
                hist_close = hist_closes[ticker].dropna()
                if hist_close.empty: continue
                hist_close.index = hist_close.index.tz_localize(None)

                try:
                    shares = stock.fast_info['shares_outstanding']
                except:
                    shares = stock.info.get('sharesOutstanding')

                if not shares: continue

                combined_market_cap[ticker] = hist_close * shares

                income_stmt = stock.financials
                income_keys = ['Net Income', 'Net Income Common Stockholders']
                net_income_row = next((income_stmt.loc[k] for k in income_keys if k in income_stmt.index), None)

                if net_income_row is None: continue

                net_income_row.index = pd.to_datetime(net_income_row.index).tz_localize(None)
                net_income_row = net_income_row.sort_index()
                combined_net_income[ticker] = net_income_row.reindex(hist_close.index, method='ffill')
                valid_tickers.append(ticker)

            except Exception:
                continue

    if combined_market_cap.empty or combined_net_income.empty:
        return None, "유효한 Market Cap 및 Net Income 데이터를 가진 종목이 없어 PER 계산이 불가능합니다."

    common_index = combined_market_cap.index.intersection(combined_net_income.index)
    total_market_cap = combined_market_cap.loc[common_index, valid_tickers].sum(axis=1)
    total_net_income = combined_net_income.loc[common_index, valid_tickers].sum(axis=1)

    group_per = total_market_cap / total_net_income.mask(total_net_income <= 0)
    group_per = group_per.astype(float).replace([np.inf, -np.inf], np.nan).dropna()

    if group_per.empty:
        return None, "순이익이 양수인 기간의 데이터가 부족하여 그룹 PER 시계열을 계산할 수 없습니다."

    return group_per, None


@st.cache_data(ttl=3600)
def calculate_multi_ticker_metrics(ticker_list, start_date, end_date):
    """여러 티커의 연환산 수익률과 변동성을 계산합니다."""
    ticker_list = [t.strip().upper() for t in ticker_list if t.strip()]
    if not ticker_list:
        return None, "티커를 입력해주세요."

    # period 인자 제거
    hist_data, error = load_historical_data(ticker_list, start_date, end_date)
    if error: return None, error

    if isinstance(hist_data.columns, pd.MultiIndex):
        returns = hist_data['Close'].pct_change().dropna(axis=0, how='all')
    else:
        returns = hist_data['Close'].pct_change().dropna()
        returns = pd.DataFrame(returns, columns=ticker_list)

    returns = returns.dropna(axis=1, how='all')

    if returns.empty or len(returns) < 20:
        return None, "데이터 부족 또는 티커 오류로 수익률 계산 불가."

    annual_factor = 252
    mean_returns = returns.mean() * annual_factor
    annual_volatility = returns.std() * np.sqrt(annual_factor)

    metrics_list = []
    for ticker in returns.columns:
        metrics_list.append({
            'Ticker': ticker,
            'Return': mean_returns.get(ticker, 0.0),
            'Volatility': annual_volatility.get(ticker, 0.0)
        })

    df_metrics = pd.DataFrame(metrics_list)
    df_metrics['Sharpe_Ratio'] = df_metrics['Return'] / df_metrics['Volatility'].mask(df_metrics['Volatility'] == 0)
    df_metrics = df_metrics.sort_values(by='Return', ascending=False).reset_index(drop=True)

    return df_metrics, None


# ==============================================================================
# 2. 핵심 계산 함수 (DCA용) (유지)
# ==============================================================================

def calculate_per_and_indicators(df, eps):
    """DCA 시뮬레이션용 간단한 계산"""
    data = df.copy()

    if isinstance(data.columns, pd.MultiIndex):
        data['Price'] = data['Close'].iloc[:, 0]
    else:
        data['Price'] = data['Close']

    return data


# ==============================================================================
# 3. 유틸리티 및 포매팅 함수 (수정: get_per_color 제거)
# ==============================================================================

@st.cache_data
def format_value(val):
    """숫자를 T (조), B (십억) 단위로 포매팅합니다."""
    if pd.isna(val) or val == 0:
        return "-"
    if abs(val) >= 1e12:
        return f"{val / 1e12:,.2f}T"
    elif abs(val) >= 1e9:
        return f"{val / 1e9:,.2f}B"
    return f"{val:,.2f}"


# get_per_color 함수는 제거됨


# ==============================================================================
# 4. Streamlit UI 및 레이아웃 설정 (Sidebar Fix) (유지)
# ==============================================================================

st.set_page_config(layout="wide", page_title="Twoziq 투자 가이드")

# --- 상태 관리 초기화 ---
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "빅테크 PER"
# DCA 티커 기본값: QQQ
if 'dca_ticker_value' not in st.session_state:
    st.session_state.dca_ticker_value = DCA_DEFAULT_TICKER
# 다중 티커 입력값 초기화
if 'multi_ticker_input_value' not in st.session_state:
    st.session_state.multi_ticker_input_value = ""

# --- 사이드바: 기본 설정 ---
with st.sidebar:
    st.header("⚙️ 기본 설정")

    ticker_symbol = None

    # 1. 티커 입력 (DCA 탭에만 표시)
    if st.session_state.active_tab == "적립식 투자":
        ticker_symbol = st.text_input(
            "DCA 분석 주식 티커:",
            value=st.session_state.dca_ticker_value,
            key="dca_ticker_input_key"
        ).upper()
        # 입력값 세션 상태에 저장
        st.session_state.dca_ticker_value = ticker_symbol
    else:
        # 다른 탭에서는 DCA 티커를 참조하지 않으므로 임시로 'N/A_Ignored' 설정 (오류 방지)
        ticker_symbol = "N/A_Ignored"

    # 2. 기간 선택 설정 (수정: YTD, 최대 기간 제거)
    period_options = {"1년": 365, "2년": 730, "3년": 3 * 365, "5년": 1825, "10년": 10 * 365}

    # DCA 탭 진입 시 기본값 '3년'으로 설정
    default_period_key = "1년"
    default_period_index = list(period_options.keys()).index(default_period_key)

    # **수정**: 기간 선택 로직 단순화
    selected_period_name = st.selectbox("기간 선택:", list(period_options.keys()), index=default_period_index,
                                        key='period_select_key')

    # 3. 날짜 계산 및 기간 인자 설정
    days = period_options.get(selected_period_name, 1 * 365)  # 기본값 3년
    start_date_default = TODAY - timedelta(days=days)

    # st.date_input의 key에 selected_period_name을 포함하여 selectbox 변경 시 강제 업데이트
    start_date_input = st.date_input(
        "시작 날짜:",
        value=start_date_default,
        max_value=TODAY,
        key=f'start_date_key_{selected_period_name}'  # Dynamic Key FIX
    )
    end_date_input = st.date_input("최종 날짜:", value=TODAY, max_value=TODAY, key='end_date_key')

    # yfinance에 전달할 최종 날짜 문자열
    start_date_final = start_date_input.strftime('%Y-%m-%d')
    end_date_final = end_date_input.strftime('%Y-%m-%d')

# ==============================================================================
# 6. 메뉴 설정 (유지)
# ==============================================================================

menu_options = ["빅테크 PER", "적립식 투자", "다중 티커 비교"]

st.markdown("""
    <style>
    /* ... (CSS 코드는 유지) ... */
    div[data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-direction: row !important;
        gap: 8px !important;
    }
    /* **추가**: st.metric 값의 글자 크기를 줄임 */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem; /* 기본값 2.5rem 보다 작게 조정 */
        font-weight: 600;
    }

    /* **추가**: st.metric 레이블의 글자 크기를 줄임 */
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem; /* 기본값 1rem 보다 작게 조정 */
        font-weight: 400;
    }
    /* 🚨 1열 강제 배치 수정 (이전에 안내해 드린 수정 사항) */
    @media (max-width: 768px) {
        /* st.columns가 만드는 Block을 1열 그리드로 강제 재정의 */
        div[data-testid="stHorizontalBlock"] {
            display: grid !important;
            grid-template-columns: 1fr !important; /* 1열로 변경 */
            gap: 6px !important;
        }

        /* st.metric을 담고 있는 column 자체도 100% 폭을 가지도록 보장 */
        div[data-testid="column"] {
            width: 100% !important;
            min-width: 0px !important;
            flex: none !important;
        }

        .stButton button p {
            font-size: 0.72rem !important;
        }
    }

    .stButton button {
        height: 2.8rem !important;
        border-radius: 8px !important;
    }
    </style>
    """, unsafe_allow_html=True)

cols = st.columns(len(menu_options))
for i, option in enumerate(menu_options):
    with cols[i]:
        is_active = (st.session_state.active_tab == option)
        btn_type = "primary" if is_active else "secondary"
        if st.button(option, key=f"resp_btn_{i}", use_container_width=True, type=btn_type):
            if st.session_state.active_tab != option:
                # 탭 전환 시 active_tab을 업데이트
                st.session_state.active_tab = option

                # 다중 티커 탭으로 전환 시 기본값 설정
                if option == "다중 티커 비교":
                    st.session_state['multi_ticker_input_value'] = MULTI_DEFAULT_TICKERS

                st.rerun()

st.markdown("---")

# ==============================================================================
# 7. Tab 구현부 (수정)
# ==============================================================================

# ------------------------------------------------------------------------------
# 탭 1: 재무 분석 (빅테크) (수정: PER 기준선, 기준표, get_per_color 호출 제거)
# ------------------------------------------------------------------------------
if st.session_state.active_tab == "빅테크 PER":  # <-- 탭 이름을 "재무 분석"으로 가정하고 수정
    st.markdown("1️⃣ Tab 1 → 지금이 투자하기 적당한 시기인가?")
    st.caption("이 페이지는 단순 매수/매도 신호가 아니라, 투자 속도를 조절하기 위한 참고 지표입니다.")
    st.caption("ETF는 개별 종목처럼 적정 가치를 계산하는 것이 쉽지 않습니다. ")
    st.caption("Top 8 빅테크를 하나의 기업이라고 가정해 PER을 산출했습니다.")
    st.caption("중위값, 평균값을 보시고 현재 주가의 적정성을 판단해보세요. ")

    tech_df_raw = load_big_tech_data(DEFAULT_BIG_TECH_TICKERS)

    if 'tech_select_state' not in st.session_state:
        st.session_state['tech_select_state'] = {t: True for t in DEFAULT_BIG_TECH_TICKERS}

    selected_tickers = [t for t, selected in st.session_state['tech_select_state'].items() if selected]
    selected_df = tech_df_raw[tech_df_raw['Ticker'].isin(selected_tickers)]

    total_market_cap = selected_df['MarketCap'].sum()
    total_net_income = selected_df['NetIncome'].sum()

    if total_net_income > 0:
        average_per = total_market_cap / total_net_income
        average_per_str = f"{average_per:,.2f}"
        # dynamic_color, position_text_raw = get_per_color(average_per) # <--- get_per_color 호출 제거
        position_text_raw = "현재 평균 PER"  # <--- 대체 문구
    else:
        average_per = np.nan
        average_per_str = "N/A"
        # dynamic_color, position_text_raw = "#gray", "데이터 없음" # <--- get_per_color 호출 제거
        position_text_raw = "데이터 없음"

    group_per_series, hist_error_tab1 = calculate_accurate_group_per_history(
        selected_tickers, start_date=start_date_final, end_date=end_date_final
    )

    if hist_error_tab1:
        st.warning(f"PER 추이 데이터를 로드할 수 없습니다: {hist_error_tab1}")
    elif group_per_series is None or group_per_series.empty:
        st.info("선택된 종목들의 유효한 데이터가 부족하여 그래프를 표시할 수 없습니다.")
    else:
        clean_per_values = group_per_series.dropna()
        avg_per_hist = clean_per_values.mean()
        median_per_hist = clean_per_values.median()

        fig_per_tab1 = go.Figure()

        fig_per_tab1.add_trace(go.Scatter(
            x=group_per_series.index, y=group_per_series,
            mode='lines', name='시총 가중 평균 PER 추이',
            line=dict(color='#1f77b4', width=2),
            showlegend=False
        ))

        # PER 레벨 기준선 추가 (제거)
        # for level, (color, name) in PER_LINE_STYLES.items():
        #     fig_per_tab1.add_hline(...)

        fig_per_tab1.add_hline(y=avg_per_hist, line_dash="dash", line_color="#d62728",
                               annotation_text=f"평균: {avg_per_hist:.2f}",
                               annotation_position="bottom left")
        fig_per_tab1.add_hline(y=median_per_hist, line_dash="dot", line_color="#ff7f0e",
                               annotation_text=f"중앙값: {median_per_hist:.2f}",
                               annotation_position="top left")

        fig_per_tab1.update_layout(
            title="미국 빅테크 Top8 평균 PER",
            xaxis_title="날짜",
            hovermode="x unified",
            template="plotly_white",
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(255, 255, 255, 0.5)"
            ),
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig_per_tab1, use_container_width=True)


    st.markdown("---")

    col_sum1, col_sum2, col_sum3 = st.columns(3)
    with col_sum1:
        st.metric(
            label="금일 기준 평균 PER",
            value=average_per_str,
            # delta=position_text_raw if average_per_str != "N/A" else None, # <--- delta 제거
            delta_color='off'
        )
    with col_sum2:
        st.metric(label="총 시가총액 합", value=format_value(total_market_cap))
    with col_sum3:
        st.metric(label="총 순이익 합", value=format_value(total_net_income))

    st.markdown("---")

    # 🚨 수정된 부분: Data Editor를 전체 폭으로 배치 (단일 컬럼)
    col_editor = st.columns(1)[0]

    # PER 기반 투자 기준표 UI는 완전히 제거됨

    with col_editor:
        editor_df = tech_df_raw.copy()
        editor_df['Select'] = editor_df['Ticker'].apply(lambda t: st.session_state['tech_select_state'].get(t, True))
        editor_df['PER'] = editor_df['TrailingPE'].apply(lambda x: f"{x:.2f}" if x > 0 else "-")  # 컬럼명 'PER' 유지
        editor_df['시가총액'] = editor_df['MarketCap'].apply(format_value)  # 컬럼명 '시가총액' 유지
        editor_df['순이익'] = editor_df['NetIncome'].apply(format_value)  # 컬럼명 '순이익' 유지

        st.markdown("**분석 포함 종목 선택(USD)**", help="체크를 해제하면 전체 평균 계산에서 제외됩니다.")

        edited_df = st.data_editor(
            editor_df[['Select', 'Ticker', '시가총액', '순이익', 'PER']],
            column_config={
                "Select": st.column_config.CheckboxColumn("선택"),
                "Ticker": st.column_config.TextColumn(disabled=True),
                "시가총액": st.column_config.TextColumn(disabled=True),
                "PER": st.column_config.TextColumn(disabled=True),
                "순이익": st.column_config.TextColumn(disabled=True),
            },
            hide_index=True,
            key='big_tech_editor_v2'
        )

        new_selections = {row['Ticker']: row['Select'] for _, row in edited_df.iterrows()}
        if new_selections != st.session_state['tech_select_state']:
            st.session_state['tech_select_state'] = new_selections
            st.rerun()


# ------------------------------------------------------------------------------
# 탭 2: 적립 모드 (DCA) (유지)
# ------------------------------------------------------------------------------
elif st.session_state.active_tab == "적립식 투자":

    # 1. 데이터 로드 (탭 진입 시점에만 실행)
    if not ticker_symbol or ticker_symbol == "N/A_Ignored":
        st.warning("DCA 분석을 위해 사이드바에 유효한 티커를 입력해 주세요.")
        st.stop()
    st.markdown("2️⃣ Tab 2 → 어떤 방식으로 투자할 것인가?")
    st.caption("거치식 투자(몰빵투자)는 큰 하락에 대응하기가 어렵습니다. ")
    st.caption("하락장은 적립식 투자자에게는 평균 매입 단가를 낮출 수 있는 구간입니다.")
    st.caption("단기 예측보다는 **장기 우상향**을 전제로 **적립식 매수 전략**을 유지하세요.")
    st.caption("바닥을 잡지 않아도, 안정적인 수익률을 기대할 수 있습니다.")

    # DCA 분석용 티커 로드 (Section 5 내용)
    with st.spinner(f"[{ticker_symbol}] 데이터 로드 중..."):
        info, info_error = load_ticker_info(ticker_symbol)
        if info_error:
            st.error(f"티커 정보를 가져오는 데 실패했습니다: {info_error}")
            st.stop()

        # FIX: period 인자 제거, start_date_final과 end_date_final만 사용
        hist_data, data_error = load_historical_data(
            ticker_symbol,
            start_date=start_date_final,
            end_date=end_date_final,
        )
        if data_error:
            st.error(f"데이터 로드 오류: {data_error}")
            st.stop()

        df_calc = calculate_per_and_indicators(hist_data, info['EPS'])

        # 2. DCA 시뮬레이션 및 플롯
    if 'dca_amount' not in st.session_state: st.session_state.dca_amount = 10.0
    if 'dca_freq' not in st.session_state: st.session_state.dca_freq = "매일"

    deposit_amount = st.session_state.dca_amount
    deposit_frequency = st.session_state.dca_freq

    dca_df = df_calc.copy()
    dca_df['WeekOfYear'] = dca_df.index.isocalendar().week.astype(int)
    dca_df['Month'] = dca_df.index.month

    if deposit_frequency == "매일":
        invest_dates = dca_df.index
    elif deposit_frequency == "매주":
        invest_dates = dca_df.groupby('WeekOfYear')['Price'].head(1).index
    elif deposit_frequency == "매월":
        invest_dates = dca_df.groupby('Month')['Price'].head(1).index

    dca_result = dca_df[dca_df.index.isin(invest_dates)].copy()
    dca_result['Shares_Bought'] = deposit_amount / dca_result['Price']
    dca_result['Total_Shares'] = dca_result['Shares_Bought'].cumsum()
    dca_result['Cumulative_Investment'] = np.arange(1, len(dca_result) + 1) * deposit_amount

    full_dca_results = dca_df.copy()
    full_dca_results['Total_Shares'] = dca_result['Total_Shares'].reindex(dca_df.index, method='ffill').fillna(0)
    full_dca_results['Cumulative_Investment'] = dca_result['Cumulative_Investment'].reindex(dca_df.index,
                                                                                            method='ffill').fillna(0)
    full_dca_results['Current_Value'] = full_dca_results['Total_Shares'] * full_dca_results['Price']

    fig_dca = go.Figure()

    fig_dca.add_trace(go.Scatter(x=full_dca_results.index, y=full_dca_results['Price'], mode='lines', name='주가 추이 (배경)',
                                 line=dict(color='gray', width=1), opacity=0.3, yaxis='y2'))

    fig_dca.add_trace(
        go.Scatter(x=full_dca_results.index, y=full_dca_results['Current_Value'], mode='lines', name='현재 평가 가치',
                   line=dict(color='green', width=2), yaxis='y1'))

    fig_dca.add_trace(
        go.Scatter(x=full_dca_results.index, y=full_dca_results['Cumulative_Investment'], mode='lines', name='총 투자 금액',
                   line=dict(color='red', width=2, dash='dash'), yaxis='y1'))

    fig_dca.update_layout(
        title=f"{ticker_symbol} 적립식 투자 백테스트", height=500, xaxis_title="날짜", hovermode="x unified",
        legend=dict(x=0.01, y=0.99, yanchor="top", xanchor="left"),
        # [수정 1] yaxis (왼쪽 축) 제목 제거
        yaxis=dict(title=dict(text="", font=dict(color="green")), side="left", showgrid=True),
        # [수정 2] yaxis2 (오른쪽 축, 배경) 제목 제거
        yaxis2=dict(title=dict(text="", font=dict(color="gray")), overlaying="y", side="right",
                    showgrid=False,
                    range=[full_dca_results['Price'].min() * 0.9, full_dca_results['Price'].max() * 1.1])
    )
    st.plotly_chart(fig_dca, use_container_width=True)


    st.markdown("---")
    st.markdown("### 🛠️ 시뮬레이션 설정")
    col_dca_config1, col_dca_config2 = st.columns(2)
    with col_dca_config1:
        st.number_input("**적립 금액 (USD)**", min_value=1.0, step=1.0, format="%.2f", key='dca_amount',
                        help="매번 투자할 금액을 입력합니다.")
    with col_dca_config2:
        current_freq_index = ["매일", "매주", "매월"].index(st.session_state.dca_freq)
        st.selectbox("**적립 주기**", ["매일", "매주", "매월"], index=current_freq_index, key='dca_freq')

    st.markdown("---")
    st.markdown("### 📊 최종 요약")

    if not full_dca_results.empty:
        final_row = full_dca_results.iloc[-1]
        current_value = final_row['Current_Value'].item()
        cumulative_investment = final_row['Cumulative_Investment'].item()
        col_dca_summary = st.columns(4)
        col_dca_summary[0].metric(label="최종 평가 가치", value=f"${current_value:,.2f}",
                                  delta=f"${current_value - cumulative_investment:,.2f}")
        col_dca_summary[1].metric("총 투자 금액", f"${cumulative_investment:,.2f}")
        col_dca_summary[2].metric("총 매수 주식 수", f"{final_row['Total_Shares'].item():,.4f} 주")


# ------------------------------------------------------------------------------
# 탭 3: 다중 티커 비교 (수정: Sharpe Ratio 색상 스케일 변경)
# ------------------------------------------------------------------------------
elif st.session_state.active_tab == "다중 티커 비교":

    # 세션 상태에서 다중 티커 입력값을 가져와 기본값으로 사용 (탭 전환 시 기본값 설정됨)
    col_multi_input, col_multi_rf = st.columns([2, 1])
    st.markdown("3️⃣ Tab 3 → 어떤 종목을 선택할 것인가?")
    st.caption(f"**Sharpe Ratio** = (수익률 - 기준 금리%) / 변동성, 통상 **1 이상:** 우수")
    st.caption("간단히, Sharpe Ratio는 리턴/리스크. 투자 매력도를 나타내는 값 입니다.")
    st.caption("수치가 높을수록, 적은 기회비용으로 높은 수익을 내는 구조입니다.")
    st.caption(
        """
        <span style='color: red; font-weight: bold;'>빨간색</span>은 한 번 더 고민하시고, 
        차라리 <span style='color: blue; font-weight: bold;'>파란색</span>을 투자하세요.
        """,
        unsafe_allow_html=True
    )
    st.caption("좌상단에 가까울수록 좋은 종목이지만, 높은 수익률을 위해 리스크를 감수하는 것도 중요합니다.")

    with col_multi_input:
        # key를 사용해 입력값의 영속성(Persistence) 유지
        multi_ticker_input = st.text_input(
            "비교할 티커 입력 (공백 또는 쉼표로 구분)",
            value=st.session_state.multi_ticker_input_value,
            key="multi_ticker_mpt_sec6"
        )
        # 사용자가 입력값을 변경하면 세션 상태에 저장하여 유지
        st.session_state.multi_ticker_input_value = multi_ticker_input

    with col_multi_rf:
        user_rf = st.number_input("기준금리(%)", value=DEFAULT_RISK_FREE_RATE * 100, step=0.1, key="rf_sec6")
        rf_multi = user_rf / 100


    ticker_list_multi = [t.strip().upper() for t in multi_ticker_input.replace(',', ' ').split() if t.strip()]

    # 사이드바의 start_date_final과 end_date_final 사용
    start_date_multi, end_date_multi = start_date_final, end_date_final

    if ticker_list_multi:
        with st.spinner("다중 분석 중..."):
            df_m, err = calculate_multi_ticker_metrics(ticker_list_multi, start_date_multi, end_date_multi)
        if err:
            st.error(err)
        elif df_m is not None and not df_m.empty:
            df_m['Sharpe_Ratio'] = (df_m['Return'] - rf_multi) / df_m['Volatility']

            st.markdown("#### 📈 자산별 위험 대비 수익 현황", help="우상단: 고위험고수익, 좌상단: 가성비(고효율)")

            # **핵심 수정**: 커스텀 색상 스케일 (Red-White-Blue) 정의
            # [정규화된 값 (0.0~1.0), 색상 코드]
            # 0.0: 최솟값 (빨강), 0.5: 중앙값 (흰색), 1.0: 최댓값 (파랑)
            custom_rwb_colorscale = [
                [0.0, 'rgb(255, 0, 0)'],  # 최소값: 순수한 빨간색
                [0.5, 'rgb(255, 255, 255)'],  # 중앙값: 순수한 흰색
                [1.0, 'rgb(0, 0, 255)']  # 최댓값: 순수한 파란색
            ]

            fig_multi = go.Figure(go.Scatter(
                x=df_m['Volatility'] * 100, y=df_m['Return'] * 100, mode='markers+text', text=df_m['Ticker'],
                marker=dict(size=15, color=df_m['Sharpe_Ratio'],
                            colorscale=custom_rwb_colorscale,  # 커스텀 색상 스케일 적용
                            showscale=False)  # 색상 바 제거 유지
            ))
            fig_multi.update_layout(xaxis_title="위험률 (%)", yaxis_title="수익률 (%)", template="plotly_white", height=600,
                                    margin=dict(
                                        b=100))  # xaxis=dict(rangemode='tozero'), yaxis=dict(rangemode='tozero'))
            st.plotly_chart(fig_multi, use_container_width=True)

            df_d = df_m.sort_values(by='Sharpe_Ratio', ascending=False).reset_index(drop=True)
            df_d.index += 1
            df_d_f = df_d.copy()
            df_d_f['Return'] = df_d_f['Return'].apply(lambda x: f"{x * 100:.2f}%")
            df_d_f['Volatility'] = df_d_f['Volatility'].apply(lambda x: f"{x * 100:.2f}%")
            df_d_f['Sharpe_Ratio'] = df_d_f['Sharpe_Ratio'].apply(lambda x: f"{x:.2f}")
            st.dataframe(df_d_f.rename(
                columns={'Ticker': '티커', 'Return': '수익률', 'Volatility': '위험률', 'Sharpe_Ratio': 'Sharpe Ratio'}),
                use_container_width=True)

            # --- 사용자 요청 반영 (Help 제거, 샤프 비율 하단 분리 및 기준 간소화) ---
            st.markdown(f"💡 **분석 결과:** 가장 효율적인 자산은 **{df_d.iloc[0]['Ticker']}**입니다.")

            st.caption(f"ℹ️ 기간: {start_date_multi}~{end_date_multi} | 기준금리 {user_rf}% 반영")

    else:
        st.info("티커를 입력해 주세요.")