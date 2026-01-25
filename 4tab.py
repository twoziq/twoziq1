import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import linregress
from datetime import date, timedelta, datetime
import time
import pytz
from plotly.subplots import make_subplots


# ==============================================================================
# 0. 전역 설정 및 상수 정의 (수정: PER 기준 삭제)
# ==============================================================================
DEFAULT_BIG_TECH_TICKERS = ['NVDA', 'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'AVGO', 'META', 'TSLA']
DCA_DEFAULT_TICKER = "QQQ"  # DCA 탭 기본 티커
MULTI_DEFAULT_TICKERS = "DIA SPY QQQ"  # 다중 티커 탭 기본값
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
# B1. 시뮬레이션분석
# ==============================================================================
@st.cache_data(ttl=3600)
def run_simulation_analysis_streamlit(ticker_symbol, start_date, end_date, 
                                       forecast_days=252, iterations=10000, 
                                       rank_mode='relative'):
    """
    확률분포 시뮬레이션 분석 (수정: 롤링 윈도우 수익률)
    """
    try:
        # 1. 데이터 로드
        hist_data, error = load_historical_data(ticker_symbol, start_date, end_date)
        if error:
            return None, error
        
        # 2. 종가 추출
        if isinstance(hist_data.columns, pd.MultiIndex):
            series = hist_data['Close'].iloc[:, 0].dropna()
        else:
            series = hist_data['Close'].dropna()
        
        series.index = pd.to_datetime(series.index).tz_localize(None)
        series = series.sort_index()
        
        if len(series) < forecast_days + 1:
            return None, "데이터가 부족합니다."
        
        # ✅ 3. 롤링 윈도우 수익률 계산 (핵심 수정)
        # 모든 시작점에서 forecast_days 후의 수익률
        returns = series.pct_change(forecast_days).dropna()
        
        # 4. 백분위 순위 계산
        if rank_mode == 'absolute':
            full_returns = returns
            sorted_values = np.sort(full_returns.values)
            rank_ts = returns.apply(
                lambda x: (np.searchsorted(sorted_values, x) / len(sorted_values)) * 100
            )
        else:
            sorted_values = np.sort(returns.values)
            rank_ts = returns.apply(
                lambda x: (np.searchsorted(sorted_values, x) / len(sorted_values)) * 100
            )
        
        # 5. 몬테카를로 시뮬레이션
        S0 = series.iloc[-1]
        log_returns = np.log(1 + series.pct_change()).dropna()
        drift = log_returns.mean() - (0.5 * log_returns.var())
        stdev = log_returns.std()
        
        daily_returns = np.exp(
            drift + stdev * np.random.normal(0, 1, (forecast_days, iterations))
        )
        
        price_list = np.zeros_like(daily_returns)
        price_list[0] = S0
        for t in range(1, forecast_days):
            price_list[t] = price_list[t - 1] * daily_returns[t]
        
        final_prices = price_list[-1]
        sim_returns_pct = ((final_prices - S0) / S0) * 100
        
        # ✅ 6. 전체 기간 롤링 윈도우 수익률 (그래프2용)
        # returns는 이미 모든 시작점에서 N일 후 수익률임
        all_returns_pct = returns.values * 100  # 퍼센트로 변환
        
        return {
            "current_price": S0,
            "data_start": series.index[0].strftime('%Y-%m-%d'),
            "data_end": series.index[-1].strftime('%Y-%m-%d'),
            "price_list": price_list,
            "returns_pct": sim_returns_pct,  # 시뮬레이션 수익률
            "all_returns_pct": all_returns_pct,  # ✅ 롤링 윈도우 수익률
            "ticker_symbol": ticker_symbol,
            "days": forecast_days,
            "percentile": float(rank_ts.iloc[-1]) if len(rank_ts) > 0 else 0,
            "rank_ts": rank_ts,
            "rank_mode": rank_mode,
            "series": series
        }, None
        
    except Exception as e:
        return None, f"시뮬레이션 분석 오류: {e}"
# ==============================================================================
# B2. 퀀트 분석 (Streamlit용)
# ==============================================================================
@st.cache_data(ttl=3600)
def run_quant_analysis_streamlit(ticker_symbol, start_date, end_date, 
                                  lookback=252, rank_mode='relative'):
    """
    퀀트 리스크 지표 분석을 실행합니다. (Streamlit 버전)
    """
    try:
        # 1. 데이터 로드
        hist_data, error = load_historical_data(ticker_symbol, start_date, end_date)
        if error:
            return None, error
        
        # 2. 종가 추출
        if isinstance(hist_data.columns, pd.MultiIndex):
            series = hist_data['Close'].iloc[:, 0].dropna()
        else:
            series = hist_data['Close'].dropna()
        
        series.index = pd.to_datetime(series.index).tz_localize(None)
        series = series.sort_index()
        
        if len(series) < lookback + 1:
            return None, "데이터가 부족합니다."
        
        # 3. 수익률 계산
        returns = series.pct_change(lookback).dropna()
        
        # 4. 백분위 순위
        if rank_mode == 'absolute':
            full_returns = returns
            sorted_values = np.sort(full_returns.values)
            percentile = returns.apply(
                lambda x: (np.searchsorted(sorted_values, x) / len(sorted_values)) * 100
            )
        else:
            sorted_values = np.sort(returns.values)
            percentile = returns.apply(
                lambda x: (np.searchsorted(sorted_values, x) / len(sorted_values)) * 100
            )
        
        # 5. Z-score 계산
        z_score = (returns - returns.mean()) / returns.std()
        z_scaled = (z_score.clip(-3, 3) + 3) / 6 * 100
        
        # 6. 복합 지수
        composite_idx = (percentile + z_scaled) / 2
        
        return {
            "percentile": percentile,
            "data_start": series.index[0].strftime('%Y-%m-%d'),
            "data_end": series.index[-1].strftime('%Y-%m-%d'),
            "z_score": z_score,
            "composite_idx": composite_idx,
            "lookback": lookback,
            "ticker_symbol": ticker_symbol,
            "current_val": composite_idx.iloc[-1] if len(composite_idx) > 0 else 0,
            "rank_mode": rank_mode,
            "series": series  # ✅ 추가
        }, None
        
    except Exception as e:
        return None, f"퀀트 분석 오류: {e}"

# ==============================================================================
# B3. 추세선 분석 (Streamlit용)
# ==============================================================================
@st.cache_data(ttl=3600)
def run_trend_analysis_streamlit(ticker_symbol, start_date, end_date):
    """
    장기 추세선 분석을 실행합니다. (Streamlit 버전)
    
    Parameters:
    -----------
    ticker_symbol : str
        분석할 티커 심볼
    start_date : str
        시작 날짜
    end_date : str
        종료 날짜
    
    Returns:
    --------
    dict : 분석 결과
    """
    try:
        # 1. 데이터 로드
        hist_data, error = load_historical_data(ticker_symbol, start_date, end_date)
        if error:
            return None, error
        
        # 2. 종가 추출
        if isinstance(hist_data.columns, pd.MultiIndex):
            series = hist_data['Close'].iloc[:, 0].dropna()
        else:
            series = hist_data['Close'].dropna()
        
        series.index = pd.to_datetime(series.index).tz_localize(None)
        series = series.sort_index()
        
        if len(series) < 100:
            return None, "최소 100일 이상의 데이터가 필요합니다."
        
        # 3. 로그 추세선 계산
        log_prices = np.log(series.values)
        x = np.arange(len(series))
        
        coeffs = np.polyfit(x, log_prices, 1)
        trend_line = np.polyval(coeffs, x)
        
        residuals = log_prices - trend_line
        std_residual = np.std(residuals)
        
        # 4. 밴드 계산
        upper_line = np.exp(trend_line + 2 * std_residual)
        middle_line = np.exp(trend_line)
        lower_line = np.exp(trend_line - 2 * std_residual)
        
        current_price = series.iloc[-1]
        band_position = ((current_price - lower_line[-1]) / 
                        (upper_line[-1] - lower_line[-1])) * 100
        
        return {
            "series": series,
            "data_start": series.index[0].strftime('%Y-%m-%d'),
            "data_end": series.index[-1].strftime('%Y-%m-%d'),
            "upper_line": upper_line,
            "middle_line": middle_line,
            "lower_line": lower_line,
            "current_price": current_price,
            "current_middle": middle_line[-1],
            "band_position": band_position,
            "ticker_symbol": ticker_symbol
        }, None
        
    except Exception as e:
        return None, f"추세선 분석 오류: {e}"

# ==============================================================================
# B4. Plotly 시뮬레이션 차트 
# ==============================================================================
def draw_plotly_simulation(data, show_label=True, max_paths=100):

    S0 = data["current_price"]
    price_list = data["price_list"]
    days = data["days"]
    ticker = data.get("ticker_symbol", "")
    start_date = data.get("data_start", "")
    end_date = data.get("data_end", "")
    
    # 티커명 매핑
    ticker_names = {
        "^GSPC": "S&P 500", "^KS11": "KOSPI", "^IXIC": "NASDAQ",
        "^DJI": "Dow Jones", "QQQ": "QQQ", "SPY": "SPY", "SCHD": "SCHD", "DIA": "DIA"
    }
    display_name = ticker_names.get(ticker, ticker)
    
    # 수익률로 변환
    returns_paths_all = (price_list / S0 - 1) * 100
    paths_subset = returns_paths_all[:, :max_paths]
    x = np.arange(days)
    
    # 백분위 계산
    p95 = np.percentile(returns_paths_all, 95, axis=1)
    p75 = np.percentile(returns_paths_all, 75, axis=1)
    p50 = np.percentile(returns_paths_all, 50, axis=1)
    p25 = np.percentile(returns_paths_all, 25, axis=1)
    p5 = np.percentile(returns_paths_all, 5, axis=1)
    
    # Figure 생성
    fig = go.Figure()
    
    # 개별 경로 (회색, 투명)
    for i in range(paths_subset.shape[1]):
        fig.add_trace(go.Scatter(
            x=x, y=paths_subset[:, i],
            mode='lines',
            line=dict(color='rgba(93, 109, 126, 0.25)', width=0.7),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # 90% 범위
    fig.add_trace(go.Scatter(
        x=x, y=p95, mode='lines',
        line=dict(color='rgba(52, 152, 219, 0)', width=0),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=x, y=p5, fill='tonexty', mode='lines',
        line=dict(color='rgba(52, 152, 219, 0)', width=0),
        fillcolor='rgba(52, 152, 219, 0.15)',
        name='90% 범위', hoverinfo='skip'
    ))
    
    # 50% 범위
    fig.add_trace(go.Scatter(
        x=x, y=p75, mode='lines',
        line=dict(color='rgba(41, 128, 185, 0)', width=0),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=x, y=p25, fill='tonexty', mode='lines',
        line=dict(color='rgba(41, 128, 185, 0)', width=0),
        fillcolor='rgba(41, 128, 185, 0.25)',
        name='50% 범위', hoverinfo='skip'
    ))
    
    # 중윗값
    fig.add_trace(go.Scatter(
        x=x, y=p50, mode='lines',
        line=dict(color='#1c4966', width=2),
        name='중윗값'
    ))
    
    # 0선
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, opacity=0.5)
    
    # 제목
    title_text = f"{display_name} ({start_date} ~ {end_date}) 시뮬레이션"
    
    # 레이아웃
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=14)),
        xaxis_title="거래일",
        yaxis_title="수익률 (%)",
        template="plotly_white",
        height=400,
        hovermode="x unified",
        showlegend=False,
        margin=dict(l=50, r=20, t=40, b=40)
    )
    
    # 레이블 (중윗값 표시)
    if show_label and len(p50) > 0:
        median_val = p50[-1]
        fig.add_annotation(
            xref="paper", yref="y",
            x=0.98, y=median_val,
            text=f"중윗값: {median_val:+.1f}%",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#1c4966",
            borderwidth=1,
            xanchor="right",
            font=dict(size=10)
        )
    
    return fig


# ==============================================================================
# B5. Plotly 확률분포 차트 (실제 252일 후 수익률, 2σ 기준 색상)
# ==============================================================================
def draw_plotly_distribution(data):
    """실제 252일 후 수익률 분포 (2σ 기준 색상 구분)"""
    # 전체 기간 롤링 윈도우 수익률 사용
    rets = data.get("all_returns_pct", data["returns_pct"])
    ticker = data.get("ticker_symbol", "")
    days = data["days"]
    start_date = data.get("data_start", "")
    end_date = data.get("data_end", "")
    
    # ✅ 현재 수익률 계산 (가장 최근 252일 수익률)
    if len(rets) > 0:
        current_return = rets[-1]  # 가장 최근 값
    else:
        current_return = None
    
    # 티커명 매핑
    ticker_names = {
        "^GSPC": "S&P 500", "^KS11": "KOSPI", "^IXIC": "NASDAQ",
        "^DJI": "Dow Jones", "QQQ": "QQQ", "SPY": "SPY", "SCHD": "SCHD", "DIA": "DIA"
    }
    display_name = ticker_names.get(ticker, ticker)
    
    # 통계 계산
    mean_ret = np.mean(rets)
    std_ret = np.std(rets)
    win_rate = np.mean(rets > 0) * 100
    var_95 = np.percentile(rets, 5)
    
    # 2σ 기준
    upper_2sigma = mean_ret + 2 * std_ret
    lower_2sigma = mean_ret - 2 * std_ret
    
    # Figure 생성
    fig = go.Figure()
    
    # 히스토그램 데이터 생성
    hist_data, bin_edges = np.histogram(rets, bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # ✅ 색상 계산: 2σ 기준으로 4가지 색상
    colors = []
    for center in bin_centers:
        if lower_2sigma <= center <= upper_2sigma:
            # 2σ 내부 (중립 구간)
            if center >= 0:
                # 수익: 초록 + 살짝 빨강
                colors.append('rgba(170, 220, 170, 0.6)')  # 연한 초록(수익)
            else:
                # 손실: 초록 + 살짝 파랑
                colors.append('rgba(170, 210, 225, 0.6)')  # 연한 초록(손실)
        elif center > upper_2sigma:
            # 과도한 상승
            colors.append('rgba(255, 0, 0, 0.8)')  # 진한 빨강
        else:
            # 과도한 하락
            colors.append('rgba(0, 0, 255, 0.8)')  # 진한 파랑
    
    # 히스토그램 추가
    fig.add_trace(go.Bar(
        x=bin_centers,
        y=hist_data,
        marker=dict(
            color=colors, 
            line=dict(color='white', width=0.5)
        ),
        showlegend=False
    ))
    
    # 0선 (검은색 실선)
    fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=2)
    
    # ✅ 평균선 (라임그린 점선)
    fig.add_vline(x=mean_ret, line_dash="dash", line_color="limegreen", line_width=2)
    
    # 2σ 경계선
    fig.add_vline(x=upper_2sigma, line_dash="dot", line_color="red", line_width=1.5, opacity=0.7)
    fig.add_vline(x=lower_2sigma, line_dash="dot", line_color="blue", line_width=1.5, opacity=0.7)
    
    # ✅ 현재 수익률 표시 (검은색 화살표)
    if current_return is not None:
        # 히스토그램 최대 높이 찾기
        max_height = np.max(hist_data)
        
        # 화살표 annotation 추가
        fig.add_annotation(
            x=current_return,
            y=max_height * 0.05,  # 히스토그램 위쪽에 배치
            text=f"현재<br>{current_return:+.1f}%",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor="black",
            ax=0,
            ay=-40,  # 화살표 길이
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="black",
            borderwidth=2,
            font=dict(size=10, color="black", family="Arial Black")
        )
    
    # 통계 텍스트
    stats_text = (
        f"승률: {win_rate:.1f}%<br>"
        f"평균 수익률: {mean_ret:+.1f}%<br>"
        f"리스크(하위5%): {var_95:.1f}%"
    )
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.98, y=0.98,
        text=stats_text,
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="black",
        borderwidth=1,
        xanchor="right",
        yanchor="top",
        font=dict(size=11)
    )
    
    # 제목
    title_text = f"{display_name} {days}일 수익률 확률분포"
    
    # 레이아웃
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=14)),
        xaxis_title="수익률 (%)",
        yaxis_title="빈도",
        template="plotly_white",
        height=400,
        margin=dict(l=50, r=20, t=40, b=40)
    )
    
    return fig
# ==============================================================================
# B6. Plotly 백분위 차트 (수정: 레이블 내부 이동, 타이틀 제거)
# ==============================================================================
def draw_plotly_percentile(data, show_price_bg=False, show_label=True):
    """백분위 순위 차트 (레이블 그래프 내부)"""
    
    # 데이터 추출
    if "rank_ts" in data:
        rank_ts = data["rank_ts"]
    elif "percentile" in data:
        rank_ts = data["percentile"]
    else:
        return None
    
    if len(rank_ts) == 0:
        return None
    
    ticker = data.get("ticker_symbol", "")
    start_date = data.get("data_start", "")  # ✅ 추가
    end_date = data.get("data_end", "")      # ✅ 추가
    
    # ✅ 티커명 매핑
    ticker_names = {
        "^GSPC": "S&P 500", "^KS11": "KOSPI", "^IXIC": "NASDAQ",
        "^DJI": "Dow Jones", "QQQ": "QQQ", "SPY": "SPY", "SCHD": "SCHD", "DIA": "DIA"
    }
    display_name = ticker_names.get(ticker, ticker)
    
    # Figure 생성 (2축)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 가격 배경 (항상 표시)
    if show_price_bg and "series" in data:
        series = data["series"]
        fig.add_trace(
            go.Scatter(
                x=series.index, y=series.values,
                mode='lines',
                line=dict(color='gray', width=1),
                opacity=0.3,
                name='가격',
                yaxis='y2',
                showlegend=False
            ),
            secondary_y=True
        )
    
    # 백분위 선
    fig.add_trace(
        go.Scatter(
            x=rank_ts.index, y=rank_ts.values,
            mode='lines',
            line=dict(color='#2980b9', width=2),
            name='백분위',
            showlegend=False
        ),
        secondary_y=False
    )
    
    # 기준선
    fig.add_hline(y=75, line_dash="dash", line_color="red", line_width=2, opacity=0.5)
    fig.add_hline(y=50, line_dash="dash", line_color="limegreen", line_width=2, opacity=0.5)
    fig.add_hline(y=25, line_dash="dash", line_color="blue", line_width=2, opacity=0.5)
    
    # 현재값 표시 (그래프 내부로)
    if show_label and len(rank_ts) > 0:
        current_val = rank_ts.iloc[-1]
        fig.add_trace(
            go.Scatter(
                x=[rank_ts.index[-1]], y=[current_val],
                mode='markers',
                marker=dict(size=10, color='black'),
                showlegend=False
            ),
            secondary_y=False
        )
        
        # 레이블을 그래프 내부로 이동
        fig.add_annotation(
            xref="paper", yref="y",
            x=0.98, y=current_val,
            text=f"{current_val:.1f}%",
            showarrow=False,
            bgcolor="rgba(255, 255, 0, 0.7)",
            bordercolor="black",
            borderwidth=1,
            xanchor="right",
            font=dict(size=10, color="black", family="Arial Black")
        )
    
    # ✅ 제목 텍스트 정의
    title_text = f"{display_name} ({start_date} ~ {end_date}) 백분위 순위"
    
    # 레이아웃
    fig.update_yaxes(title_text="백분위 (%)", range=[-10, 110], secondary_y=False)
    if show_price_bg:
        fig.update_yaxes(title_text="", showticklabels=False, secondary_y=True)
    
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=14)),  # ✅ 제목 추가
        xaxis_title="",
        template="plotly_white",
        height=400,
        hovermode="x unified",
        showlegend=False,
        margin=dict(l=50, r=20, t=40, b=40)  # ✅ 상단 여백 증가
    )
    
    return fig


# ==============================================================================
# Z-Score 시계열 차트 (백분위 차트 대체용)
# ==============================================================================
def draw_plotly_zscore(data, show_price_bg=False, show_label=True):
    """Z-Score 시계열 차트 (표준편차 기준 과열/저평가 분석)"""
    
    # 데이터 추출
    if "z_score" not in data:
        return None
    
    z_score = data["z_score"]
    
    if len(z_score) == 0:
        return None
    
    ticker = data.get("ticker_symbol", "")
    start_date = data.get("data_start", "")
    end_date = data.get("data_end", "")
    
    # 티커명 매핑
    ticker_names = {
        "^GSPC": "S&P 500", "^KS11": "KOSPI", "^IXIC": "NASDAQ",
        "^DJI": "Dow Jones", "QQQ": "QQQ", "SPY": "SPY", "SCHD": "SCHD", "DIA": "DIA"
    }
    display_name = ticker_names.get(ticker, ticker)
    
    # Figure 생성 (2축)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 가격 배경 (옵션) - ✅ 세로 눈금 숨김
    if show_price_bg and "series" in data:
        series = data["series"]
        fig.add_trace(
            go.Scatter(
                x=series.index, y=series.values,
                mode='lines',
                line=dict(color='gray', width=1),
                opacity=0.3,
                name='가격',
                yaxis='y2',
                showlegend=False
            ),
            secondary_y=True
        )
    
    # 배경 색상 영역 (과열/저평가 구간)
    # 과열 구간 (+2σ 이상)
    fig.add_hrect(
        y0=2, y1=3,
        fillcolor="rgba(255, 0, 0, 0.1)",
        layer="below",
        line_width=0,
        secondary_y=False
    )
    
    # 주의 구간 (+1σ ~ +2σ)
    fig.add_hrect(
        y0=1, y1=2,
        fillcolor="rgba(255, 165, 0, 0.1)",
        layer="below",
        line_width=0,
        secondary_y=False
    )
    
    # 정상 구간 (-1σ ~ +1σ)
    fig.add_hrect(
        y0=-1, y1=1,
        fillcolor="rgba(0, 255, 0, 0.05)",
        layer="below",
        line_width=0,
        secondary_y=False
    )
    
    # 주의 구간 (-2σ ~ -1σ)
    fig.add_hrect(
        y0=-2, y1=-1,
        fillcolor="rgba(0, 165, 255, 0.1)",
        layer="below",
        line_width=0,
        secondary_y=False
    )
    
    # 저평가 구간 (-2σ 이하)
    fig.add_hrect(
        y0=-3, y1=-2,
        fillcolor="rgba(0, 0, 255, 0.1)",
        layer="below",
        line_width=0,
        secondary_y=False
    )
    
    # Z-Score 라인
    fig.add_trace(
        go.Scatter(
            x=z_score.index, y=z_score.values,
            mode='lines',
            line=dict(color='#2c3e50', width=2.5),
            name='Z-Score',
            showlegend=False
        ),
        secondary_y=False
    )
    
    # 기준선 (수평선)
    fig.add_hline(y=2, line_dash="dot", line_color="red", line_width=2, opacity=0.7)
    #fig.add_hline(y=1, line_dash="dot", line_color="orange", line_width=1.5, opacity=0.5)
    # ✅ 0선을 점선으로 변경
    fig.add_hline(y=0, line_dash="dash", line_color="limegreen", line_width=2, opacity=0.7)
    #fig.add_hline(y=-1, line_dash="dot", line_color="dodgerblue", line_width=1.5, opacity=0.5)
    fig.add_hline(y=-2, line_dash="dot", line_color="blue", line_width=2, opacity=0.7)
    
    # 현재값 표시
    if show_label and len(z_score) > 0:
        current_val = z_score.iloc[-1]
        
        # 현재 포인트
        fig.add_trace(
            go.Scatter(
                x=[z_score.index[-1]], y=[current_val],
                mode='markers',
                marker=dict(size=12, color='black', line=dict(color='yellow', width=2)),
                showlegend=False
            ),
            secondary_y=False
        )
        
        # 상태 판단
        if current_val >= 2:
            status = "🔴 과열"
            color = "red"
        elif current_val >= 1:
            status = "🟠 주의"
            color = "orange"
        elif current_val >= -1:
            status = "🟢 정상"
            color = "limegreen"
        elif current_val >= -2:
            status = "🔵 관심"
            color = "dodgerblue"
        else:
            status = "💎 저평가"
            color = "blue"
        
        # ✅ 레이블을 그래프 내부로 이동 (x=0.02로 변경)
        fig.add_annotation(
            xref="paper", yref="y",
            x=0.90, y=current_val,  # 왼쪽 내부로 이동
            text=f"{status}<br>{current_val:+.2f}σ",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor=color,
            borderwidth=2,
            xanchor="left",  # 왼쪽 정렬
            font=dict(size=11, color=color, family="Arial Black")
        )
    
    # 제목
    title_text = f"{display_name} ({start_date} ~ {end_date}) Z-Score 분석"
    
    # 레이아웃
    fig.update_yaxes(
        title_text="Z-Score (표준편차)", 
        range=[-4.0, 4.0], 
        secondary_y=False
    )
    
    # ✅ 가격축 완전히 숨김 (눈금 + 타이틀 모두 제거)
    fig.update_yaxes(
        title_text="", 
        showticklabels=False,
        showgrid=False,  # 격자선도 숨김
        secondary_y=True
    )
    
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=14)),
        xaxis_title="",
        template="plotly_white",
        height=400,
        hovermode="x unified",
        showlegend=False,
        margin=dict(l=50, r=20, t=40, b=40)
    )
    
    return fig


# ==============================================================================
# B7. Plotly 추세선 차트 (수정: 레이블 내부 이동)
# ==============================================================================
def draw_plotly_trend(data):
    """추세선 차트 (레이블 그래프 내부)"""
    series = data["series"]
    upper = data["upper_line"]
    middle = data["middle_line"]
    lower = data["lower_line"]
    current_price = data["current_price"]
    band_pos = data["band_position"]
    ticker = data.get("ticker_symbol", "")
    start_date = data.get("data_start", "")
    end_date = data.get("data_end", "")
    
    ticker_names = {
        "^GSPC": "S&P 500", "^KS11": "KOSPI", "^IXIC": "NASDAQ",
        "^DJI": "Dow Jones", "QQQ": "QQQ", "SPY": "SPY", "SCHD": "SCHD", "DIA": "DIA"
    }
    display_name = ticker_names.get(ticker, ticker)
    
    # Figure 생성
    fig = go.Figure()
    
    # 정상 범위 (채우기)
    fig.add_trace(go.Scatter(
        x=series.index, y=upper,
        mode='lines',
        line=dict(color='rgba(231, 76, 60, 0)', width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=series.index, y=lower,
        fill='tonexty',
        mode='lines',
        line=dict(color='rgba(231, 76, 60, 0)', width=0),
        fillcolor='rgba(128, 128, 128, 0.1)',
        name='정상 범위',
        hoverinfo='skip',
        showlegend=False
    ))
    
    # 추세선들
    fig.add_trace(go.Scatter(
        x=series.index, y=upper,
        mode='lines',
        line=dict(color='red', width=2, dash='dot'),
        name='상한선',
        opacity=0.8,
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=series.index, y=middle,
        mode='lines',
        line=dict(color='limegreen', width=2, dash='dash'),
        name='중앙선',
        opacity=0.8,
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=series.index, y=lower,
        mode='lines',
        line=dict(color='blue', width=2, dash='dot'),
        name='하한선',
        opacity=0.8,
        showlegend=False
    ))
    
    # 실제 가격
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values,
        mode='lines',
        line=dict(color='black', width=1.5),
        name=display_name,
        showlegend=False
    ))
    
    # 현재 위치
    fig.add_trace(go.Scatter(
        x=[series.index[-1]], y=[current_price],
        mode='markers',
        marker=dict(size=15, color='red', line=dict(color='black', width=2)),
        showlegend=False
    ))
    
    # 현재가 주석 (그래프 내부로 이동)
    fig.add_annotation(
        xref="paper", yref="y",
        x=0.98, y=current_price,
        text=f"${current_price:,.0f}<br>{band_pos:.0f}%",
        showarrow=False,
        bgcolor="rgba(255, 255, 0, 0.8)",
        bordercolor="black",
        borderwidth=1,
        xanchor="right",
        font=dict(size=10, color="black", family="Arial Black")
    )
    
    # 경고 메시지 (그래프 내부 좌상단)
    if band_pos > 80:
        warning = "⚠️ 과열"
        color = '#e74c3c'
    elif band_pos < 20:
        warning = "💡 저평가"
        color = '#3498db'
    else:
        warning = "✅ 정상"
        color = '#2ecc71'
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        text=warning,
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor=color,
        borderwidth=2,
        xanchor="left",
        yanchor="top",
        font=dict(size=11, color=color, family="Arial Black")
    )
    title_text = f"{display_name} ({start_date} ~ {end_date}) 장기 추세 로그스케일"
    


    # 레이아웃 (타이틀 제거, 로그 스케일)
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=14)),
        xaxis_title="",
        yaxis_title=f"{display_name} (로그)",
        yaxis_type="log",
        template="plotly_white",
        height=500,
        hovermode="x unified",
        showlegend=False,
        margin=dict(l=50, r=20, t=20, b=40)
    )
    
    return fig



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
    st.session_state.active_tab = "Tab 1 빅테크 PER"
# DCA 티커 기본값: QQQ
if 'dca_ticker_value' not in st.session_state:
    st.session_state.dca_ticker_value = DCA_DEFAULT_TICKER
# 다중 티커 입력값 초기화
if 'multi_ticker_input_value' not in st.session_state:
    st.session_state.multi_ticker_input_value = ""


# ==============================================================================
# 사이드바: 탭별 설정 이원화
# ==============================================================================
with st.sidebar:
    st.header("⚙️ 기본 설정")
    
    current_tab = st.session_state.active_tab
    
    # ========================================================================
    # 탭 1, 2, 3용 설정 (기존 단기 분석)
    # ========================================================================
    if current_tab in ["Tab 1 빅테크 PER", "Tab 2 적립식 투자", "Tab 3 다중 티커 비교"]:
        
        st.markdown("### 📅 단기 분석 설정")
        
        ticker_symbol = None
        
        # 1. 티커 입력 (DCA 탭에만 표시) - ✅ 5개 티커 + 직접입력
        if current_tab == "Tab 2 적립식 투자":
            ticker_options_dca = {
                "^IXIC": "^IXIC (Nasdaq)",
                "^GSPC": "^GSPC (S&P500)",
                "^DJI": "^DJI (Dow Jones)",
                "^KS11": "^KS11 (KOSPI)",
                "SCHD": "SCHD"
            }
            
            if 'dca_ticker_value' not in st.session_state:
                st.session_state.dca_ticker_value = "^GSPC"  # 기본값 S&P500
            
            # 현재 값이 옵션에 있으면 드롭다운에서 선택, 없으면 "직접 입력" 선택
            current_ticker = st.session_state.dca_ticker_value
            if current_ticker in ticker_options_dca:
                default_index = list(ticker_options_dca.keys()).index(current_ticker)
            else:
                default_index = 0  
            
            selected_option = st.selectbox(
                "DCA 분석 티커:",
                list(ticker_options_dca.values()) + ["직접 입력"],
                index=default_index,
                key="dca_ticker_dropdown"
            )
            
            if selected_option == "직접 입력":
                ticker_symbol = st.text_input(
                    "티커 입력:",
                    value=current_ticker if current_ticker not in ticker_options_dca else "",
                    key="dca_ticker_manual",
                    help="예: QQQ, AAPL, TSLA"
                ).upper().strip()
            else:
                # 드롭다운에서 선택한 경우
                ticker_symbol = next(k for k, v in ticker_options_dca.items() if v == selected_option)
            
            st.session_state.dca_ticker_value = ticker_symbol
        else:
            ticker_symbol = "N/A_Ignored"
        
        # 2. 기간 선택
        period_options = {"1년": 365, "2년": 730, "3년": 3 * 365, "5년": 1825, "10년": 10 * 365}
        
        default_period_key = "1년"
        default_period_index = list(period_options.keys()).index(default_period_key)
        
        selected_period_name = st.selectbox(
            "기간 선택:", 
            list(period_options.keys()), 
            index=default_period_index,
            key='period_select_key'
        )
        
        # 3. 날짜 계산
        days = period_options.get(selected_period_name, 365)
        start_date_default = TODAY - timedelta(days=days)
        
        start_date_input = st.date_input(
            "시작 날짜:",
            value=start_date_default,
            max_value=TODAY,
            key=f'start_date_key_{selected_period_name}'
        )
        end_date_input = st.date_input(
            "최종 날짜:", 
            value=TODAY, 
            max_value=TODAY, 
            key='end_date_key'
        )
        
        start_date_final = start_date_input.strftime('%Y-%m-%d')
        end_date_final = end_date_input.strftime('%Y-%m-%d')
        end_date_common = end_date_input
    
    # ========================================================================
    # 탭 4용 설정 (장기 퀀트 분석)
    # ========================================================================
    elif current_tab == "Tab 4 퀀트 분석":
        
        st.markdown("### 📊 장기 퀀트 분석 설정")
        
        # 1. 티커 선택 - ✅ 5개 티커 + 직접입력
        ticker_options_quant = {
            "^IXIC": "^IXIC (Nasdaq)",
            "^GSPC": "^GSPC (S&P500)",
            "^DJI": "^DJI (Dow Jones)",
            "^KS11": "^KS11 (KOSPI)",
            "SCHD": "SCHD"
        }
        
        if 'quant_ticker_value' not in st.session_state:
            st.session_state.quant_ticker_value = "^GSPC"  # 기본값 S&P500
        
        # 현재 값이 옵션에 있으면 드롭다운에서 선택, 없으면 "직접 입력" 선택
        current_ticker_quant = st.session_state.quant_ticker_value
        if current_ticker_quant in ticker_options_quant:
            default_index_quant = list(ticker_options_quant.keys()).index(current_ticker_quant)
        else:
            default_index_quant = 0
        
        selected_option_quant = st.selectbox(
            "분석 티커:",
            list(ticker_options_quant.values()) + ["직접 입력"],
            index=default_index_quant,
            key="quant_ticker_dropdown"
        )
        
        if selected_option_quant == "직접 입력":
            ticker_quant = st.text_input(
                "티커 입력:",
                value=current_ticker_quant if current_ticker_quant not in ticker_options_quant else "",
                key="quant_ticker_manual",
                help="예: QQQ, AAPL, TSLA"
            ).upper().strip()
        else:
            # 드롭다운에서 선택한 경우
            ticker_quant = next(k for k, v in ticker_options_quant.items() if v == selected_option_quant)
        
        st.session_state.quant_ticker_value = ticker_quant
        
        # 2. 티커 시작 날짜 조회
        try:
            temp_data, _ = load_historical_data(ticker_quant, start_date="1990-01-01", end_date=TODAY.strftime('%Y-%m-%d'))
            if temp_data is not None and not temp_data.empty:
                ticker_first_date = temp_data.index[0].date()
            else:
                ticker_first_date = None
        except:
            ticker_first_date = None
        
        # 3. 전체 기간 분석 체크박스
        use_full_period = st.checkbox(
            "티커 전체 기간 분석",
            value=False,
            help="체크 시 해당 티커의 최초 거래일부터 분석합니다",
            key="quant_use_full_period"
        )
        
        # 4. 시작 날짜 설정
        if use_full_period and ticker_first_date:
            start_date_quant = ticker_first_date
            st.info(f"✅ 전체 기간 분석: {ticker_first_date.strftime('%Y-%m-%d')}부터")
        else:
            default_start_quant = TODAY - timedelta(days=15*365)
            start_date_quant = st.date_input(
                "시작 날짜:",
                value=default_start_quant,
                max_value=TODAY,
                key='quant_start_date_key'
            )
        
        # 5. 종료 날짜
        end_date_common = st.date_input(
            "최종 날짜:",
            value=TODAY,
            max_value=TODAY,
            key='quant_end_date_key'
        )
        
        # 6. 분석 기간 (일수)
        lookback_days = st.number_input(
            "분석 기간(일):",
            min_value=30,
            max_value=1000,
            value=252,
            step=1,
            help="백분위 순위 및 시뮬레이션 계산에 사용할 기간",
            key="quant_lookback_input"
        )
        
        st.markdown("---")
        
        # 안내 문구
        ticker_start_info = f"\n티커 시작: {ticker_first_date.strftime('%Y-%m-%d')}" if ticker_first_date else ""
        
        st.info(
            f"📌 **현재 설정**\n\n"
            f"티커: {ticker_quant}{ticker_start_info}\n\n"
            f"기간: {start_date_quant.strftime('%Y-%m-%d')} ~ {end_date_common.strftime('%Y-%m-%d')}\n\n"
            f"분석 일수: {lookback_days}일"
        )
        
        ticker_symbol = "N/A_Ignored"
        start_date_final = start_date_quant.strftime('%Y-%m-%d')
        end_date_final = end_date_common.strftime('%Y-%m-%d')
    
    else:
        ticker_symbol = "N/A_Ignored"
        start_date_final = (TODAY - timedelta(days=365)).strftime('%Y-%m-%d')
        end_date_final = TODAY.strftime('%Y-%m-%d')
        end_date_common = TODAY
        ticker_quant = "^GSPC"
        start_date_quant = TODAY - timedelta(days=15*365)
        lookback_days = 252

# ==============================================================================
# 6. 메뉴 설정 (유지)
# ==============================================================================

menu_options = ["Tab 1 빅테크 PER", "Tab 2 적립식 투자", "Tab 3 다중 티커 비교", "Tab 4 퀀트 분석"]

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
                if option == "Tab 3 다중 티커 비교":
                    st.session_state['multi_ticker_input_value'] = MULTI_DEFAULT_TICKERS

                st.rerun()

st.markdown("---")

# ==============================================================================
# 7. Tab 구현부 (수정)
# ==============================================================================

# ------------------------------------------------------------------------------
# 탭 1: 재무 분석 (빅테크) (수정: PER 기준선, 기준표, get_per_color 호출 제거)
# ------------------------------------------------------------------------------
if st.session_state.active_tab == "Tab 1 빅테크 PER":  


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
elif st.session_state.active_tab == "Tab 2 적립식 투자":

    # 1. 데이터 로드 (탭 진입 시점에만 실행)
    if not ticker_symbol or ticker_symbol == "N/A_Ignored":
        st.warning("DCA 분석을 위해 사이드바에 유효한 티커를 입력해 주세요.")
        st.stop()


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
elif st.session_state.active_tab == "Tab 3 다중 티커 비교":

    


    
    # 세션 상태에서 다중 티커 입력값을 가져와 기본값으로 사용 (탭 전환 시 기본값 설정됨)
    col_multi_input, col_multi_rf = st.columns([2, 1])

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
                x=df_m['Volatility'] * 100,
                y=df_m['Return'] * 100,
                mode='markers+text',
                text=df_m['Ticker'],
                textposition='top center',

                marker=dict(
                    size=15,
                    color=df_m['Sharpe_Ratio'],
                    colorscale=custom_rwb_colorscale,
                    showscale=False,
                    line=dict(color="black", width=1.5)
                ),

                hovertemplate=(
                    "수익률 : %{y:.1f}%<br>"
                    "위험률 : %{x:.1f}%"
                    "<extra></extra>"
                )
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
            st.markdown(f"💡 **분석 결과:** 가장 매력적인 종목은 **{df_d.iloc[0]['Ticker']}**입니다.")

            st.caption(f"ℹ️ 기간: {start_date_multi}~{end_date_multi} | 기준금리 {user_rf}% 반영")

    else:
        st.info("티커를 입력해 주세요.")


# ==============================================================================
# Tab 4: 퀀트 분석 (장기 통계 분석) - 미니멀 버전
# ==============================================================================
elif st.session_state.active_tab == "Tab 4 퀀트 분석":
    
    # 티커 코드 추출
    ticker_code = ticker_quant
    end_date_quant = end_date_common.strftime('%Y-%m-%d')
    start_date_quant_str = start_date_quant.strftime('%Y-%m-%d')
    
    # ✅ 캐시 키 생성 (티커, 시작일, 종료일, 분석기간 모두 포함)
    cache_key = f"{ticker_code}_{start_date_quant_str}_{end_date_quant}_{lookback_days}"
    
    # ✅ 설정이 변경되면 자동으로 재분석
    if 'quant_cache_key' not in st.session_state or st.session_state.quant_cache_key != cache_key:
        
        with st.spinner(f"📊 {ticker_code} 분석 중..."):
            
            # 시뮬레이션 분석
            sim_data, sim_error = run_simulation_analysis_streamlit(
                ticker_code, 
                start_date_quant_str, 
                end_date_quant,
                forecast_days=lookback_days,
                iterations=10000,
                rank_mode='relative'
            )
            
            # 퀀트 분석
            quant_data, quant_error = run_quant_analysis_streamlit(
                ticker_code,
                start_date_quant_str,
                end_date_quant,
                lookback=lookback_days,
                rank_mode='relative'
            )
            
            # 추세선 분석
            trend_data, trend_error = run_trend_analysis_streamlit(
                ticker_code,
                start_date_quant_str,
                end_date_quant
            )
            
            # 에러 체크
            if sim_error:
                st.error(f"시뮬레이션 분석 오류: {sim_error}")
                st.stop()
            if quant_error:
                st.error(f"퀀트 분석 오류: {quant_error}")
                st.stop()
            if trend_error:
                st.error(f"추세선 분석 오류: {trend_error}")
                st.stop()
            
            # 성공 시 캐시에 저장
            st.session_state['quant_data_cache'] = {
                'sim': sim_data,
                'quant': quant_data,
                'trend': trend_data,
                'ticker': ticker_code,
                'lookback': lookback_days
            }
            st.session_state['quant_cache_key'] = cache_key  # ✅ 캐시 키 저장

    
    # 결과 시각화 (헤더 없이 그래프만)
    if 'quant_data_cache' in st.session_state:
        
        cache = st.session_state['quant_data_cache']
        sim_data = cache['sim']
        quant_data = cache['quant']
        trend_data = cache['trend']
        
        # 그래프 1: 시뮬레이션 (세로 배열)
        if sim_data:
            fig_sim = draw_plotly_simulation(sim_data, show_label=True)
            if fig_sim:
                st.plotly_chart(fig_sim, use_container_width=True)
        
        # 그래프 2: 확률분포
        if sim_data:
            fig_dist = draw_plotly_distribution(sim_data)
            if fig_dist:
                st.plotly_chart(fig_dist, use_container_width=True)
        
        # 그래프 3: Z-Score 시계열 (백분위 차트 대체) ⭐
        if quant_data:
            fig_zscore = draw_plotly_zscore(
                quant_data, 
                show_price_bg=True,  # 가격 배경 표시
                show_label=True      # 레이블 표시
            )
            if fig_zscore:
                st.plotly_chart(fig_zscore, use_container_width=True)

        # 그래프 4: 추세선
        if trend_data:
            fig_trend = draw_plotly_trend(trend_data)
            if fig_trend:
                st.plotly_chart(fig_trend, use_container_width=True)
        
        st.markdown("---")
        
        # 4-4. 요약 지표
        st.markdown("#### 📋 분석 요약")
        
        col_summary1, col_summary2, col_summary3, col_summary4 = st.columns(4)
        
        with col_summary1:
            if sim_data:
                current_percentile = sim_data.get('percentile', 0)
                st.metric(
                    label=f"{lookback_days}일 백분위 순위",
                    value=f"{current_percentile:.1f}%"
                )
        
        with col_summary2:
            if quant_data:
                composite_val = quant_data.get('current_val', 0)
                st.metric(
                    label="복합 리스크 지수",
                    value=f"{composite_val:.1f}"
                )
        
        with col_summary3:
            if quant_data and 'z_score' in quant_data:
                current_z = quant_data['z_score'].iloc[-1] if len(quant_data['z_score']) > 0 else 0
                st.metric(
                    label="Z-Score (표준편차)",
                    value=f"{current_z:+.2f}σ"
                )
        
        with col_summary4:
            if trend_data:
                band_pos = trend_data.get('band_position', 0)
                st.metric(
                    label="추세선 밴드 위치",
                    value=f"{band_pos:.1f}%"
                )
        
        # 데이터 기간 정보
        if sim_data:
            data_range = f"📅 데이터 기간: {sim_data['data_start']} ~ {sim_data['data_end']}"
            st.caption(data_range)
    
    else:
            # 초기 상태 (분석 전)
            pass
