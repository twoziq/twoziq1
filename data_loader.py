import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time

@st.cache_data(ttl=3600)
def load_ticker_info(ticker, max_retries=3):
    for attempt in range(max_retries):
        try:
            data = yf.Ticker(ticker)
            info = data.info
            eps = info.get('trailingEps') or info.get('forwardEps') or 0
            return {'EPS': eps, 'CompanyName': info.get('longName', ticker)}, None
        except Exception as e:
            if attempt < max_retries - 1: time.sleep(2 * (attempt + 1))
            else: return None, f"Info load error: {e}"

@st.cache_data(ttl=3600)
def load_historical_data(ticker_or_list, start_date=None, end_date=None, max_retries=3):
    for attempt in range(max_retries):
        try:
            hist = yf.download(ticker_or_list, start=start_date, end=end_date, progress=False)
            if hist.empty: return None, "데이터가 없습니다."
            return hist, None
        except Exception as e:
            if attempt < max_retries - 1: time.sleep(2 * (attempt + 1))
            else: return None, f"Data load error: {e}"

@st.cache_data(ttl=3600)
def load_big_tech_data(tickers):
    data_list = []
    tickers_obj = yf.Tickers(tickers)
    for t in tickers:
        try:
            info = tickers_obj.tickers[t].info
            m_cap = info.get('marketCap', np.nan)
            pe = info.get('trailingPE', np.nan)
            income = m_cap / pe if m_cap and pe and pe > 0 else np.nan
            data_list.append({'Ticker': t, 'MarketCap': m_cap, 'TrailingPE': pe, 'NetIncome': income})
        except:
            data_list.append({'Ticker': t, 'MarketCap': np.nan, 'TrailingPE': np.nan, 'NetIncome': np.nan})
    return pd.DataFrame(data_list)
