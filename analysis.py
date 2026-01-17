import pandas as pd
import numpy as np

def calculate_dca_simulation(df, deposit_amount, frequency):
    dca_df = df.copy()
    if isinstance(dca_df.columns, pd.MultiIndex):
        dca_df['Price'] = dca_df['Close'].iloc[:, 0]
    else:
        dca_df['Price'] = dca_df['Close']

    dca_df['Date'] = dca_df.index
    if frequency == "매일":
        invest_dates = dca_df.index
    elif frequency == "매주":
        invest_dates = dca_df.groupby(dca_df.index.isocalendar().week)['Price'].head(1).index
    else: # 매월
        invest_dates = dca_df.groupby(dca_df.index.month)['Price'].head(1).index

    dca_df['Shares_Bought'] = 0.0
    mask = dca_df.index.isin(invest_dates)
    dca_df.loc[mask, 'Shares_Bought'] = deposit_amount / dca_df.loc[mask, 'Price']
    dca_df['Total_Shares'] = dca_df['Shares_Bought'].cumsum()
    dca_df['Cumulative_Investment'] = mask.astype(int).cumsum() * deposit_amount
    dca_df['Current_Value'] = dca_df['Total_Shares'] * dca_df['Price']
    return dca_df

def calculate_multi_metrics(hist_data, rf_rate):
    if isinstance(hist_data.columns, pd.MultiIndex):
        returns = hist_data['Close'].pct_change().dropna(axis=0, how='all')
    else:
        returns = pd.DataFrame(hist_data['Close'].pct_change().dropna())
    
    annual_factor = 252
    df_m = pd.DataFrame({
        'Return': returns.mean() * annual_factor,
        'Volatility': returns.std() * np.sqrt(annual_factor)
    })
    df_m['Sharpe_Ratio'] = (df_m['Return'] - rf_rate) / df_m['Volatility']
    return df_m.reset_index().rename(columns={'index': 'Ticker'})
