import streamlit as st
import plotly.graph_objects as go
import seaborn as sns

def inject_custom_css():
    st.markdown("""
        <style>
        div[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 600; }
        div[data-testid="stMetricLabel"] { font-size: 0.85rem; }
        @media (max-width: 768px) {
            div[data-testid="stHorizontalBlock"] { display: grid !important; grid-template-columns: 1fr !important; }
        }
        </style>
    """, unsafe_allow_html=True)

def plot_dca_chart(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Price'], name='주가', opacity=0.3, yaxis='y2', line=dict(color='gray')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Current_Value'], name='평가가치', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative_Investment'], name='투자원금', line=dict(color='red', dash='dash')))
    fig.update_layout(title=f"{ticker} DCA 결과", hovermode="x unified",
                      yaxis=dict(side="left"), yaxis2=dict(overlaying="y", side="right"))
    return fig

def format_value(val):
    if val >= 1e12: return f"{val / 1e12:,.2f}T"
    if val >= 1e9: return f"{val / 1e9:,.2f}B"
    return f"{val:,.2f}"

def plot_distribution_chart(df, target_col, years):
    # 최근 n개년 데이터 추출 (결측치 제외)
    data = df.iloc[:years][target_col].dropna()
    
    fig = plt.Figure(figsize=(7, 5), dpi=100)
    ax = fig.add_subplot(111)
    
    # 히스토그램 및 밀도함수(KDE)
    sns.histplot(data, kde=True, ax=ax, color='#2E59A7', bins=15, stat="density", alpha=0.5)
    
    mean_val = data.mean()
    ax.axvline(mean_val, color='red', linestyle='--', label=f'평균: {mean_val:.2f}%')
    ax.axvline(0, color='black', linewidth=1.5)
    
    ax.set_title(f'{target_col} 수익률 분포 (최근 {len(data)}년)')
    ax.set_xlabel('연간 수익률 (%)')
    ax.set_ylabel('밀도')
    ax.legend()
    
    return fig
