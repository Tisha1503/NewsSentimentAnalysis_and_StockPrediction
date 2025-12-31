import streamlit as st
import yfinance as yf
from news_analysis import SentimentEngine
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import plotly.express as px
st.set_page_config(page_title="FinBERT Intelligence Pro", layout="wide", page_icon="")

@st.cache_resource
def get_model():
    return SentimentEngine()

engine = get_model()

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { border: 1px solid #333; padding: 10px; border-radius: 5px; background-color: #161b22; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #00ffcc; }
    </style>
    """, unsafe_allow_html=True)

st.title("FinBERT Real-Time Market Intelligence")
st.caption("Quantitative NLP Pipeline | Sentiment-Price Convergence & Pearson Correlation")
st.markdown("---")

with st.sidebar:
    st.header(" Configuration")
    ticker = st.text_input("Stock Ticker", value="NVDA").upper()
    api_key = st.text_input("NewsAPI Key", type="password", help="Get yours at newsapi.org")
    period = st.selectbox("Historical Window", ["1mo", "3mo", "6mo", "1y"], index=0)
    
    st.info("**Quant Tip:** Correlation is calculated by aligning daily mean sentiment with next-day price returns.")
    st.markdown("---")
    analyze_btn = st.button("Generate Intelligence Report", use_container_width=True)

if analyze_btn:
    if not api_key:
        st.error("Please provide a NewsAPI Key.")
    else:
        with st.spinner(f"Generating Quant Report for {ticker}..."):
            raw_news = engine.fetch_news(ticker, api_key)
            sent_df, avg_score = engine.process_sentiment(raw_news) 
            stock_data = yf.download(ticker, period=period)
            
            if isinstance(stock_data.columns, pd.MultiIndex):
                stock_data.columns = stock_data.columns.get_level_values(0)

            if sent_df.empty or stock_data.empty:
                st.warning("Insufficient data. Check ticker symbol or API limits.")
            else:
                corr_val = engine.get_correlation_analysis(sent_df, stock_data)
                last_close = float(stock_data['Close'].iloc[-1])
                first_close = float(stock_data['Close'].iloc[0])                
                price_change = ((last_close - first_close) / first_close) * 100
                
                c1, c2, c3, c4 = st.columns(4)
                bias = "BULLISH" if avg_score > 0.05 else "BEARISH" if avg_score < -0.05 else "NEUTRAL"
                
                c1.metric("Market Bias", bias)
                c2.metric("Mean Sentiment", f"{avg_score:.2f}")
                c3.metric("Pearson Correlation", f"{corr_val:.2f}")
                c4.metric("Period Return", f"{price_change:.1f}%", delta=f"{price_change:.1f}%")

                st.markdown("---")

                st.subheader("Sentiment-Price Convergence Analysis")
                
                daily_sent = sent_df.groupby('Date')['Score'].mean().reset_index()
                daily_sent['Date'] = pd.to_datetime(daily_sent['Date'])

                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig.add_trace(
                    go.Scatter(x=stock_data.index, y=stock_data['Close'], name="Price", line=dict(color='#00ffcc', width=2)),
                    secondary_y=False,
                )
                
                fig.add_trace(
                    go.Bar(x=daily_sent['Date'], y=daily_sent['Score'], name="Daily Sentiment", marker_color='rgba(255, 255, 255, 0.2)'),
                    secondary_y=True,
                )

                fig.update_layout(template="plotly_dark", height=500, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                fig.update_yaxes(title_text="Stock Price ($)", secondary_y=False)
                fig.update_yaxes(title_text="Sentiment Score", secondary_y=True, range=[-1, 1])
                
                st.plotly_chart(fig, use_container_width=True)

                col_left, col_right = st.columns([2, 1])
                
                with col_left:
                    st.subheader(" Real-Time Intelligence Feed")
                    st.dataframe(sent_df.style.background_gradient(subset=['Score'], cmap='RdYlGn'), use_container_width=True)
                
                with col_right:
                    st.subheader("Polarity Distribution")
                    fig_pie = px.pie(sent_df, names="Sentiment", color="Sentiment",
                                     color_discrete_map={'POSITIVE':'#00cc66', 'NEGATIVE':'#ff3333', 'NEUTRAL':'#666666'},
                                     hole=0.4)
                    fig_pie.update_layout(template="plotly_dark")
                    st.plotly_chart(fig_pie, use_container_width=True)

                st.info(f"Report generated. Analysis based on {len(sent_df)} data points across the selected window.")