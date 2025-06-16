import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Trading Bot Dashboard", page_icon="🚀", layout="wide")
st.title("🚀 ETH Trading Bot Dashboard")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Account Balance", "$1,247.83", "+$247.83")
with col2:
    st.metric("Win Rate", "67.3%", "+2.3%")
with col3:
    st.metric("Total Trades", "156", "+12")
with col4:
    st.metric("Sharpe Ratio", "1.67", "+0.23")

dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=30)
values = [1000 + i*10 + (i%3)*20 for i in range(30)]

fig = go.Figure()
fig.add_trace(go.Scatter(x=dates, y=values, name="Portfolio Value"))
fig.update_layout(title="Portfolio Performance", height=400)
st.plotly_chart(fig, use_container_width=True)

st.success("✅ Bot Status: ONLINE")
st.info("📊 Current Position: LONG ETH (0.85 ETH)")
