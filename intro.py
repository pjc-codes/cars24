import streamlit as st
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import numpy as np

st.title("Sales Dashboard 4165465465")

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    date_range = st.date_input("Date range:", [])
    region = st.selectbox("Region:", ["North", "South", "East", "West"])

# Main content
col1, col2, col3 = st.columns(3)
col1.metric("Total Sales", "$1.2M", "+12%")
col2.metric("Orders", "3,456", "+8%")
col3.metric("Avg Order", "$347", "+3%")

# Data visualization
st.subheader("Sales Trend")
chart_data = pd.DataFrame({
    "date": pd.date_range("2024-01-01", periods=30),
    "sales": np.random.randint(1000, 5000, 30)
})
st.line_chart(chart_data, x="date", y="sales")
