import streamlit as st
import yfinance as yf
from datetime import date


st.title("Stock Price Viewer")
ticker_symbol = st.text_input("Enter Stock Ticker Symbol:", "AAPL")
if ticker_symbol:
    c1, c2 = st.columns(spec=[1, 1])
    with c1:
        start_date = st.date_input(
            "Select a start date", date.today().replace(month=date.today().month - 3))
    with c2:
        end_date = st.date_input("Select end date", date.today())

    stock_data = yf.Ticker(ticker_symbol)
    hist = stock_data.history(start=start_date, end=end_date)

    st.subheader(f"Historical Data for {ticker_symbol}")
    st.line_chart(hist['Close'])

    # st.subheader(f"Company Info for {ticker_symbol}")
    # st.write(stock_data.info)

    fig = hist['Volume'].plot(
        title=f"Volume Traded for {ticker_symbol}", kind='bar').get_figure()
    st.pyplot(fig)

else:
    st.write("Please enter a valid stock ticker symbol.")
