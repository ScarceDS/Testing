#Import Libraries

import yfinance as yf
import streamlit as st
import datetime 
import ta
import pandas as pd
import requests
yf.pdr_override()







st.write("""
# Stock Finance Analysis Web Application 
""")

st.sidebar.header('Please Enter Your Parameters')

today = datetime.date.today()
def user_input_features():
    ticker = st.sidebar.text_input("Ticker", 'AAPL')
    price_type=st.sidebar.selectbox('Price Type',('Close', 'Open','High','Low','Adj Close'))
    start_date = st.sidebar.text_input("Start Date", '2019-01-01')
    end_date = st.sidebar.text_input("End Date", f'{today}')
    return ticker,price_type,start_date, end_date

symbol,price_type,start, end = user_input_features()

def get_symbol(symbol):
    url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)
    result = requests.get(url,timeout=5).json()
    for x in result['ResultSet']['Result']:
        if x['symbol'] == symbol:
            return x['name']
company_name = get_symbol(symbol.upper())

start = pd.to_datetime(start)
end = pd.to_datetime(end)

# Read data 
data = yf.download(symbol,start,end)

# Plotting Stock of selected Price type
st.header(f"{price_type} Price\n {company_name}")
st.line_chart(data[price_type])


#Simple Moving Average
sma = st.sidebar.checkbox('Simple Moving Average')
if sma:
    
    period= st.sidebar.slider('SMA period', min_value=5, max_value=50,
                             value=20,  step=1)
    data[f'SMA {period}'] = data[price_type].rolling(period ).mean()
    st.subheader('SMA')
    st.line_chart(data[[price_type,f'SMA {period}']])   

## CCI (Commodity Channel Index)

cci = ta.trend.cci(data['High'], data['Low'], data['Close'], window=31, constant=0.015)

# Plotting (Commodity Channel Index)
st.header(f"Commodity Channel Index\n {company_name}")
st.line_chart(cci)

st.sidebar.title("About")
st.sidebar.info('This app is a simple example of '
                    'using Streamlit & Heroku to create a financial data web app.\n'
                    '\nIt is maintained by [Anas Alfadhel]\n'
                    'Check the code at https://github.com/ScarceDS/Streamlit-Heroku-App')


st.write("""
#   Thanks For Using My Application
For more information, you can contact me on alfadhel.anas@gmail.com
""")
