#Import Libraries

import yfinance as yf
import streamlit as st
import datetime 
import time
import ta
import pandas as pd
import requests
import os
import plotly.graph_objects as go
from fbprophet import Prophet
#from fbprophet.plot import plot_plotly
import yfinance as yf
import numpy as np
#from load_data import load_data
import pandas_market_calendars as mcal
#from pmdarima.arima import ADFTest
#from pmdarima.arima import ndiffs
#from pmdarima.arima import auto_arima
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

#Forecasting using LSTM model
Forecasting = st.sidebar.checkbox('Forecasting')
if Forecasting:
    
    n_periods= st.sidebar.slider('Forecasting period', min_value=1, max_value=30,
                             value=5,  step=1)
    
    #prepare the new index
    nyse=mcal.get_calendar('NYSE')
    new_index=nyse.valid_days(start_date=start, end_date='2030-01-01 00:00:00')
    new_index=new_index[0:(len(data)+n_periods)]

    #Close Model
    df_close=data['Close']
    df_close.index.rename('ds',True)
    df_close=df_close.reset_index()
    df_close.columns=['ds','y']
    model_close=Prophet()
    model_close.fit(df_close)
    future_dates_close=model_close.make_future_dataframe(periods=n_periods)
    prediction_close=model_close.predict(future_dates_close)
    #Open Model
    df_open=data['Open']
    df_open.index.rename('ds',True)
    df_open=df_open.reset_index()
    df_open.columns=['ds','y']
    model_open=Prophet()
    model_open.fit(df_open)
    future_dates_open=model_open.make_future_dataframe(periods=n_periods)
    prediction_open=model_open.predict(future_dates_open)
    #High Model
    df_high=data['High']
    df_high.index.rename('ds',True)
    df_high=df_high.reset_index()
    df_high.columns=['ds','y']
    model_high=Prophet()
    model_high.fit(df_high)
    future_dates_high=model_high.make_future_dataframe(periods=n_periods)
    prediction_high=model_high.predict(future_dates_high)
    #Low Model
    df_low=data['Low']
    df_low.index.rename('ds',True)
    df_low=df_low.reset_index()
    df_low.columns=['ds','y']
    model_low=Prophet()
    model_low.fit(df_low)
    future_dates_low=model_low.make_future_dataframe(periods=n_periods)
    prediction_low=model_low.predict(future_dates_low)
    fig = go.Figure(data=[go.Candlestick(x=new_index,
                open=prediction_open.yhat,
                high=prediction_high.yhat,
                low=prediction_low.yhat,
                close=prediction_close.yhat)])

    st.plotly_chart(fig)
    
    
#Simple Moving Average
sma = st.sidebar.checkbox('Simple Moving Average')
if sma:
    
    period= st.sidebar.slider('SMA period', min_value=5, max_value=50,
                             value=20,  step=1)
    data[f'SMA {period}'] = data[price_type].rolling(period ).mean()
    st.subheader('SMA')
    st.line_chart(data[[price_type,f'SMA {period}']])   
CCI=st.sidebar.checkbox('Commodity Channel Index')
if CCI:
    ## CCI (Commodity Channel Index)

    cci = ta.trend.cci(data['High'], data['Low'], data['Close'], window=31, constant=0.015)

    # Plotting (Commodity Channel Index)
    st.header(f"Commodity Channel Index\n {company_name}")
    st.line_chart(cci)
ADC=st.sidebar.checkbox('Average Daily Change')
if ADC:
    data['day']=data.index.day_name()
    monday=data[data['day']=='Monday']['High']-data[data['day']=='Monday']['Low']
    tuesday=data[data['day']=='Tuesday']['High']-data[data['day']=='Tuesday']['Low']
    wednesday=data[data['day']=='Wednesday']['High']-data[data['day']=='Wednesday']['Low']
    thursday=data[data['day']=='Thursday']['High']-data[data['day']=='Thursday']['Low']
    friday=data[data['day']=='Friday']['High']-data[data['day']=='Friday']['Low']
    day_mean=[monday.mean(),tuesday.mean(),wednesday.mean(),thursday.mean(),friday.mean()]
    day=['Monday','Tuesday','Wednesday','Thursday','Friday']
    day_of_week=pd.DataFrame(data=day_mean,index=day,columns=['Average Daily Change'])
    try:
        day_of_week.plot()
    except:
        st.error("Pandas plot is not working")
    try:
        st.line_chart(day_of_week)
    except:
        st.error("streamlit plot is not working")
    
    
    
st.sidebar.title("About")
st.sidebar.info('This app is a simple example of '
                    'using Streamlit & Heroku to create a financial data web app.\n'
                    '\nIt is maintained by [Anas Alfadhel]\n'
                    'Check the code at https://github.com/ScarceDS/Streamlit-Heroku-App')


st.write("""
#   Thanks For Using My Application
For more information, you can contact me on alfadhel.anas@gmail.com
""")
