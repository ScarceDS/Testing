#Import Libraries

import yfinance as yf
import itertools
import streamlit as st
import datetime 
import time
import ta
import pandas as pd
import requests
import os
import plotly.graph_objects as go
from fbprophet.diagnostics import cross_validation,performance_metrics
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import yfinance as yf
import numpy as np
#from load_data import load_data
import pandas_market_calendars as mcal
#from pmdarima.arima import ADFTest
#from pmdarima.arima import ndiffs
#from pmdarima.arima import auto_arima
yf.pdr_override()
pd.options.plotting.backend = "plotly"





st.write("""
# Stock Finance Analysis Web Application 
""")

st.sidebar.header('Please Enter Your Parameters')
number_of_tickers=st.sidebar.slider('No of Tickers to be plotted', min_value=1, max_value=5,
                                 value=1,  step=1)

today = datetime.date.today()
def user_input_features(number_of_tickers):
    Num={1:'First',2:'Second',3:'Third',4:'Forth',5:'Fifth'}
    #tickers=[1:'First Ticker',2:'Second Ticker',3:'Third Ticker',4:'Forth Ticker',5:'Fifth Ticker']
    tickers=[]
    for i in range(1,int(number_of_tickers)+1):
        ticker =st.sidebar.text_input(Num[i]+" Ticker",'AAPL')
        ticker=ticker.upper()
        tickers.append(ticker)
    price_type=st.sidebar.selectbox('Price Type',('Close', 'Open','High','Low','Adj Close'))
    start_date = st.sidebar.text_input("Start Date", '2019-01-01')
    end_date = st.sidebar.text_input("End Date", f'{today}')
    return tickers,price_type,start_date, end_date

symbol,price_type,start, end = user_input_features(number_of_tickers)

def get_symbol(symbol):
    company_names=[]
    for i in symbol:
        url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(i)
        result = requests.get(url,timeout=5).json()
        for x in result['ResultSet']['Result']:
            if x['symbol'] == i:
                company_names.append(x['name'])
    return company_names 


company_name = get_symbol(symbol)

start = pd.to_datetime(start)
end = pd.to_datetime(end)

# Read data 
def read_data(symbol):
    DATA={}
    for i in symbol:
      print(i)
      data= yf.download(i,start,end)
      DATA[i]=data
    return DATA
combined_data=read_data(symbol) 
# Plotting Stock/s of selected Price type
def Plot_data(combined_data,symbol,company_name):
    st.header(f"{price_type} Price")
    zip_=zip(symbol,company_name)
    dictionary=dict(zip_)
    df=pd.DataFrame()
    for i,j in dictionary.items():
        df[price_type+' '+j]=combined_data[i][price_type]  
    fig=df.plot()
    #fig.set_ylabel(price_type+" Price")
    #fig.set_xlabel("Date")
    #fig.set(xlabel="Date", ylabel=price_type+" Price")
    return fig
    
    
fig=Plot_data(combined_data,symbol,company_name)
st.plotly_chart(fig)
#Forecasting using fbprophet model
Forecasting = st.sidebar.checkbox('Forecasting')
if Forecasting:
    stock=st.sidebar.selectbox('Ticker',(symbol))
    n_periods= st.sidebar.slider('Forecasting period', min_value=1, max_value=30,
                                 value=5,  step=1)
    #changepoint_prior_scale=[np.arange(0.001, 0.5,0.1),0.05]
    #seasonality_prior_scale=[np.arange(0.01,10.51,0.5)]
   # holidays_prior_scale=[np.arange(0.01,10.51,0.5)]
   # seasonality_mode=['additive', 'multiplicative']
    @st.cache(suppress_st_warning=True)
    def forecast(data,price_type,n_periods):
        
        #Automatic_tuning=st.sidebar.checkbox('Automatic Tuning')

        #prepare the new index
        nyse=mcal.get_calendar('NYSE')
        new_index=nyse.valid_days(start_date=start, end_date='2030-01-01 00:00:00')
        new_index=new_index[0:(len(data)+n_periods)]
        
        #Prophet Model
        df=data[price_type]
        df.index.rename('ds',True)
        df=df.reset_index()
        df.columns=['ds','y']
        changepoint_prior_scale=st.sidebar.number_input('Changepoint_prior_scale')
        seasonality_prior_scale=st.sidebar.number_input('Seasonality_prior_scale')
        seasonality_mode=st.sidebar.selectbox('Seasonality_mode',('additive', 'multiplicative'))
        model=Prophet(changepoint_prior_scale=changepoint_prior_scale,seasonality_mode=seasonality_mode,seasonality_prior_scale=seasonality_prior_scale).fit(df)
        
        cutoffs = pd.to_datetime([df['ds'][int(0.3*len(df))],df['ds'][int(0.7*len(df))]])
        df_cv = cross_validation(model, cutoffs=cutoffs, horizon='30 days', parallel="processes")
        df_p = performance_metrics(df_cv, rolling_window=1)
        
        future_dates=model.make_future_dataframe(periods=n_periods)
        prediction=model.predict(future_dates)
        '''if Automatic_tuning:
          param_grid = {  
              'changepoint_prior_scale': [0.001,0.50],
              'seasonality_prior_scale': [0.01,2.51,10],
              'seasonality_mode':['additive', 'multiplicative']
          }
          cutoffs = pd.to_datetime([df['ds'][int(0.3*len(df))],df['ds'][int(0.7*len(df))]])
          # Generate all combinations of parameters
          all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
          rmses = []  # Store the RMSEs for each params here

          # Use cross validation to evaluate all parameters
          for params in all_params:
              m = Prophet(**params).fit(df)  # Fit model with given params
              df_cv = cross_validation(m, cutoffs=cutoffs, horizon='30 days', parallel="processes")
              df_p = performance_metrics(df_cv, rolling_window=1)
              rmses.append(df_p['rmse'].values[0])

          # Find the best parameters
          tuning_results = pd.DataFrame(all_params)
          tuning_results['rmse'] = rmses
          #print(tuning_results)
          st.sidebar.subheader(f"RMSE for the best Model is\n{tuning_results['rmse'].max()}")
          changepoint_prior_scale=tuning_results[tuning_results['rmse']==tuning_results['rmse'].max()]['changepoint_prior_scale']
          seasonality_prior_scale=tuning_results[tuning_results['rmse']==tuning_results['rmse'].max()]['seasonality_prior_scale']
          seasonality_mode=tuning_results[tuning_results['rmse']==tuning_results['rmse'].max()]['seasonality_mode']
        else:
          changepoint_prior_scale=st.sidebar.number_input('Changepoint_prior_scale')
          seasonality_prior_scale=st.sidebar.number_input('Seasonality_prior_scale')
          seasonality_mode=st.sidebar.selectbox('Seasonality_mode',('additive', 'multiplicative'))
        model=Prophet(changepoint_prior_scale=changepoint_prior_scale,seasonality_mode=seasonality_mode,seasonality_prior_scale=seasonality_prior_scale)
        model.fit(df)
        future_dates=model.make_future_dataframe(periods=n_periods)
        prediction=model.predict(future_dates)'''
         
        #st.plotly_chart(fig)
        return model,prediction  
    model,prediction=forecast(combined_data[stock],price_type,n_periods)    
    fig=plot_plotly(model,prediction,trend=True)
    st.plotly_chart(fig)
        
    
#Simple Moving Average
sma = st.sidebar.checkbox('Simple Moving Average')
if sma:
    stock=st.sidebar.selectbox('Ticker',(symbol))
    period= st.sidebar.slider('SMA period', min_value=5, max_value=50,
                             value=20,  step=1)
    data[f'SMA {period}'] = stock[price_type].rolling(period).mean()
    st.subheader('SMA')
    st.line_chart(stock[[price_type,f'SMA {period}']])   
#CCI=st.sidebar.checkbox('Commodity Channel Index')
#if CCI:
    ## CCI (Commodity Channel Index)

    #cci = ta.trend.cci(data['High'], data['Low'], data['Close'], window=31, constant=0.015)

    # Plotting (Commodity Channel Index)
    #st.header(f"Commodity Channel Index\n {company_name}")
    #st.line_chart(cci)
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
        st.header(f"Average Daily Change\n {company_name}")
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
