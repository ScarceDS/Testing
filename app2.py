#Import Libraries

import yfinance as yf
import base64
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
import nltk
#from load_data import load_data
import pandas_market_calendars as mcal
#from pmdarima.arima import ADFTest
#from pmdarima.arima import ndiffs
#from pmdarima.arima import auto_arima
import holidays
yf.pdr_override()
pd.options.plotting.backend = "plotly"



#@st.cache(suppress_st_warning=True)
def initial():
  #Initialize lists for user inputs data to be downloaded
  RMSE_Values=[]
  Changepoint_prior_scale_values=[]
  seasonality_prior_scale_values=[]
  seasonality_mode_value=[]
  return RMSE_Values,Changepoint_prior_scale_values,seasonality_prior_scale_values,seasonality_mode_value

if st.sidebar.button('Start Logging'):
  st.sidebar.write('Started Logging')
  RMSE_Values,Changepoint_prior_scale_values,seasonality_prior_scale_values,seasonality_mode_value=[],[],[],[]




st.write("""
# Stock Finance Analysis Web Application 
""")


st.sidebar.header('Please Enter Your Parameters')
number_of_tickers=st.sidebar.slider('No of Tickers to be plotted', min_value=1, max_value=4,
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
        company_name=yf.Ticker(i).info['longName']
        company_names.append(company_name)
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


  
def Plot_data(combined_data,symbol,company_name,number_of_tickers):
  """Plotting multiple stocks.
  Parameters
  ----------
  combined_data: Dictionary object which include all stocks dataframes to be plotted. 
  symbol: list with all stocks symbols.
  company_name: list with all stocks company names.
  number_of_tickers: int indicating the number of stocks to be plotted on the same plot.

  """
   # Create figure object
  fig = go.Figure()
  fig.update_layout(
    autosize=False,
    width=1000,
    height=550)
  ctrl=0 # iterator
        
  fig.update_layout(
      xaxis=dict(
          domain=[0.18, 0.75]))
  fig.update_xaxes(title_text='Date')
  #Layout_preparation
  if number_of_tickers>0:
   
    fig = fig.add_trace(go.Scatter(y = combined_data[symbol[ctrl]]['Close'],
                                  x = combined_data[symbol[ctrl]].index, 
                                  name = company_name[ctrl]))
    yaxis=dict(
            title=company_name[ctrl]+' '+price_type+' Price',
            titlefont=dict(color="blue"),tickfont=dict(
            color="blue"))
   
    fig.update_layout(yaxis=yaxis)
    ctrl+=1
    
    if number_of_tickers > 1:
      
      fig = fig.add_trace(go.Scatter(y = combined_data[symbol[ctrl]]['Close'],
                                    x = combined_data[symbol[ctrl]].index, 
                                    name = company_name[ctrl],yaxis="y"+str(ctrl+1)))
      yaxis2=dict(
              title=company_name[ctrl]+' '+price_type+' Price',
              titlefont=dict(color="red"),tickfont=dict(color="red")
              ,anchor="x",overlaying="y",side="right")
      
      fig.update_layout(yaxis2=yaxis2)
      ctrl+=1
      
      if number_of_tickers>2:
        
        fig = fig.add_trace(go.Scatter(y = combined_data[symbol[ctrl]]['Close'],
                                      x = combined_data[symbol[ctrl]].index, 
                                      name = company_name[ctrl],yaxis="y"+str(ctrl+1)))
        yaxis3=dict(
                title=company_name[ctrl]+' '+price_type+' Price',
                titlefont=dict(color="forestgreen"),tickfont=dict(color="forestgreen")
                ,anchor="free",overlaying="y",side="left",position=0.05)
        
        fig.update_layout(yaxis3=yaxis3)
        ctrl+=1
         
        if number_of_tickers>3:
          fig = fig.add_trace(go.Scatter(y = combined_data[symbol[ctrl]]['Close'],
                                        x = combined_data[symbol[ctrl]].index, 
                                        name = company_name[ctrl],yaxis="y"+str(ctrl+1)))
          yaxis4=dict(
                  title=company_name[3]+' '+price_type+' Price',
                  titlefont=dict(color="#9467bd"),tickfont=dict(color="#9467bd")
                  ,anchor="free",overlaying="y",side="right",position=0.86)
          
          fig.update_layout(yaxis4=yaxis4)
           
  return fig

fig=Plot_data(combined_data,symbol,company_name,number_of_tickers)



  
st.plotly_chart(fig)



#fig=Plot_data(combined_data,symbol,company_name)
#st.plotly_chart(fig)

#Forecasting using fbprophet model
Forecasting = st.sidebar.checkbox('Forecasting')
if Forecasting:
    stock=st.sidebar.selectbox('Ticker',(symbol))
    n_periods= st.sidebar.slider('Forecasting period', min_value=1, max_value=5,
                                 value=5,  step=1)
    changepoint_prior_scale=st.sidebar.number_input('Changepoint_prior_scale',value=0.005,min_value=0.001,max_value=0.500)
    seasonality_prior_scale=st.sidebar.number_input('Seasonality_prior_scale',value=5.00,min_value=0.01,max_value=10.00)
    seasonality_mode=st.sidebar.selectbox('Seasonality_mode',('additive', 'multiplicative'))
    #changepoint_prior_scale=[np.arange(0.001, 0.5,0.1),0.05]
    #seasonality_prior_scale=[np.arange(0.01,10.51,0.5)]
   # holidays_prior_scale=[np.arange(0.01,10.51,0.5)]
   # seasonality_mode=['additive', 'multiplicative']
    @st.cache(suppress_st_warning=True)
    def forecast(data,price_type,n_periods,changepoint_prior_scale,seasonality_prior_scale,seasonality_mode):
        
        #prepare the new index
        nyse=mcal.get_calendar('NYSE')
        new_index=nyse.valid_days(start_date=start, end_date='2030-01-01 00:00:00')
        new_index=new_index[0:(len(data)+n_periods)]
        
        #Prophet Model
        df=data[price_type]
        df.index.rename('ds',True)
        df=df.reset_index()
        df.columns=['ds','y']

        model=Prophet(changepoint_prior_scale=changepoint_prior_scale,seasonality_mode=seasonality_mode,seasonality_prior_scale=seasonality_prior_scale).add_country_holidays(country_name='US').fit(df)
        
        st.sidebar.text('Model Training is completed')
        
     
        future_dates=model.make_future_dataframe(periods=n_periods)
        prediction=model.predict(future_dates)
        prediction.index=new_index
        
        return model,prediction,df
    
    model,prediction,real=forecast(combined_data[stock],price_type,n_periods,changepoint_prior_scale,seasonality_prior_scale,seasonality_mode)   
    
    #Update lists for user inputs data to be downloaded
    
    Changepoint_prior_scale_values.append(changepoint_prior_scale)
    seasonality_prior_scale_values.append(seasonality_prior_scale)
    seasonality_mode_value.append(seasonality_mode)
    
    fig=plot_plotly(model,prediction,trend=True)
    fig.update_layout(width=700,
    height=500)
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text=price_type+' Price')
    st.plotly_chart(fig)
    
    
# Model Cross Validation    
model_validation = st.sidebar.checkbox('Cross Validation')
if model_validation:
  if Forecasting:
    st.sidebar.header('Cross Validation Customization')
    def Cross_validation(Model,Real_data):

      Training_size=st.sidebar.number_input('Training Set Size in Days',value=int(0.45*len(real)),min_value=int(0.25*len(real)),max_value=int(0.75*len(real)))
      cutoff_seperation=st.sidebar.number_input('Rolling Window Size in Days',value=int(0.1*len(real)),min_value=int(0.05*len(real)),max_value=int(0.5*len(real)))
      Validation_size=st.sidebar.number_input('Validation/Forecasting Set Size in Days',value=int(0.05*len(real)),min_value=int(0.05*len(real)),max_value=int(0.2*len(real)))

             #cutoffs = pd.to_datetime([df['ds'][int(0.55*len(df))],df['ds'][int(0.75*len(df))]])
      df_cv = cross_validation(model,initial=str(Training_size)+' days',period=str(cutoff_seperation)+' days', horizon=str(Validation_size)+' days', parallel="processes")
             #df_cv = cross_validation(model, cutoffs=cutoffs, horizon='30 days', parallel="processes")
      df_p = performance_metrics(df_cv, rolling_window=1)
      return df_p['rmse'].mean(),df_cv
    rmse,cv=Cross_validation(model,real)      
    st.sidebar.subheader(f"RMSE =\n {rmse}")
    RMSE_Values.append(rmse)
  else:
    st.error('CV can be done only after forecasting is completed')
    
    
  
#Technical Analysis

TA=st.sidebar.checkbox('Technical Analysis')
if TA:
  #User Stock Selection
  stock_ta=st.sidebar.selectbox('Ticker_Name',(symbol))
  #TA Menu
  sma = st.sidebar.checkbox('Simple Moving Average')
  ADC=st.sidebar.checkbox('Average Daily Price Change')
  #daily_return=st.sidebar.checkbox('Daily Return')
  daily_return=False
  #vortex_indicator=st.sidebar.checkbox('Vortex Indicator')
  vortex_indicator=False
  #Simple Moving Average
  if sma:
    data_sma=combined_data[stock_ta]
    period= st.sidebar.slider('SMA period', min_value=5, max_value=50,
                             value=20,  step=1)
    data_sma[f'SMA {period}'] = data_sma[price_type].rolling(period).mean()
    
    st.header('Simple Moving Average')
    fig=data_sma[[price_type,f'SMA {period}']].plot()
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text=price_type+' Price')
    st.plotly_chart(fig)  
    
  
  if ADC:
    def ADC(combined_data,symbol):
        all_data={}
        for i in symbol:
            data=combined_data[i]
            data['day']=data.index.day_name()
            monday=data[data['day']=='Monday']['High']-data[data['day']=='Monday']['Low']
            tuesday=data[data['day']=='Tuesday']['High']-data[data['day']=='Tuesday']['Low']
            wednesday=data[data['day']=='Wednesday']['High']-data[data['day']=='Wednesday']['Low']
            thursday=data[data['day']=='Thursday']['High']-data[data['day']=='Thursday']['Low']
            friday=data[data['day']=='Friday']['High']-data[data['day']=='Friday']['Low']
            day_mean=[monday.mean(),tuesday.mean(),wednesday.mean(),thursday.mean(),friday.mean()]
            day=['Monday','Tuesday','Wednesday','Thursday','Friday']
            day_of_week=pd.DataFrame(data=day_mean,index=day,columns=['Average Daily Price Change $'])
            all_data[i]=day_of_week
        return all_data
    all_data=ADC(combined_data,symbol)
    
    #Plotting the average daily change
    def Plot_data_adc(combined_data,symbol,company_name,number_of_tickers):
      """Plotting multiple stocks.
      Parameters
      ----------
      combined_data: Dictionary object which include all stocks dataframes to be plotted. 
      symbol: list with all stocks symbols.
      company_name: list with all stocks company names.
      number_of_tickers: int indicating the number of stocks to be plotted on the same plot.
      """
    
    # Create figure object
      fig = go.Figure()
      fig.update_layout(
        title="Average Price Daily Change ($)",
        autosize=False,
        width=1000,
        height=550)
      ctrl=0 # iterator

      fig.update_layout(
          xaxis=dict(
              domain=[0.18, 0.75]))
      fig.update_xaxes(title_text='Day')
      #Layout_preparation
      if number_of_tickers>0:

        fig = fig.add_trace(go.Scatter(y = combined_data[symbol[ctrl]]['Average Daily Price Change $'],
                                      x = combined_data[symbol[ctrl]].index, 
                                      name = company_name[ctrl]))
        yaxis=dict(
                title=company_name[ctrl],
                titlefont=dict(color="blue"),tickfont=dict(
                color="blue"))

        fig.update_layout(yaxis=yaxis)
        ctrl+=1

        if number_of_tickers > 1:

          fig = fig.add_trace(go.Scatter(y = combined_data[symbol[ctrl]]['Average Daily Price Change $'],
                                        x = combined_data[symbol[ctrl]].index, 
                                        name = company_name[ctrl],yaxis="y"+str(ctrl+1)))
          yaxis2=dict(
                  title=company_name[ctrl],
                  titlefont=dict(color="red"),tickfont=dict(color="red")
                  ,anchor="x",overlaying="y",side="right")

          fig.update_layout(yaxis2=yaxis2)
          ctrl+=1

          if number_of_tickers>2:

            fig = fig.add_trace(go.Scatter(y = combined_data[symbol[ctrl]]['Average Daily Price Change $'],
                                          x = combined_data[symbol[ctrl]].index, 
                                          name = company_name[ctrl],yaxis="y"+str(ctrl+1)))
            yaxis3=dict(
                    title=company_name[ctrl],
                    titlefont=dict(color="forestgreen"),tickfont=dict(color="forestgreen")
                    ,anchor="free",overlaying="y",side="left",position=0.05)

            fig.update_layout(yaxis3=yaxis3)
            ctrl+=1

            if number_of_tickers>3:
              fig = fig.add_trace(go.Scatter(y = combined_data[symbol[ctrl]]['Average Daily Price Change $'],
                                            x = combined_data[symbol[ctrl]].index, 
                                            name = company_name[ctrl],yaxis="y"+str(ctrl+1)))
              yaxis4=dict(
                      title=company_name[ctrl],
                      titlefont=dict(color="#9467bd"),tickfont=dict(color="#9467bd")
                      ,anchor="free",overlaying="y",side="right",position=0.86)

              fig.update_layout(yaxis4=yaxis4)

      return fig
    fig2=Plot_data_adc(all_data,symbol,company_name,len(symbol))
    try:
        #st.header(f"Average Daily Change\n {company_name[stock_ADC]}")
        st.header(f"Average Price Daily Change\n")
        st.plotly_chart(fig2)
    except:
        st.error("streamlit plot is not working")
    
    
  #Daily Return  
  if daily_return:
    dr=ta.others.daily_return(combined_data[stock_ta][price_type]).plot()
    st.plotly_chart(dr)
  #Vortex Indicator
  if vortex_indicator:
    vor=ta.trend.vortex_indicator_pos(combined_data[stock_ta]['High'], combined_data[stock_ta]['Low'], combined_data[stock_ta]['Close'], window=14, fillna=True).plot()
    st.plotly_chart(vor)
   

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
    return href
  
download=st.sidebar.checkbox('Download_User_Log')    
if download:
  dic={'RMSE_Values':RMSE_Values,'Changepoint_prior_scale_values':Changepoint_prior_scale_values,'seasonality_prior_scale_values':seasonality_prior_scale_values,'seasonality_mode_value':seasonality_mode_value}
  user_log=pd.DataFrame(dic,columns=['RMSE','Changepoint_Scale','Seasonality_prior_scale','Seasonality_mode'])
  #stock_download=st.sidebar.selectbox('Select Ticker Name',(symbol))
  st.markdown(get_table_download_link(user_log), unsafe_allow_html=True)
  

      
#CCI=st.sidebar.checkbox('Commodity Channel Index')
#if CCI:
    ## CCI (Commodity Channel Index)

    #cci = ta.trend.cci(data['High'], data['Low'], data['Close'], window=31, constant=0.015)

    # Plotting (Commodity Channel Index)
    #st.header(f"Commodity Channel Index\n {company_name}")
    #st.line_chart(cci)

    
    
st.sidebar.title("About")
st.sidebar.info('This app is a simple example of '
                    'using Streamlit & Heroku to create a financial data web app.\n'
                    '\nIt is maintained by [Anas Alfadhel]\n'
                    'Check the code at https://github.com/ScarceDS/Streamlit-Heroku-App')


st.write("""
    #   Thanks For Using My Application
    For more information, you can contact me on alfadhel.anas@gmail.com
""")

