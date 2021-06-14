#Import Libraries

import yfinance as yf
import streamlit as st
import datetime 
import ta
import pandas as pd
import requests
from pmdarima.arima import ADFTest
from pmdarima.arima import ndiffs
from pmdarima.arima import auto_arima
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
    
    n_periods= st.sidebar.slider('Forecasting period', min_value=1, max_value=5,
                             value=5,  step=1)
    import matplotlib.pyplot as plt
    #Splitting the data into testing and training
    def test_train_split(data,test_split=0.2):
        x=int((1-test_split)*len(data))
        training_set = data.iloc[:x].values
        test_set = data.iloc[x:].values
        return training_set,test_set,x

    # multi-step data preparation
    from numpy import array

    # split a univariate sequence into samples
    def split_sequence(sequence, n_steps_in, n_steps_out):
        X, y = list(), list()
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            # check if we are beyond the sequence
            if out_end_ix > len(sequence):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)

    from sklearn.preprocessing import MinMaxScaler
    import pandas_market_calendars as mcal
    df=data[price_type]
    Train,Test,index=test_train_split(df,0.2)

    sc = MinMaxScaler(feature_range = (0, 1))
    # define input sequence
    df1=Train.reshape(-1,1)
    Raw_data =sc.fit_transform(df1)
    # choose a number of time steps
    n_steps_in, n_steps_out = 30, n_periods
    # Prepare the new index
    n_periods=n_steps_out
    nyse=mcal.get_calendar('NYSE')
    new_index=nyse.valid_days(start_date=start, end_date='2030-01-01 00:00:00')
    new_index=new_index[0:(len(data)+n_periods)]
    # split into samples
    X, y = split_sequence(Raw_data, n_steps_in, n_steps_out)
    # summarize the data
    #for i in range(len(X)):
       # print('\nX_train\n',X[i],'\n Y_train\n', y[i])
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    # univariate multi-step vector-output stacked lstm example
    from numpy import array
    from keras.models import Sequential
    from keras.layers import LSTM
    from keras.layers import Dense
    # define model
    model1 = Sequential()
    model1.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
    model1.add(LSTM(100, activation='relu'))
    model1.add(Dense(n_steps_out))
    model1.compile(optimizer='adam', loss='mse')
    #model1.summary()
    # fit model
    history1=model1.fit(X, y, epochs=50, verbose=0,validation_split=0.15)
    #Testing
    #Prepare Testing data
    Test_data=Test[-n_steps_in:]
    print(Test_data.shape)
    Test_data=Test_data.reshape(-1,1)
    print(Test_data.shape)
    Test_data =sc.fit_transform(Test_data)
    print(Test_data.shape)
    Test_data=Test_data.reshape((Test_data.shape[1],Test_data.shape[0], n_features))
    print(Test_data.shape)
    ## Testing our model
    yhat = model1.predict(Test_data, verbose=1)
    print('yhat shape',np.shape(yhat))
    yhat=sc.inverse_transform(yhat)
    yhat=yhat.reshape(-1,1)
    print(yhat)
    New_df=pd.DataFrame(columns=['Close'],data=yhat,index=new_index[-n_periods:])
    Old_df=pd.DataFrame(columns=['Close'],data=df.values,index=new_index[:-n_periods])
    Final_df=pd.concat([Old_df,New_df],axis=0)
    plt.plot(Final_df[-100:-n_periods+1])
    plt.plot(Final_df[-n_periods:])

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
