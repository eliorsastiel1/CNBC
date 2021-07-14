import os
import pandas as pd
from datetime import datetime,timedelta
from talib import RSI
data_file_name =  os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data/tickers_data.pkl')
sectors_file_name =  os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data/tickers_info.csv')
df=pd.read_pickle(data_file_name)
sectors=pd.read_csv(sectors_file_name)
def get_stock_at(ticker,date):
    result=df.loc[(df['Short_Ticker']==ticker) & (df['Date']== date),['Date','Short_Ticker','Adj Close','Daily Change','Sell_Ind','Buy_Ind','Volume']]
    if(result.empty):
        return '-'
    #print(result['Daily Change'].values)
    return result['Daily Change'].values[0]

def get_stocks_sectors(tickers):
    subset=sectors[sectors['Short_Ticker'].isin(tickers)]
    stats=subset.groupby(['Sector']).size().reset_index(name='counts')
    return stats
    #print(stats)

def get_history_performance(day,tickers,dl=22):
    
    d = timedelta(days = dl)
    start_date = datetime.strftime(day-d,"%Y-%m-%d")
    end_date = datetime.strftime(day,"%Y-%m-%d")
    after_start_date = df["Date"] >= start_date
    before_end_date = df["Date"] <= end_date
    between_two_dates = after_start_date & before_end_date
    filtered_dates = df.loc[between_two_dates]
    stocks=filtered_dates[filtered_dates['Short_Ticker'].isin(tickers)]
    #reduced=pd.melt(stocks, id_vars=['Date','Short_Ticker'], value_vars=['Adj Close'])
    #print("get history data")
    #print(reduced)
    return stocks

def get_RSI(day,tickers,dl=22):
    d = timedelta(days = dl)
    window = timedelta(days = 100)
    start_date = datetime.strftime(day-d,"%Y-%m-%d")
    start_date_window = datetime.strftime(day-window,"%Y-%m-%d")
    end_date = datetime.strftime(day,"%Y-%m-%d")
    after_start_date = df["Date"] >= start_date_window
    before_end_date = df["Date"] <= end_date
    between_two_dates = after_start_date & before_end_date
    filtered_dates = df.loc[between_two_dates]
    res=[]
    for ticker in tickers:
        fin=filtered_dates[filtered_dates['Short_Ticker']==ticker]
        fin=fin.copy()
        real=RSI(fin['Adj Close'],timeperiod=14)
        real=real.tolist()
        fin["RSI"]=real
        start_date_idx = fin["Date"] >= start_date
        fin = fin.loc[start_date_idx]
        #print(fin)
        res.append(fin)
    #print("RSI")
    #print(pd.concat(res))
    return pd.concat(res)

def get_bollinger_bands(day,tickers,dl=22):
    period=20
    multiplier=2
    d = timedelta(days = dl)
    window = timedelta(days = 100)
    start_date = datetime.strftime(day-d,"%Y-%m-%d")
    start_date_window = datetime.strftime(day-window,"%Y-%m-%d")
    end_date = datetime.strftime(day,"%Y-%m-%d")
    after_start_date = df["Date"] >= start_date_window
    before_end_date = df["Date"] <= end_date
    between_two_dates = after_start_date & before_end_date
    filtered_dates = df.loc[between_two_dates]
    res=[]
    for ticker in tickers:
        fin=filtered_dates[filtered_dates['Short_Ticker']==ticker]
        fin=fin.copy()
        fin['UpperBand'] = fin['Adj Close'].rolling(period).mean() + fin['Adj Close'].rolling(period).std() * multiplier
        fin['LowerBand'] = fin['Adj Close'].rolling(period).mean() - fin['Adj Close'].rolling(period).std() * multiplier
        start_date_idx = fin["Date"] >= start_date
        fin = fin.loc[start_date_idx]
        res.append(fin)
    return pd.concat(res)