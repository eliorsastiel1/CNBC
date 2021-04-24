import yfinance as yf
import pandas as pd
import numpy as np
import pickle
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import talib
from IPython.display import clear_output
###Get ETF names'
# alphabet_list=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P',
#               'Q','R','S','T','U','V','W','X','Y','Z']
headers = {
    'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.183 Safari/537.36"}
# fdf = pd.DataFrame(columns=['Short_Ticker','Name','Sector'])
# for letter in alphabet_list:
#     for page_num in range(1,11):
#         try:
#             response = requests.get(
#                 f'https://etfdb.com/alpha/{letter}/#etfs&sort_name=symbol&sort_order=asc&page={page_num}',headers=headers)
#         except Exception:
#             continue
#         soup = BeautifulSoup(response.content.decode(),features='lxml')
#         data = []
#         table = soup.find('table', attrs={'id':'etfs'})
#         table_body = table.find('tbody')
#         rows = table_body.find_all('tr')
#         for row in rows:
#             cols = row.find_all('td')
#             cols = [ele.text.strip() for ele in cols]
#             data.append([ele for ele in cols if ele]) # Get rid of empty values
#         df = pd.DataFrame(data)   
#         df = df.rename(columns={0: "Short_Ticker", 1: "Name",2:"Sector"})
#         fdf = pd.concat([fdf, df[['Short_Ticker','Name','Sector']]], ignore_index=True)
# fdf.to_csv('ETF_List.csv')
fdf= pd.read_csv('ETF_List.csv')
###Get NYSE stocks' names
# fdf2 = pd.DataFrame(columns=['Short_Ticker','Name'])
# for letter in alphabet_list:
#         response = requests.get(
#             f'https://eoddata.com/stocklist/NYSE/{letter}.htm',headers=headers)
#         soup = BeautifulSoup(response.content.decode(),features='lxml')
#         data2 = []
#         table = soup.find('table', attrs={'class':'quotes'})
#         rows = table.find_all('tr')
#         for row in rows[1:]:
#             cols = row.find_all('td')
#             cols = [ele.text.strip() for ele in cols]
#             data2.append([ele for ele in cols if ele]) # Get rid of empty values
#         df2 = pd.DataFrame(data2)   
#         df2 = df2.rename(columns={0: "Short_Ticker", 1: "Name"})
#         fdf2 = pd.concat([fdf2, df2[['Short_Ticker','Name']]], ignore_index=True)
# sectors=[]
# for ticker in list(fdf2['Short_Ticker']):
#     response = requests.get(f'https://finance.yahoo.com/quote/{ticker}/profile?p={ticker}',headers=headers)
#     soup = BeautifulSoup(response.content.decode(),features='lxml')
#     try:
#         sector = soup.find('span', attrs={'class':'Fw(600)'})
#         sectors.append(sector.text)
#     except Exception:
#         sectors.append(np.nan)
# fdf2['Sector'] = sectors
# fdf2.to_csv('fdf2.csv')
fdf2 = pd.read_csv('fdf2.csv')
###Get NASDAQ stocks' names
nasdaq = pd.read_csv('nasdaq.csv')
nasdaq=nasdaq.rename(columns={'Symbol': "Short_Ticker"})
###Unite within one df
tickers_df = pd.concat([fdf,fdf2,nasdaq]).drop_duplicates().reset_index(drop=True)
"""Functions to find best short-long SMA periods for Train dataset"""
#Create a Function to signal when to buy and sell
def buy_sell_fun(data):
    sig_price_buy = []
    sig_price_sell = []
    flag = -1

    for i in range(len(data)):
        if data['SMA_short'][i] > data['SMA_long'][i]:
            if flag != 1:
                sig_price_buy.append(data['stock_adj_close'][i])
                sig_price_sell.append(np.nan)
                flag = 1
            else:
                sig_price_buy.append(np.nan)
                sig_price_sell.append(np.nan)
        elif data['SMA_short'][i] < data['SMA_long'][i]:
            if flag !=0:
                sig_price_buy.append(np.nan)
                sig_price_sell.append(data['stock_adj_close'][i])
                flag = 0
            else:
                sig_price_buy.append(np.nan)
                sig_price_sell.append(np.nan)
        else:
            sig_price_buy.append(np.nan)
            sig_price_sell.append(np.nan)

    return (sig_price_buy, sig_price_sell)

def chart_data(data):
    #store the buy and sell data into a variable
    buy_sell = buy_sell_fun(data)
    data['buy_signal_price'] = buy_sell[0]
    data['sell_signal_price']=buy_sell[1]

    #Looking for the dates of the buy signals
    buy_dates = data.loc[data['buy_signal_price']>0]

    #Looking for the dates of the sell signals
    sell_dates = data.loc[data['sell_signal_price']>0]

    #Dropping uneccessary columns
    buy_dates.drop(columns = ['sell_signal_price', 'SMA_short', 'SMA_long'], inplace = True)
    sell_dates.drop(columns = ['buy_signal_price', 'SMA_short', 'SMA_long'], inplace = True)

    # DF of dates with prices
    dates = pd.concat([buy_dates, sell_dates],sort = True)

    # Now to organize by the dates, drop adj_close price column, and make date the index
    dates.sort_values(by = 'Date', inplace = True)
    dates.drop(columns = ['stock_adj_close'],inplace = True)
    dates.set_index('Date', inplace = True)

def get_profit(data,s_sum): 
    sellList = [x for x in data['sell_signal_price'] if str(x) != 'nan']
    buyList = [x for x in data['buy_signal_price'] if str(x) != 'nan']
    current=0
    if buyList:
        stocks= s_sum/(buyList[0]*1.04)
        if len(buyList)>len(sellList):
            for i in range(0,len(sellList)):
                current = sellList[i]*0.96*stocks
                if current < 0:
                    profit = 0 - s_sum
                    return profit
                else:
                    if i == (len(buyList)-1):
                        stocks= current/(buyList[-1]*1.04)
                    else:
                        stocks= current/(buyList[i+1]*1.04)
            current = (data['stock_adj_close'].iloc[-1])*0.96*stocks
            if current < 0:
                profit = 0 - s_sum
                return profit
        else:
            for i in range(0,len(buyList)):
                current = sellList[i]*0.96*stocks
                if current <0:
                    profit= 0 - s_sum
                    return profit
                else:
                    if i == (len(buyList)-1):
                        stocks= current/(buyList[-1]*1.04)
                    else:
                        stocks= current/(buyList[i+1]*1.04)

        profit = current - s_sum
        return profit
    else:
        profit=0
        return profit

def best_profit(data,s_sum,short_list,long_list):
    best_periods={}
    for ticker in tqdm(data, position = 0, leave = True):
        top_prof=[0,0]
        max_profit=0
        try:
            perf = yf.download(ticker, start='2005-01-01', end='2015-12-31',interval='1d',progress=False)
        except:
            continue
        for short in short_list:
            sma_short = pd.DataFrame()
            sma_short['Adj_close'] = perf['Adj Close'].rolling(window=short).mean()
            for long in long_list:
                ticker_data=[]
                sma_long = pd.DataFrame()
                sma_long['Adj_close'] = perf['Adj Close'].rolling(window=long).mean()
                # Create a dataframe of all adj close prices
                udata = pd.DataFrame(index=perf.index)
                udata['stock_adj_close'] = perf['Adj Close']
                udata['SMA_short'] = sma_short['Adj_close']
                udata['SMA_long']= sma_long['Adj_close']
                udata.reset_index(inplace=True)
                chart_data(udata)
                profit = get_profit(udata,s_sum)
                ticker_data=[ticker,short,long,profit]
                clear_output(wait=True)
                if profit > max_profit:
                    max_profit = profit
                    top_prof=[short,long]
                    
        best_periods[ticker]=top_prof

    fdf= pd.DataFrame.from_dict(
        best_periods,columns=['Short Period','Long Period'],
                               orient='index')
    fdf.drop(fdf.loc[fdf['Short Period']==0].index, inplace=True)
    fdf.reset_index(drop=True)
    fdf.reset_index(inplace=True)
    fdf = fdf.rename(columns={'index': 'Short_Ticker'})
    return fdf
###SMA data
s_list=[7,14,21,30]
l_list=[50,100,150,200]
s_sum=1000000
sma_df=best_profit(list(tickers_df['Short_Ticker'].unique()),s_sum,s_list,l_list)

###Get stocks' data for training
tickers_data = []
for ticker in tqdm(list(tickers_df['Short_Ticker'].unique()), position = 0, leave = True):
    try:
        data = yf.download(ticker, start='2005-01-01', end='2021-01-05',interval="1d",progress=False)
    except:
        continue
    clear_output(wait=True)
    data.reset_index(inplace=True)
    period = 20
    multiplier = 2
    if ticker in list(sma_df['Short_Ticker']):
        index = sma_df[sma_df['Short_Ticker']==ticker].index.values[0]
        data['SMA_Short'] = data['Adj Close'].rolling(
            window=sma_df['Short Period'].iloc[index]).mean()
        data['SMA_Long']= data['Adj Close'].rolling(
            window=sma_df['Long Period'].iloc[index]).mean()
    else:
        data['SMA_Short'] = data['Adj Close'].rolling(
            window= int(sma_df['Short Period'].mean())).mean()
        data['SMA_Long']= data['Adj Close'].rolling(
            window= int(sma_df['Long Period'].mean())).mean()
    data['UpperBand'] = data['Adj Close'].rolling(period).mean() + data['Adj Close'].rolling(period).std() * multiplier
    data['LowerBand'] = data['Adj Close'].rolling(period).mean() - data['Adj Close'].rolling(period).std() * multiplier
    data['Date']=data['Date'].astype(str)
    data['Daily_Change'] = pd.DataFrame(data['Adj Close']).pct_change(periods=1, limit=None, freq=None)
    try:
        real = RSI(data['Adj Close'],timeperiod=14)
        data['RSI'] = real
    except Exception:
        data['RSI'] = np.nan
    for index, row in data.iterrows():
        s_indication = 0
        b_indication = 0
        if row['Adj Close'] >= row['UpperBand']:
            s_indication += 1/3
            b_indication -=1/3
        if row['Adj Close'] <=row['LowerBand']:
            s_indication -= 1/3
            b_indication += 1/3
        if row['SMA_Long'] > row['SMA_Short']:
            s_indication += 1/3
            b_indication -= 1/3
        if row['SMA_Long'] < row['SMA_Short']:
            s_indication -= 1/3
            b_indication += 1/3
        if row['RSI'] < 30:
            s_indication -= 1/3
            b_indication += 1/3
        if row['RSI'] > 70:
            s_indication += 1/3
            b_indication -= 1/3
        tickers_data.append([
                ticker,row['Date'][:10],row['Adj Close'],
                row['Daily_Change'],s_indication, b_indication,row['Volume']])
df1=pd.DataFrame(tickers_data, columns=[
    'Short_Ticker','Date','Adj Close','Daily Change','Sell_Ind','Buy_Ind','Volume'])  
#df1.to_csv('training_data.csv')
tickers_df.to_csv('tickers_info.csv')
df1 = df1.sort_values(by='Date', ascending=True)
df1.reset_index(inplace=True,drop=True)
df1.to_pickle("training_data.pkl")
