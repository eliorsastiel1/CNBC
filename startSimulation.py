import pandas as pd
from trade_simulator import train_test_split,train_model,simulator
from tqdm import tqdm
import yfinance as yf
from dataWrapper.sentiment_loader import get_sentiment_data
sentiment=get_sentiment_data()
stock_data=pd.read_pickle('./Data/training_data.pkl')
isum=1000000
split_date = '2015-01-02'
train,test = train_test_split(stock_data,split_date)
n, coefs =train_model(train,sentiment,isum)
profit=simulator(stock_data,sentiment,n,coefs,isum)
print(profit)

###Compare profits with indices
compare = {'^GSPC':'','^RUT':'','^IXIC':'','^DJI':'','^RUI':''}
years=range(2015,2021)
for index in compare:
    index_dict={}
    for year in years:
      if year != 2021:
          data = yf.download(
              index, start=split_date, end=f'{year+1}-01-02',interval="1d",progress=False)
      else:
        data = yf.download(
              index, start=split_date, end='2021-03-27',interval="1d",progress=False)
      j = data.iloc[[0, -1]]
      l = j['Adj Close'].pct_change(periods=1, limit=None, freq=None)        
      index_dict[year]= round(l.iloc[-1]*100,3)
    compare[index]=index_dict
compare['My Program']= profit
pd.DataFrame(compare)
