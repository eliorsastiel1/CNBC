import pandas as pd
from tradeSimulatorV14 import train_test_split
import numpy as np
stock_data=pd.read_pickle('./Data/stocks_filtered.pkl')
unwanted=stock_data[stock_data['Adj Close']<0]
unwanted=np.unique(unwanted['Short_Ticker'])
stock_data = stock_data[~stock_data['Short_Ticker'].isin(unwanted)]
split_date = '2015-01-02'
print(stock_data.head())
train,test = train_test_split(stock_data,split_date)
print(train.head())
print(test.head())
train.to_pickle('./Data/train.pkl')
test.to_pickle('./Data/test.pkl')