import pandas as pd
from trade_simulator import train_test_split,train_model,simulator
from tqdm import tqdm
import yfinance as yf
import os
import pickle
from dataWrapper.sentiment_loader import get_sentiment_data

simulation_training_data =  os.path.join(os.path.dirname(__file__), 'Data/simulation_train_data.pkl')
    
simulator_params =  os.path.join(os.path.dirname(__file__), 'Data/simulator_params.pkl')

sentiment=get_sentiment_data()
stock_data=pd.read_pickle('./Data/training_data.pkl')
isum=1000000
split_date = '2015-01-02'
if not os.path.isfile(simulation_training_data):
    train,test = train_test_split(stock_data,split_date)
    with open(simulation_training_data, 'wb') as out_file:
        pickle.dump(train,out_file, protocol=-1)
        pickle.dump(test,out_file, protocol=-1)
else:
    with open(simulation_training_data, 'rb') as in_file: 
            train = pickle.load(in_file)
            test = pickle.load(in_file)  

if not os.path.isfile(simulator_params):    
    n, coefs =train_model(train,sentiment,isum)
    with open(simulator_params, 'wb') as out_file:
        pickle.dump(n,out_file, protocol=-1)
        pickle.dump(coefs,out_file, protocol=-1)
else:
    with open(simulator_params, 'rb') as in_file: 
        n = pickle.load(in_file)
        coefs = pickle.load(in_file) 
 
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
