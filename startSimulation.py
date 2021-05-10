import pandas as pd
from trade_simulator import train_test_split,train_model,simulator
from tqdm import tqdm
from dataWrapper.sentiment_loader import get_sentiment_data
sentiment=get_sentiment_data()
stock_data=pd.read_pickle('./Data/training_data.pkl')
isum=1000000
train,test=train_test_split(stock_data)
n, coefs=train_model(train,sentiment,isum)
profit=simulator(stock_data,sentiment,n,coefs,isum)
print(profit)

