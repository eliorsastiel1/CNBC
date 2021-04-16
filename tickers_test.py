import os
import pandas as pd
import pickle
import datetime

raw_file_name =  os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data/tickers_data.pkl')
df=pd.read_pickle(raw_file_name)

#pd.show_versions(as_json=False)

result=df.loc[(df['Short_Ticker']=='AAPL') & (df['Date']== '2018-05-01'),['Date','Short_Ticker','Adj Close','Daily Change','Sell_Ind','Buy_Ind','Volume']]
print(result)