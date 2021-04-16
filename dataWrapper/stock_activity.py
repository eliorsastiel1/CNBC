import os
import pandas as pd

data_file_name =  os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data/tickers_data.pkl')
df=pd.read_pickle(data_file_name)

def get_stock_at(ticker,date):
    result=df.loc[(df['Short_Ticker']==ticker) & (df['Date']== date),['Date','Short_Ticker','Adj Close','Daily Change','Sell_Ind','Buy_Ind','Volume']]
    if(result.empty):
        return '-'
    #print(result['Daily Change'].values)
    return result['Daily Change'].values[0]

