from load_russel3000_stocks import load_russel3000
import pandas as pd
from tqdm import tqdm
import numpy as np
def norm_volume(df):
    n_vol_list = []
    for day in tqdm(list(df['Date'].unique()), position = 0, leave = True):
        day_df = df[df['Date']==day]
        day_df['Normalized_Volume'] = [(x-day_df['Volume'].mean())/day_df['Volume'].std() for x in day_df['Volume']]
        n_vol_list.extend(day_df['Normalized_Volume']) 
    df['Normalized_Volume'] = n_vol_list
    return df

russel_stocks=load_russel3000()
stocks=pd.read_pickle('./Data/stocks_data.pkl')
#print(russel_stocks['Ticker'])

#print(stocks[stocks['Short_Ticker']=='AAPL'])
filtered=stocks[stocks['Short_Ticker'].isin(russel_stocks['Ticker'])]
unwanted=filtered[filtered['Adj Close']<0]
unwanted=np.unique(unwanted['Short_Ticker'])
filtered = filtered[~filtered['Short_Ticker'].isin(unwanted)]
filtered = filtered.reset_index(drop=True)
print('filtering')
#print(list(filtered['Date'].unique()))
normalized=norm_volume(filtered)
print(filtered.head())
print(normalized.head())
normalized.to_pickle('stocks_filtered.pkl')