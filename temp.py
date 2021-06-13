import os
import pickle
import numpy as np
from dataWrapper.sentiment_loader import get_sentiment_data
import pandas as pd
simulator_params2 =  os.path.join(os.path.dirname(__file__), 'Data/simulator_params2.pkl')
simulator_params =  os.path.join(os.path.dirname(__file__), 'Data/simulator_params.pkl')
sentiment=get_sentiment_data()
train=pd.read_pickle('./Data/train.pkl')
unwanted=train[train['Adj Close']<0]
unwanted=np.unique(unwanted['Short_Ticker'])
df = train[~train['Short_Ticker'].isin(unwanted)]
#print(df.head())
with open(simulator_params2, 'rb') as in_file: 
    n_list = pickle.load(in_file)
    k_list = pickle.load(in_file)
    t_coefs = pickle.load(in_file)
print(np.mean(n_list))
print(np.mean(k_list))
#print(t_coefs)
exit()
test=pd.read_pickle('./Data/test.pkl')

df = test.copy(deep=True)
sentiment_df = sentiment.copy(deep=True)
sentiment_df.reset_index(inplace=True)
sentiment_df.drop(sentiment_df.columns[2:7], axis = 1, inplace = True)
sentiment_df = sentiment_df.rename(columns={'level_0':'Date','level_1':'Short_Ticker'})
day='2018-06-13'
#print(sentiment.loc[day])
day_sentiment = sentiment_df[sentiment_df['Date']==day]
#print(day_sentiment)
day_df = df[df['Date']==day]
#print(day_df)
day_sentiment.to_csv('Data/day_sent.csv')
day_df.to_csv('Data/day.csv')
exit()
with open(simulator_params2, 'rb') as in_file: 
    n_list = pickle.load(in_file)
    k_list = pickle.load(in_file)
    t_coefs = pickle.load(in_file) 
    final_n = np.median(n_list)
    final_k = np.median(k_list)
    f_coefs = []
    B_I =  np.nanmean([x[0] for x in t_coefs])
    SA1 =  np.nanmean([x[1] for x in t_coefs])    
    nVol1 = np.nanmean([x[2] for x in t_coefs])
    f_coefs.extend([B_I,SA1, nVol1])
            #f_coefs.extend([S_I,nVol1])
            #return final_n, f_coefs
        #n, coefs =train_model(train,sentiment,isum)
    #with open(simulator_params, 'wb') as out_file:
    #    pickle.dump(final_n,out_file, protocol=-1)
    #    pickle.dump(final_k,out_file, protocol=-1)
    #    pickle.dump(f_coefs,out_file, protocol=-1)
    print(final_n)
    print(final_k)
    print(f_coefs)