import pandas as pd
import os
from data_wrapper.get_stocks_data import get_stock_data
from data_wrapper.sentiment_loader import get_sentiment_for_day
from trader_helper_functions import weighted_score,buy,get_portfolio_value,sell

import numpy as np

n=0.01#buy percentage
k=0.09#sell percentage
coefs=[-0.114, -0.049, 0.016, 0.114]
current=1000000

portfolio={}
profits_dict={}

def get_portfolio():
    global portfolio
    df=pd.DataFrame.from_dict(portfolio,orient='index')
    if(df.empty):
        return df
    df=df[df[df.columns[1]]!=0]
    return df

def get_current():
    global current
    return current
def add_to_current(val):
    global current
    current=current+val
    return current

def get_portfolio_val(day):
    global portfolio
    stocks=get_stock_data(day)
    df = stocks.copy(deep=True)
    return get_portfolio_value(df,portfolio)

def get_recommendations(day):
    sentiment=get_sentiment_for_day(day)
    stocks=get_stock_data(day)
    df = stocks.copy(deep=True)
    sentiment_df = sentiment.copy(deep=True)
    sentiment_df.reset_index(inplace=True)
    sentiment_df.drop(sentiment_df.columns[1:6], axis = 1, inplace = True)
    sentiment_df = sentiment_df.rename(columns={'index':'Short_Ticker'})
    if np.isnan(list(df['Adj Close'].unique())[0])==True and len(list(df['Adj Close'].unique()))==1:
        return None
    
    day_df = weighted_score(df,sentiment_df,coefs)
    mdf = day_df.dropna(subset=['Score']) 
    top_n = round(n*len(list(mdf['Short_Ticker'])))
    bottom_k = round(k*len(list(mdf['Short_Ticker'])))   
    top_df = mdf.nlargest(top_n, 'Score')
    bottom_df = mdf.nsmallest(bottom_k, 'Score')
    top_df['Percent'] = 1/top_n
    doubt_list = [ticker for ticker in list(top_df['Short_Ticker']) if ticker in list(bottom_df['Short_Ticker'])]
    top_df = top_df[~top_df['Short_Ticker'].isin(doubt_list)]
    bottom_df = bottom_df[~bottom_df['Short_Ticker'].isin(doubt_list)]
    holding_tickers=portfolio.keys()
    #only show tickers taht we are holding for sale
    bottom_df = bottom_df[bottom_df['Short_Ticker'].isin(holding_tickers)]
    top_df_clone = top_df.copy(deep=True)
    bottom_df_clone = bottom_df.copy(deep=True)
    top_df.set_index('Short_Ticker', inplace=True)
    bottom_df.set_index('Short_Ticker', inplace=True)
    return top_df[['Score','Normalized_Volume','Percent']],bottom_df[['Score','Normalized_Volume']],top_df_clone,bottom_df_clone
    #return top_df,portfolio,current,doubt_list


def buy_action(top_df,day):
    global portfolio
    global current
    current2,portfolio2 = buy(top_df,portfolio,current,[],day)
    print(portfolio2)
    if np.isnan(current2)==False:
        current = current2
        portfolio = portfolio2

def sell_action(bottom_df,day):
    global portfolio
    global current
    stocks=get_stock_data(day)
    df = stocks.copy(deep=True)
    current2,portfolio2 = sell(bottom_df,df,portfolio,current,[],day)
    if np.isnan(current2)==False:
        current = current2
        portfolio = portfolio2
