import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy.stats
"""Simulation Functions"""
def train_test_split(df):
    train_performance=[]
    test_performance=[]
    daily_mean=[]
    daily_std=[]
    split_ix = df.index[df['Date'] == '2015-01-01'].tolist()[0]
    for day in tqdm(list(df['Date'].unique()), position = 0, leave = True):
        start_ix = df.index[df['Date'] == day].tolist()[0]
        end_ix = df.index[df['Date'] == day].tolist()[-1]
        daily_mean.append(
        df['Volume'].iloc[start_ix:end_ix].mean())
        daily_std.append(
        df['Volume'].iloc[start_ix:end_ix].std())
    for index, row in tqdm(df.iterrows(), position = 0, leave = True,total=mdf.shape[0]):
            j=[]                                
            if index < split_ix:
                day_ix = list(df['Date'].unique()).index(row['Date'])
                vol_mean = daily_mean[day_ix]
                vol_std = daily_std[day_ix]
                j.extend([
                row['Date'],row['Short_Ticker'],row['Adj Close'],
                row['Daily Change'],row['Buy_Ind'],row['Sell_Ind'],
                row['Volume'],(row['Volume']-vol_mean)/vol_std])
                train_performance.append(j)
            else:
                day_ix = list(df['Date'].unique()).index(row['Date'])
                vol_mean = daily_mean[day_ix]
                vol_std = daily_std[day_ix]
                j.extend([
                row['Date'],row['Short_Ticker'],row['Adj Close'],
                row['Daily Change'],row['Buy_Ind'],row['Sell_Ind'],
                row['Volume'],(row['Volume']-vol_mean)/vol_std])
                test_performance.append(j)     
    #Forming Train and Test corpuses
    train_corpus = pd.DataFrame(train_performance,columns=[
        'Date','Short_Ticker','Adj_Close','Daily_Change','Buy_Ind','Sell_Ind',
        'Volume','Normalized_Volume'])
    test_corpus = pd.DataFrame(test_performance,columns=[
        'Date','Short_Ticker','Adj_Close','Daily_Change','Buy_Ind','Sell_Ind',
        'Volume','Normalized_Volume'])
    return train_corpus,test_corpus

#Measuring portfolio value
def get_portfolio_value(df,portfolio):
    portfolio_value = 0
    for ticker in list(portfolio.keys()):
        if portfolio.get(ticker)>0:
            try:
                ticker_price = float(df.loc[df['Short_Ticker'] == ticker, 'Adj_Close'].iloc[0])
            except Exception:
                ticker_price = 0
            try:
                ticker_value = float(ticker_price*portfolio.get(ticker))
            except Exception:
                ticker_value = 0
            portfolio_value += ticker_value
    return portfolio_value
        
def simulator(df,sentiment_df,n,coefs,s_sum):
    profit = 0
    current = s_sum
    portfolio={}
    profits_dict={}
    first_day = list(df['Date'].unique())[0]
    last_day = list(df['Date'].unique())[-1]
    days = list(df['Date'].unique())
    for day in tqdm(days , position = 0, leave = True):
        is_day = df['Date']==day
        day_df = df[is_day]
        day_df = weighted_score(day_df,sentiment_df[['Date']==day],coefs)
        top_n = round(n*len(list(day_df['Short_Ticker'])))
        top_df = day_df.nlargest(top_n, 'Weighted_Score')
        top_df['Percent'] = [
            float(
                x/(top_df['Weighted_Score'].notna().sum())) for x in top_df['Weighted_Score']]
        if day == first_day:
            current,portfolio = buy(top_df,portfolio,current)
        elif day == last_day:
            current,portfolio = sell(top_df,day_df,portfolio,current,'YES')
            final_profit = round(100*(current - s_sum)/s_sum,3)
            profits_dict[(int(day[:4])-1)] = final_profit
        else:
            current,portfolio = sell(top_df,day_df,portfolio,current)
            if current <=0:
                print(f'Ran out of cash on {day}')
                return None
            if day[5:]=='01-01':
                pfl_value = get_portfolio_value(day_df,portfolio)
                annual_profit = round(100*(current + pfl_value - s_sum)/s_sum,3)
                profits_dict[(int(day[:4])-1)] = annual_profit
            current,portfolio = buy(top_df,portfolio,current)

    return profits_dict

def buy(top_df,portfolio,current):
    base = current
    for ticker in list(top_df['Short_Ticker']):
        if ticker in list(portfolio.keys()):
            stocks = float(portfolio.get(ticker))
        else:
            stocks = 0
        invest = float(top_df.loc[top_df['Short_Ticker'] == ticker,\
                                   'Percent'].iloc[0])*base
        price = float(top_df.loc[top_df['Short_Ticker'] == ticker, 'Adj_Close'].iloc[0])*1.004
        stocks += float(invest/price)
        portfolio[ticker] = stocks
        current -= invest
    return current,portfolio

def sell(top_df,day_df,portfolio,current,indicator='NO'):
    if indicator == 'NO':
        for ticker in list(portfolio.keys()):
            if ticker not in list(top_df['Short_Ticker']):
                if pd.isnull(portfolio.get(ticker)):
                    continue
                else:
                    try:
                        price = 0.996*float(day_df.loc[
                            day_df['Short_Ticker'] == ticker,'Adj_Close'].iloc[0])
                    except Exception:
                        continue
                    sell = float(portfolio.get(ticker))*price
                    current += sell
                    portfolio[ticker]=0
    else:
        for ticker in list(portfolio.keys()):
            try:
                price = 0.996*float(day_df.loc[
                            day_df['Short_Ticker'] == ticker,'Adj_Close'].iloc[0])
            except Exception:
                continue
            sell = float(portfolio.get(ticker))*price                
            current += sell
            portfolio[ticker]=0       
    return current,portfolio

def train_model(train,s_sum):
    pd.set_option('mode.chained_assignment', None)
    df = train
    profit = 0
    n_list=[]
    current = s_sum
    portfolio={}
    t_coefs=[]
    first_day = list(df['Date'].unique())[0]
    last_day = list(df['Date'].unique())[-1]
    for n in tqdm(np.arange(0.01, 0.31, 0.01), position = 0, leave = True):
        for day in list(df['Date'].unique()):
            is_day = df['Date']==day
            day_df = df[is_day]
            coefs = train_coefs(day_df,sentiment_df[['Date']==day])
            t_coefs.append(coefs)
            day_df = weighted_score(day_df,sentiment_df[['Date']==day],coefs)
            top_n = round(n*len(list(day_df['Short_Ticker'])))
            top_df = day_df.nlargest(top_n, 'Weighted_Score')
            top_df['Percent'] = [
                float(x/(top_df['Weighted_Score'].sum())) for x in top_df['Weighted_Score']]
            if day == first_day:
                current,portfolio = buy(top_df,portfolio,current)
            elif day == last_day:
                current,portfolio = sell(top_df,day_df,portfolio,current,'YES')
            else:
                current,portfolio = sell(top_df,day_df,portfolio,current)
                if current <=0:
                    profit = -100
                    del t_coefs[-1]
                    break
                current,portfolio = buy(top_df,portfolio,current)
        profit = 100*(current - s_sum)/s_sum
        if profit >0:
            n_list.append(n)
        
    final_n = np.median(n_list)
    f_coefs = []
    SA1 = np.nanmean([x[0] for x in t_coefs])
    B_I =  np.nanmean([x[1] for x in t_coefs])
    S_I =  np.nanmean([x[2] for x in t_coefs])
    nVol1 =  np.nanmean([x[4] for x in t_coefs])
    f_coefs.extend([SA1,B_I,S_I,nVol1])
    return final_n, f_coefs

def train_coefs(train,sentiment_df):
    coefs=[]
    df= train
    sentiments = []
    for ticker in list(df['Short_Ticker'].unique()):
        if ticker in list(sentiment_df['Short_Ticker'].unique()):
            sentiments.append(sentiment_df.loc[sentiment_df['Ticker'] == ticker, 'Sentiment'].iloc[0])
        else:
            sentiments.append(0)
    SA = scipy.stats.pearsonr(df['Daily_Change'].notna(),sentiments.notna())[1]
    B_I = scipy.stats.pearsonr(df['Daily_Change'].notna(),df['Buy_Ind'].notna())[1]
    S_I = scipy.stats.pearsonr(df['Daily_Change'].notna(),df['Sell_Ind'].notna())[1]
    nVol = scipy.stats.pearsonr(
        df['Daily_Change'].notna(),df['Normalized_Volume'])[1]
    if not isinstance(SA, float):
        SA = np.nanmean([B_I,S_I,nVol])   
    if not isinstance(SMA, float):
        B_I = np.nanmean([SA,S_I,nVol])
    if not isinstance(U_I, float):
        U_I = np.nanmean([SA,B_I,nVol])
    if not isinstance(nVol, float):
        nVol = np.nanmean([SA,B_I,S_I])
    if [SA,B_I,S_I,nVol] == [np.nan,np.nan,np.nan,np.nan]:
        coefs.extend([0.2,0.2,0.2,0.2])
    else:
        coefs.extend([SA,B_I,S_I,nVol])
    return coefs

def weighted_score(df,sentiment_df,coefs):
    SA = coefs[0]
    B_I = coefs[1]
    S_I = coefs[2]
    nVol = coefs[3]
    weighted_scores = []
    for index,row in df.iterrows():
        Buy_score = 0
        Sell_score = 0
        Sentiment_score = 0
        Volume_score = 0
        if row['Buy_Ind']==np.nan:
            Buy_score = 0
        else:
            Buy_score = B_I*row['Buy_Ind']
        if row['Sell_Ind']==np.nan:
            Sell_score = 0
        else:
            Sell_score = B_I*row['Buy_Ind'] 
        Volume_score = nVol*row['Normalized_Volume']
        if row['Short_Ticker'] in list(sentiment_df['Ticker'].unique()):
            Sentiment_score = SA*sentiment_df.loc[sentiment_df['Ticker'] == ticker, 'Sentiment'].iloc[0]
        else:
            Sentiment_score = 0
        weighted_scores.append(Buy_score+Sell_score+Volume_score+Sentiment_score)           

    df['Weighted_Score'] = weighted_scores

    return df