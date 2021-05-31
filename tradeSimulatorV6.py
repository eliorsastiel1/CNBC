import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy.stats
import concurrent.futures
import multiprocessing
pd.set_option('mode.chained_assignment', None)

"""Simulation Functions"""
#Splitting stock_data to train and test dataframes
def train_test_split(df,split_date):
    split_ix = df.index[df['Date'] == split_date].tolist()[0]
    train_corpus = df.iloc[:split_ix]
    train_corpus.reset_index(inplace=True,drop=True)
    test_corpus = df.iloc[split_ix:]
    test_corpus.reset_index(inplace=True,drop=True)
    
    return train_corpus,test_corpus

#Measuring portfolio value
def get_portfolio_value(df,portfolio):
    portfolio_value = 0
    for ticker in list(portfolio.keys()):
        if portfolio.get(ticker)>0:
            try:
                ticker_price = float(df.loc[df['Short_Ticker'] == ticker, 'Adj Close'].iloc[0])
            except Exception:
                ticker_price = 0
            try:
                ticker_value = float(ticker_price*portfolio.get(ticker))
            except Exception:
                ticker_value = 0
            portfolio_value += ticker_value
    return portfolio_value

#Simulating trading of trained model
def simulator(df,sentiment_df,n,k,coefs,s_sum):
    profit = 0
    current = s_sum
    portfolio={}
    profits_dict={}
    first_day = list(df['Date'].unique())[0]
    last_day = list(df['Date'].unique())[-1]
    days = list(df['Date'].unique())
    for day in days:
        day_df = df[df['Date']==day]
        coefs = train_coefs(day_df)
        day_df = weighted_score(day_df,sentiment_df,day,coefs)
        mdf1 = day_df.dropna(subset=['Buy_Score'])
        mdf2 = day_df.dropna(subset=['Sell_Score'])   
        top_n = round(n*len(list(mdf1['Short_Ticker'])))
        bottom_k = round(k*len(list(mdf2['Short_Ticker'])))     
        top_df = mdf1.nlargest(top_n, 'Buy_Score')
        bottom_df = mdf2.nlargest(bottom_k, 'Sell_Score')
        if top_df['Buy_Score'].sum() == 0:
            top_df['Percent'] = 0
            doubt_list = []
        else:
            top_df['Percent'] = [
                round(float(x/(top_df['Buy_Score'].sum())),2) for x in top_df['Buy_Score']]
            doubt_list = [ticker for ticker in top_df['Short_Ticker'] if ticker in bottom_df['Short_Ticker']]
        if day == first_day:
            doubt_list=[]
            current,portfolio = buy(top_df,portfolio,current,doubt_list)
        elif day == last_day:
            doubt_list=[]
            current,portfolio = sell(bottom_df,day_df,portfolio,current,doubt_list,'YES')
            final_profit = round(100*(current - s_sum)/s_sum,3)
            profits_dict[(int(day[:4])-1)] = final_profit
        else:
            current,portfolio = sell(bottom_df,day_df,portfolio,current,doubt_list)
            current,portfolio = buy(top_df,portfolio,current,doubt_list)
            status = ((current+get_portfolio_value(day_df,portfolio))/s_sum)-1
            if  status <=-1:
                print(f'Ran out of cash on {day}')
                return None
            if day[5:]=='01-01':
                pfl_value = get_portfolio_value(day_df,portfolio)
                annual_profit = round(100*(current + pfl_value - s_sum)/s_sum,3)
                profits_dict[(int(day[:4])-1)] = annual_profit
                
    return profits_dict

#Buying function
def buy(top_df,portfolio,current,doubt_list):
    base = current
    if top_df['Buy_Score'].sum()!=0:
        for ticker in list(top_df['Short_Ticker']):
            if ticker in doubt_list:
                continue
            else:
                if ticker in list(portfolio.keys()):
                    stocks = float(portfolio.get(ticker))
                else:
                    stocks = 0
                invest = float(top_df.loc[top_df['Short_Ticker'] == ticker,\
                                        'Percent'].iloc[0])*base
                if invest > current:
                    return current,portfolio
                else:
                    price = float(top_df.loc[top_df['Short_Ticker'] == ticker, 'Adj Close'].iloc[0])*1.004
                    stocks += float(invest/price)
                    portfolio[ticker] = stocks
                    current -= invest   
        return current,portfolio
    else:
        return current,portfolio

#Selling function
def sell(bottom_df,day_df,portfolio,current,doubt_list,indicator='NO'):
    if indicator == 'NO':
        for ticker in list(portfolio.keys()):
            if ticker in doubt_list:
                continue
            else:
                if ticker in list(bottom_df['Short_Ticker']):
                    if pd.isnull(portfolio.get(ticker)):
                        continue
                    else:
                        try:
                            price = 0.996*float(day_df.loc[
                                day_df['Short_Ticker'] == ticker,'Adj Close'].iloc[0])
                        except Exception:
                            continue
                        sell = float(portfolio.get(ticker))*price
                        current += sell
                        portfolio[ticker]=0
    else:
        for ticker in list(portfolio.keys()):
            try:
                price = 0.996*float(day_df.loc[
                            day_df['Short_Ticker'] == ticker,'Adj Close'].iloc[0])
            except Exception:
                continue
            sell = float(portfolio.get(ticker))*price                
            current += sell
            portfolio[ticker]=0   
            
    return current,portfolio

#Model train process (one of n, one of k)
def train_process(df,sentiment_df,s_sum,n,k):
    profit = 0
    current = s_sum
    portfolio={}
    d_coefs=[]
    days=list(df['Date'].unique())
    first_day = list(df['Date'].unique())[0]
    last_day = list(df['Date'].unique())[-1]
    for day in days:
        day_df = df[df['Date']==day]
        coefs = train_coefs(day_df)
        day_df = weighted_score(day_df,sentiment_df,day,coefs)
        mdf1 = day_df.dropna(subset=['Buy_Score'])
        mdf2 = day_df.dropna(subset=['Sell_Score'])   
        top_n = round(n*len(list(mdf1['Short_Ticker'])))
        bottom_k = round(k*len(list(mdf2['Short_Ticker'])))     
        top_df = mdf1.nlargest(top_n, 'Buy_Score')
        bottom_df = mdf2.nlargest(bottom_k, 'Sell_Score')
        if top_df['Buy_Score'].sum() == 0:
            top_df['Percent'] = 0
            doubt_list = []
        else:
            top_df['Percent'] = [
                round(float(x/(top_df['Buy_Score'].sum())),2) for x in top_df['Buy_Score']]
            doubt_list = [ticker for ticker in top_df['Short_Ticker'] if ticker in bottom_df['Short_Ticker']]
        if day == first_day:
            current,portfolio = buy(top_df,portfolio,current,doubt_list)
        elif day == last_day:
            doubt_list=[]
            current,portfolio = sell(bottom_df,day_df,portfolio,current,doubt_list,'YES')
        else:
            current,portfolio = sell(bottom_df,day_df,portfolio,current,doubt_list)
            if current > 1:
                current,portfolio = buy(top_df,portfolio,current,doubt_list)
            status = ((current+get_portfolio_value(day_df,portfolio))/s_sum)-1
            if  status <=-1:
                print(f'Ran out of cash on {day} with top {n}% and bottom {k}%')
                return [None,None,None]
            elif status > 0: 
                d_coefs.append(coefs)
    profit = 100*(current - s_sum)/s_sum
    if profit >0:
        return [n, k,d_coefs]
    else:
        return [None,None,None]
        
#Single Process model    
# def train_model(train,sentiment_df,s_sum):
#     n_list=[]
#     k_list=[]
#     t_coefs = [] 
#     for n in tqdm(np.arange(0.01,0.11,0.01) , position = 0, leave = True):
#         for k in tqdm(np.arange(0.01,0.11,0.01), position = 1, leave = True):
#             result = train_process(train,sentiment_df,s_sum,n,k)
#             n_list.append(result[0])
#             k_list.append(result[1])
#             t_coefs.extend(result[2])
#     final_n = np.median(n_list)
#     final_k = np.median(k_list)
#     f_coefs = []
#     S_I =  np.nanmean([x[0] for x in t_coefs])
#     nVol1 =  np.nanmean([x[1] for x in t_coefs])
#     f_coefs.extend([S_I,nVol1])
#     return final_n,final_k,f_coefs

#Training function - retrieves top % and coefficients
def train_model(train,sentiment_df,s_sum):
    n_list = []
    k_list = []
    t_coefs = []
    num_processes = multiprocessing.cpu_count()-2
    #num_processes = 4
    if __name__ == '__main__':
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:

        #results = list(tqdm(executor.map(train_process, train), total=len(my_iter)))

            futures = [executor.submit(train_process,train,sentiment_df,x,y) for x in np.arange(0.01, 0.11, 0.01)
            for y in np.arange(0.01,0.11,0.01)]
            for future in tqdm(concurrent.futures.as_completed(futures),total=len(futures)):
                if future.result()[0] is not None:
                    n_list.append(future.result()[0])
                    k_list.append(future.result()[1])
                    t_coefs.extend(future.result()[2])     
                
        final_n = np.median(n_list)
        final_k = np.median(k_list)
        f_coefs = []
        S_I =  np.nanmean([x[0] for x in t_coefs])
        nVol1 =  np.nanmean([x[1] for x in t_coefs])
        f_coefs.extend([S_I,nVol1])
        return final_n,final_k,f_coefs

#Function that creates training coefficients
def train_coefs(train):
    coefs=[]
    dchange1 = []
    dchange2=[]
    sell_ind = []
    vol_list = []
    for ticker in list(train['Short_Ticker'].unique()):
        if np.isnan(
                float(train[train['Short_Ticker']==ticker]['Daily Change']))==False:
            dchange1.append(float(train[train['Short_Ticker']==ticker]['Daily Change']))
            sell_ind.append(float(train[train['Short_Ticker']==ticker]['Sell_Ind']))
            if np.isnan(
                float(train[train['Short_Ticker']==ticker]['Normalized_Volume']))==False:
                vol_list.append(float(train[train['Short_Ticker']==ticker]['Normalized_Volume']))
                dchange2.append(float(train[train['Short_Ticker']==ticker]['Daily Change']))
    if not dchange1 and not dchange2:
        return [0.5,0.5]   
    if not dchange1 and not dchange2:
            return [0.5,0.5]
    try:
        S_I = scipy.stats.pearsonr(dchange1,sell_ind)[1]
        nVol = scipy.stats.pearsonr(dchange2,vol_list)[1]
    except Exception:
        return[0.5,0.5]
    coefs.extend([S_I,nVol])
    return coefs

#Calculating weighted scores (according to coefficients)
def weighted_score(df,sentiment_df,day,coefs):
    is_sentiment = True
    try:
        sentiment_df = sentiment_df.loc[day]
    except Exception:
        is_sentiment = False
    S_I = coefs[0]
    nVol = coefs[1]
    buy_scores = []
    sell_scores=[]
    for index,row in df.iterrows():
        Sell_score = 0
        Sentiment_score = 0
        Volume_score = 0
        Sell_score = S_I*row['Sell_Ind'] 
        Volume_score = nVol*row['Normalized_Volume']
        if is_sentiment==True:
            if row['Short_Ticker'] in sentiment_df.index:
                Sentiment_score = float(
                    sentiment_df.loc[row['Short_Ticker']]['Sentiment Score']/sentiment_df["Sentiment Score"].max())
            else:
                Sentiment_score = 0
        else:
            Sentiment_score = 0
        sell_scores.append(Sell_score+Volume_score)
        buy_scores.append(Sentiment_score)          
    
    #Just making sure...
    df['Sell_Score'] = sell_scores
    df['Sell_Score'] = pd.to_numeric(df['Sell_Score'])
    df['Buy_Score'] = buy_scores
    df['Buy_Score'] = pd.to_numeric(df['Buy_Score'])

    return df