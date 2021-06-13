import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy.stats
import concurrent.futures
import multiprocessing
pd.set_option('mode.chained_assignment', None)
from collections import OrderedDict

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
                if ticker_price < 0:
                    ticker_price = 0
            except Exception:
                ticker_price = 0
            try:
                ticker_value = float(ticker_price*portfolio.get(ticker))
            except Exception:
                ticker_value = 0
            portfolio_value += ticker_value
    return portfolio_value

#Simulating trading of trained model
def simulator(stocks,sentiment,n,k,coefs,s_sum):
#def simulator(df,sentiment_df,n,k,s_sum):
    profit = 0
    current = s_sum
    portfolio={}
    profits_dict={}
    df = stocks.copy(deep=True)
    sentiment_df = sentiment.copy(deep=True)
    sentiment_df.reset_index(inplace=True)
    sentiment_df.drop(sentiment_df.columns[2:7], axis = 1, inplace = True)
    sentiment_df = sentiment_df.rename(columns={'level_0':'Date','level_1':'Short_Ticker'})
    first_day = list(df['Date'].unique())[0]
    last_day = list(df['Date'].unique())[-1]
    days = list(df['Date'].unique())
    allyears = [d[:4] for d in days]
    years =  list(OrderedDict.fromkeys(allyears))[1:]
    first_days=[]
    for year in years:
        days_in_year = [d for d in days if d[:4]==year]
        first_days.append(days_in_year[0])
    for day in tqdm(days , position = 0, leave = True):
        day_df = df[df['Date']==day]
        day_sentiment = sentiment_df[sentiment_df['Date']==day]
        if np.isnan(list(day_df['Adj Close'].unique())[0])==True and len(list(day_df['Adj Close'].unique()))==1:
            continue
        day_df = weighted_score(day_df,day_sentiment,coefs)
        #day_df = weighted_score(day_df,sentiment_df,day)
        mdf1 = day_df.dropna(subset=['Buy_Score'])
        mdf2 = day_df.dropna(subset=['Sell_Score'])   
        top_n = round(n*len(list(mdf1['Short_Ticker'])))
        bottom_k = round(k*len(list(mdf2['Short_Ticker'])))     
        top_df = mdf1.nlargest(top_n, 'Buy_Score')
        bottom_df = mdf2.nlargest(bottom_k, 'Sell_Score')
        if top_df['Buy_Score'].max() == 0:
        #if top_df['Buy_Score'].min() == 0:
            top_df['Percent'] = 0
            doubt_list = []
        else:
            pos_sums = sum(x for x in top_df['Buy_Score'] if x > 0)
            #score_sum = top_df['Buy_Score'].sum()
            top_df['Percent'] = [
                round(float(x/(pos_sums)),3) if x>0 and top_df['Buy_Score'].max()>0 
                else 0 for x in top_df['Buy_Score']]
            doubt_list = [ticker for ticker in list(top_df['Short_Ticker']) if ticker in list(bottom_df['Short_Ticker'])]
        if day == first_day:
            current2,portfolio2 = buy(top_df,portfolio,current,doubt_list)
            if np.isnan(current2)==False:
                current = current2
                portfolio = portfolio2            
        elif day == last_day:
            doubt_list=[]
            current2,portfolio2 = sell(bottom_df,day_df,portfolio,current,doubt_list,'YES')
            if np.isnan(current2)==False:
                current = current2
                portfolio = portfolio2
            final_profit = round(100*(current - s_sum)/s_sum,3)
            profits_dict[(int(day[:4]))] = final_profit            
        else:
            current2,portfolio2 = sell(bottom_df,day_df,portfolio,current,doubt_list)
            if np.isnan(current2)==False:
                current = current2
                portfolio = portfolio2
            if current > 1:
                current2,portfolio2 = buy(top_df,portfolio,current,doubt_list)
                if np.isnan(current2)==False:
                    current = current2
                    portfolio = portfolio2
            status = ((current+get_portfolio_value(day_df,portfolio))/s_sum)-1
            if  status <=-1:
                print(f'Ran out of cash on {day}')
                return None
            if day in first_days:
                pfl_value = get_portfolio_value(day_df,portfolio)
                annual_profit = round(100*(current + pfl_value - s_sum)/s_sum,3)
                pos_pfl = {k:v for (k,v) in portfolio.items() if v > 0}
                print(pos_pfl)
                print(annual_profit,(int(day[:4])-1))
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
                if invest < 0 :
                    continue
                if np.isnan(invest)==True:
                    continue
                if invest > current:
                    return current,portfolio
                else:
                    price = float(top_df.loc[top_df['Short_Ticker'] == ticker, 'Adj Close'].iloc[0])*1.004
                    if np.isnan(price)==True:
                        continue
                    if price < 0:
                        continue
                    else:
                        stocks += round(float(invest/price),2)
                        portfolio[ticker] = stocks
                        current -= stocks*price  
        return current,portfolio
    else:
        return current,portfolio

#Selling function
def sell(bottom_df,day_df,portfolio,current,doubt_list,indicator='NO'):
    if indicator == 'NO':
        mean_volume = np.nanmean(list(day_df['Volume']))
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
                        try:
                            volume = float(day_df.loc[day_df['Short_Ticker'] == ticker,'Volume'].iloc[0])
                        except Exception:
                            continue
                        if np.isnan(price)==True:
                            continue
                        if price < 0:
                            continue
                        else:
                            if volume >= mean_volume:
                                sell = float(portfolio.get(ticker))*price
                                portfolio[ticker]=0
                            else:
                                sell = float(portfolio.get(ticker)*(volume/mean_volume))*price
                                stocks = float(portfolio.get(ticker)*(1-(volume/mean_volume)))
                                portfolio[ticker]= stocks
                            current += sell
    else:
        for ticker in list(portfolio.keys()):
            try:
                price = 0.996*float(day_df.loc[
                            day_df['Short_Ticker'] == ticker,'Adj Close'].iloc[0])
            except Exception:
                continue
            if np.isnan(price)==True:
                continue
            else:
                sell = float(portfolio.get(ticker))*price                
                current += sell
                portfolio[ticker]=0   
            
    return current,portfolio

#Model train process (one of n, one of k)
def train_process(stocks,sentiment,s_sum,n,k):
    profit = 0
    current = s_sum
    portfolio={}
    d_coefs=[]
    df = stocks.copy(deep=True)
    sentiment_df = sentiment.copy(deep=True)
    sentiment_df.reset_index(inplace=True)
    sentiment_df.drop(sentiment_df.columns[2:7], axis = 1, inplace = True)
    sentiment_df = sentiment_df.rename(columns={'level_0':'Date','level_1':'Short_Ticker'})
    days=list(df['Date'].unique())
    first_day = days[0]
    last_day = days[-1]
    prev_day = 0
    current_day = 0
    for day in days:
        day_df = df[df['Date']==day]
        day_sentiment = sentiment_df[sentiment_df['Date']==day]
        if np.isnan(list(day_df['Adj Close'].unique())[0])==True and len(list(day_df['Adj Close'].unique()))==1:
            prev_day = 0
            continue
        coefs = train_coefs(day_df,day_sentiment)
        d_coefs.append(coefs)
        day_df = weighted_score(day_df,day_sentiment,coefs)
        mdf1 = day_df.dropna(subset=['Buy_Score'])
        mdf2 = day_df.dropna(subset=['Sell_Score'])   
        top_n = round(n*len(list(mdf1['Short_Ticker'])))
        bottom_k = round(k*len(list(mdf2['Short_Ticker'])))     
        top_df = mdf1.nlargest(top_n, 'Buy_Score')
        bottom_df = mdf2.nlargest(bottom_k, 'Sell_Score')
        if top_df['Buy_Score'].max() == 0:
        #if top_df['Buy_Score'].min() == 0:
            top_df['Percent'] = 0
            doubt_list = []
        else:
            pos_sums = sum(x for x in top_df['Buy_Score'] if x > 0)
            #score_sum = top_df['Buy_Score'].sum()
            top_df['Percent'] = [
                round(float(x/(pos_sums)),3) if x>0 and top_df['Buy_Score'].max()>0 
                else 0 for x in top_df['Buy_Score']]
            doubt_list = [ticker for ticker in list(top_df['Short_Ticker']) if ticker in list(bottom_df['Short_Ticker'])]
        if day == first_day:
            current2,portfolio2 = buy(top_df,portfolio,current,doubt_list)
            if np.isnan(current2)==False:
                current = current2
                portfolio = portfolio2
            # prev_day = ((current+get_portfolio_value(day_df,portfolio))/s_sum)-1
            # prev_coefs = coefs
        elif day == last_day:
            doubt_list=[]
            current2,portfolio2 = sell(bottom_df,day_df,portfolio,current,doubt_list,'YES')
            if np.isnan(current2)==False:
                current = current2
                portfolio = portfolio2
        else:
            current2,portfolio2 = sell(bottom_df,day_df,portfolio,current,doubt_list)
            if np.isnan(current2)==False:
                current = current2
                portfolio = portfolio2
            if current >1: 
                current2,portfolio2 = buy(top_df,portfolio,current,doubt_list)
                if np.isnan(current2)==False:
                    current = current2
                    portfolio = portfolio2
            current_day = ((current+get_portfolio_value(day_df,portfolio))/s_sum)-1
            if  current_day <=-1:
                print(portfolio)
                print(day_df)
                print(current)
                print(get_portfolio_value(day_df,portfolio))
                print(f'Ran out of cash on {day} with top {n}% and bottom {k}%')
                return [None,None,None]
            # elif current_day > prev_day: 
            #     d_coefs.append(prev_coefs)
            # prev_day = current_day
            # prev_coefs = coefs
    print(current)
    if current/s_sum > 3:
        return [n,k,d_coefs]
    else:
        return [None,None,None]
        
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

            futures = [executor.submit(train_process,train,sentiment_df,s_sum,x,y) for x in np.arange(0.01, 0.21, 0.01)
            for y in np.arange(0.01,0.21,0.01)]
            for future in tqdm(concurrent.futures.as_completed(futures),total=len(futures)):
                if future.result()[0] is not None:
                    n_list.append(future.result()[0])
                    k_list.append(future.result()[1])
                    t_coefs.extend(future.result()[2])     
                
        # final_n = np.nanmean(n_list)
        # final_k = np.nanmean(k_list)
        final_n = np.median(n_list)
        final_k = np.median(k_list)
        f_coefs = []
        B_I =  np.nanmean([x[0] for x in t_coefs])
        SA1 =  np.nanmean([x[1] for x in t_coefs])     
        nVol1 = np.nanmean([x[2] for x in t_coefs])
        f_coefs.extend([B_I,SA1, nVol1])
        return final_n,final_k,f_coefs

#Function that creates training coefficients
def train_coefs(day_df,sentiment):
    SA = 0
    B_I = 0
    nVol = 0
    if len(list(sentiment['Short_Ticker'])) > 1:
        day_df = day_df.merge(sentiment, how='outer', on='Short_Ticker')
        column_1 = day_df['Daily Change']
        column_2 = day_df['Sentiment Score']
        SA = round(column_1.corr(column_2),3)
        if np.isnan(SA)==True:
            SA = 0
    if len(day_df['Daily Change'].dropna()) > 1:
        column_1 = day_df['Daily Change']
        if len(day_df['Buy_Ind'].dropna()) > 1:
            column_2 = day_df['Buy_Ind']
            B_I = round(column_1.corr(column_2),3)
            if np.isnan(B_I)==True:
                B_I = 0
        if len(day_df['Normalized_Volume'].dropna()) > 1:  
            column_3 = day_df['Normalized_Volume']
            nVol = round(column_1.corr(column_3),3)
            if np.isnan(nVol)==True:
                nVol = 0    
    return [B_I,SA,nVol]


#Calculating weighted scores (according to coefficients)
def weighted_score(df,sentiment,coefs):
    max_sell = df['Sell_Ind'].max()
    B_I = coefs[0]
    SA = coefs[1]
    nVol = coefs[2]
    buy_scores = []
    sell_scores=[]
    for index,row in df.iterrows():
        Sell_score = 0
        Buy_score = 0
        Sentiment_score = 0
        Volume_score = 0
        if max_sell>0:
            Sell_score = row['Sell_Ind']/max_sell
        else:
            Sell_score = row['Sell_Ind']
        if len(list(sentiment['Short_Ticker'])) > 0:
            if row['Short_Ticker'] in list(sentiment['Short_Ticker']):
                Sentiment_score = SA*float(
                    sentiment.loc[sentiment['Short_Ticker']== row['Short_Ticker']]['Sentiment Score'])
        if np.isnan(float(row['Buy_Ind']))==False:
            Buy_score = B_I*float(row['Buy_Ind'])
        if np.isnan(float(row['Normalized_Volume']))==False:
            Volume_score = nVol*float(row['Normalized_Volume'])
        sell_scores.append(Sell_score)
        buy_scores.append(Sentiment_score + Buy_score + Volume_score)          
    
    #Just making sure...
    df['Sell_Score'] = sell_scores
    df['Sell_Score'] = pd.to_numeric(df['Sell_Score'])
    df['Buy_Score'] = buy_scores
    df['Buy_Score'] = pd.to_numeric(df['Buy_Score'])

    return df