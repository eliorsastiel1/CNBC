import numpy as np
import pandas as pd
#Measuring portfolio value
def get_portfolio_value(df,portfolio):
    portfolio_value = 0
    for ticker in list(portfolio.keys()):
        if portfolio.get(ticker)[0]>0:
            try:
                ticker_price = float(df.loc[df['Short_Ticker'] == ticker, 'Adj Close'].iloc[0])
                if ticker_price < 0:
                    ticker_price = 0
            except Exception:
                ticker_price = 0
            try:
                ticker_value = float(ticker_price*portfolio.get(ticker)[0])
            except Exception:
                ticker_value = 0
            portfolio_value += ticker_value
    return portfolio_value



#Buying function
def buy(top_df,portfolio,current,doubt_list):
    base = current
    #if top_df['Buy_Score'].sum()!=0:
    if top_df['Score'].sum()!=0:
        for ticker in list(top_df['Short_Ticker']):
            if ticker in doubt_list:
                continue
            else:
                if ticker in list(portfolio.keys()):
                    stocks = float(portfolio.get(ticker)[0])
                else:
                    portfolio[ticker] = [0,0]
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
                        portfolio[ticker][0] = stocks
                        portfolio[ticker][1] +=1
                        current -= stocks*price  
        return current,portfolio
    else:
        return current,portfolio

#Selling function
def sell(bottom_df,day_df,portfolio,current,doubt_list,indicator='NO'):
    mean_vol = np.nanmean(day_df['Volume'])
    if indicator == 'NO':
        for ticker in list(portfolio.keys()):
            if ticker in doubt_list:
                continue
            else:
                if ticker in list(bottom_df['Short_Ticker']):
                    if pd.isnull(portfolio.get(ticker)[0]):
                        continue
                    else:
                        try:
                            price = 0.996*float(day_df.loc[
                                day_df['Short_Ticker'] == ticker,'Adj Close'].iloc[0])
                        except Exception:
                            continue
                        if np.isnan(price)==True:
                            continue
                        if price < 0:
                            continue
                        else:
                            if portfolio.get(ticker)[1] < 22:
                                continue
                            else:
                                ticker_volume = day_df.loc[day_df['Short_Ticker'] == ticker,'Volume'].iloc[0]
                                if np.isnan(ticker_volume):
                                    ticker_volume = mean_vol
                                if ticker_volume >=mean_vol:
                                    sell = float(portfolio.get(ticker)[0])*price
                                    current += sell
                                    portfolio[ticker] = [0,0]
                                else:
                                    stocks_to_sell = round(
                                        (ticker_volume/mean_vol)*float(portfolio.get(ticker)[0]),2)
                                    current += float(stocks_to_sell)*price
                                    portfolio[ticker][0] -= stocks_to_sell
                                    portfolio[ticker][1] = 0
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
                sell = float(portfolio.get(ticker)[0])*price                
                current += sell
                portfolio[ticker] = [0,0] 
            
    return current,portfolio

#Calculating weighted scores (according to coefficients)
def weighted_score(df,sentiment,coefs):
    #max_sell = df['Sell_Ind'].max()
    B_I = coefs[0]
    SA = coefs[1]
    nVol = coefs[2]
    S_I = coefs[3]
    f_scores = []
    day_df = df.merge(sentiment, how='outer', on='Short_Ticker')
    column_1 = day_df['Buy_Ind']
    column_2 = day_df['Sentiment Score']
    # try:
    #     bs_cor = round(column_1.corr(column_2),3)
    # except Exception:
    #     bs_cor = np.nan
    for index,row in df.iterrows():
        Sell_score = 0
        Buy_score = 0
        Sentiment_score = 0
        Volume_score = 0
        # if max_sell>0:
        #     Sell_score = row['Sell_Ind']/max_sell
        # else:
        #     Sell_score = row['Sell_Ind']
        if len(list(sentiment['Short_Ticker'])) > 0:
            if row['Short_Ticker'] in list(sentiment['Short_Ticker']):
                Sentiment_score = SA*float(
                    sentiment.loc[sentiment['Short_Ticker']== row['Short_Ticker']]['Sentiment Score'])
        if np.isnan(float(row['Buy_Ind']))==False:
            Buy_score = B_I*float(row['Buy_Ind'])
        if np.isnan(float(row['Sell_Ind']))==False:
            Sell_score = S_I*float(row['Buy_Ind'])            
        if np.isnan(float(row['Normalized_Volume']))==False:
            Volume_score = nVol*float(row['Normalized_Volume'])
        f_scores.append(Buy_score + Sell_score + Sentiment_score + Volume_score)
        # sell_scores.append(Sell_score)
        # if np.isnan(bs_cor)==False:
        #     buy_scores.append(Sentiment_score + bs_cor*(Buy_score + Volume_score)) 
        # else:
        #     buy_scores.append(Sentiment_score + Buy_score + Volume_score)        
    
    #Just making sure...
    df['Score'] = f_scores
    df['Score'] = pd.to_numeric(df['Score'])
    # df['Sell_Score'] = sell_scores
    # df['Sell_Score'] = pd.to_numeric(df['Sell_Score'])
    # df['Buy_Score'] = buy_scores
    # df['Buy_Score'] = pd.to_numeric(df['Buy_Score'])

    return df