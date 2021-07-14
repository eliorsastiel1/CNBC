import streamlit as st
import streamlit.components.v1 as components
from trader import get_portfolio, get_recommendations,buy_action,get_current,get_portfolio_val,sell_action,add_to_current
from datetime import datetime, timedelta,date
from data_wrapper.cnbc_data_loader import get_cnbc_data,get_cnbc_data_with_sentiment
from data_wrapper.sentiment_loader import get_sentiment_for_day
from data_wrapper.stock_activity import get_stocks_sectors,get_history_performance,get_RSI,get_bollinger_bands
import plotly.express as px
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



st.sidebar.title('Investment Recommender')
selected_date = st.sidebar.date_input('Select Date', date(2020,3,5))


raw_data=None
article_sentiment=None
st.sidebar.write('Navigation')
page = st.sidebar.radio("Go To",('Explorer','Investor'))


date_str=datetime.strftime(selected_date,"%Y-%m-%d")
if page == 'Explorer':
    explore_page = st.radio("View",('Articles','Sentiments','Financial Sentiment'))
    if raw_data is None:
        raw_data=get_cnbc_data()
    if explore_page=='Articles':
        day_articles=raw_data[raw_data['Publish Date']==date_str]
        st.dataframe(day_articles)
    elif explore_page=='Sentiments':
        if article_sentiment is None:
            article_sentiment=get_cnbc_data_with_sentiment()
        #print(article_sentiment.head())
        day_sentiment=article_sentiment[article_sentiment['Publish Date']==date_str]
        st.dataframe(day_sentiment)
    elif explore_page=='Financial Sentiment':
        sentiment=get_sentiment_for_day(date_str)
        st.dataframe(sentiment)
elif page == 'Investor':
    
    sector_chart=None
    portfolio_col, sector_col = st.beta_columns(2)
    portfolio_col.write('Portfolio')
    portfolio=get_portfolio()
    portfolio_preview=portfolio_col.dataframe(portfolio) 
    if(not portfolio.empty):
        sectors=get_stocks_sectors(portfolio.index.values)
        fig = px.pie(sectors, values='counts', names='Sector', title='Investment By Sector')
        sector_chart=sector_col.plotly_chart(fig)
    current=get_current()
    portfolio_col.write('Balance')
    current_preview=portfolio_col.text(current)
    
    
  


    invest,sell,top_df,bottom_df=get_recommendations(date_str)
 
    portfolio_col.write('Portfolio Value') 
    pf_val=get_portfolio_val(date_str)
    portfolio_val_preview=portfolio_col.text(pf_val) 

    components.html("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """)

    col1, col2 = st.beta_columns(2)

    col1.header("Buy")
    col1.dataframe(invest)
    
    if col1.button("BUY NOW"):
        buy_action(top_df,date_str)
        portfolio=get_portfolio()
        portfolio_preview.empty()
        portfolio_preview.dataframe(portfolio) 
        current=get_current()
        current_preview.text(current) 
        pf_val=get_portfolio_val(date_str)
        portfolio_val_preview.text(pf_val)
        sectors=get_stocks_sectors(portfolio.index.values)
        fig = px.pie(sectors, values='counts', names='Sector', title='Investment By Sector')
        if(sector_chart is None):
            sector_chart=sector_col.plotly_chart(fig)
        else:
            sector_chart.plotly_chart(fig)

    col2.header("Sell")
    col2.dataframe(sell)
    if col2.button("SELL NOW"):
        sell_action(bottom_df,date_str)
        portfolio=get_portfolio()
        portfolio_preview.empty()
        portfolio_preview.dataframe(portfolio) 
        current=get_current()
        current_preview.text(current)
        pf_val=get_portfolio_val(date_str)
        portfolio_val_preview.text(pf_val) 
        sectors=get_stocks_sectors(portfolio.index.values)
        fig = px.pie(sectors, values='counts', names='Sector', title='Investment By Sector')
        if(sector_chart is None):
            sector_chart=sector_col.plotly_chart(fig)
        else:
            sector_chart.plotly_chart(fig)
    
    stocks_to_investigate=[]
    stocks_to_investigate.extend(invest.index)
    stocks_to_investigate.extend(sell.index)
    stocks_to_investigate.extend(portfolio.index)
    selected_indices = st.multiselect('Select Tickers:', stocks_to_investigate)

    if(len(selected_indices)>0):    
        stocks_to_show=get_history_performance(selected_date,selected_indices)
        if(not stocks_to_show.empty):
            fig = px.line(stocks_to_show, x='Date', y="Adj Close",color='Short_Ticker')
            st.plotly_chart(fig)
        
        rsi=get_RSI(selected_date,selected_indices)
        if(not rsi.empty):
            fig = px.line(rsi, x='Date', y="RSI",color='Short_Ticker')
            st.plotly_chart(fig)
        
        bollinger=get_bollinger_bands(selected_date,selected_indices)
        print(bollinger)
        if(not bollinger.empty):
            fig = px.line(bollinger, x='Date', y=["UpperBand","LowerBand"],color='Short_Ticker',labels={
                     "Date": "Date",
                     "value": "Bands",
                     "Short_Ticker": "Ticker"
                 })
            st.plotly_chart(fig)
        
        
    balance_input = st.sidebar.number_input("Add to balance", 3000)
    if st.sidebar.button("Add To Balance"):
        current=add_to_current(balance_input)
        current_preview.text(current)
        sectors=get_stocks_sectors(portfolio.index.values)
        fig = px.pie(sectors, values='counts', names='Sector', title='Investment By Sector')
        if(sector_chart is None):
            sector_chart=sector_col.plotly_chart(fig)
        else:
            sector_chart.plotly_chart(fig)