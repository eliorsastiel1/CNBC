import streamlit as st
from trader import get_portfolio, get_recommendations,buy_action,get_current,get_portfolio_val,sell_action
from datetime import datetime, timedelta,date
st.title('Investment Recommender')

st.sidebar.write('Portfolio')
portfolio=get_portfolio()
portfolio_preview=st.sidebar.dataframe(portfolio) 
current=get_current()
st.sidebar.write('Balance')
current_preview=st.sidebar.text(current)


selected_date = st.date_input('Select Date', date(2020,3,5))
date_str=datetime.strftime(selected_date,"%Y-%m-%d")
invest,sell,top_df,bottom_df=get_recommendations(date_str)
 
st.sidebar.write('Portfolio Value') 
pf_val=get_portfolio_val(date_str)
portfolio_val_preview=st.sidebar.text(pf_val) 

col1, col2 = st.beta_columns(2)

col1.header("Buy")
col1.dataframe(invest)
if col1.button("BUY NOW"):
    buy_action(top_df)
    portfolio=get_portfolio()
    #portfolio_preview.empty()
    portfolio_preview.dataframe(portfolio) 
    current=get_current()
    current_preview.text(current) 
    pf_val=get_portfolio_val(date_str)
    portfolio_val_preview.text(pf_val) 

col2.header("Sell")
col2.dataframe(sell)
if col2.button("SELL NOW"):
    sell_action(bottom_df,date_str)
    portfolio=get_portfolio()
    #portfolio_preview.empty()
    portfolio_preview.dataframe(portfolio) 
    current=get_current()
    current_preview.text(current)
    pf_val=get_portfolio_val(date_str)
    portfolio_val_preview.text(pf_val) 