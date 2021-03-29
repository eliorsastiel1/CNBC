import streamlit as st
import pandas as pd
import numpy as np
from dataWrapper.yahoo_finance_api import get_distribution_among_sectors, get_performance_info
from dataWrapper.cnbc_data_loader import get_cnbc_data
#pip install spacy-streamlit
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
#st.markdown(hide_menu_style, unsafe_allow_html=True)

st.title('Financial recommender')
st.sidebar.write('Navigation')
page = st.sidebar.radio("Go To",('Market Explorer','Data Explorer','Training'))
articles=get_cnbc_data()
if page == 'Market Explorer':
    ticker=st.text_input('ETFS Ticker', 'angl')
    perf=get_performance_info(ticker)
    chart=pd.DataFrame(perf, index=[0])
    st.dataframe(chart) 
    dist=get_distribution_among_sectors(ticker)
    st.bar_chart(pd.DataFrame(dist, index=[0]))
elif page == 'Data Explorer':
    article_id= st.slider('Article Index', 0,int(articles.shape[0]), 0)
    st.header(articles.loc[article_id,"Headline"])
    st.write(articles.loc[article_id,"Text"])
