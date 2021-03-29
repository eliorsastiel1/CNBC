import streamlit as st
import pandas as pd
import numpy as np
from dataWrapper.yahoo_finance_api import get_distribution_among_sectors, get_performance_info
from dataWrapper.cnbc_data_loader import get_cnbc_data
from NER import plot_ner
from sentiment_analysis import extract_sentiment_from_text

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
#st.markdown(hide_menu_style, unsafe_allow_html=True)

st.title('Financial recommender')
st.sidebar.write('Navigation')
page = st.sidebar.radio("Go To",('ETFS Explorer','Data Explorer','Training'))
articles=get_cnbc_data()
if page == 'ETFS Explorer':
    ticker=st.text_input('ETFS Ticker', 'angl')
    perf=get_performance_info(ticker)
    chart=pd.DataFrame(perf, index=[0])
    st.dataframe(chart) 
    dist=get_distribution_among_sectors(ticker)
    st.bar_chart(pd.DataFrame(dist, index=[0]))
elif page == 'Data Explorer':
    article_id= st.slider('Article Index', 0,int(articles.shape[0]), 0)
    st.header(articles.loc[article_id,"Headline"])
    plot_ner(articles.loc[article_id,"Text"])
    sentiment=extract_sentiment_from_text(articles.loc[article_id,"Text"])
    st.dataframe(sentiment) 
    #st.write(articles.loc[article_id,"Text"])

