import streamlit as st
import pandas as pd
import numpy as np
from dataWrapper.yahoo_finance_api import get_distribution_among_sectors, get_performance_info
from dataWrapper.cnbc_data_loader import get_cnbc_data
from dataWrapper.stock_activity import get_stock_at
from NER import ner
from sentiment_analysis import extract_sentiment_from_text
from SVO import extract_svo
from search_for_ticker import get_ticker_by_company #uses google search
from extract_companies_from_text import search_companies #uses fuzzywuzzy
from fuzzywuzzy import process
#from annotated_text import annotated_text
import re
import datetime

def replace(old, new, str, caseinsentive = False):
    if caseinsentive:
        return str.replace(old, new)
    else:
        return re.sub(re.escape(old), new, str, flags=re.IGNORECASE)


hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
#st.markdown(hide_menu_style, unsafe_allow_html=True)

st.title('Financial recommender')
st.sidebar.write('Navigation')
page = st.sidebar.radio("Go To",('ETFS Explorer','Data Explorer','Training','Mentioned Stocks'))
articles=get_cnbc_data()
print(articles.head())
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
    st.write(articles.loc[article_id,"Publish Date"])
    [entities,doc]=ner(articles.loc[article_id,"Text"])
    orginazations=[]
    article=articles.loc[article_id,"Text"]
    #print(article)
    annotated=[]
    for i,e in enumerate(entities):
        if(e[1]=='ORG'):
            ticker=search_companies(e[0],90)
            #ticker=get_ticker_by_company(e[0])
            if(ticker is not None):
                #orginazations.append([ticker,e[0]])
                orginazations.append(ticker)
                if(e[0] not in annotated):
                    #print(e[0])
                    annotated.append(e[0])
                    article=article.replace(e[0],"{}<span style='height:15px;background-color:coral;padding:5px;border-radius:5px'>Russel3000</span>".format(e[0]))
                    #print(article)
                #article = replace(e[0], "<span style='color:red'>{}</span><span style='height:15px;background-color:coral;padding:5px;border-radius:5px'>Russel3000</span>".format(e[0]),article)
    print(orginazations)
    #print text to screen
    st.markdown(article, unsafe_allow_html=True)
    #annotated_text(articles.loc[article_id,"Text"])
    #entities=plot_ner(articles.loc[article_id,"Text"])
    org_df = pd.DataFrame(orginazations, columns =[ 'Ticker','Name'])
    org_df.drop_duplicates(subset ="Ticker",keep = "first", inplace = True)
    st.dataframe(org_df)
    sentiment=extract_sentiment_from_text(articles.loc[article_id,"Text"])
    subjects=[]
    sentiment_per_org={}
    for org in annotated:
        sentiment_per_org[org]=0

    for index, row in sentiment.iterrows():
        discovered_comapnies=process.extract(row['sentence'], annotated)
        subject, verb, attribute = extract_svo(row['sentence'])
        subjects.append(subject)
        if(row['prediction']=='neutral'):
            continue
        for suggestion in discovered_comapnies:
            if(suggestion[1]>50):
                sentiment_per_org[suggestion[0]]=sentiment_per_org[suggestion[0]]+row['sentiment_score']
    sentiment['subject']=subjects
    st.dataframe(sentiment)
    st.write(sentiment_per_org)
    #st.write(articles.loc[article_id,"Text"])
elif page == 'Mentioned Stocks':
    date = st.date_input('Select Date', datetime.date(2020,1,1))
    tomorrow = date + datetime.timedelta(days=1)
    subset=articles[articles['Publish Date']==date.strftime("%Y-%m-%d")]
    st.dataframe(subset)
    orginizations=[]
    for index, row in subset.iterrows():
        [entities,doc]=ner(row["Text"])
        for i,e in enumerate(entities):
            if(e[1]=='ORG'):
                ticker=search_companies(e[0],90)
                if(ticker is not None):
                    orginizations.append([ticker[0],ticker[1],get_stock_at(ticker[0],date.strftime("%Y-%m-%d")),get_stock_at(ticker[0],tomorrow.strftime("%Y-%m-%d"))])
    org_df = pd.DataFrame(orginizations, columns =[ 'Ticker','Name','Change','Next Day Change'])
    org_df.drop_duplicates(subset ="Ticker",keep = "first", inplace = True)
    st.dataframe(org_df)