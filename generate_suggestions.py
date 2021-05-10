import pandas as pd
import os
import pickle
from NER import ner
from extract_companies_from_text import search_companies #uses fuzzywuzzy
from fuzzywuzzy import process
#from dataWrapper.cnbc_data_loader import get_cnbc_data

file_name =  os.path.join(os.path.dirname(__file__), 'Data/sentiment_data.pkl')
output =  os.path.join(os.path.dirname(__file__), 'Data/suggestions_dict.pkl')

def add_sentiment(row,suggestions,s):
    if(row['Sentiment']=='positive'):
        suggestions[row['Effective_Date']][s]={"Positive Count":1,"Negative Count":0,"Positive Score":row['Sentiment_confidence'],"Negative Score":0}
    elif(row['Sentiment']=='negative'):
        suggestions[row['Effective_Date']][s]={"Negative Count":1,"Positive Count":0,"Negative Score":row['Sentiment_confidence'],"Positive Score":0}
    return suggestions

def sum_sentiment(row,suggestions,s):
    if(row['Sentiment']=='positive'):
        suggestions[row['Effective_Date']][s]={"Positive Count":suggestions[row['Effective_Date']][s]["Positive Count"]+1,"Positive Score":suggestions[row['Effective_Date']][s]["Positive Score"]+row['Sentiment_confidence'],"Negative Score":suggestions[row['Effective_Date']][s]["Negative Score"],"Negative Count":suggestions[row['Effective_Date']][s]["Negative Count"]}
    elif(row['Sentiment']=='negative'):
        suggestions[row['Effective_Date']][s]={"Negative Count":suggestions[row['Effective_Date']][s]["Negative Count"]+1,"Negative Score":suggestions[row['Effective_Date']][s]["Negative Score"]+row['Sentiment_confidence'],"Positive Count":suggestions[row['Effective_Date']][s]["Positive Count"],"Positive Score":suggestions[row['Effective_Date']][s]["Positive Score"]}
    return suggestions

#articles=get_cnbc_data()
with open(file_name, 'rb') as in_file: 
    cnbc_data = pickle.load(in_file)

suggestions={}
for index, row in cnbc_data.iterrows():
    [entities,doc]=ner(row.content)
    orginazations=[]
    annotated=[]
    for i,e in enumerate(entities):
        if(e[1]=='ORG'):
            ticker=search_companies(e[0],90)
            if(ticker is not None):
                orginazations.append(ticker[0])
    orginazations = list(dict.fromkeys(orginazations))
    #print(orginazations)
    for s in orginazations:
        if(row['Effective_Date'] in suggestions):
            if(s in suggestions[row['Effective_Date']]):
                suggestions=sum_sentiment(row,suggestions,s)
            else:
                suggestions=add_sentiment(row,suggestions,s)
        else:
            suggestions[row['Effective_Date']]={}
            suggestions=add_sentiment(row,suggestions,s)
with open(output, 'wb') as out_file:
    pickle.dump(suggestions,out_file, protocol=-1)

    
