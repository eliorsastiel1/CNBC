import pandas as pd
import os
import pickle

from statistics import mean
from tqdm import tqdm
raw_file_name =  os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data/suggestions_dict.pkl')
processed_file_name =  os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Processed/sentiment_weighted.pkl')
sentimentRef=None
def get_sentiment_data():
    global sentimentRef
    if not os.path.isfile(processed_file_name):
        print("Normalizing Sentiment")
        sentiment=pd.read_pickle(raw_file_name)
        cummulative={}
        cummulativeAvg=[]
        for index,row in tqdm(sentiment.iterrows(),total=sentiment.shape[0]):
            if index[1] not in cummulative:
                cummulative[index[1]]=[]
            cummulative[index[1]].append(row['Negative Count']+row['Positive Count'])
            cummulativeAvg.append(mean(cummulative[index[1]]))
        sentiment['Weight']=cummulativeAvg        
        sentiment.to_pickle(processed_file_name)
    else:
        sentiment=pd.read_pickle(processed_file_name)
    sentiment['Sentiment Score'] = (sentiment['Positive Score']-sentiment['Negative Score'])/(sentiment['Weight'])
    #sentiment['Sentiment Score'] = (sentiment['Positive Score']-sentiment['Negative Score'])/(sentiment['Positive Score']-sentiment['Negative Score']+1)
    return sentiment

def get_sentiment_for_day(day):
    global sentimentRef
    if(sentimentRef is None):
        sentimentRef=get_sentiment_data()
    return sentimentRef.loc[day]
