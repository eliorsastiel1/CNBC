import pandas as pd
import os
import pickle

from statistics import mean
from tqdm import tqdm
raw_file_name =  os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data/suggestions_dict.pkl')
processed_file_name =  os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Processed/sentiment_weighted.pkl')
def get_sentiment_data():
    if not os.path.isfile(processed_file_name):
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
        sentiment=pd.read_pickle('./Processed/sentiment_weighted.pkl')
    sentiment['Sentiment Score'] = (sentiment['Positive Score']-sentiment['Negative Score'])/(sentiment['Weight'])
    return sentiment
