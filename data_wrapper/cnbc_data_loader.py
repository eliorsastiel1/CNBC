import pandas as pd
import os
import pickle
import json
from datetime import datetime

raw_file_name =  os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data/cnbc_news.json')
parsed_file_name =  os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data/cnbc_news.pickle')
preloaded_file=None
def flatten_data(data):
    flattened=[]
    for year in data:
        if(data[year] is None):
            continue
        for month in data[year]:
            if(data[year][month] is None):
                continue
            for day in data[year][month]:
                if(data[year][month][day] is None):
                    continue
                for url in data[year][month][day]:
                    publish_date = datetime.strptime("{}-{}-{}".format(year,month,day), "%Y-%B-%d")
                    flattened.append({'Publish Date': publish_date,'Headline':data[year][month][day][url]['Headline'],'URL':url,'Author':data[year][month][day][url]['Author'],'Text':data[year][month][day][url]['Article_Text']})
    df = pd.DataFrame(flattened)
    return df

def get_cnbc_data():
    global preloaded_file
    if not os.path.isfile(parsed_file_name):
        print('Parsed data unavailable, parsing JSON file')
        with open(raw_file_name, encoding="utf8") as f:
            data = json.load(f)
            cnbc_data=flatten_data(data)
            with open(parsed_file_name, 'wb') as out_file:
                pickle.dump(cnbc_data,out_file, protocol=-1)
    else:
        print('Reading Parsed Data')
        if(preloaded_file is not None):
            print('preloaded file')
            return preloaded_file
        with open(parsed_file_name, 'rb') as in_file: 
            cnbc_data = pickle.load(in_file)
            preloaded_file=cnbc_data
    #print(cnbc_data.tail())
    return cnbc_data

def get_cnbc_data_with_sentiment():
    df= pd.read_pickle(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data/sentiment_data.pkl'))
    df['Publish Date'] = df["PubDate"].dt.strftime("%Y-%m-%d")
    return df
    