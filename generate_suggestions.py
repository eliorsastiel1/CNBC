import pandas as pd
import os
import pickle
from NER import ner
from extract_companies_from_text import search_companies #uses fuzzywuzzy
from fuzzywuzzy import process
from tqdm import tqdm
import multiprocessing
import numpy as np
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

def iterator(d):
    suggestions={}
    for index, row in d.iterrows():
        [entities,doc]=ner(row.content)
        orginazations=[]
        for i,e in enumerate(entities):
            if(e[1]=='ORG'):
                ticker=search_companies(e[0],90)
                if(ticker is not None):
                    orginazations.append(ticker[0])
        orginazations = list(dict.fromkeys(orginazations))
        for s in orginazations:
            if(row['Effective_Date'] in suggestions):
                if(s in suggestions[row['Effective_Date']]):
                    suggestions=sum_sentiment(row,suggestions,s)
                else:
                    suggestions=add_sentiment(row,suggestions,s)
            else:
                suggestions[row['Effective_Date']]={}
                suggestions=add_sentiment(row,suggestions,s)
    return pd.DataFrame.from_dict({(i,j): suggestions[i][j] for i in suggestions.keys() for j in suggestions[i].keys()},orient='index')

    
if __name__ == '__main__':

    with open(file_name, 'rb') as in_file: 
        cnbc_data = pickle.load(in_file)

    #cnbc_data=cnbc_data.iloc[0:5000,:]
    print(cnbc_data.shape)
    # create as many processes as there are CPUs on your machine
    num_processes = multiprocessing.cpu_count()-2

    # calculate the chunk size as an integer
    chunk_size = int(cnbc_data.shape[0]/num_processes)
    #print(chunk_size)
    indices=[]
    for partition in range(num_processes-1):
        indices.append((partition+1)*chunk_size)
    #print(indices)

    #need to make sure we are not cutting within daetes for multiprocessing
    idx=0
    while idx<len(indices):
        if(cnbc_data.iloc[indices[idx]]['Effective_Date'] == cnbc_data.iloc[indices[idx]-1]['Effective_Date'] ):
            indices[idx]=indices[idx]+1
        else:
            idx=idx+1

    print(indices)

    df_split = np.array_split(cnbc_data, indices)
    pool = multiprocessing.Pool(num_processes)
    result = pd.concat(pool.map(iterator, df_split))

#print(df_split.shape())
#suggestions={}
    with open(output, 'wb') as out_file:
        pickle.dump(result,out_file, protocol=-1)

    
