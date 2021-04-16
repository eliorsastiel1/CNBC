import pandas as pd
import os
import pickle
import yfinance as yf
import sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
from NER import ner
data_file_name =  os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data/russel3000.pickle')
source_file_name =  os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data/russel3000.csv')

# this function is used to add special records of stocks where trade name and legal name are not identical
def fill_special_cases(dataset):
    modified=dataset.append({'Name' : 'Google','Ticker' : 'GOOG'} , ignore_index=True)
    return modified


def load_russel3000():
    if not os.path.isfile(data_file_name):
        if not os.path.isfile(source_file_name):
            print("Missing russel 3000 source file")
            return None
        raw=pd.read_csv(source_file_name)
        details=[]
        for index, row in raw.iterrows():
            stock=yf.Ticker(row["Ticker"])
            try:
                ent=ner(stock.info['longBusinessSummary'])
            except:
                details.append(row["Name"])
                continue
            org=''
            for token in ent:
                if(token[1]=='ORG'):
                    if(org!=''):
                        org=org+'|'
                    org=org+token[0]
            details.append(org)
        raw['Details']=details
        with open(data_file_name, 'wb') as out_file:
                pickle.dump(raw,out_file, protocol=-1)
        print(raw.head())
        return raw
    else:
        with open(data_file_name, 'rb') as in_file: 
            russel3000 = pickle.load(in_file)
            return fill_special_cases(russel3000)
        

if __name__ == '__main__': 
    load_russel3000()