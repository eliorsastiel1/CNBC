import pandas as pd
import os
import pickle

file_name =  os.path.join(os.path.dirname(__file__), 'Data/sentiment_data.pkl')

with open(file_name, 'rb') as in_file: 
    cnbc_data = pickle.load(in_file)
print(cnbc_data.iloc[0].content)
