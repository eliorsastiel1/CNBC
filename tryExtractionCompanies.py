import pandas as pd
import os
import pickle
from NER import ner
from extract_companies_from_text import search_companies #uses fuzzywuzzy
from fuzzywuzzy import process
from tqdm import tqdm
import multiprocessing
import numpy as np
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()

from fuzzywuzzy import process
from dataWrapper.load_russel3000_stocks import load_russel3000
companies=load_russel3000()
details=companies['Details'].tolist()
names=companies['Name'].tolist()

file_name =  os.path.join(os.path.dirname(__file__), 'Data/sentiment_data.pkl')
output =  os.path.join(os.path.dirname(__file__), 'Data/suggestions_dict.pkl')

with open(file_name, 'rb') as in_file: 
    cnbc_data = pickle.load(in_file)
for index, row in tqdm(cnbc_data.iterrows(), total=cnbc_data.shape[0]):
    doc=nlp(row.content)
    entities=[(X.text, X.label_) for X in doc.ents]
    #[entities,doc]=ner(row.content)
    orgs=[]
    for i,e in enumerate(entities):
        if(e[1]=='ORG'):
            #result=process.extractOne(e[0], names)
            orgs.append(e[0])
            #ticker=search_companies(e[0],90)
