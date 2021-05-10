import pandas as pd
from trade_simulator import train_test_split,train_model,simulator
from tqdm import tqdm
import yfinance as yf
import os
import pickle
from dataWrapper.sentiment_loader import get_sentiment_data
import multiprocessing
from trade_simulator import train_process
import concurrent.futures
import numpy as np

simulation_training_data =  os.path.join(os.path.dirname(__file__), 'Data/simulation_train_data.pkl')
    
simulator_params =  os.path.join(os.path.dirname(__file__), 'Data/simulator_params.pkl')

sentiment=get_sentiment_data()
stock_data=pd.read_pickle('./Data/training_data.pkl')
isum=1000000

with open(simulation_training_data, 'rb') as in_file: 
    train = pickle.load(in_file)
    test = pickle.load(in_file)  

n,coeff=train_process(train,sentiment,isum,0.01)
print(n)
print(coeff)