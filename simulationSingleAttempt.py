import pandas as pd
from trade_simulator import train_test_split,train_model,simulator
from tqdm import tqdm
import yfinance as yf
import os
import pickle
from dataWrapper.sentiment_loader import get_sentiment_data
import multiprocessing
from tradeSimulatorV8 import train_process
import concurrent.futures
import numpy as np

if __name__ == '__main__':
    simulation_training_data =  os.path.join(os.path.dirname(__file__), 'Data/simulation_train_data.pkl')
    
    simulator_params =  os.path.join(os.path.dirname(__file__), 'Data/simulator_params.pkl')
    simulator_params2 =  os.path.join(os.path.dirname(__file__), 'Data/simulator_params2.pkl')

    sentiment=get_sentiment_data()
    train=pd.read_pickle('./Data/train.pkl')
    isum=1000000
    split_date = '2007-02-02'
    tracking=train_process(train,sentiment,isum,0.01,0.01)
    tracking.to_pickle("./Data/track.pkl")
