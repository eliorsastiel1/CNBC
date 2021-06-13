from tradeSimulatorV14 import train_process
from dataWrapper.sentiment_loader import get_sentiment_data
import pandas as pd
import pickle
#sentiment=get_sentiment_data()
#train=pd.read_pickle('./Data/train.pkl')
#train_process(train,sentiment,1000000,0.03,0.03)\

with open('./Data/simulator_params2.pkl', 'rb') as in_file:
    n_list = pickle.load(in_file)
    k_list = pickle.load(in_file)
    t_coefs = pickle.load(in_file)
    current_list=pickle.load(in_file)
#print(n_list)
#print(current_list)
#print(k_list.index(0.09))
idx=k_list.index(0.09)
print(n_list[idx])
print(k_list[idx])
print(t_coefs[idx])
#print(current_list[idx])
#print(t_coefs[0])
#print(t_coefs[1])
#print(current_list.index(max(current_list)))