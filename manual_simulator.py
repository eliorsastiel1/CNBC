import pandas as pd
from tradeSimulatorV15 import train_process,simulator
import os
import pickle
from dataWrapper.sentiment_loader import get_sentiment_data
sentiment=get_sentiment_data()
n=0.01
k=0.09
#coefs=[-0.165, 0.164, 0.001, 0.165]
coefs=[-0.114, -0.049, 0.016, 0.114]
test=pd.read_pickle('./Data/test.pkl')

profit=simulator(test,sentiment,n,k,coefs,1000000)
simulation_result =  os.path.join(os.path.dirname(__file__), 'Data/simulation_result.pkl')

with open(simulation_result, 'wb') as out_file:
    pickle.dump(profit,out_file, protocol=-1)
print(profit)