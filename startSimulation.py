import pandas as pd
#from trade_simulator import train_test_split,train_model
from tqdm import tqdm
import yfinance as yf
import os
import pickle
from dataWrapper.sentiment_loader import get_sentiment_data
import multiprocessing
from tradeSimulatorV15 import train_process,simulator
import concurrent.futures
import numpy as np

if __name__ == '__main__':
    simulation_training_data =  os.path.join(os.path.dirname(__file__), 'Data/simulation_train_data.pkl')
    
    simulator_params =  os.path.join(os.path.dirname(__file__), 'Data/simulator_params.pkl')
    simulator_params2 =  os.path.join(os.path.dirname(__file__), 'Data/simulator_params2.pkl')

    sentiment=get_sentiment_data()
    

    

    isum=1000000
    split_date = '2007-02-02'
    #if not os.path.isfile(simulation_training_data):
    #    stock_data=pd.read_pickle('./Data/stocks_data.pkl')
    #    print(stock_data.head())
    #    train,test = train_test_split(stock_data,split_date)
    #    with open(simulation_training_data, 'wb') as out_file:
    #        pickle.dump(train,out_file, protocol=-1)
    #        pickle.dump(test,out_file, protocol=-1)
    #else:
    #    with open(simulation_training_data, 'rb') as in_file: 
    #            train2 = pickle.load(in_file)
    #            test2 = pickle.load(in_file)  

    if not os.path.isfile(simulator_params):
        train=pd.read_pickle('./Data/train.pkl')
        #unwanted=train[train['Adj Close']<0]
        #unwanted=np.unique(unwanted['Short_Ticker'])
        #train = train[~train['Short_Ticker'].isin(unwanted)]

        n_list = []
        k_list = []
        t_coefs = []
        current_list=[]
        #frames=[]
        num_processes = multiprocessing.cpu_count()-1
        #num_processes =1
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(train_process,train,sentiment,1000000,x,y) for x in np.arange(0.01, 0.21, 0.01) for y in np.arange(0.01,0.21,0.01)]
            #futures = [executor.submit(train_process,train,sentiment,isum,x) for x in np.arange(0.01, 0.11, 0.01)]
            for future in tqdm(concurrent.futures.as_completed(futures),total=len(futures)):
                #print(future.result())
                #frames.append(future.result()[2])
                #frame=future.result()[2]
                #frame.to_pickle('Data/Debug/{}-{}.pkl'.format(future.result()[0],future.result()[1]))
                if future.result()[0] is not None:
                    n_list.append(future.result()[0])
                    k_list.append(future.result()[1])
                    t_coefs.extend(future.result()[2])   
                    current_list.append(future.result()[3])    
            #final_n = np.mean(n_list)
            #final_k = np.mean(k_list)    
            #debug_output = pd.concat(frames)
            #debug_output.to_pickle('Data/debug.pkl')
            final_n = np.median(n_list)
            final_k = np.median(k_list)
            f_coefs = []
            #S_I =  np.nanmean([x[0] for x in t_coefs])
            #nVol1 =  np.nanmean([x[1] for x in t_coefs])
            try:
                B_I =  np.nanmean([x[0] for x in t_coefs])
                SA1 =  np.nanmean([x[1] for x in t_coefs])     
                nVol1 = np.nanmean([x[2] for x in t_coefs])
                S_I = np.nanmean([x[3] for x in t_coefs])
                f_coefs.extend([B_I,SA1, nVol1,S_I])
            #f_coefs.extend([S_I,nVol1])
            #return final_n, f_coefs
        #n, coefs =train_model(train,sentiment,isum)
                with open(simulator_params, 'wb') as out_file:
                    pickle.dump(final_n,out_file, protocol=-1)
                    pickle.dump(final_k,out_file, protocol=-1)
                    pickle.dump(f_coefs,out_file, protocol=-1)
                n=final_n
                k=final_k
                coefs=f_coefs
            except:
                print("Error saving coeff")
            with open(simulator_params2, 'wb') as out_file:
                pickle.dump(n_list,out_file, protocol=-1)
                pickle.dump(k_list,out_file, protocol=-1)
                pickle.dump(t_coefs,out_file, protocol=-1)
                pickle.dump(current_list,out_file, protocol=-1)
            
    else:
        test=pd.read_pickle('./Data/test.pkl')
        #unwanted=test[test['Adj Close']<0]
        #unwanted=np.unique(unwanted['Short_Ticker'])
        #test = test[~test['Short_Ticker'].isin(unwanted)]
        simulation_result =  os.path.join(os.path.dirname(__file__), 'Data/simulation_result.pkl')

        with open(simulator_params, 'rb') as in_file: 
            n = pickle.load(in_file)
            k = pickle.load(in_file)
            coefs = pickle.load(in_file) 
            print(n)
            print(k)
        profit=simulator(test,sentiment,n,k,coefs,1000000)
        #profit=simulator(test,sentiment,n,k,1000000)
        with open(simulation_result, 'wb') as out_file:
            pickle.dump(profit,out_file, protocol=-1)
        print(profit)

    ###Compare profits with indices
    #compare = {'^GSPC':'','^RUT':'','^IXIC':'','^DJI':'','^RUI':''}
    #years=range(2015,2021)
    #for index in compare:
    #    index_dict={}
    #    for year in years:
    #      if year != 2021:
    #          data = yf.download(
    #              index, start=split_date, end=f'{year+1}-01-02',interval="1d",progress=False)
    #      else:
    #        data = yf.download(
    #              index, start=split_date, end='2021-03-27',interval="1d",progress=False)
    #      j = data.iloc[[0, -1]]
    #      l = j['Adj Close'].pct_change(periods=1, limit=None, freq=None)        
    #      index_dict[year]= round(l.iloc[-1]*100,3)
    #    compare[index]=index_dict
    #compare['My Program']= profit
    #pd.DataFrame(compare)
