from bs4 import BeautifulSoup
import requests
def get_performance_info(ticker):
    #find etfs performance from Yahoo Finance
    title = ['YTD','1M','3M','1Y','3Y','5Y','10Y','Last Bull Market','Last Bear Market']
    performance_info = requests.get(f'https://finance.yahoo.com/quote/{ticker}/performance?p={ticker}')
    soup = BeautifulSoup(performance_info.content, 'html.parser')
    info = soup.find_all('section', class_ = 'Pb(20px) smartphone_Px(20px) smartphone_Mt(20px) undefined')
    performance = {}
    for ind,i in enumerate(info[0].find_all('div')[6].find_all('div')[2:]):
        try:
            performance[title[ind]] = float(i.find_all('span')[3].get_text()[:-1])
        except:
            performance[title[ind]]= None
    return performance  


def get_distribution_among_sectors(ticker):  
    #find etfs distribution among sectors from Yahoo Finance
    etf_info_yahoo = requests.get(f'https://finance.yahoo.com/quote/{ticker}/holdings?p={ticker}')
    soup_yahoo = BeautifulSoup(etf_info_yahoo.content, 'html.parser')
    sectors = {}
    info = soup_yahoo.find_all('div', class_ = 'W(48%) smartphone_W(100%) Fl(start)')
    for i in info[0].find_all('div')[5].find_all('div')[1:]:
        try:
            sectors[i.find_all('span')[0].get_text()] = float(i.find_all('span')[4].get_text()[:-1])
        except:
            continue
    return sectors