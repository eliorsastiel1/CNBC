from googlesearch import search
def get_ticker_by_company(company_name):
    #use quotation marks to search for exact match
    results=search('"{}" stock'.format(company_name), num_results=5)
    for url in results:
        if('finance.yahoo.com' in url):
            start_idx=url.rfind("quote/")
            if(start_idx==-1):
                return None
            ticker=url[start_idx+6:]
            pquery=ticker.rfind("?")
            if(pquery>=0):
                ticker=ticker[0:pquery]
            ticker=ticker.replace("/", "")
            return ticker
    return None
