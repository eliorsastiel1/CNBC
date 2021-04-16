import yfinance as yf
import matplotlib.pyplot as plt
from NER import ner
msft=yf.Ticker("GOOG")
ent=ner(msft.info['longBusinessSummary'])
org=''
for token in ent:
    if(token[1]=='ORG'):
        if(org!=''):
            org=org+'|'
        org=org+token[0]
print(org)
#print(ent)
#hist=msft.history(period="max")
#plt.plot(hist)
#print(msft.info)
#plt.show()
