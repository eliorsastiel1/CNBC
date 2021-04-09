import yfinance as yf
import matplotlib.pyplot as plt
msft=yf.Ticker("MSFT")
hist=msft.history(period="max")
plt.plot(hist)
#print(msft.info)
plt.show()
