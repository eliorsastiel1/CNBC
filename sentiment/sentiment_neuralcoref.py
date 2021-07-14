import pandas as pd
import neuralcoref
import en_core_web_lg
from tqdm import tqdm
nlp = en_core_web_lg.load()
neuralcoref.add_to_pipe(nlp)

df = pd.read_pickle('cnbc_newsDF_explode_sentiment.pkl')

#Run neuralcoref on every content in df
df['content'] = [nlp(x)._.coref_resolved for x in tqdm(df['content'], position=0, leave=True)]

#A boolean function - check if a day is a business day
def is_business_day(date):
    return bool(len(pd.bdate_range(date, date)))

#Returns the next business day
bd = pd.tseries.offsets.BusinessDay(n = 1)

#Assigns articles/sentences to the correct business day by publication hour
df['Effective_Date'] = [x.strftime('%Y-%m-%d') if (
    x.hour < 16 and is_business_day(x.strftime('%Y-%m-%d'))==True) else (x+bd).strftime('%Y-%m-%d')
    for x in df['PubDate']]

df.to_pickle('sentiment_data.pkl')