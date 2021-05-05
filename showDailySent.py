import pandas as pd
from pandasgui import show
sugg=pd.read_pickle('Processed/suggestions_dict.pkl')
print(sugg.loc['2006-11-27'].iloc[0])
#show(sugg)