import spacy
import unittest 
from spacy_streamlit import visualize_ner

import en_core_web_md
nlp = en_core_web_md.load()
#pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.0.0/en_core_web_md-3.0.0.tar.gz
#pip install spacy-streamlit
def ner(text):
    """Extract named entities from a text input.
    Returns a list of a word from the text and matching entity
    """
    doc=nlp(text)
    dict=[(X.text, X.label_) for X in doc.ents]
    return [dict,doc]

def plot_ner(text,doc=None,table=False,tit=''):
    if(doc is None):
        doc=nlp(text)
    visualize_ner(doc, labels=nlp.get_pipe("ner").labels,show_table=table,title=tit)
    dict=[(X.text, X.label_) for X in doc.ents]
    return dict

class NerTest(unittest.TestCase): 
    # Returns True or False.  
    def test(self):
        text=" Goldman Sachs says stocks most owned by the smart money are crushing the market this year, leading to the strongest start for hedge funds since 2009. The bank's hedge fund VIP list consists of the 50 stocks that appear most often among the top 10 holdings of fundamentally driven hedge fund portfolios for the June quarter. Goldman analyzed the positions of 803 hedge funds with total long and short stock holdings of $1.9 trillion.   Our Hedge Fund VIP list of the most popular long positions has outperformed the SP 500, strategist Ben Snider wrote in a note to clients Thursday. From an implementation standpoint, the hedge fund VIP list represents a tool for investors seeking to 'follow the smart money' based on 13-F filings. Snider noted that technology is a favorite sector for the hedge fund managers. Popular stocks such as  Facebook ,  Amazon ,  Alibaba  and  Apple  are among the top names on the list.    The firm's basket of the top holdings of hedge funds is up 19 percent this year through Aug. 14 versus the   12 percent return, according to Goldman Sachs. As a result, the average hedge fund is up 7 percent, the industry's best start in eight years, according to the strategist.  The VIP list has beaten the market's performance in 66 percent of quarters and by 64 basis points on average per quarter since 2001. Turnover for the VIP list was slightly below normal with 12 new names in the June quarter compared with the historical average turnover of 16 stocks.   Here are the top 10 stocks on the Goldman's hedge fund VIP list.  — CNBC's Patricia Martell contributed to this report.   "
        expected=[('Goldman Sachs', 'ORG'), ('this year', 'DATE'), ('2009', 'DATE'), ('50', 'CARDINAL'), ('10', 'CARDINAL'), ('the June quarter', 'DATE'), ('Goldman', 'ORG'), ('803', 'CARDINAL'), ('$1.9 trillion', 'MONEY'), ('500', 'CARDINAL'), ('Ben Snider', 'PERSON'), ('Thursday', 'DATE'), ('13', 'CARDINAL'), ('Snider', 'PERSON'), ('Amazon', 'ORG'), ('19 percent', 'PERCENT'), ('this year', 'DATE'), ('Aug. 14', 'DATE'), ('12 percent', 'PERCENT'), ('Goldman Sachs', 'ORG'), ('7 percent', 'PERCENT'), ('eight years', 'DATE'), ('66 percent', 'PERCENT'), ('64', 'CARDINAL'), ('2001', 'DATE'), ('12', 'CARDINAL'), ('the June quarter', 'DATE'), ('16', 'CARDINAL'), ('10', 'CARDINAL'), ('Goldman', 'ORG'), ('CNBC', 'ORG'), ('Patricia Martell', 'PERSON')]
        result=ner(text)
        self.assertCountEqual(result,expected)



if __name__ == '__main__': 
    unittest.main() 
