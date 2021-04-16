import os
import requests
from finbert.finbert import predict
from transformers import AutoModelForSequenceClassification
import nltk
nltk.download('punkt')

def load_model():
    model_url='https://prosus-public.s3-eu-west-1.amazonaws.com/finbert/finbert-sentiment/pytorch_model.bin'
    model_dir=os.path.join(os.path.dirname(__file__), 'models')
    sentiment_model_dir=model_dir+'/finbert-sentiment'
    model_path=sentiment_model_dir+'/pytorch_model.bin'
    config_path=sentiment_model_dir+'/config.json'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(sentiment_model_dir):
        os.mkdir(sentiment_model_dir)
    if not os.path.exists(model_path):
        r = requests.get(model_url, allow_redirects=True)
        open(model_path, 'wb').write(r.content)
    if not os.path.exists(config_path):
        open(config_path, 'w').write('{"attention_probs_dropout_prob": 0.1,"hidden_act": "gelu","hidden_dropout_prob": 0.1,"hidden_size": 768,"initializer_range": 0.02,"intermediate_size": 3072,"max_position_embeddings": 512,"num_attention_heads": 12,"num_hidden_layers": 12,"type_vocab_size": 2,"vocab_size": 30522}')
    return sentiment_model_dir
        
def extract_sentiment_from_text(text,labels=3):
    model_path=load_model()
    model = AutoModelForSequenceClassification.from_pretrained(model_path,num_labels=labels,cache_dir=None)
    prediction=predict(text,model)
    print(prediction)
    return prediction
