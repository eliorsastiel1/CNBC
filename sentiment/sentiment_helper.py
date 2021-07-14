from sentiment.finbert import predict
from transformers import AutoTokenizer, AutoModelForSequenceClassification



def get_sentiment_from_text(text,model_name="ProsusAI/finbert"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return predict(text,model,tokenizer)
