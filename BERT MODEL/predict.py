from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained('./sentiment_model')
model = BertForSequenceClassification.from_pretrained('./sentiment_model')

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1)
    sentiment = "positive" if prediction.item() == 1 else "negative"
    return sentiment

text = "This movie was fantastic! I loved it."
print(f"Sentiment: {predict_sentiment(text)}")

text = "I did not enjoy this movie at all."
print(f"Sentiment: {predict_sentiment(text)}")
