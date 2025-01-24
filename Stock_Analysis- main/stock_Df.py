"""Dataframe"""

pip install finbert_embedding

from finbert_embedding import FinbertEmbedding
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pandasai import Agent

os.environ["PANDASAI_API_KEY"] = " "
# Initialize FinBERT for embeddings
finbert = FinbertEmbedding()

# Sample news data
news_data = titles
# Get embeddings (assuming the library provides a method for this)
embeddings = finbert.process_text(news_summary)

# Load pre-trained FinBERT model for sentiment classification
tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone', cache_dir='./models')
model = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', cache_dir='./models')

# Tokenize and prepare inputs
inputs = tokenizer(news_data, return_tensors='pt', truncation=True, padding=True)

# Perform sentiment analysis
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1).tolist()

# Map predictions to sentiment labels (assuming 0 = negative, 1 = neutral, 2 = positive)
sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
sentiments = [sentiment_labels[pred] for pred in predictions]

# Convert results to DataFrame
sentiment_df = pd.DataFrame({'Text': news_data, 'Sentiment': sentiments})
print(sentiment_df)

sentiment_df

agent = Agent(sentiment_df)
agent.chat('Which is the first news?')

agent.chat('can I invest now?')

agent.chat('Sentiment Barchart')
