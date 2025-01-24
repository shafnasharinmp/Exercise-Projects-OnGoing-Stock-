"""fINBert -nEWS"""

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from yahoo_fin import news
from newspaper import Article

# Fetch the news data for Apple (AAPL)
news_data = news.get_yf_rss('AAPL')

# Extract headlines and URLs
headlines = [article['title'] for article in news_data]
urls = [article['link'] for article in news_data]

# Initialize FinBert tokenizer and model
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

# Function to fetch and extract the article text
def fetch_article_text(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

# Fetch and analyze articles
for i, (headline, url) in enumerate(zip(headlines, urls)):
    try:
        text = fetch_article_text(url)

        # Tokenize text
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

        # Predict sentiment
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()

        # Map prediction to sentiment label
        labels = ['positive', 'neutral', 'negative']
        sentiment = labels[prediction]

        # Print results
        print(f"Headline: {headline}")
        print(f"URL: {url}")
        print(f"Sentiment: {sentiment}")
        print()

        # Provide recommendation based on sentiment
        if sentiment == 'positive':
            print("View: Consider buying or holding the stock.")
        elif sentiment == 'neutral':
            print("View: Monitor the stock and wait for more news.")
        else:
            print("View: Consider selling or avoiding the stock.")

        print('-' * 80)
    except Exception as e:
        print(f"Error processing article {i}: {e}")

