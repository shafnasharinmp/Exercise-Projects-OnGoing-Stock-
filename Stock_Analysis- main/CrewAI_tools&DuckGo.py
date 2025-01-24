import os
from typing import Optional
from pydantic import BaseModel, Field
from crewai_tools import BaseTool
from duckduckgo_search import DDGS

OPENAI_API_KEY=' '
os.environ["OPENAI_MODEL_NAME"] = "gpt-3.5-turbo"

class NewsSearchTool(BaseTool):
    name: str = Field(default="NewsSearchTool", description="Name of the news search tool")
    description: str = Field(default="A tool to search for news articles on a given topic", description="Description of the tool")

    def _run(self, topic: str) -> str:
        ddgs = DDGS()
        results = ddgs.news(keywords=topic, timelimit="d", max_results=20)
        result_string = ""
        for result in results:
            result_string += f"Title: {result['title']}\nDescription: {result['body']}\n\n"
        if len(result_string.strip()) < 2:
            result_string = "No news found for the given keyword"
        return result_string.strip()

news_search_tool = NewsSearchTool()

news_summary = news_search_tool.run("AAPL")
print(news_summary)

news_summary[0]



'''duckduckgo_search& FinBERT'''

import os
import pandas as pd
import torch
from pydantic import BaseModel, Field
from crewai_tools import BaseTool
from duckduckgo_search import DDGS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import google.generativeai as gemini
from finbert_embedding import FinbertEmbedding

# Set up the environment
os.environ["OPENAI_MODEL_NAME"] = "gpt-3.5-turbo"
gemini.configure(api_key=" ")

# Define the custom tool with Pydantic validation
class NewsSearchTool(BaseTool):
    name: str = Field(default="NewsSearchTool", description="Name of the news search tool")
    description: str = Field(default="A tool to search for news articles on a given topic", description="Description of the tool")

    def _run(self, topic: str) -> list:
        ddgs = DDGS()
        results = ddgs.news(keywords=topic, timelimit="d", max_results=20)
        return results

# Instantiate the news search tool
news_search_tool = NewsSearchTool()

# Define a function to get news and generate insights
def get_news_insights(selected_stock):
    # Fetch stock-related news articles
    news_results = news_search_tool.run(selected_stock)
    if not news_results:
        st.write("No recent news found for the selected stock.")
        return

    # Extract titles and descriptions separately
    titles = [result['title'] for result in news_results]
    descriptions = [result.get('body', 'No description available') for result in news_results]
    urls = [result['link'] for result in news_results]

    # Initialize FinBERT for sentiment analysis
    tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone', cache_dir='./models')
    model = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', cache_dir='./models')

    # Tokenize and prepare inputs for sentiment analysis
    inputs = tokenizer(titles, return_tensors='pt', truncation=True, padding=True)

    # Perform sentiment analysis
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1).tolist()

    # Map predictions to sentiment labels
    sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    sentiments = [sentiment_labels[pred] for pred in predictions]

    # Create a DataFrame to display the news and sentiments
    sentiment_df = pd.DataFrame({
        'Title': titles,
        'Description': descriptions,
        'URL': urls,
        'Sentiment': sentiments
    })

    # Display the DataFrame in Streamlit
    print(sentiment_df)

    # Display the news, URLs, and sentiments in Streamlit
    for title, description, url, sentiment in zip(titles, descriptions, urls, sentiments):
        print(f"**Title:** {title}")
        print(f"**Description:** {description}")
        print(f"**URL:** {url}")
        print(f"**Sentiment:** {sentiment}")

    # Generate insights and recommendations using Google Gemini
    news_summary = "\n".join([f"{title}: {description}" for title, description in zip(titles, descriptions)])
    input_text = f"Based on the following news titles and summary, provide insights and recommendations for investors:\n\n{news_summary}\n\n"
    prompt = input_text + "Please provide detailed insights and actionable recommendations for investors based on the above information."

    response = gemini.generate_text(
        prompt=prompt,
        max_output_tokens=150,
        temperature=0.7
    )

    insights_recommendations = response.result

    # Display insights and recommendations
    st.write("## Insights and Recommendations")
    st.write(insights_recommendations)


selected_stock = 'AAPL'
get_news_insights(selected_stock)




import re
text = news_summary
pattern = r"Title:\s*(.*)\nDescription:\s*(.*)(?=\nTitle:|$)"
matches = re.findall(pattern, text, re.DOTALL)

titles_and_descriptions = []
for match in matches:
    title = match[0].strip()
    description = match[1].strip()
    titles_and_descriptions.append((title, description))

for idx, (title, description) in enumerate(titles_and_descriptions):
    print(f"Article {idx+1}:")
    print(f"Title: {title}")
    print(f"Description: {description}")
    print("-" * 40)


'''Insight'''
import google.generativeai as gemini
gemini.configure(api_key=" ")

input_text = "Based on the following news titles and summary, provide insights and recommendations for investors:\n\n"

prompt = input_text + "Please provide detailed insights and actionable recommendations for investors based on the above information."

response = gemini.generate_text(
    prompt=prompt,
    max_output_tokens=150,
    temperature=0.7
)

insights_recommendations1 = response.result

print("Insights and Recommendations:\n")
print(insights_recommendations1)


'''BERT TOKENIZER'''

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re

# Load the pre-trained FinBERT model
model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)


# Extract titles from the news summary
titles = re.findall(r'Title: (.+)', news_summary)

# Perform sentiment analysis for each title
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        softmax_probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted_label = torch.argmax(softmax_probs, dim=1).item()

    sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_mapping[predicted_label]

# Analyze sentiment for each title
sentiment_results = {title: analyze_sentiment(title) for title in titles}

# Print sentiment results
for title, sentiment in sentiment_results.items():
    print(f"Title: {title}")
    print(f"Sentiment: {sentiment}\n")

sentiment_results

import google.generativeai as gemini
gemini.configure(api_key=" ")
input_text = "Based on the following news titles and their sentiment analysis results, provide insights and recommendations for investors:\n\n"

for title, sentiment in sentiment_results.items():
    input_text += f"Title: {title}\nSentiment: {sentiment}\n\n"

prompt = input_text + "Please provide detailed insights and actionable recommendations for investors based on the above information."

response = gemini.generate_text(
    prompt=prompt,
    max_output_tokens=150,
    temperature=0.7
)

insights_recommendations = response.result

print("Insights and Recommendations:\n")
print(insights_recommendations)

insights_recommendations = response

print("Insights and Recommendations:\n")
print(insights_recommendations)

"""END"""












































