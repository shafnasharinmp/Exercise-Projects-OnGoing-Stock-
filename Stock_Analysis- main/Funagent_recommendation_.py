
from openai import OpenAI
import openai
import os

os.environ['OPENAI_API_KEY'] = ' '
openai.api_key = os.getenv('OPENAI_API_KEY')

# Function to generate a response using CrewAI's GPT model
def generate_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50
    )
    return response.choices[0].text.strip()

# Function to provide recommendations
def provide_recommendation(data):
    last_signal = data['Signal'].iloc[-1]
    if last_signal == 1:
        return "Recommendation: Buy"
    elif last_signal == -1:
        return "Recommendation: Sell"
    else:
        return "Recommendation: Hold"

# Example of integrating CrewAI
def main_agent(query):
    if "trend" in query.lower():
        trend = provide_recommendation(stock_data)
        return trend
    elif "buy" in query.lower() or "sell" in query.lower():
        recommendation = provide_recommendation(stock_data)
        return recommendation
    else:
        return generate_response(query)

user_query = "Should I buy or sell AAPL?"
print(main_agent(user_query))


'''Gemini Response'''
import os
import google.generativeai as gemini
gemini.configure(api_key=" ")

# Function to generate a response using Google's Generative AI
def generate_response(prompt):
    response = gemini.generate_text(
        model='models/text-bison-001',
        prompt=prompt,
        temperature=0.7,
        max_output_tokens=50
    )
    return response.result



def main_agent(query):
    if "trend" in query.lower():
        # Assuming the query asks for a specific stock, which we'll simplify here
        # Analyze trends and generate insights
        recent_trend = stock_data['Signal'].iloc[-1]
        if recent_trend == 1:
            return "The stock is currently in an upward trend."
        elif recent_trend == 0:
            return "The stock is currently in a neutral or unclear trend."
        else:
            return "The stock is currently in a downward trend."
    elif "buy" in query.lower() or "sell" in query.lower():
        # Provide basic buy/sell signals
        buy_signal = stock_data[stock_data['Position'] == 1].index[-1]
        sell_signal = stock_data[stock_data['Position'] == -1].index[-1]
        if buy_signal > sell_signal:
            return "It might be a good time to buy based on the latest signals."
        else:
            return "It might be a good time to sell based on the latest signals."
    else:
        return generate_response(query)

# Example usage
user_query = "What is the recent trend for the stock?"
print(main_agent(user_query))
user_query = "Should I buy or sell Stock?"
print(main_agent(user_query))
user_query = "What are the insights and recommendation over recent stock data?"
print(main_agent(user_query))


def generate_response(prompt):
    response = gemini.generate_text(
        model='models/text-bison-001',
        prompt=prompt,
        temperature=0.7,
        max_output_tokens=50
    )
    return response.result
prompt = "What is the capital of France?"
print(generate_response(prompt))


