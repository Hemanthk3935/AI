# project_apollo_sentiment_risk_backend.py
# Core backend for Real-Time Market Sentiment & News-Driven Risk Dashboard

import os
import requests # For making HTTP requests to news APIs
import pandas as pd # For data manipulation and analysis
from transformers import AutoTokenizer, AutoModelForSequenceClassification # For sentiment analysis
import torch # PyTorch, required by transformers
from datetime import datetime, timedelta # For handling dates and times
import time # For rate limiting
import logging # For logging messages

# --- Configuration & Setup ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# API Key for Alpha Vantage (Replace with your actual key)
# IMPORTANT: Store your API key securely, e.g., as an environment variable.
# For GitHub: Do NOT commit your API key directly into the code.
# Use GitHub Secrets and load it as an environment variable in your deployment.
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', "YOUR_API_KEY_HERE")

# Hugging Face Model for Financial Sentiment Analysis
SENTIMENT_MODEL_NAME = "ProsusAI/finbert"

# --- Helper Functions ---

def validate_api_key():
    """Checks if the API key is set and not the placeholder."""
    if ALPHA_VANTAGE_API_KEY == "YOUR_API_KEY_HERE" or not ALPHA_VANTAGE_API_KEY:
        logging.warning("Alpha Vantage API key is not set or is using the placeholder. News fetching will be disabled.")
        return False
    return True

# --- Sentiment Analysis Module ---

class SentimentAnalyzer:
    """
    Handles sentiment analysis using a pre-trained Hugging Face model.
    """
    _instance = None # For Singleton pattern

    # Making SentimentAnalyzer a singleton to avoid reloading the model multiple times
    # This is beneficial in a Streamlit app context where scripts might re-run.
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SentimentAnalyzer, cls).__new__(cls)
            cls._instance._initialized = False # Add an initialization flag
        return cls._instance

    def __init__(self, model_name=SENTIMENT_MODEL_NAME):
        """
        Initializes the tokenizer and model.
        Downloads the model from Hugging Face if not already cached.
        Ensures initialization happens only once for the singleton instance.
        """
        if hasattr(self, '_initialized') and self._initialized: # Check if already initialized
            return

        self.tokenizer = None # Initialize attributes
        self.model = None
        try:
            logging.info(f"Loading sentiment analysis model: {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval() # Set model to evaluation mode
            logging.info("Sentiment analysis model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading sentiment model: {e}")
            # self.tokenizer and self.model remain None if loading fails
        self._initialized = True # Mark as initialized


    def analyze_sentiment(self, text):
        """
        Analyzes the sentiment of a given text.

        Args:
            text (str): The text to analyze.

        Returns:
            dict: A dictionary containing 'label' (Positive, Negative, Neutral)
                  and 'score' (confidence score). Returns error dict if model not loaded or other issues.
        """
        if not self.model or not self.tokenizer:
            logging.warning("Sentiment model not available. Cannot analyze sentiment.")
            return {'label': 'Error', 'score': 0.0, 'error': 'Sentiment model not loaded'}


        if not text or not isinstance(text, str) or not text.strip(): # Added strip() check
            logging.warning("Invalid or empty text input for sentiment analysis.")
            return {'label': 'Neutral', 'score': 0.0, 'error': 'Invalid or empty input'}

        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            with torch.no_grad(): # Disable gradient calculations for inference
                outputs = self.model(**inputs)

            scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # FinBERT labels: 0: positive, 1: negative, 2: neutral
            label_map = {0: 'Positive', 1: 'Negative', 2: 'Neutral'}

            max_score_index = torch.argmax(scores).item()
            max_score = scores[0, max_score_index].item()

            sentiment_label = label_map.get(max_score_index, 'Unknown')

            return {'label': sentiment_label, 'score': max_score, 'error': None}

        except Exception as e:
            logging.error(f"Error during sentiment analysis for text '{text[:50]}...': {e}")
            return {'label': 'Error', 'score': 0.0, 'error': str(e)}

# --- News Fetching Module ---

class NewsFetcher:
    """
    Fetches financial news from Alpha Vantage.
    """
    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key):
        self.api_key = api_key

    def fetch_news_sentiments(self, tickers=None, topics=None, limit=50):
        """
        Fetches news and sentiment data from Alpha Vantage.

        Args:
            tickers (list, optional): List of stock tickers (e.g., ["AAPL", "MSFT"]).
            topics (list, optional): List of topics (e.g., ["technology", "earnings"]).
            limit (int, optional): Number of results to return (max 1000, but be mindful of API limits).

        Returns:
            pd.DataFrame: DataFrame containing news articles, or empty DataFrame if fetching fails.
        """
        # Define expected columns for consistent empty DataFrame returns
        expected_columns = ['title', 'url', 'time_published', 'summary', 'source',
                            'overall_sentiment_label', 'overall_sentiment_score', 'ticker_sentiment']

        if not validate_api_key(): # Validate API key just before making the call
            logging.warning("Cannot fetch news: API key is not valid or not set.")
            return pd.DataFrame(columns=expected_columns)

        params = {
            "function": "NEWS_SENTIMENT",
            "apikey": self.api_key,
            "limit": str(limit)
        }
        # Ensure tickers/topics are properly formatted if provided
        if tickers and any(t.strip() for t in tickers): # Check if list is not empty and contains non-empty strings
            params["tickers"] = ",".join([t.strip() for t in tickers if t.strip()])
        if topics and any(t.strip() for t in topics):
            params["topics"] = ",".join([t.strip() for t in topics if t.strip()])

        logging.info(f"Fetching news with params: {params}")

        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            if "Information" in data:
                logging.warning(f"API Information: {data['Information']}")
                return pd.DataFrame(columns=expected_columns)

            if "feed" not in data or not data["feed"]:
                logging.info("No news feed found for the given parameters.")
                return pd.DataFrame(columns=expected_columns)

            news_items = data["feed"]
            df = pd.DataFrame(news_items)

            if df.empty: # If DataFrame is empty after creation
                logging.info("Fetched news but resulted in an empty DataFrame.")
                return pd.DataFrame(columns=expected_columns) # Return with expected columns

            # Standardize data types and handle missing values
            df['time_published'] = pd.to_datetime(df['time_published'], format='%Y%m%dT%H%M%S', errors='coerce')
            df['overall_sentiment_score'] = pd.to_numeric(df['overall_sentiment_score'], errors='coerce')

            # Ensure all expected columns are present
            for col in expected_columns:
                if col not in df.columns:
                    df[col] = pd.NA # Use pandas NA for missing values consistently

            logging.info(f"Successfully fetched {len(df)} news articles.")
            return df[expected_columns] # Return only expected columns

        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching news: {e}")
        except ValueError as e: # Catches JSON decoding errors
            logging.error(f"Error decoding JSON response from news API: {e}")
            logging.error(f"Response text was: {response.text}")
        except Exception as e:
            logging.error(f"An unexpected error occurred while fetching news: {e}")

        return pd.DataFrame(columns=expected_columns)


# --- Risk Assessment Module (Simplified) ---

class RiskAssessor:
    """
    Calculates a simplified risk score based on news sentiment and other factors.
    """
    def calculate_risk_score(self, sentiment_score_numeric, volatility_factor=0.5):
        """
        Calculates a risk score. A more negative sentiment leads to a higher risk score.
        - sentiment_score_numeric is between -1 (very negative) and +1 (very positive).
        - A score of +1 should result in low risk.
        - A score of -1 should result in high risk.
        """
        # Ensure sentiment_score_numeric is float, default to 0.0 if None or NaN
        sentiment_score_numeric = float(sentiment_score_numeric) if pd.notna(sentiment_score_numeric) else 0.0

        # Normalize sentiment so that +1 -> 0 and -1 -> 1. This is our base risk.
        normalized_sentiment_risk = (1 - sentiment_score_numeric) / 2

        # Apply a volatility factor (placeholder for now, could be based on market data)
        # This enhances the risk based on perceived market stability.
        risk_score = normalized_sentiment_risk * (1 + volatility_factor)

        # Clamp the risk score to be between 0 (no risk) and 1 (max risk)
        return min(max(risk_score, 0), 1)

    def map_sentiment_to_numeric(self, sentiment_label, sentiment_confidence):
        """Maps categorical sentiment label and confidence to a single numeric score from -1 to 1."""
        # Ensure confidence is float, default to 0.0 if None or NaN
        sentiment_confidence = float(sentiment_confidence) if pd.notna(sentiment_confidence) else 0.0

        if sentiment_label == 'Positive':
            return 1.0 * sentiment_confidence
        elif sentiment_label == 'Negative':
            return -1.0 * sentiment_confidence
        elif sentiment_label == 'Neutral':
            return 0.0
        else: # Error or Unknown
            return 0.0

# --- Main Application Logic ---

def process_financial_data(tickers=None, topics=None, news_limit=20):
    """
    Main function to fetch news, analyze sentiment, and assess risk.
    """
    # Standardize input: ensure tickers and topics are lists of strings, handle None or empty.
    tickers_processed = [t.strip().upper() for t in tickers if t and t.strip()] if tickers else []
    topics_processed = [t.strip().lower() for t in topics if t and t.strip()] if topics else []

    # Define expected columns for the final DataFrame to ensure consistency
    final_df_columns = [
        'title', 'summary', 'time_published', 'source', 'url',
        'finbert_sentiment_label', 'finbert_sentiment_score', 'finbert_error',
        'av_sentiment_label', 'av_sentiment_score',
        'numeric_sentiment', 'risk_score', 'ticker_sentiment'
    ]

    if not validate_api_key():
        logging.error("Exiting due to invalid or missing API key for process_financial_data.")
        return pd.DataFrame(columns=final_df_columns)

    # Initialize components
    news_fetcher = NewsFetcher(ALPHA_VANTAGE_API_KEY)
    sentiment_analyzer = SentimentAnalyzer() # Gets the singleton instance
    risk_assessor = RiskAssessor()

    # 1. Fetch News
    logging.info(f"Processing financial data for tickers: {tickers_processed}, topics: {topics_processed}...")
    news_df = news_fetcher.fetch_news_sentiments(tickers=tickers_processed, topics=topics_processed, limit=news_limit)

    if news_df.empty:
        logging.warning("No news data fetched by NewsFetcher. Returning empty DataFrame.")
        return pd.DataFrame(columns=final_df_columns)

    # 2. Analyze Sentiment for each news item using our model
    logging.info("Analyzing sentiment for fetched news articles...")
    finbert_sentiments_data = []
    for index, row in news_df.iterrows():
        text_to_analyze = row['summary'] if pd.notna(row['summary']) and row['summary'].strip() else row['title']
        if not text_to_analyze or not (isinstance(text_to_analyze, str) and text_to_analyze.strip()):
            logging.warning(f"Skipping sentiment analysis for empty or invalid content (index {index}).")
            finbert_sentiments_data.append({'label': 'Neutral', 'score': 0.0, 'error': 'Empty or invalid content'})
            continue

        sentiment_result = sentiment_analyzer.analyze_sentiment(text_to_analyze)
        finbert_sentiments_data.append(sentiment_result)

    finbert_df = pd.DataFrame(finbert_sentiments_data)
    finbert_df.rename(columns={'label': 'finbert_sentiment_label',
                               'score': 'finbert_sentiment_score',
                               'error': 'finbert_error'}, inplace=True)

    # Concatenate DataFrames
    news_df.reset_index(drop=True, inplace=True)
    finbert_df.reset_index(drop=True, inplace=True)
    processed_df = pd.concat([news_df, finbert_df], axis=1)


    # 3. Calculate Risk Score
    logging.info("Calculating risk scores...")
    risk_scores = []
    numeric_sentiments = []
    for index, row in processed_df.iterrows():
        current_sentiment_label = row.get('finbert_sentiment_label', 'Neutral')
        current_sentiment_score = row.get('finbert_sentiment_score', 0.0)

        if pd.isna(current_sentiment_score): current_sentiment_score = 0.0

        # Fallback to Alpha Vantage sentiment if our model fails
        if pd.isna(current_sentiment_label) or current_sentiment_label == 'Error':
            av_label = row.get('overall_sentiment_label')
            av_score = row.get('overall_sentiment_score')
            if pd.notna(av_label) and isinstance(av_label, str) and pd.notna(av_score):
                if 'bullish' in av_label.lower(): current_sentiment_label = 'Positive'
                elif 'bearish' in av_label.lower(): current_sentiment_label = 'Negative'
                else: current_sentiment_label = 'Neutral'
                current_sentiment_score = abs(float(av_score))
            else:
                current_sentiment_label = 'Neutral'
                current_sentiment_score = 0.0

        numeric_sentiment = risk_assessor.map_sentiment_to_numeric(current_sentiment_label, current_sentiment_score)
        numeric_sentiments.append(numeric_sentiment)

        # For this version, volatility is a mock constant. In a future version, this could be dynamic.
        mock_volatility = 0.3
        risk = risk_assessor.calculate_risk_score(numeric_sentiment, mock_volatility)
        risk_scores.append(risk)

    processed_df['numeric_sentiment'] = numeric_sentiments
    processed_df['risk_score'] = risk_scores

    # Rename Alpha Vantage sentiment columns for clarity
    processed_df.rename(columns={
        'overall_sentiment_label': 'av_sentiment_label',
        'overall_sentiment_score': 'av_sentiment_score'
    }, inplace=True, errors='ignore')

    # Ensure all final columns exist, add if not (with NA), and select them in order
    for col in final_df_columns:
        if col not in processed_df.columns:
            processed_df[col] = pd.NA

    processed_df = processed_df[final_df_columns]

    logging.info("Financial data processing complete.")
    return processed_df


# --- Example Usage (for direct script execution) ---
if __name__ == "__main__":
    logging.info("Starting Project Apollo: Sentiment & Risk Backend Demo (Direct Execution)")

    # --- IMPORTANT ---
    # Before running, set your Alpha Vantage API key as an environment variable:
    # export ALPHA_VANTAGE_API_KEY="YOUR_ACTUAL_KEY"

    if not validate_api_key():
        print("\nWARNING: Alpha Vantage API key is not configured correctly.")
        print("The script will attempt to run but news fetching will fail.")
        print("Please set the ALPHA_VANTAGE_API_KEY environment variable.")

    print("\n--- Scenario 1: General Technology News (Example) ---")
    tech_topics_input = ["technology", "ipo"]
    df_tech_news = process_financial_data(topics=tech_topics_input, news_limit=5)
    if not df_tech_news.empty:
        print(f"Processed {len(df_tech_news)} technology news articles:")
        print(df_tech_news[['title', 'finbert_sentiment_label', 'risk_score', 'time_published']].head())
    else:
        print("No technology news data processed (check API key, API limits, or parameters).")

    if validate_api_key():
        logging.info("Waiting briefly to respect API rate limits...")
        time.sleep(15) # Be cautious with Alpha Vantage free tier

    print("\n--- Scenario 2: News for specific Tickers (Example: AAPL, MSFT) ---")
    specific_tickers_input = ["AAPL", "MSFT"]
    df_ticker_news = process_financial_data(tickers=specific_tickers_input, news_limit=5)
    if not df_ticker_news.empty:
        print(f"Processed {len(df_ticker_news)} ticker-specific news articles:")
        print(df_ticker_news[['title', 'finbert_sentiment_label', 'risk_score', 'time_published']].head())
    else:
        print("No ticker-specific news data processed (check API key, API limits, or parameters).")

    logging.info("Project Apollo: Sentiment & Risk Backend Demo Finished.")
