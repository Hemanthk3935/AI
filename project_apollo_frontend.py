# project_apollo_frontend.py
# Streamlit UI for Real-Time Market Sentiment & News-Driven Risk Dashboard

import streamlit as st
import pandas as pd
from project_apollo_sentiment_risk_backend import process_financial_data, validate_api_key

# --- UI Helper Functions ---

def display_metrics(df):
    """Displays the key metrics in columns."""
    if df.empty:
        st.info("No data to display metrics for.")
        return

    # Calculate overall metrics
    avg_risk_score = df['risk_score'].mean()
    avg_sentiment_score = df['numeric_sentiment'].mean()
    
    # Determine overall sentiment label based on the average numeric sentiment
    if avg_sentiment_score > 0.15: # Threshold for being positive
        overall_sentiment = "Positive"
        sentiment_emoji = "😊"
    elif avg_sentiment_score < -0.15: # Threshold for being negative
        overall_sentiment = "Negative"
        sentiment_emoji = "😟"
    else:
        overall_sentiment = "Neutral"
        sentiment_emoji = "😐"

    # Display in columns
    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Sentiment", f"{overall_sentiment} {sentiment_emoji}")
    col2.metric("Average Numeric Sentiment", f"{avg_sentiment_score:.2f}")
    col3.metric("Average Risk Score", f"{avg_risk_score:.2f}")

def format_risk_score(score):
    """Formats the risk score with a color indicator."""
    if pd.isna(score):
        return "N/A"
    color = "red" if score > 0.65 else "orange" if score > 0.4 else "green"
    return f'<span style="color:{color}; font-weight:bold;">{score:.2f}</span>'

def format_sentiment(label):
    """Formats the sentiment label with an emoji."""
    if label == "Positive":
        return "Positive 😊"
    elif label == "Negative":
        return "Negative 😟"
    elif label == "Neutral":
        return "Neutral 😐"
    return "N/A"

# --- Main Streamlit App ---

st.set_page_config(page_title="Project Apollo: Sentiment & Risk Dashboard", layout="wide")

# --- Page Header ---
st.title("🚀 Project Apollo")
st.subheader("Real-Time Market Sentiment & News-Driven Risk Dashboard")
st.markdown("---")

# --- API Key Check ---
if not validate_api_key():
    st.error("🚨 **CRITICAL:** `ALPHA_VANTAGE_API_KEY` is not configured!")
    st.warning("Please set this environment variable to fetch live news data. The app will not function without it.")
    st.stop() # Stop the app from running further if no key is found

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("🔍 Analysis Parameters")

    # Use a text_input for tickers and topics for easier copy-pasting
    tickers_input = st.text_input("Enter Stock Tickers (comma-separated)", "AAPL, GOOGL, TSLA")
    topics_input = st.text_input("Enter News Topics (comma-separated)", "technology, finance")
    
    # Convert comma-separated strings to lists
    tickers = [t.strip() for t in tickers_input.split(',') if t.strip()]
    topics = [t.strip() for t in topics_input.split(',') if t.strip()]

    news_limit = st.slider("Number of News Articles to Analyze", 10, 100, 25)

    analyze_button = st.button("Analyze Sentiment & Risk", type="primary")

# --- Main Content Area ---

if analyze_button:
    # Basic validation
    if not tickers and not topics:
        st.warning("Please enter at least one ticker or topic.")
    else:
        with st.spinner(f"Fetching news and analyzing {news_limit} articles... This may take a moment."):
            try:
                # Call the backend processing function
                results_df = process_financial_data(tickers=tickers, topics=topics, news_limit=news_limit)
                st.session_state['results_df'] = results_df # Cache results in session state
            except Exception as e:
                st.error(f"An unexpected error occurred during analysis: {e}")
                st.session_state['results_df'] = pd.DataFrame() # Clear results on error

# Check if results exist in session state to display them
if 'results_df' in st.session_state and not st.session_state['results_df'].empty:
    df = st.session_state['results_df']
    
    st.success(f"✅ Analysis complete. Displaying results for {len(df)} articles.")

    # --- Display Key Metrics ---
    st.markdown("### 📊 Key Metrics Overview")
    display_metrics(df)
    st.markdown("---")

    # --- Display Detailed Results Table ---
    st.markdown("### 📰 Detailed News Analysis")
    
    # Format the DataFrame for display
    display_df = df[['time_published', 'title', 'source', 'finbert_sentiment_label', 'risk_score']].copy()
    display_df['time_published'] = display_df['time_published'].dt.strftime('%Y-%m-%d %H:%M')
    display_df['risk_score'] = df['risk_score'].apply(format_risk_score)
    display_df['finbert_sentiment_label'] = df['finbert_sentiment_label'].apply(format_sentiment)
    
    display_df.rename(columns={
        'time_published': 'Published',
        'title': 'Article Title',
        'source': 'Source',
        'finbert_sentiment_label': 'Sentiment',
        'risk_score': 'Risk Score'
    }, inplace=True)

    # Use st.dataframe with HTML rendering enabled
    st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
    
    # --- Expander for Raw Data ---
    with st.expander("🔬 View Raw Data & Full Details"):
        st.dataframe(df)

elif 'results_df' in st.session_state and st.session_state['results_df'].empty:
    st.info("No news articles were found for the specified tickers or topics. Please try different search terms or check the API limits.")

else:
    st.info("Enter tickers or topics in the sidebar and click 'Analyze' to begin.")
