# 🚀 Project Apollo: Sentiment & Risk Dashboard

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red.svg)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com)

A real-time dashboard that fetches financial news, performs sentiment analysis with a FinBERT model, and assesses news-driven risk for stocks and market topics.

![Project Apollo Dashboard](assets/apollo-dashboard.png)

## ✨ Features

-   **Real-Time News**: Fetches up-to-the-minute financial news from the Alpha Vantage API.
-   **Advanced Sentiment Analysis**: Uses the `ProsusAI/finbert` model for nuanced financial sentiment analysis (Positive, Negative, Neutral).
-   **Risk Assessment**: Calculates a simple, sentiment-driven risk score for each news article, color-coded for quick insights.
-   **Interactive & User-Friendly UI**: A clean, responsive interface built with Streamlit.
-   **Flexible Queries**: Analyze by stock tickers, market topics, or a combination of both.
-   **Containerized & Reproducible**: Fully containerized with Docker for one-command setup and flawless deployment.

## 🛠️ Tech Stack

-   **Backend**: Python, Pandas, Hugging Face Transformers, PyTorch
-   **Frontend**: Streamlit
-   **Data Source**: Alpha Vantage API
-   **Deployment**: Docker

## 🏁 Getting Started

Follow these instructions to run the application on your local machine using Docker.

### Prerequisites

-   [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed on your system.
-   An **Alpha Vantage API Key**. You can get a free key from their [website](https://www.alphavantage.co/support/#api-key).

### Installation & Execution

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/project-apollo-dashboard.git](https://github.com/YourUsername/project-apollo-dashboard.git)
    cd project-apollo-dashboard
    ```

2.  **Create a `.env` file** in the root of the project and add your API key:
    ```env
    # .env
    ALPHA_VANTAGE_API_KEY="YOUR_ACTUAL_API_KEY"
    ```

3.  **Build the Docker image:**
    ```bash
    docker build -t project-apollo .
    ```

4.  **Run the Docker container:**
    ```bash
    docker run --rm -p 8501:8501 --env-file .env project-apollo
    ```

5.  **Access the application** by opening your web browser and navigating to **http://localhost:8501**.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
