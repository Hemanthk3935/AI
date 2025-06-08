# 🚀 Project Apollo: Real-Time Market Sentiment & News-Driven Risk Dashboard

Project Apollo is a web application that fetches the latest financial news, performs sentiment analysis using a state-of-the-art FinBERT model, and calculates a simplified risk score for each news article. It provides a high-level dashboard for quickly assessing the sentiment landscape for specific stocks or market topics.

This repository contains code that has been verified to run perfectly given the correct setup.

## ✨ Features

- **Real-Time News**: Fetches up-to-the-minute financial news from Alpha Vantage.
- **Advanced Sentiment Analysis**: Uses the `ProsusAI/finbert` model for nuanced financial sentiment analysis.
- **Risk Assessment**: Calculates a simple, sentiment-driven risk score for each article.
- **Interactive UI**: A clean, user-friendly interface built with Streamlit.
- **Containerized**: Fully containerized with Docker for easy, flawless deployment.

---

## 🏁 How to Run This Application Perfectly

Follow these steps precisely to get a running version on your local machine.

### Prerequisites

1.  **Docker**: Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) on your system.
2.  **Alpha Vantage API Key**: Get a free API key from the [Alpha Vantage website](https://www.alphavantage.co/support/#api-key). The application will **not** run without it.

### Step 1: Create the Project Directory

Create a folder for the project and place the following four files inside it:

1.  `project_apollo_sentiment_risk_backend.py`
2.  `project_apollo_frontend.py`
3.  `requirements.txt`
4.  `Dockerfile`

### Step 2: Configure Your API Key

This is the most common point of failure. You **must** provide your API key.

1.  In the same project folder, create a new file named `.env`.
2.  Open the `.env` file and add your API key like this (replace `YOUR_KEY_HERE` with your actual key):

    ```env
    # .env
    ALPHA_VANTAGE_API_KEY="YOUR_KEY_HERE"
    ```

3.  Save the file.

### Step 3: Build and Run with Docker

Using Docker is the recommended way to run this application as it handles all dependencies and configurations within a controlled environment.

1.  Open your terminal or command prompt.
2.  Navigate into your project folder.
3.  **Build the Docker image**. This command reads the `Dockerfile`, downloads the necessary base image, installs the Python packages from `requirements.txt`, and packages your application.

    ```bash
    docker build -t project-apollo .
    ```
    *(Note: The first time you build this, it will download the base Python image and the PyTorch/Transformers libraries, which may take a few minutes.)*

4.  **Run the Docker container**. This command starts your application.

    ```bash
    docker run --rm -p 8501:8501 --env-file .env project-apollo
    ```
    - `--rm`: Automatically removes the container when it's stopped.
    - `-p 8501:8501`: Maps your computer's port 8501 to the container's port 8501.
    - `--env-file .env`: Tells Docker to load the environment variables (your API key) from the `.env` file.

### Step 4: Access the Application

Once the container is running, open your web browser and go to:

**http://localhost:8501**

You will see the Project Apollo dashboard, ready to use. The first time you run an analysis, the backend will download the `ProsusAI/finbert` model, which might take a moment. Subsequent analyses will be much faster.

By following these precise steps, you will have a perfectly running instance of the application.AI
