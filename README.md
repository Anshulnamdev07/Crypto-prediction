# 📊 Crypto Historical Data Analysis and Prediction

This project is a comprehensive data analysis and machine learning pipeline for retrieving, processing, and predicting cryptocurrency metrics based on historical data.  
It focuses on working with APIs, handling large data, calculating trading metrics, and making predictions using machine learning models.

---

## 📚 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Features](#features)
- [API Research](#api-research)
- [Data Retrieval and Processing](#data-retrieval-and-processing)
- [Machine Learning Models](#machine-learning-models)
- [Results](#results)
- [Challenges](#challenges)
- [Next Steps](#next-steps)

---

## 📌 Overview

This project retrieves historical trading data for frequently traded cryptocurrency pairs from the CoinGecko API.  
It calculates analytical metrics (like historical highs and lows), and uses machine learning to predict future percentage differences based on recent data.

---

## 🗂️ Project Structure

```bash
├── crypto_calc.ipynb       # Data retrieval + metric calculations (Jupyter)
├── ml_model.py             # ML training & evaluation scripts
├── crypto.xlsx             # Excel file of processed data + metrics
├── README.md               # Project documentation (this file)
├── requirements.txt        # All required Python libraries
🚀 Getting Started
1. Clone the Repository
bash
Copy
Edit
git clone <repository-url>
cd <repository-directory>
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
✨ Features
✅ Data Retrieval
Fetches historical data for any crypto pair via CoinGecko API

Data includes Open, High, Low, Close (OHLC) prices

Aggregated at daily intervals

📈 Metric Calculation
Historical High/Low over a rolling window

Future High/Low prediction targets

Percentage difference from historical and future metrics

🤖 Machine Learning Predictions
Models Used:

LinearRegression

Lasso, Ridge

SVR (Support Vector Regressor)

RandomForestRegressor

XGBRegressor

Targets:

% Diff From High Next 5 Days

% Diff From Low Next 5 Days

Evaluation using Mean Squared Error (MSE)

Best model saved as .pkl

🌐 API Research
Source: CoinGecko API

Supported Pairs: All top-traded crypto pairs

Timeframe: Daily data

Limitation: Free users limited to last 365 days

🛠️ Data Retrieval and Processing
Uses fetch_crypto_data() to fetch & format data

Data saved as crypto.xlsx with:

OHLC prices

Rolling metrics

Percentage changes

🧠 Machine Learning Models
🔹 Features:
Days Since High (Last 7 Days)

% Diff From High (Last 7 Days)

Days Since Low (Last 7 Days)

% Diff From Low (Last 7 Days)

🔹 Targets:
% Diff From High (Next 5 Days)

% Diff From Low (Next 5 Days)

🔹 Models:
Model	Purpose
LinearRegression	Baseline model
Lasso / Ridge	Regularization & feature control
SVR	Handles non-linearity
RandomForest	Robust structured data predictions
XGBoost	Optimized gradient boosting

✅ Results
Models compared via MSE

Predictions saved

Metrics viewable in crypto.xlsx

⚠️ Challenges
API Rate Limits → Used max 365 days

Missing Data → Cleaned with dropna()

Model Choice → Based on accuracy + efficiency

🔮 Next Steps
Hyperparameter tuning

Add multi-crypto support

Add RSI, MACD, volume-based metrics

🔗 Connect
💼 Add this to your LinkedIn & Resume
⭐ Give a star on GitHub if you find it useful