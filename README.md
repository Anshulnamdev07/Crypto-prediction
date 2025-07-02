Crypto Historical Data Analysis and Prediction

This project is a comprehensive data analysis and machine learning pipeline for retrieving, processing, and predicting cryptocurrency metrics based on historical data. The task focuses on working with APIs, handling large data, calculating trading metrics, and making predictions using machine learning models.

Table of Contents

Overview
Project Structure
Getting Started
Features
API Research
Data Retrieval and Processing
Machine Learning Models
Results
Challenges
Next Steps
Overview
This project retrieves historical trading data for frequently traded cryptocurrency pairs from the CoinGecko API. It then calculates several analytical metrics, such as historical highs and lows, and uses a machine learning model to predict future percentage differences based on recent historical data.

Crypto Historical Data Analysis and Prediction
This project is a comprehensive data analysis and machine learning pipeline for retrieving, processing, and predicting cryptocurrency metrics based on historical data. The task focuses on working with APIs, handling large data, calculating trading metrics, and making predictions using machine learning models.

Project Structure
crypto_calc.ipynb: Jupyter notebook containing code for data retrieval and metric calculations.
ml_model.py: Python script for training and evaluating machine learning models.
crypto.xlsx: Excel file containing historical data and calculated metrics.
README.md: Documentation for the project.
requirements.txt: Dependencies required for the project.
Getting Started
To get started with this project, follow these steps:

Clone the Repository:
git clone <repository-url>
cd <repository-directory>
Install Dependencies:
pip install -r requirements.txt

Features
Data Retrieval
Fetches historical data for specified cryptocurrency pairs from the CoinGecko API.
Data includes Open, High, Low, Close prices, aggregated daily.
Metric Calculation
Calculates historical high/low and future high/low prices over specific periods.
Adds percentage differences from historical and future highs/lows.
Machine Learning Predictions
Uses LinearRegression,Lasso,Ridge,Support Vector Regressor,RandomForestRegressor and XGBRegressor to predict:
% Difference from High Next 5 Days
% Difference from Low Next 5 Days
Compares models based on Mean Squared Error (MSE) and selects the best-performing model for predictions.
API Research
We used the CoinGecko API for cryptocurrency data. The API provides free access to daily historical data on top-traded crypto pairs.

Supported Pairs: Covers a range of popular cryptocurrency pairs.
Timeframes: Daily data; hourly data not available in this configuration.
Data Range: Available from CoinGecko’s historical data start date to present day.
Data Retrieval and Processing
The fetch_crypto_data function retrieves daily price data starting from a specified date. The data is then processed and saved in crypto.xlsx with key calculated metrics for easy access and visualization.

Machine Learning Models
Feature Columns:
Days Since High Last 7 Days
% Diff From High Last 7 Days
Days Since Low Last 7 Days
% Diff From Low Last 7 Days
Target Columns:
% Diff From High Next 5 Days
% Diff From Low Next 5 Days
Models Used:
LinearRegression : Basic regression model to establish a baseline for predicting price movements.
Lasso and Ridge : Regularized linear models to reduce overfitting and manage feature importance.
Support Vector Regressor : Effective for capturing complex relationships in non-linear data..
RandomForestRegressor: Used for general accuracy in structured data.
XGBRegressor: For optimized tree-based regression with custom learning rates.
Training and Evaluation:
Both models are trained, and the best-performing one (lowest MSE) is saved as {mode_name}_diff_from_high_next_{}_days_model.pkl and {mode_name}_diff_from_low_next_{}_days_model.pkl.

Results
Models are evaluated on Mean Squared Error (MSE), and predictions for new data points can be generated using the predict_outcomes function. Metrics are saved in crypto.xlsx, which can be visualized in Excel or further analyzed.

Challenges
API Rate Limits: Ensured data retrieval compliance within free API limitations.
Data Completeness: Handled missing data cases with dropna functions.
Model Selection: Balanced accuracy with model efficiency using two regressors.
Next Steps
Implement hyperparameter tuning for model optimization.
Expand functionality to handle multiple cryptocurrencies simultaneously.
Integrate additional indicators or metrics for deeper analysis.