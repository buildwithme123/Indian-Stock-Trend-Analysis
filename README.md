# Indian-Stock-Trend-Analysis
# Sensex Stock Market Trend Prediction & ML Backtesting

This project analyzes historical stock market data (focusing on the **Sensex**, along with Nifty 50, SBI, HDFC Bank, and TCS) using **Python**, and builds Machine Learning classification models to predict daily stock price movements and backtest trading strategies.



## Project Overview:-

The project follows a standard quantitative finance and machine learning pipeline:
1. **Data Acquisition**: Downloads historical adjusted close prices from 2015-01-01 onwards using `yfinance`.
2. **Feature Engineering**: Calculates quantitative features including 1-day returns, 5-day returns, moving average gaps (`ma_gap`), and rolling volatility.
3. **Target Creation**: Defines a binary target (`1` for price increase, `0` for decrease) based on the next day's closing price shift].
4. **Model Training & Evaluation**: 
   - Splits data into chronological training and testing sets (20% test size).
   - Trains and evaluates **Logistic Regression** and **Random Forest Classifier** models.
   - Evaluates performance using Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.
5. **Backtesting**: Computes a portfolio growth simulation starting with an initial cash value of ₹1,00,000, comparing the ML strategies against a standard Buy-and-Hold benchmark.



##  Tech Stack & Libraries:-

* **Python**
* **Pandas** & **NumPy** (Data manipulation and feature engineering)
* **yfinance** (Financial data retrieval)
* **Scikit-Learn** (Preprocessing, ML models, and metrics)
* **SciPy** (Statistical functions)





* **Logistic Regression**: Achieved an accuracy of ~47.4% with a high recall of ~93.1% for the positive class.
* **Random Forest Classifier**: Achieved an accuracy of ~52.4% with a balanced precision/recall trade-off after applying class weighting.
* **Backtesting**: Portfolio value over the test period was tracked and compared daily against actual market performance.

---

