# Foreign Exchange Volatility and Trend Analysis

## Overview
Quantitative analysis of foreign exchange rate data to identify market trends, estimate risk metrics, and visualize price movements using historical daily exchange rates for three major currency pairs.

The repository contains a Jupyter Notebook covering data acquisition, preprocessing, indicators (volatility, SMA, RSI) and visualization.

## Currency Pairs Analyzed
The analysis focuses on the following three major currency pairs:
- **EUR/USD** (Euro / US Dollar)
- **GBP/USD** (British Pound / US Dollar)
- **USD/JPY** (US Dollar / Japanese Yen)

## Methodology

### 1. Data Collection
- **Source:** Historical daily OHLC (Open, High, Low, Close) data is retrieved using the `yfinance` API (Yahoo Finance).
- **Period:** January 1, 2022 to December 31, 2024.
- **Frequency:** Daily observations.

### 2. Data Preprocessing
- **Alignment:** Time series from different pairs are aligned to a single calendar.
- **Missing Values:** Missing data points (e.g., weekends, holidays) are handled using a **sequential 5-day rolling mean** imputation method to preserve trends and ensure continuous data for analysis.

## Quantitative Metrics & Indicators

The project computes several key financial metrics:

### 1. Daily Returns
Calculated as the percentage change in closing prices between consecutive trading days. This is the foundation for volatility estimation.

### 2. Rolling Volatility (Annualized)
Volatility is estimated using the standard deviation of daily returns over rolling windows, annualized by a factor of $\sqrt{252}$ (trading days in a year).
- **30-Day Volatility:** Short-term risk assessment.
- **90-Day Volatility:** Medium-term risk assessment.

### 3. Simple Moving Averages (SMA)
Used to identify trends and smooth out price noise.
- **20-Day SMA:** Short-term trend indicator.
- **50-Day SMA:** Medium-term trend indicator.

### 4. Relative Strength Index (RSI)
A momentum oscillator ranging from 0 to 100, used to identify overbought (>70) or oversold (<30) conditions.
- **Period:** 14 days.

## Project Structure
```
forex-volatility-analysis/
├── fx_volatility_analysis.ipynb   # Main analysis notebook
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## Installation & Setup

1. **Clone the repository (or navigate to the project folder):**
   ```bash
   cd forex-volatility-analysis
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To run the analysis:
1. Ensure your virtual environment is activated.
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open `fx_volatility_analysis.ipynb` and run all cells.

The notebook will download the latest data, perform calculations, and generate visualization dashboards for each currency pair.

## Dependencies
- **pandas:** Data manipulation and time series analysis.
- **numpy:** Numerical computations.
- **matplotlib:** Data visualization.
- **yfinance:** Financial data retrieval.

## Auteur
Dan Allouche
