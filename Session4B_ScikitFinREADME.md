# Scikit-Learn Stock Market Analysis Demos

This directory contains a series of Python scripts that demonstrate machine learning concepts applied to financial data analysis and algorithmic trading. The files are organized in a progressive learning sequence, starting with basic scikit-learn concepts and advancing to sophisticated trading strategies.

## Prerequisites

### Required Libraries
```bash
pip install numpy scikit-learn yfinance matplotlib pandas
```

### Python Packages Used
- `numpy`: Numerical computing
- `scikit-learn`: Machine learning algorithms
- `yfinance`: Yahoo Finance data retrieval
- `matplotlib`: Data visualization
- `pandas`: Data manipulation

## File Overview

### 1. scikit-learn-1-basicFunctions.py
**Purpose**: Introduction to fundamental scikit-learn preprocessing and regression concepts.

**Key Concepts**:
- Data standardization and normalization
- Handling missing values with imputation
- Train-test data splitting
- Basic linear regression implementation

**Features**:
- StandardScaler for data standardization
- SimpleImputer for missing value handling
- Linear regression with synthetic data
- Visualization of regression results

### 2. scikit-learn-2-regressing-aapl.py
**Purpose**: Apply linear regression to real stock market data (Apple Inc.).

**Key Concepts**:
- Real financial data retrieval using Yahoo Finance
- Time series prediction with linear regression
- Model evaluation with actual vs predicted values

**Parameters**:
- Stock: AAPL (Apple Inc.)
- Training period: May 2023
- Prediction target: June 1, 2023

**Output**: Comparison of predicted vs actual stock prices with visualization.

### 3. scikit-learn-3-sliding.py
**Purpose**: Introduce sliding window technique for time series prediction.

**Key Concepts**:
- Sliding window approach for sequential data
- Creating training sequences from time series
- Understanding feature engineering for temporal data

**Parameters**:
- Window size: 5 data points
- Sample data: Synthetic price array
- Demonstrates coefficient interpretation

### 4. scikit-learn-4-sliding-stock.py
**Purpose**: Apply sliding window technique to real stock data with advanced preprocessing.

**Key Concepts**:
- MinMaxScaler for data normalization
- Multi-day ahead predictions
- Real-time model retraining
- Advanced data reshaping techniques

**Parameters**:
- Stock: AAPL
- Data period: June 2023 - June 2024
- Prediction window: 30 days
- Training window: 30 previous days

### 5. scikit-learn-5-sliding-stock-withTrading.py
**Purpose**: Implement basic trading strategy based on price predictions.

**Key Concepts**:
- Trading signal generation from predictions
- Account value tracking
- Performance metrics (accuracy, profitability)
- Risk-free trading simulation

**Parameters**:
- Stock: SPY (S&P 500 ETF)
- Window size: 10 days
- Training range: 100 days
- Period: 2019-2021
- Trading logic: Long if prediction > current price, Short otherwise

**Trading Strategy**:
- Buy 100 shares if predicting price increase
- Sell 100 shares if predicting price decrease
- Track cumulative returns

### 6. scikit-learn-6-sliding-stock-withTrading-and-Volume.py
**Purpose**: Advanced trading strategy incorporating volume data and realistic position management.

**Key Concepts**:
- Multi-feature model (price + volume)
- Random Forest regression for improved predictions
- Realistic position sizing and management
- Advanced trading metrics

**Parameters**:
- Stock: SPY
- Features: Close prices + Volume
- Model: RandomForestRegressor
- Window size: 10 days
- Training range: 100 days
- Period: 2023-2024

**Advanced Features**:
- Position reversal logic (long to short transitions)
- Share quantity tracking
- Comprehensive performance metrics
- Account value visualization

## Key Learning Progression

1. **Basic ML Concepts** → **Real Data Application** → **Time Series Techniques**
2. **Simple Prediction** → **Trading Integration** → **Multi-Feature Models**
3. **Linear Models** → **Ensemble Methods** (Random Forest)
4. **Static Analysis** → **Dynamic Trading Strategies**

## Usage Instructions

### Running Individual Scripts
```bash
python scikit-learn-1-basicFunctions.py
python scikit-learn-2-regressing-aapl.py
# ... etc
```

### Customizing Parameters
Each script contains configurable parameters at the top:
- `ticker_symbol`: Change stock symbol
- `start_date`/`end_date`: Modify data range
- `window_size`: Adjust prediction window
- `training_range`: Change training data size

### Understanding Output

**Trading Scripts (5 & 6) Output Metrics**:
- **Account Value**: Current portfolio value
- **Accuracy**: Percentage of correct directional predictions
- **Long/Short Ratio**: Distribution of trading positions
- **Final Performance**: Total return and prediction accuracy

## Important Notes

### Limitations
- Historical data only - no real-time trading
- No transaction costs or slippage considered
- Simplified position management
- Past performance doesn't guarantee future results

### Educational Purpose
These scripts are designed for learning machine learning and algorithmic trading concepts. They should not be used for actual trading without significant additional development and risk management.

### Data Dependencies
- Requires internet connection for Yahoo Finance data
- Market data availability may vary by date range
- Some historical data might be adjusted for splits/dividends

## Troubleshooting

### Common Issues
1. **Network errors**: Check internet connection for yfinance data retrieval
2. **Date range errors**: Ensure start_date < end_date and dates are valid trading days
3. **Insufficient data**: Increase date range if getting empty datasets
4. **Package versions**: Update packages if encountering compatibility issues

### Performance Considerations
- Larger training ranges increase computation time
- Random Forest models are slower than Linear Regression
- Consider reducing data range for faster execution during development
