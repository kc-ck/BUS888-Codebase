# ML for Finance Workshop: Stock Price Prediction with Artificial Neural Networks

## Overview

This workshop introduces students to applying Artificial Neural Networks (ANNs) for stock price prediction using Python. The lab progresses from basic prediction to advanced trading strategies and backtesting.

## Prerequisites

### Required Libraries

See pre-reqs-for-spyder.md

### Python Version
- Python 3.7 or higher recommended

## Workshop Files

### 📄 File 1: `/demos/NeuralNets/simpleannprediction.py`
**Basic ANN for Stock Price Prediction**

#### Key Concepts:
- Data fetching using `yfinance`
- Time series data preparation with lookback windows
- Data scaling with MinMaxScaler
- Building a simple feedforward neural network
- Visualization of predictions vs actual prices

#### Features:
- Downloads Google (GOOG) stock data
- 70/30 train-test split
- 3-layer neural network (64-32-1 neurons)
- MSE evaluation
- Multiple visualization options (matplotlib & plotly)

#### Parameters to Experiment:
- `STOCK_SYMBOL`: Try different stocks (AAPL, MSFT, TSLA)
- `LOOKBACK_PERIOD`: Window size for predictions (default: 5)
- `EPOCHS`: Training iterations (default: 10)

---

### 📄 File 2: `/demos/NeuralNets/simpleannprediction_w_functions.py`
**Modularized ANN Implementation**

#### Key Concepts:
- Code organization with functions
- Reusable components for ML pipeline
- Improved model architecture

#### Improvements over File 1:
- Modular design with separate functions
- Enhanced neural network (64-64-32-1 architecture)
- Mixed activation functions (ReLU and tanh)
- 80/20 train-test split
- Better code maintainability

#### Main Functions:
- `download_stock_data()`: Fetches stock prices
- `prepare_dataset()`: Creates time series sequences
- `initialize_model()`: Builds the neural network
- `plot_combined_actual_vs_predicted()`: Visualizes results

---

### 📄 File 3: `/demos/NeuralNets/ann_predict_backtest.py`
**ANN with Trading Strategy Backtesting**

#### Key Concepts:
- Trading strategy implementation
- Portfolio simulation
- Performance tracking
- Proper data preprocessing to avoid leakage

#### New Features:
- Initial balance simulation ($10,000)
- Buy/sell decision logic based on predictions
- Account value tracking over time
- Dual-axis plotting (prices + portfolio value)

#### Trading Strategy:
- Buy when predicted price > current price
- Sell when predicted price < current price
- Track total portfolio value

#### Important Note:
This file correctly handles data scaling to prevent data leakage by fitting scalers only on training data.

---

### 📄 File 4: `/demos/NeuralNets/ann_rollingwindowprediction.py`
**Advanced Rolling Window Prediction**

#### Key Concepts:
- Rolling window predictions
- Multi-step ahead forecasting
- Advanced trading strategy with visual indicators
- Buy/sell point visualization

#### Advanced Features:
- 60-day lookback window
- Larger network architecture (128-64-32-1)
- More training epochs (100)
- Visual buy/sell signals on charts
- Refined trading logic (single stock position)

#### Trading Improvements:
- Prevents multiple positions
- Clear entry/exit points
- Final position liquidation

---

## Workshop Flow

### Session 1: Basic Prediction (File 1)
1. Understanding time series data structure
2. Creating input sequences from historical data
3. Building and training a simple ANN
4. Evaluating prediction accuracy

### Session 2: Code Organization (File 2)
1. Refactoring code into functions
2. Improving model architecture
3. Experimenting with hyperparameters

### Session 3: Backtesting Strategies (File 3)
1. Implementing trading logic
2. Simulating portfolio performance
3. Understanding data leakage prevention

### Session 4: Advanced Techniques (File 4)
1. Rolling window predictions
2. Complex trading strategies
3. Performance visualization

## Common Parameters Across Files

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `LOOKBACK_PERIOD` | Days of history for prediction | 5-60 |
| `EPOCHS` | Training iterations | 10-100 |
| `BATCH_SIZE` | Samples per gradient update | 1-32 |
| `TRAIN_RATIO` | Training data percentage | 0.7-0.8 |

## Potential Issues & Solutions

### 1. Data Fetching Errors
**Issue**: Network connection or invalid ticker symbols
```python
# Add error handling:
try:
    data = yf.download(STOCK_SYMBOL, start=START_DATE, end=END_DATE)
except Exception as e:
    print(f"Error fetching data: {e}")
```

### 2. Insufficient Data
**Issue**: Not enough historical data for the specified period
- Solution: Adjust `START_DATE` or reduce `LOOKBACK_PERIOD`

### 3. Overfitting
**Issue**: Model performs well on training but poorly on test data
- Solutions:
  - Add dropout layers
  - Reduce model complexity
  - Increase training data
  - Use regularization

### 4. Unrealistic Trading Assumptions
**Issue**: No transaction costs or slippage
- Note: These are simplified educational examples
- Real trading requires more sophisticated strategies

## Hands-On Exercises

### Exercise 1: Stock Comparison
Run File 1 with different stocks and compare prediction accuracy:
- Tech stocks: AAPL, GOOGL, MSFT
- Traditional: JPM, WMT, XOM

### Exercise 2: Hyperparameter Tuning
Experiment with File 2:
- Change `LOOKBACK_PERIOD`: [5, 10, 20, 30]
- Modify network architecture
- Try different optimizers (adam, sgd, rmsprop)

### Exercise 3: Strategy Development
Using File 3, implement:
- Position sizing based on confidence
- Stop-loss mechanisms
- Multiple stock portfolios

### Exercise 4: Advanced Predictions
With File 4, explore:
- Multi-step ahead predictions
- Different technical indicators
- Ensemble methods

## Best Practices

1. **Always split data before scaling** to prevent data leakage
2. **Validate on out-of-sample data** for realistic performance
3. **Consider transaction costs** in real applications
4. **Don't rely solely on MSE** - use multiple evaluation metrics
5. **Understand the limitations** of point predictions in finance

## Extensions for Advanced Students

1. **LSTM Implementation**: Replace Dense layers with LSTM for better sequence modeling
2. **Feature Engineering**: Add technical indicators (RSI, MACD, Bollinger Bands)
3. **Ensemble Methods**: Combine multiple models for robust predictions
4. **Risk Metrics**: Implement Sharpe ratio, maximum drawdown analysis
5. **Real-time Data**: Modify for live trading simulation

## Troubleshooting

### Common Errors:

1. **ImportError**: Install missing packages
   ```bash
   pip install -r requirements.txt
   ```

2. **ValueError in reshape**: Check data dimensions
   ```python
   print(f"Data shape: {data.shape}")
   ```

3. **Memory Issues**: Reduce batch size or data size

4. **NaN in predictions**: Check for missing data
   ```python
   data = data.dropna()
   ```

## Resources

- [Keras Documentation](https://keras.io/)
- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [Time Series Prediction Guide](https://machinelearningmastery.com/time-series-prediction-with-deep-learning-in-python-with-keras/)

## Disclaimer

**Important**: These examples are for educational purposes only. Real financial trading involves significant risk. Always:
- Consult with financial professionals
- Thoroughly backtest strategies
- Understand the risks involved
- Never trade with money you cannot afford to lose

## Workshop Feedback

After completing the workshop, students should be able to:
- ✅ Build ANNs for time series prediction
- ✅ Implement basic trading strategies
- ✅ Evaluate model performance
- ✅ Visualize predictions and portfolio performance
- ✅ Understand the limitations and challenges of algorithmic trading

---

*Workshop developed for ML for Finance course*  
*Last updated: July 2025*