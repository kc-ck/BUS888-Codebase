# Neural Nets for Quant Trading: Student Guide

This workshop uses four Jupyter notebooks to show how a **feed-forward artificial neural network (ANN)** can be applied to lagged stock prices, evaluated out of sample, and connected to a deliberately simplified trading rule.

> **Important:** These notebooks are teaching demonstrations, not production trading systems and not evidence that an ANN can generate abnormal returns. The most important learning goal is to inspect the entire pipeline—from information available at prediction time to information used by the backtest.

## Recommended path through the files

| Status | Notebook | How to use it |
|---|---|---|
| **Core** | `1_SimpleANNPrediction.ipynb` | Run slowly and use it to understand every stage of the prediction pipeline. |
| **Reference / compare** | `2_SimpleANNPrediction_w_functions.ipynb` | Read after Notebook 1 to see the same workflow organized into reusable functions. It does not introduce a fundamentally new forecasting method. |
| **Optional code audit** | `3_ANN_Predict_Backtest.ipynb` | Use to identify why a plausible-looking backtest can still have a signal-timing problem. Do not interpret its account-value curve as a valid strategy result. |
| **Core extension** | `4_ANN_RollingWindowPrediction.ipynb` | Use to discuss next-day signal alignment, position rules, and the difference between a moving input window and a true walk-forward re-fit. |


## Learning objectives

By the end of the workshop, you should be able to:

1. Convert a price series into supervised-learning rows using a lookback window.
2. Explain why financial time series must be split chronologically rather than randomly.
3. Fit preprocessing transformations on training data only.
4. Build and train a dense ANN for one-step-ahead price prediction.
5. Distinguish predictive fit from a tradeable signal.
6. Detect data leakage, target misalignment, and unrealistic backtest assumptions.
7. Compare an ANN with a simple benchmark such as “tomorrow equals today.”

## Setup

### Option A: Google Colab

Upload the notebooks to Colab. Most libraries are already available; install `yfinance` if needed:

```python
!pip install -q yfinance
```

### Option B: Local Jupyter

```bash
pip install yfinance numpy pandas scikit-learn matplotlib plotly keras tensorflow
```

Then start Jupyter Notebook or JupyterLab and open the files in numerical order.

### Before running a notebook

- Restart the kernel/runtime, then run cells from top to bottom.
- Confirm that the ticker and date range return data.
- Keep the chronological order of observations.
- Change only one parameter at a time when experimenting.
- Expect results to vary between runs because the notebooks do not set random seeds.
- An internet connection is required when `yfinance` downloads data.

The included examples use `GOOG` over approximately one trading year. The end date supplied to `yfinance` is normally treated as an exclusive boundary, so the final downloaded observation may be earlier than the date written in the constants.

## The forecasting problem used in all four notebooks

For a lookback of five days, one row is constructed as:

```text
X_t = [Price_(t-5), Price_(t-4), Price_(t-3), Price_(t-2), Price_(t-1)]
y_t = Price_t
```

The ANN learns a nonlinear function that maps the lagged prices in `X_t` to the next price `y_t`.

These are **dense feed-forward networks**. They are not LSTMs, recurrent neural networks, reinforcement-learning agents, or transformer models. Time is represented only through the ordered lag columns supplied as features.

## At-a-glance comparison

| Feature | Notebook 1 | Notebook 2 | Notebook 3 | Notebook 4 |
|---|---|---|---|---|
| Primary purpose | Make every pipeline step visible | Refactor the basic pipeline into functions | Add a first portfolio simulation | Add a better-aligned next-day rule and position markers |
| Code style | Expanded, cell-by-cell | Modular functions plus a short main block | Functions for data, model, strategy, and plot | Functions plus staged main execution |
| Lookback | 5 days | 5 days | 10 days | 60 days |
| Chronological split | 70% train / 30% test | 80% train / 20% test | 80% train / 20% test | 80% train / 20% test |
| Windows created | Separately inside train and test slices | Separately inside train and test slices | Before splitting the supervised rows | Before splitting the supervised rows |
| Consequence at boundary | First test lookback days are not forecast | First test lookback days are not forecast | First test row can use prior historical observations | First test row can use prior historical observations |
| Scaling | One price scaler, fit on train and applied to test | One price scaler, fit on train and applied to test | Separate `X` and `y` scalers, fit on train | Separate `X` and `y` scalers, fit on train |
| Hidden layers | 64 → 32 | 64 → 64 → 32 | 64 → 32 | 128 → 64 → 32 |
| Approx. trainable parameters | 2,497 | 6,657 | 2,817 | 18,177 |
| Training settings | 10 epochs; batch 1 | 10 epochs; batch 1 | 10 epochs; batch 1 | 100 epochs; batch 32 |
| Saved-run training rows | About 169 | About 194 | About 191 | About 152 |
| Prediction metric | Train and test MSE | Test MSE | None printed | None printed |
| Backtest | No | No | Yes; inventory can accumulate one share at a time | Yes; maximum one share and final liquidation |
| Backtest signal | Not applicable | Not applicable | Compares `prediction_i` with `actual_i` | Compares `prediction_(i+1)` with `actual_i` |
| Best teaching use | Data preparation and ANN basics | Software design and reuse | Backtest code review | Forecast-to-strategy mapping and limitations |
| Main caution | Date labels in the output table are not aligned to the target rows | Larger model is not automatically a better model | Signal is based on same-day prediction error, not a next-day expected return | Not a true rolling re-fit; highly over-parameterized for the included sample |

The parameter counts above include weights and biases. A large number of parameters relative to observations makes overfitting especially likely; a more complex network should not be assumed to be an improvement.

Two implementation details also differ across the files:

- Notebooks 1–3 use `range(len(data) - lookback - 1)` when constructing sequences, which omits one additional valid target at the end. Notebook 4 uses all available windows.
- In Notebooks 3 and 4, `MinMaxScaler` is fitted to the two-dimensional `X_train` matrix, so each lag column receives its own training minimum and maximum. Notebooks 1 and 2 instead apply one scaler to the underlying one-dimensional price series. Both avoid test-set fitting, but they create different feature representations.

---

## Notebook 1 — transparent ANN pipeline

### What it adds

`1_SimpleANNPrediction.ipynb` exposes each step:

1. Download adjusted stock data.
2. Extract and reshape closing prices.
3. Split the raw time series chronologically.
4. Fit `MinMaxScaler` on training prices only.
5. Construct lagged training and test rows.
6. Build a `64 → 32 → 1` dense ANN.
7. Generate train and test predictions.
8. Inverse-transform predictions to price units.
9. Calculate train and test mean squared error.
10. Plot actual prices, predicted prices, and errors.

### What to pay attention to

- **Fit on train, transform both:** The scaler is fitted only on training data and then used to transform both training and test data. Do not fit a separate scaler on the test set.
- **Test values can exceed 1:** This is not an error. If test prices exceed the maximum training price, their transformed values can be greater than 1.
- **MSE is in squared price units:** Convert it to RMSE for an error measure in approximate price units.
- **The test window starts inside the test slice:** The notebook does not use the last five training observations to predict the first available test target, so several boundary observations are skipped.
- **Current date table caveat:** The table’s `Date` column is not aligned correctly with the corresponding training and test targets. Use the plots and arrays for the workshop; do not interpret those table dates as exact forecast dates without fixing the index offsets.
- **Several plots repeat the same information:** Once you understand the first combined plot, the later Matplotlib and Plotly versions are optional.

### Suggested experiment

Run the notebook once unchanged. Then change only one of the following and rerun from the top:

- `STOCK_SYMBOL`
- `LOOKBACK_PERIOD`
- `EPOCHS`
- Number of hidden neurons

Record both train and test error. A lower training error alone is not evidence of a better model.

---

## Notebook 2 — the same workflow as Notebook 1 but organized into functions

### What changes from Notebook 1

`2_SimpleANNPrediction_w_functions.ipynb` packages the pipeline into reusable functions:

- `download_stock_data()`
- `prepare_dataset()`
- `initialize_model()`
- `plot_combined_actual_vs_predicted()`

It also changes the split from 70/30 to 80/20 and uses a larger `64 → 64 → 32 → 1` network with ReLU and tanh activations.

### What does **not** change

- It still predicts the next price from lagged prices.
- It still uses a fixed chronological holdout set.
- It still fits the scaler on training prices only.
- It still builds train and test windows separately, so the first few test-period targets are skipped.
- A larger network does not establish better out-of-sample performance.

### Why this file exists

The main lesson is **software organization**, not a new financial model. Compare the functions with the expanded cells in Notebook 1 rather than spending class time retraining both notebooks end to end.

The Colab badge embedded in the notebook points to a different course repository name. Use the notebook supplied with this workshop unless your instructor provides an updated repository link.

---

## Notebook 3 — first backtest and a signal-alignment audit

### What it adds

`3_ANN_Predict_Backtest.ipynb`:

- Creates supervised rows before the train/test split.
- Uses separate feature and target scalers fitted on training data.
- Trains a `64 → 32 → 1` ANN with a 10-day lookback.
- Starts with `$10,000` cash.
- Buys or sells one share per signal and tracks marked-to-market account value.

Creating the supervised rows before the chronological split lets the first test target use earlier prices in its lookback. That is appropriate because those earlier prices would already have been known.

### Critical issue to find

For row `i`, the model predicts `y_i` from the preceding ten closes. The strategy then compares:

```text
predicted y_i  versus  actual y_i
```

That comparison is the sign of the model’s **same-day forecast error**. It is not the comparison required for a next-day trading decision, which would be closer to:

```text
predicted Price_(i+1)  versus  known Price_i
```

The resulting account-value curve should therefore be treated as a **code-review example**, not as evidence of strategy performance.

Additional simplifications:

- The rule can accumulate multiple shares over consecutive buy signals.
- Remaining inventory is not explicitly liquidated at the final test price.
- No benchmark, return, drawdown, Sharpe ratio, fees, spread, slippage, or execution delay is included.

### Code-audit question

Explain precisely what information is available when each prediction is made, what price is used to generate the signal, and what price is used for execution. A backtest is invalid when those timestamps do not line up.

---

## Notebook 4 — moving input windows and a better next-day mapping

### What it adds

`4_ANN_RollingWindowPrediction.ipynb`:

- Uses a 60-day lookback.
- Trains a larger `128 → 64 → 32 → 1` ANN.
- Generates one prediction for each test input window.
- Compares the next row’s prediction with the current realized price.
- Restricts the strategy to at most one share.
- Records buy and sell markers.
- Liquidates a remaining share at the end.

The strategy uses `prediction_(i+1)` against `actual_i`. Because the next input window includes the current price and only earlier prices, this is a better mapping from known information to a next-day forecast than the rule in Notebook 3.

### What “rolling” means here

The input window moves forward through the test set, but the model is trained once and then held fixed. This is **not** a full rolling-window or expanding-window backtest in which the ANN is repeatedly re-fitted as new observations arrive.

The function includes a `steps_ahead` argument, but the workshop path uses `steps_ahead=1`. Treat it as a looped one-step forecast, not as a validated multi-horizon forecasting system.

### Major cautions

- The included one-year sample leaves only about 152 training rows after a 60-day lookback, while the model has roughly 18,177 trainable parameters.
- One hundred epochs on such a small sample can produce severe overfitting.
- The signal uses the current close and assumes execution at that same close. A realistic backtest needs a one-bar execution lag or next-open execution.
- The strategy holds at most one share, so most of the `$10,000` remains uninvested. Its account curve is not directly comparable with an all-in buy-and-hold strategy.
- It still excludes transaction costs, spread, slippage, liquidity limits, dividends, taxes, and short selling.

---

## Prediction accuracy is not trading performance

A price-level model can appear accurate simply because stock prices are persistent: tomorrow’s price is often close to today’s price. Always compare the ANN with a **naive persistence benchmark**:

```text
Predicted Price_t = Price_(t-1)
```

At minimum, compare:

- ANN test RMSE versus naive test RMSE
- Directional accuracy versus a naive direction or majority-class benchmark
- Strategy return versus cash and buy-and-hold
- Maximum drawdown
- Turnover and number of trades
- Results before and after plausible costs

A model can have lower RMSE but still produce poor trading signals. Conversely, a useful directional signal does not need to predict the exact price level well.

## Checklist for a defensible time-series experiment

Before believing any result, verify:

1. **Target:** What exactly is predicted—price, return, excess return, or direction?
2. **Timing:** Are all features known before the prediction and trade?
3. **Split:** Is the test period strictly later than the training period?
4. **Scaling:** Were scalers fitted using training data only?
5. **Baseline:** Does the ANN beat a simple persistence or linear model?
6. **Validation:** Were hyperparameters selected without looking at the final test period?
7. **Execution:** Is there a realistic delay between signal construction and trade execution?
8. **Costs:** Are spread, commissions, slippage, and turnover considered?
9. **Benchmark:** Is performance compared with cash and buy-and-hold on the same dates?
10. **Stability:** Does the result survive different assets, dates, and random seeds?

## questions to ask:

1. Why should time-series data not be randomly shuffled before a final holdout test?
2. Why is fitting the scaler on the entire dataset a form of leakage?
3. Why can a properly transformed test value be greater than 1?
4. What does one row of `X_train` mean economically?
5. Does a lower MSE imply higher trading profits?
6. What is the information timestamp for a closing-price forecast?
7. Which notebook has the most defensible prediction-to-signal alignment?
8. Why is Notebook 4 not a true rolling re-training procedure?
9. What simple benchmark should every price forecast beat?
10. How would transaction costs change a high-turnover rule?

## Exercises for home

### Core exercise 1 — explain the data matrix

Using Notebook 1, select one row of `X_train` and its matching `y_train`. Write down the five input prices and the target price in plain language.

### Core exercise 2 — benchmark the ANN

Add a persistence forecast using the final observed price in each lookback window. Compare its test RMSE with the ANN’s test RMSE.

### Core exercise 3 — change one design choice

Change one parameter only—ticker, lookback, number of neurons, or epochs. Explain why the result changed and whether the change improved out-of-sample performance.

### Core exercise 4 — audit the backtest

For Notebooks 3 and 4, draw a timeline showing:

```text
last feature observation → prediction time → signal time → execution time → realized target
```

Identify any same-bar execution assumption.

### Advanced extension

Redesign the target as next-day return or direction, add a validation period, compare against logistic/linear baselines, and implement an expanding-window or rolling-window re-fit with transaction costs.

## Troubleshooting

### No data or an empty dataframe

- Check the ticker symbol and date range.
- Confirm internet access.
- Try a wider date range.
- Explicitly set `auto_adjust=True` in `yf.download()` for consistent adjusted-price behavior across the notebooks.

### Keras warning about `input_dim`

The notebooks may display a warning recommending an explicit `Input(shape=(...))` layer. The current code can still run, but the recommended modern form is:

```python
from keras import Input

model = Sequential([
    Input(shape=(LOOKBACK_PERIOD,)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1),
])
```

### Test values exceed the scaler’s 0–1 range

This can happen when test prices lie outside the minimum or maximum observed in training. It is expected when the scaler is correctly fitted only on training data.

### Results change after rerunning

Neural-network weights are initialized randomly. For a controlled comparison, set seeds for NumPy and the neural-network backend and keep all other settings fixed.

### Model looks excellent on training data but weak on test data

This is overfitting. Try a smaller network, fewer epochs, more historical data, early stopping, regularization, and a separate validation period.

## Educational disclaimer

These notebooks omit many elements required for a credible quantitative trading study. They are intended to teach supervised-learning and backtesting concepts, including how easy it is to produce an attractive but invalid result. They should not be used to make investment decisions.

---

*Neural Nets for Quant Trading workshop*  
*Last updated: July 22, 2026*
