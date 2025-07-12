# Apple Stock Price Trend Prediction using LSTM

This project uses a Long Short-Term Memory (LSTM) neural network to predict the **closing price** of Apple Inc. (AAPL) stock. The model is trained on historical stock data and achieves high **trend prediction accuracy**.

---

## Project Overview

- **Goal**: Predict the future trend of Apple's stock prices using deep learning (LSTM).
- **Method**: Create input sequences using a sliding window technique and train an LSTM-based model.
- **Key Features**:
  - Time series sequence modeling
  - Stacked LSTM architecture
  - High trend prediction accuracy
  - Real-world application to financial forecasting

---

##  Performance

| Metric                    | Value     |
|--------------------------|-----------|
| **Trend Prediction Accuracy** | **98.62%** |
| **Trend Accuracy (Upward)**   | 98.65%    |
| **Trend Accuracy (Downward)** | 98.58%    |

> *Trend accuracy* indicates how well the model predicts whether the price will go up or down, rather than the exact price value.

---

##  Model Architecture

```python
model = Sequential([
    LSTM(128, activation="tanh", return_sequences=True, input_shape=(window_size, 1)),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(1)
])
