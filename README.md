# Apple Stock Price Prediction using Stacked LSTM

This project implements a **Stacked Long Short-Term Memory (LSTM)** neural network to predict Apple stock closing prices using historical time-series data. The model learns temporal patterns in stock price movements and forecasts future values based on past sequences.

The project was built as a **learning-focused deep learning experiment** to understand time-series forecasting using LSTM networks.

---

# Project Overview

Stock market prices are inherently **sequential and influenced by historical patterns**. Traditional machine learning models struggle to capture long-term dependencies in such data.

To address this, this project uses **Recurrent Neural Networks (RNNs)** — specifically **Stacked LSTM layers** — which are designed to remember patterns over long sequences.

The model is trained on **historical Apple stock price data** and learns to predict future closing prices based on the **previous 100 time steps**.

---

# Dataset

The dataset used for training and evaluation comes from Kaggle.

Dataset Link:  
https://www.kaggle.com/datasets/varpit94/apple-stock-data-updated-till-22jun2021

## Dataset Description

The dataset contains historical Apple stock market data including:

- Date
- Open
- High
- Low
- Close
- Volume

For this project, the **Close price** was used as the **target variable for prediction**.

---

# Problem Statement

The goal of this project is to:

- Learn temporal patterns in historical Apple stock prices
- Train a deep learning model capable of forecasting future values
- Understand how LSTM networks work for time-series prediction

The model predicts **future closing prices using the previous 100 days of stock data**.

---

# Exploratory Data Analysis

Initial data analysis included:

- Checking for missing values
- Identifying duplicate records
- Observing general trends in the stock price
- Visualizing historical closing price movements

EDA helps understand the **structure and stability of the time-series data** before applying deep learning models.

---

# Data Preprocessing

Several preprocessing steps were applied before feeding the data into the neural network.

## 1. Feature Selection

Only the **Close price** was used for training.

## 2. Data Scaling

The data was normalized using **MinMaxScaler** to bring values into the range:

```
[0,1]
```

Scaling is important because neural networks perform better when features are normalized.

## 3. Train-Test Split

The dataset was divided into:

- **Training Data:** 65%
- **Testing Data:** 35%

The split was performed **without shuffling** to preserve time order.

## 4. Time Series Windowing

A **sliding window approach** was used to convert the sequence into supervised learning format.

Time Step:

```
100
```

This means:

```
Previous 100 days → Predict next day price
```

---

# Model Architecture

The model uses a **Stacked LSTM architecture**, which allows the network to capture deeper temporal relationships.

## Architecture

```
Input Layer
   ↓
LSTM Layer (50 units, return_sequences=True)
   ↓
LSTM Layer (50 units, return_sequences=True)
   ↓
LSTM Layer (50 units)
   ↓
Dense Layer (1 unit)
```

## Model Configuration

**Loss Function**

```
Mean Squared Error (MSE)
```

**Optimizer**

```
Adam
```

Stacking multiple LSTM layers allows the model to **learn higher-level temporal patterns in stock movements**.

---

# Model Training

The model was trained on the processed dataset using **sequential time-series input**.

Key aspects of training include:

- Feeding sequences of **100 previous stock prices**
- Learning to **predict the next price**
- Minimizing prediction error using **backpropagation through time**

After training, predictions were generated on the **test dataset**.

---

# Prediction and Evaluation

The model produces predicted stock prices which are then:

- Rescaled back to **original price values**
- Compared with **actual stock prices**
- **Visualized** to evaluate prediction accuracy

This helps observe how well the model captures the **trend and direction of stock price movement**.

---

# Visualization

The project visualizes:

- Historical stock prices
- Model predictions
- Actual vs predicted price comparisons

Visualization helps understand whether the model captures **real market trends**.

---

# Technologies Used

## Programming Language

- Python

## Libraries

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow
- Keras

## Model Type

- Recurrent Neural Network (RNN)
- Stacked LSTM

---

# Learning Outcomes

This project helped build understanding of:

- Time-series forecasting
- Sequence modeling
- LSTM networks
- Data preprocessing for neural networks
- Sliding window sequence generation
- Deep learning model training and evaluation

---

# Future Improvements

Possible extensions for this project include:

- Using multiple features (Open, High, Low, Volume)
- Hyperparameter tuning
- Increasing model depth
- Adding **Bidirectional LSTM**
- Implementing **GRU networks**
- Testing **Transformer-based time-series models**
- Building a **real-time prediction dashboard**
