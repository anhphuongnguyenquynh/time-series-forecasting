# ⏳ Time Series Forecasting with Classical and Deep Learning Models

This project **Time Series Forecasting** explores a wide range of forecasting techniques, from classical statistical models to cutting-edge deep learning architectures. It aims to provide a unified, modular framework to experiment, compare, and evaluate multiple forecasting methods on time series data.

---
## 📈 Project Overview

Time series forecasting is essential in many domains such as finance, energy, retail, and climate science. This repository includes:

* **Deep Learning models**:

  * RNN (Recurrent Neural Network)
  * LSTM (Long Short-Term Memory)
  * GRU (Gated Recurrent Unit)
  * Encoder-Decoder architectures
  * Attention-based models

* **Evaluation metrics**: RMSE, MAE, MAPE, sMAPE
* **Visualization tools**: Interactive plots for predictions vs actuals

---

## 🛠️ Features

* Unified pipeline for preprocessing, training, and evaluation
* Support for multivariate and univariate time series
* Train/test split with rolling window or walk-forward validation
* GPU support for deep learning models
* Easily extendable to new models or datasets

---

## 📂 Repository Structure

```
time-series-forecasting/
│
├── data/                  # Sample datasets
├── notebooks/             # Jupyter notebooks for exploration and experimentation
├── models/                # Model definitions (ARIMA, LSTM, GRU, etc.)
├── utils/                 # Utility functions (preprocessing, metrics, plotting)
├── experiments/           # Scripts for training and testing models
├── requirements.txt       # Required Python packages
└── README.md              # Project overview and documentation
```

---

## 🧠 Deep Learning Architectures

| Model           | Description                                           |
| --------------- | ----------------------------------------------------- |
| RNN             | Basic recurrent network, captures short-term patterns |
| LSTM            | Handles long-term dependencies better than RNN        |
| GRU             | A lighter, faster alternative to LSTM                 |
| Encoder-Decoder | Seq2Seq model useful for multi-step forecasting       |
| Attention       | Focuses on important parts of input sequence          |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/anhphuongnguyenquynh/time-series-forecasting.git
cd time-series-forecasting
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run an Example

```bash
python experiments/train_lstm.py
```

Or explore with notebooks:

```bash
jupyter notebook notebooks/
```

---

## 📊 Sample Dataset

In this project, I use real-world datasets such as:

* Source: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce?select=olist_orders_dataset.csv)

---

## 📈 Evaluation Metrics

Models are evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Percentage Error (MAPE)
* Symmetric Mean Absolute Percentage Error (sMAPE)

---

## 📚 References

* [Dive into Deep Learning book](https://d2l.ai/)
* [Time-series Forecasting using Deep Learning](https://github.com/bpbpublications/Time-Series-Forecasting-using-Deep-Learning)

---

## ⭐️ If you find this useful...

Please ⭐️ the repo and share it with your peers to support the project!

---

Let me know if you want to include diagrams, specific datasets, benchmarking results, or HuggingFace/TensorFlow/Keras versions too!
