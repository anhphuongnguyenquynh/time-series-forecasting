# ⏳ Time Series Forecasting with Classical and Deep Learning Models

Welcome to the **Time Series Forecasting** repository! This project explores a wide range of forecasting techniques, from classical statistical models to cutting-edge deep learning architectures. It aims to provide a unified, modular framework to experiment, compare, and evaluate multiple forecasting methods on time series data.

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

We provide toy and real-world datasets such as:

* 
* Source

---

## 📈 Evaluation Metrics

Models are evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Percentage Error (MAPE)
* Symmetric Mean Absolute Percentage Error (sMAPE)

---

## 📚 References

* [DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks (Amazon)](https://arxiv.org/abs/1704.04110)
* [Attention is All You Need](https://arxiv.org/abs/1706.03762)
* [Hands-On Time Series Analysis with Python](https://www.oreilly.com/library/view/hands-on-time-series/9781788624561/)

---

## 🧑‍💻 Maintainer

Created and maintained by [Your Name](https://github.com/yourusername). Feel free to reach out via Issues or Discussions for questions or suggestions!

---

## ⭐️ If you find this useful...

Please ⭐️ the repo and share it with your peers to support the project!

---

Let me know if you want to include diagrams, specific datasets, benchmarking results, or HuggingFace/TensorFlow/Keras versions too!
