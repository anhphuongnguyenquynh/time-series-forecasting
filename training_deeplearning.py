#Import libraries
import torch.nn as nn
import matplotlib.pyplot as plt 
import copy 
import random
import sys
import numpy as np 
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler 
from models.rnn import RNN

#Load dataset
olist_orders = pd.read_csv("data/olist_dataset/olist_orders_dataset.csv")

### Extract monthly sales
# Convert order purchase timestamp to_datetime
olist_orders['purchase_datetime'] = pd.to_datetime(olist_orders['order_purchase_timestamp'])
# Get date only
olist_orders['purchase_date'] = olist_orders['purchase_datetime'].dt.date
# Get month only
olist_orders['purchase_month'] = olist_orders['purchase_datetime'].dt.strftime('%Y-%m')

#Group by month, number of orders
orders_bymonth = olist_orders.groupby('purchase_month')['order_id'].nunique().reset_index()
orders_bymonth.columns = ['month', 'no_orders']
#Sort ascending month
orders_bymonth = orders_bymonth.sort_values('month', ascending = False)

#Group by date, number of orders
orders_bydate = olist_orders.groupby('purchase_date')['order_id'].nunique().reset_index()
orders_bydate.columns = ['date', 'no_orders']
orders_bydate = orders_bydate.sort_values('date', ascending = False)

# Remove outliers
orders_bydate_filtered = orders_bydate.loc[orders_bydate['no_orders'] < 400]

#Prepare dataset for training
data_rnn = orders_bydate_filtered.copy()
data_rnn.sort_values('date', inplace = True)
#Normalize time series in range [0,1] 
scaler = MinMaxScaler()
#df_scaled = scaler.fit_transform(orders_bydate_rnn)
data_rnn['scaled_orders'] = scaler.fit_transform(data_rnn[['no_orders']])


def create_sequences(data, window_size):
    sequences = []
    labels = []
    for i in range(len(data) - window_size):
        seq = data[i: i + window_size]
        label = data[i+ window_size]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

window_size = 10
data = data_rnn['scaled_orders'].values
X, y = create_sequences(data, window_size)
print(X.shape)
print(y.shape)

# Reshape X to (samples, seq_len, 1) for RNN input
X = X.reshape(X.shape[0], X.shape[1], 1)
y = y.reshape(-1, 1)

#Create a custom dataset for class for Pytorch DataLoader

# Split the data into training and test sets
train_size = int(0.7 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
print(X_train.shape)
print(y_train.shape)

from torch.utils.data import DataLoader, Dataset
class RNNDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype = torch.float32)
        self.y = torch.tensor(y, dtype = torch.float32)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

#Deep Learning Model
rnn_model = RNN(hidden_size = 32)

#Hyperparameters
hidden_size = 32
learning_rate = 0.02
training_epochs = 500
batch_size = 64

#Data Loaders
train_dataset = RNNDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_dataset = RNNDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

#Initialize the model, loss function and optimizer
model = RNN(hidden_size=hidden_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#Training
##Training loop
num_epochs = 100
for epoch in range(num_epochs):
    ###TRAINING

    #Put model in training mode
    model.train()

    train_loss = 0
    #Loop in batch
    for X_batch, y_batch in train_loader:
        #1. Forward pass on train data using the forward() method inside
        y_pred = model(X_batch)

        #2. Calculate the loss (how different are our models predictions to the ground truth)
        loss = criterion(y_pred, y_batch)

        #3. Zero grad of the optimizer
        optimizer.zero_grad()

        #4. Loss backwards
        loss.backward()
        
        #5. Progress the optimizer
        optimizer.step()

        #6. Accumulate loss
        train_loss += loss.item()

    ###TESTING
    model.eval()

    test_loss = 0

    with torch.inference_mode():
        #Create loop in batch
        for X_batch, y_batch in test_loader:
            #1. Forward pass on the test data
            y_pred = model(X_batch)

            #2. Calculate loss on test data
            loss = criterion(y_pred, y_batch)

            #3. Accumulate loss
            test_loss += loss.item()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

#Predict on test set
model.eval()
with torch.no_grad():
    test_inputs = torch.tensor(X_test, dtype=torch.float32)
    predictions = model(test_inputs).numpy()

# Inverse scale predictions and actuals
y_test_inv = scaler.inverse_transform(y_test)
predictions_inv = scaler.inverse_transform(predictions)

#Plot
plt.figure(figsize=(10,5))
plt.plot(y_test_inv, label='Actual')
plt.plot(predictions_inv, label='Predicted')
plt.legend()
plt.title("RNN Time Series Forecast")
plt.show()

