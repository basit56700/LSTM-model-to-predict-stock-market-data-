#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the necessary libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


# <font color='green'>
# <b>pandas</b> is a library for reading and manipulating data. It provides functions for reading data from various sources (such as CSV files) into a data structure called a DataFrame, which allows you to easily manipulate and analyze the data.
# 
# <b>numpy</b> is a library for numerical operations. It provides functions for performing mathematical operations on arrays and matrices of data, such as calculating mean and standard deviation.
# 
# <b>matplotlib</b> is a library for plotting data. It provides functions for creating a variety of plots, such as line plots, scatter plots, and bar plots.
# 
# <b>sklearn</b> (short for Scikit-Learn) is a library for machine learning in Python. It provides functions for tasks such as data preprocessing, model training and evaluation, and feature selection.
# 
# <b>tensorflow</b> is a library for machine learning and deep learning. It provides functions for building and training neural network models, including LSTM models.
# 
# <b>keras</b> is a high-level library for building and training neural network models. It provides a user-friendly interface for building models using tensorflow or other deep learning libraries.
# </font>

# In[20]:


# Read in the data
df = pd.read_csv('HistoricalPrices.csv')
df = df.drop(' Open', axis=1)
df.head(15)


# In[21]:


# Convert the date column to a datetime type
df['Date'] = pd.to_datetime(df['Date'])
print(df['Date'])


# In[22]:


# Set the date column as the index
df.set_index('Date', inplace=True)
df.head(15)


# In[41]:


# Normalize the data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[[' High', ' Low', ' Close', ' Volume']])
print(df_scaled)


# In[24]:


# Split the data into training and testing sets
train_size = int(len(df_scaled) * 0.8)
test_size = len(df_scaled) - train_size
train, test = df_scaled[0:train_size, :], df_scaled[train_size:len(df_scaled), :]


# In[25]:


# Split the data into input and output sets
X_train, y_train = train[:, 1:], train[:, 0]
X_test, y_test = test[:, 1:], test[:, 0]


# In[26]:


# Reshape the input data for the LSTM model
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])


# In[29]:


# Build the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(1, 3)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')


# 
# # Internal Structure of LSTM Model
# 
# ![Jupyter logo](https://www.researchgate.net/profile/Fan-Liu-58/publication/332790290/figure/fig2/AS:954364717973505@1604549688939/An-illustration-of-the-internal-structure-of-LSTM-cell.png)
# 
# In an LSTM (Long Short-Term Memory) model, the "gates" are used to control the flow of information through the model and allow the model to remember information for longer periods of time. There are three types of gates in an LSTM model:
# 
# Input gate: The input gate determines which pieces of information from the current input should be allowed to pass through to the cell state. It is composed of a sigmoid layer that determines the "openness" of the gate and a dot product operation that scales the input by the sigmoid output.
# 
# Forget gate: The forget gate determines which pieces of information from the previous cell state should be "forgotten" or discarded. It is also composed of a sigmoid layer that determines the "openness" of the gate and a dot product operation that scales the previous cell state by the sigmoid output.
# 
# Output gate: The output gate determines which pieces of information from the current cell state should be allowed to pass through to the output. It is composed of a sigmoid layer that determines the "openness" of the gate and a tanh layer that scales the current cell state. The output is then passed through the sigmoid layer to determine which parts of the output should be allowed to pass through.
# 
# By controlling the flow of information through the gates, the LSTM model is able to remember information for longer periods of time and make more informed predictions.
# 
# In an LSTM (Long Short-Term Memory) model, the "tanh" function is used to scale the current cell state before it is passed through the output gate. The tanh function is a mathematical function that maps any real value to a value between -1 and 1. It is defined as:
# 
# tanh(x) = 2 * sigmoid(2x) - 1
# 
# where "sigmoid" is the sigmoid function, which maps any real value to a value between 0 and 1. It is defined as:
# 
# sigmoid(x) = 1 / (1 + e^-x)
# 
# In an LSTM model, the tanh function is used to squash the current cell state into a range between -1 and 1, which helps to stabilize the gradients during training and prevent the model from "exploding" or "vanishing".
# 
# The sigmoid function, on the other hand, is used to determine the "openness" of the input, forget, and output gates. It is used to create a gate that can take on any value between 0 and 1, with values closer to 0 representing a "closed" gate and values closer to 1 representing an "open" gate. The sigmoid function is often used in neural networks as a way to "squash" output into a range between 0 and 1, which can be useful for binary classification tasks or for creating probabilities.
# 

# In[30]:


# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2, shuffle=False)


# 
# 
# An <b>LSTM model </b>is being defined using the Sequential model from the keras library. The model has a single LSTM layer with 50 units, and an output layer with a single neuron. The input shape of the LSTM layer is defined as (1, 4), which means that the model expects a 3D input array with a shape of (batch_size, 1, 4).
# 
# The <b>batch_size</b> is the number of samples that will be processed by the model at a time. The 1 in the input shape corresponds to the time steps, and the 4 corresponds to the number of features in the input data. In this case, the input data has 4 features, which are the opening price, highest price, lowest price, and volume of trades for each day.
# 
# The model is compiled with the <b>mean_squared_error </b>loss function and the adam optimizer. The mean_squared_error loss function measures the average squared difference between the predicted and actual values, and the adam optimizer is an algorithm for adjusting the model's weights to minimize the loss.
# 
# <b>Mean Squared Error (MSE) </b>is a commonly used loss function for regression tasks, such as predicting stock prices. It measures the average squared difference between the predicted and actual values. MSE is calculated by taking the sum of the squared differences between the predicted and actual values, dividing by the number of samples, and then taking the square root of the result.
# 
# Here is the formula for MSE:
# 
# <b>MSE = sqrt(1/n * sum((predicted_i - actual_i)^2))</b>
# 
# where n is the number of samples and predicted_i and actual_i are the predicted and actual values for sample i.
# 
# <b>Adam (Adaptive Moment Estimation)</b> is an optimization algorithm for training neural networks. It is an extension of the Stochastic Gradient Descent (SGD) algorithm and combines the benefits of SGD with the adaptive learning rates of Adagrad. Adam adjusts the learning rate for each weight based on the historical gradient descent updates of that weight, which can result in faster convergence and better performance.

# <b>The LSTM model</b> is being trained using the<b> fit() function</b> from the keras library. The fit() function takes the following arguments:
# 
# <b>X_train</b>: The input data for the training set. This should be a 3D array with a shape of (batch_size, time_steps, n_features), where batch_size is the number of samples, time_steps is the number of time steps, and n_features is the number of features in the input data.
# 
# <b>y_train</b>: The target data for the training set. This should be a 2D array with a shape of (batch_size, n_outputs), where batch_size is the number of samples and n_outputs is the number of outputs.
# 
# <b>epochs</b>: The number of epochs to train the model for. An epoch is a full pass through the training data.
# 
# <b>batch_size</b>: The number of samples to process at a time.
# 
# <b>verbose</b>: The verbosity mode. Setting verbose=2 will print the loss and accuracy for each epoch.
# 
# <b>shuffle</b>: Whether to shuffle the training data before each epoch.
# 
# The fit() function will train the model on the training data and return a history object that contains information about the training process, such as the loss and accuracy for each epoch. This information can be used to plot the training and validation curves to visualize the model's performance.

# In[45]:


# Make predictions on the test data
predictions = model.predict(X_test)


# In[46]:


# Inverse transform the predictions and test data

predictions = predictions.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)


# In[47]:


# Calculate the mean squared error
mse = mean_squared_error(y_test, predictions)
print('Mean Squared Error:', mse)


# In[48]:


# Plot the actual and predicted values
plt.plot(y_test, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()


# In[49]:


# Plot the actual and predicted values as a scatter plot
plt.scatter(y_test, predictions)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()


# In[38]:


input_shape = model.layers[0].input_shape
print(input_shape)


# In[ ]:


import numpy as np

# Normalize the new data
new_data = np.array([[100, 50, 75, 200,8]])
new_data_scaled = scaler.transform(new_data)

# Make a prediction on the normalized data
prediction = model.predict(new_data_scaled)

# Inverse transform the prediction
prediction_inverted = scaler.inverse_transform(prediction)


# #  **References**
# [1] https://www.wsj.com/market-data/quotes/PK/NESTLE/historical-prices <br>
# [2] https://www.researchgate.net/profile/Fan-Liu-58/publication/332790290/figure/fig2/AS:954364717973505@1604549688939/An-illustration-of-the-internal-structure-of-LSTM-cell.png<br>
