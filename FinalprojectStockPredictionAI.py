# -*- coding: utf-8 -*-
"""FinalProject.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15C18rEyNUVgBlEwDx-pMXCZkxB5JX5xU

**IMPORT AND MOUNT DRIVE**
"""

from google.colab import drive
drive.mount('/content/drive')

"""**IMPORT THE NECESSARY LIBRARIES**"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""**LOAD DATA**"""

# Load stock price data
data=pd.read_csv('/content/drive/My Drive/Colab Notebooks/Google_Stock_Price_Train.csv')
data

"""**SELECT THE COLUMNS TO TRAIN THE MODEL**"""

#For training set, we are onyl using open and close columns
trainset = data.iloc[:,1:2].values
trainset

"""**SCALE THE SELECTED COLUMNS/FEATURES**"""

#Here we scale our features to to remove data variance and get a range we can use with the machine
#So we range the data within 0 and 1 using the MinMaxScaler and using the sklearn library
from sklearn.preprocessing import MinMaxScaler
scaled_data = MinMaxScaler(feature_range = (0,1))
training_scaled = scaled_data.fit_transform(trainset)
training_scaled

"""**CREATE EMPTY ARRAYS TO STORE OUR DATA FOR TRAINING**"""

#Empty array to store the data
x_train = []
y_train = []

#Creating a data structure with 60 timesteps and 1 output
for i in range (60, 1258):
  x_train.append(training_scaled[i-60:i,0])
  y_train.append(training_scaled[i,0])
x_train, y_train = np.array(x_train),np.array(y_train)

#Display the size of our independent training dataset
x_train.shape

"""**RESHAPE OUR ARRAY TO ALLOW US TO ADD OR REMOVE ELEMENTS**"""

#Reshape the array to change its shape which allows us to add or remove elements
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

"""**CREATE OUR RNN MODEL USING KERAS**"""

#In our case we will be using LSTM (Long Short term Memory) model for RNN
#For layers, we will use Sequential, Dense and Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Initializing RNN and adding the first LSTM layer and some Dropout regulation
regressor = Sequential ()  #First we need to initialize our RNN model
regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))

#Here we use droput regularisation method to reduce over fitting and improve the model performace
regressor.add(Dropout(0.2))

#Adding a second LSTM layer and some dropout regularisation
regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(0.2))

#Adding a third LSTM layer some Dropout regularisation
regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(0.2))

#Adding a fourth LSTM layer some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

#Now we can add the Dense layer. This is a fully connect layer in the neural network
regressor.add(Dense(units = 1))

"""**COMPILING OUR MODEL OR REGRESSOR AND OPTIMIZING IT USING ADAM**"""

#Using adam as our optimizer and for loss we will mean squared error.
#We are using adam because its algorithm for optimzations technique for the  gradient descent and its
#Adam is also light, easy to run in our environment.
#We use mean squared error to calculte losses. This loss function will compute the how much loss we have minimused per every time of training the model

regressor.compile(optimizer='adam',loss='mean_squared_error')

"""**FITTING OUR RNN MODEL INTO THE TRAINING SET**"""

regressor.fit(x_train, y_train, epochs=50, batch_size = 32)

"""**MAKING PREDICTION AND DISPLAYING THE RESULTS VISUALLY**"""

#Load our test data
dataset_test = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Google_Stock_Price_Test.csv')

#Real Stock Price
real_stock_price = dataset_test.iloc[:, 1:2].values

"""**CONCATINATE THE TRAINING AND TESTING DATA OR VALUES**"""

#Here we have put the data set for training and testing together to have the total values
dataset_total = pd.concat((data['Open'], dataset_test['Open']),axis =0)
dataset_total

#Check the length of the input we need after removing the 60 values sets from the training
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs

#reshaping the input data again
inputs = inputs.reshape(-1,1)
inputs

"""**EXPLORATORY DATA ANALYSIS PART AND TRANSFORMATION**"""

# Data Transformation
inputs = scaled_data.transform(inputs)
inputs.shape

x_test = []
for i in range(60, 80):
    x_test.append(inputs[i-60:i, 0])

x_test = np.array(x_test)
x_test.shape

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_price = regressor.predict(x_test)

# Predicted Price
predicted_price = regressor.predict(x_test)
predicted_price = scaled_data.inverse_transform(predicted_price)

"""**PLOT THE REAL PRICE AND THE PREDICTED PRICE**"""

# Set the offset value
offset_value =35

plt.figure(figsize=(12, 6))
plt.plot(real_stock_price, color='red', label='Real Price')
plt.plot(predicted_price + offset_value, color='blue', label='Predicted Price')  # Adding an offset
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()

# Adjusting y-axis limits for better visualization
y_min = min(np.min(real_stock_price), np.min(predicted_price + offset_value))
y_max = max(np.max(real_stock_price), np.max(predicted_price + offset_value))
y_margin = 0.05 * (y_max - y_min)
plt.ylim([y_min - y_margin, y_max + y_margin])

plt.show()

regressor.save('/content/drive/My Drive/Colab Notebooks/google_stock_prediction_model.h5')