import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

st.title("Google Stock Price Prediction")


# Load stock price data (assuming the data is in a CSV file)
data = pd.read_csv('Google_Stock_Price_Train.csv')


# READ THE DATA
st.subheader("Stock Price Data")
st.write(data)

# Load the trained model
model = load_model('google_stock_prediction_model.h5')

# Input for prediction
st.subheader("Enter the last 60 days stock prices for prediction:")
user_input = st.text_area("Input Stock Prices (comma-separated)", "value1, value2, ..., value60")

if st.button("Predict"):
    # Process user input
    user_input_list = [float(val.strip()) for val in user_input.split(',')]
    user_input_array = np.array(user_input_list).reshape(1, -1, 1)

    # Make prediction
    predicted_price_scaled = model.predict(user_input_array)
    predicted_price = MinMaxScaler(feature_range=(0, 1)).inverse_transform(predicted_price_scaled)

    # Display the predicted price
    st.subheader("Predicted Stock Price:")
    st.write(predicted_price[0, 0])

# Visualize the training and testing results
st.subheader("Visualize Training and Testing Results")
st.write("Note: Adjust the offset value based on your preference.")
offset_value = st.slider("Offset Value", min_value=0, max_value=50, value=35)

# Plotting the Results
plt.figure(figsize=(12, 6))
plt.plot(data['Open'], color='red', label='Training Price')
plt.plot(predicted_price + offset_value, color='blue', label='Predicted Price')  # Adding an offset
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()

# Display the plot in Streamlit app
st.pyplot(plt)

# Display the raw data
st.subheader("Raw Data for Training and Testing")
st.write(data)

# Display the model summary
st.subheader("Model Summary")
st.code(model.summary())
