# Final_Project_Group_28
This repository contains the code for predicting Google stock prices using a Long Short-Term Memory (LSTM) recurrent neural network (RNN). The project focuses on utilizing the open stock prices for training and testing, employing the MinMaxScaler for feature scaling, and LSTM layers for modeling temporal dependencies.

#Usage
Training the Model:
Open the notebooks/FinalProject.ipynb notebook in a Jupyter environment.
Execute each cell sequentially to load data, preprocess it, train the LSTM model, and visualize predictions.
The trained model will be saved in the models/ directory.

#Key Features
STM Model: The core of the project is an LSTM-based neural network that leverages historical open stock prices to predict future values.

Data Scaling: MinMaxScaler is applied to scale the open stock prices between 0 and 1, facilitating the training process.

Visualization: The project includes a visual representation of both real and predicted stock prices, providing an insightful comparison.

#Instructions
Open FinalProject.ipynb in the notebooks/ directory using Jupyter.
Run each cell in the notebook sequentially to load data, preprocess, train the model, and make predictions.
The final predictions are visualized alongside the real stock prices for analysis.

#Files
i. data/: Contains datasets used for model training and testing.
ii. models/: Stores the trained LSTM model (google_stock_prediction_model.h5).
iii. notebooks/: Jupyter notebooks for data exploration, model training, and evaluation.
iv. src/: Source code for data preprocessing, LSTM model architecture, and training scripts.


