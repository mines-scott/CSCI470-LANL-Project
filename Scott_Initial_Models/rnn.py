import scipy
import numpy
import tensorflow as tf
from tensorflow import keras
from tensorflow. keras import Sequential, layers, callbacks
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from IPython.core.pylabtools import figsize
import seaborn as sns
import pickle

READ_PICKLE = True;
CHUNK_SIZE = 5_000

#Range, Std, IQR, Median

labels = np.array([])
features = np.array([])
if READ_PICKLE:
    pickle_file = open('pickle.txt', 'rb')     
    db = pickle.load(pickle_file)
    temp = []
    for key in db:
        temp.append(key)
    labels, features = temp
    pickle_file.close()
else:
    for i, dfChunk in enumerate(pd.read_csv('data/train.csv', chunksize=5_000)):
        stats = dfChunk['acoustic_data'].describe()
        size = len(dfChunk['time_to_failure'])
        labels.append(dfChunk['time_to_failure'].iloc[int(size / 2)])

        range = stats['max'] - stats['min']
        std = stats['std']
        iqr = stats['75%'] - stats['25%']
        median = stats['50%']
        mean = stats['mean']

        features.append((mean, median, std, range, iqr))
        if i % 1000 == 0 and i != 0:
            print(i)
    print('Done with parsing')
    labels = np.array(labels)
    features = np.array(features)
    pickle_file = open('pickle.txt', 'ab')
    pickle.dump((labels, features), pickle_file)
    pickle_file.close()


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(y_train)
y_train = np.reshape(y_train, (1, y_train.shape[0]))
y_test = np.reshape(y_test, (1, y_test.shape[0]))
#############################################
# Set random seed for reproducibility
tf.random.set_seed(1234)

# Different scaler for input and output
scaler_x = MinMaxScaler(feature_range = (0,1))
scaler_y = MinMaxScaler(feature_range = (0,1))
# Fit the scaler using available training data
input_scaler = scaler_x.fit(X_train)
output_scaler = scaler_y.fit(y_train)
# Apply the scaler to training data
train_y_norm = output_scaler.transform(y_train)
train_x_norm = input_scaler.transform(X_train)
# Apply the scaler to test data
test_y_norm = output_scaler.transform(y_test)
test_x_norm = input_scaler.transform(X_test)

