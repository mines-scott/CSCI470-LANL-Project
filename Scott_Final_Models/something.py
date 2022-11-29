import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD, Adam
import pickle

READ_PICKLE = True;

#Range, Std, IQR, Median

labels = []
features = []
if READ_PICKLE:
    pickle_file = open('pickle.txt', 'rb')     
    db = pickle.load(pickle_file)
    temp = []
    for key in db:
        temp.append(key)
    labels, features = temp
    pickle_file.close()
else:
    for i, dfChunk in enumerate(pd.read_csv('../train.csv', chunksize=150_000)):
        stats = dfChunk['acoustic_data'].describe()
        quantiles = dfChunk['acoustic_data'].quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).to_list()
        size = len(dfChunk['time_to_failure'])
        labels.append(dfChunk['time_to_failure'].iloc[size - 1])

        range = stats['max'] - stats['min']
        std = stats['std']
        iqr = stats['75%'] - stats['25%']
        median = stats['50%']
        mean = stats['mean']
        feature = (mean, std, range, iqr) + tuple(quantiles)
        features.append(feature)
        if i % 1000 == 0 and i != 0:
            print(i)
    print('Done with parsing')
    labels = np.array(labels)
    features = np.array(features)
    pickle_file = open('pickle.txt', 'ab')
    pickle.dump((labels, features), pickle_file)
    pickle_file.close()

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)

layers = [
    Dense(10, input_shape = (13,), activation = 'tanh'),
    Dropout(0.1),
    Dense(10, activation = 'relu'),
    Dropout(0.1),
    Dense(10, activation = 'tanh'), 
    Dense(1, activation = 'relu')
]
model = Sequential(layers)

optimizer = Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss='mae', metrics=['mae', "mse"])
n_epochs = 1000
history = model.fit(x_train, y_train, batch_size=32, epochs=n_epochs, verbose=1)

print(model.evaluate(x_test, y_test))