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

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)

layers = [
          Dense(100, input_shape = (5,), activation = 'relu'), Dropout(0.05),
          Dense(100, activation = 'tanh'), Dropout(0.05),
          Dense(100, activation = 'relu'), Dense(1, activation = 'tanh')
          ]
model = Sequential(layers)

optimizer = Adam(learning_rate=1e-7)
model.compile(optimizer=optimizer, loss='mae', metrics=['mae', "mse"])
n_epochs = 100
history = model.fit(x_train, y_train, batch_size=1000, epochs=n_epochs, verbose=1)

print(model.evaluate(x_test, y_test))

