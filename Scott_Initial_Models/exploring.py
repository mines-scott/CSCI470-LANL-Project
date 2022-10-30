import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# i = 0
# for dfChunk in pd.read_csv('LANL-Earthquake-Prediction/train.csv', chunksize=1000000):
#     i += 1
#     print(dfChunk['time_to_failure'])
#     if i == 2:
#         break

  
x = [(dfChunk.iloc[4999]['time_to_failure'], dfChunk['acoustic_data'].std()) for dfChunk in pd.read_csv('LANL-Earthquake-Prediction/train.csv', chunksize=10000)]
x, y = list(map(list, zip(*x)))
plt.plot(x,y, "bo", markersize = 1)
plt.xlabel("Avg. time to failure (s)")
plt.ylabel("Standard deviation of acoustic data")
plt.title("Deviation of subsets")
plt.show()