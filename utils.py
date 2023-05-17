import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess(filename):
    preprocessed_filename = os.path.splitext(filename)[0] + "_preprocessed.npy"

    if os.path.exists(preprocessed_filename):
        data = np.load(preprocessed_filename)
    else:
        data = pd.read_csv(filename)

        columns_to_keep = ['G2temperature', 'G2salinity', 'G2oxygen', 'G2aou', 'G2talk', 'G2cfc11', 'G2cfc12', 'G2phosphate', 'G2pcfc12', 'G2nitrate', 'G2silicate', 'G2phtsinsitutp']
        data = data[columns_to_keep]

        data = data[~(data == -9999).any(axis=1)]

        data = data.astype(float).values

        scaler = MinMaxScaler()
        data[:, :-1] = scaler.fit_transform(data[:, :-1])

        np.save(preprocessed_filename, data)

    return data
