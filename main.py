import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
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
        data = scaler.fit_transform(data)

        np.save(preprocessed_filename, data)

    return data


def main():
    data = preprocess('/Users/aadyant/Desktop/GLODAPDATA.csv')

    inputs = data[:, :-1]
    outputs = data[:, -1]

    train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(inputs, outputs, test_size=0.2, random_state=103)

    model = LinearRegression()
    model.fit(train_inputs, train_outputs)

    predictions = model.predict(test_inputs)

    r2 = r2_score(test_outputs, predictions)
    print("R^2 Score:", r2)

if __name__ == '__main__':
    main()
