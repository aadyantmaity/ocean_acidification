import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def preprocess(filename):
    preprocessed_filename = os.path.splitext(filename)[0] + "_preprocessed.csv"

    if os.path.exists(preprocessed_filename):
        # preprocessed file already exists, so just return its data
        data = pd.read_csv(preprocessed_filename)
    else:
        # read the original CSV file
        data = pd.read_csv(filename)

        # filter columns to keep only the specified columns
        columns_to_keep = ['G2temperature', 'G2salinity', 'G2oxygen', 'G2aou', 'G2talk', 'G2cfc11', 'G2cfc12', 'G2phosphate', 'G2pcfc12', 'G2nitrate', 'G2silicate', 'G2phtsinsitutp']
        data = data[columns_to_keep]

        # delete rows with any columns containing -9999
        data = data[~(data == -9999).any(axis=1)]

        # save the preprocessed data to a new CSV file
        data.to_csv(preprocessed_filename, index=False)

    return data


def main():
    # Preprocess the data file
    data = preprocess('/Users/aadyant/Desktop/GLODAPDATA.csv')

    # initializing input variables
    inputs = data[['G2temperature', 'G2salinity', 'G2oxygen', 'G2aou', 'G2talk', 'G2cfc11', 'G2cfc12', 'G2phosphate', 'G2pcfc12', 'G2nitrate', 'G2silicate']].values

    # initializing output variable
    outputs = data['G2phtsinsitutp'].values

    # making testing data 20% of the available and the rest is for training
    train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(inputs, outputs, test_size=0.2, random_state=103)

    # creating linear regression model and training it with the training data
    model = LinearRegression()
    model.fit(train_inputs, train_outputs)

    # making predictions based on testing input
    predictions = model.predict(test_inputs)

    # determining r^2 value for this seed's model
    r2 = r2_score(test_outputs, predictions)
    print("R^2 Score:", r2)

if __name__ == '__main__':
    main()
