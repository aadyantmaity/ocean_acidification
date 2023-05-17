from utils import preprocess
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

def linear_regression_main():
    current_dir = os.getcwd()
    filename = os.path.join(current_dir, 'GLODAPDATA.csv')
    data = preprocess(filename)

    inputs = data[:, :-1]
    outputs = data[:, -1]

    train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(inputs, outputs, test_size=0.2, random_state=103)

    model = LinearRegression()
    model.fit(train_inputs, train_outputs)

    predictions = model.predict(test_inputs)

    r2 = r2_score(test_outputs, predictions)
    print("Linear Regression R^2 Score:", r2)

def random_forest_main():
    data = preprocess('GLODAPDATA.csv')

    inputs = data[:, :-1]
    outputs = data[:, -1]

    train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(inputs, outputs, test_size=0.2, random_state=103)

    model = RandomForestRegressor()
    model.fit(train_inputs, train_outputs)

    predictions = model.predict(test_inputs)

    r2 = r2_score(test_outputs, predictions)
    print("Random Forest R^2 Score:", r2)

if __name__ == '__main__':
    # linear_regression_main()
    random_forest_main()
