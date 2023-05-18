from utils import preprocess
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

def perform_regression(model):
    current_dir = Path.cwd()
    filename = current_dir / 'GLODAPDATA.csv'
    data = preprocess(filename)

    inputs = data[:, :-1]
    outputs = data[:, -1]

    train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(inputs, outputs, test_size=0.2, random_state=103)

    model.fit(train_inputs, train_outputs)

    predictions = model.predict(test_inputs)

    r2 = r2_score(test_outputs, predictions)
    print(model.__class__.__name__, "R^2 Score:", r2)

if __name__ == '__main__':
    linear_regression_model = LinearRegression()
    random_forest_model = RandomForestRegressor()

    perform_regression(linear_regression_model)
    perform_regression(random_forest_model)
