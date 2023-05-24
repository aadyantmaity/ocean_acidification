from utils import preprocess
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def perform_regression(model):
    current_dir = Path(__file__).parent
    filename = current_dir / 'GLODAPv2.2022_Atlantic_Ocean.csv'
    data = preprocess(filename)

    inputs = data[:, :-1]
    outputs = data[:, -1]

    train_inputs, dev_inputs, train_targets, dev_targets = train_test_split(inputs, outputs, test_size=0.1, random_state=103)

    model.fit(train_inputs, train_targets)

    dev_predictions = model.predict(dev_inputs)
    train_predictions = model.predict(train_inputs)

    print(model.__class__.__name__, "Training R^2 Score:", r2_score(train_targets, train_predictions))
    print(model.__class__.__name__, "Dev R^2 Score:", r2_score(dev_targets, dev_predictions))

if __name__ == '__main__':
    linear_regression_model = LinearRegression()
    random_forest_model = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=3, verbose=100, n_jobs=-1)

    perform_regression(linear_regression_model)
    perform_regression(random_forest_model)
