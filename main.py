from utils import preprocess
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

def perform_regression(model, model_name):
    current_dir = Path(__file__).parent
    filename = current_dir / 'GLODAPv2.2022_Atlantic_Ocean.csv'
    data = preprocess(filename)

    inputs = data[:, :-1]
    outputs = data[:, -1]

    train_inputs, dev_inputs, train_targets, dev_targets = train_test_split(inputs, outputs, test_size=0.1, random_state=103)

    model.fit(train_inputs, train_targets)

    dev_predictions = model.predict(dev_inputs)
    train_predictions = model.predict(train_inputs)

    print(model_name, "Training R^2 Score:", r2_score(train_targets, train_predictions))
    print(model_name, "Dev R^2 Score:", r2_score(dev_targets, dev_predictions))

if __name__ == '__main__':
    linear_regression_model = LinearRegression()
    random_forest_regressor = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=3, verbose=100, n_jobs=-1)
    svm_model = SVR(kernel='linear')
    mlp_model = MLPRegressor(hidden_layer_sizes=(25, 50, 10), batch_size=64, learning_rate_init=0.001, max_iter=200, verbose=True, random_state=0)
    random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=0)

    perform_regression(linear_regression_model, "Linear Regression")
    perform_regression(random_forest_regressor, "Random Forest Regressor")
    perform_regression(svm_model, "Support Vector Machine")
    perform_regression(mlp_model, "MLP Regressor")
