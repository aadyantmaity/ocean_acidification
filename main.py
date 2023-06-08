import matplotlib.pyplot as plt
import numpy as np
from utils import preprocess
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

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

    train_r2_score = r2_score(train_targets, train_predictions)
    dev_r2_score = r2_score(dev_targets, dev_predictions)

    print(model_name, "Training R^2 Score:", train_r2_score)
    print(model_name, "Dev R^2 Score:", dev_r2_score)

    return train_r2_score, dev_r2_score, train_inputs, train_targets


if __name__ == '__main__':
    linear_regression_model = LinearRegression()
    random_forest_regressor = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=3, n_jobs=-1)
    svm_model = SVR(kernel='linear')
    mlp_model = MLPRegressor(hidden_layer_sizes=(25, 50, 10), batch_size=64, learning_rate_init=0.001, max_iter=200, random_state=0)

    models = [linear_regression_model, random_forest_regressor, svm_model, mlp_model]
    model_names = ["Linear Regression", "Random Forest Regressor", "Support Vector Machine", "MLP Regressor"]
    train_scores = []
    dev_scores = []
    train_inputs_list = []
    train_targets_list = []

    for model, model_name in zip(models, model_names):
        train_score, dev_score, train_inputs, train_targets = perform_regression(model, model_name)
        train_scores.append(train_score)
        dev_scores.append(dev_score)
        train_inputs_list.append(train_inputs)
        train_targets_list.append(train_targets)

        if isinstance(model, MLPRegressor):
            plt.figure()
            plt.plot(model.loss_curve_)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Loss Curve - {}'.format(model_name))

    x = np.arange(len(models))

    fig, ax = plt.subplots()
    ax.plot(x, train_scores, marker='o', label='Training')
    ax.plot(x, dev_scores, marker='o', label='Development')

    ax.set_ylabel('R^2 Score')
    ax.set_title('Model Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.show()
