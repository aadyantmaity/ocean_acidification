import numpy as np
import matplotlib.pyplot as plt
from utils import preprocess
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance

def preprocess_data(filename):
    data = preprocess(filename)
    inputs = data[:, :-1]
    outputs = data[:, -1]
    return inputs, outputs

def split_data(inputs, outputs):
    return train_test_split(inputs, outputs, test_size=0.1, random_state=103)

def train_model(model, train_inputs, train_targets):
    model.fit(train_inputs, train_targets)

def evaluate_model(model, inputs, targets, model_name):
    predictions = model.predict(inputs)
    r2 = r2_score(targets, predictions)
    print(model_name, "R^2 Score:", r2)
    return r2

def plot_loss_curve(model, model_name):
    if isinstance(model, MLPRegressor):
        plt.figure()
        plt.plot(model.loss_curve_)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curve - {}'.format(model_name))
        plt.show()

def plot_model_performance(model_names, train_scores, dev_scores):
    x = np.arange(len(model_names))

    fig, ax = plt.subplots()
    ax.bar(x, train_scores, width=0.4, label='Training')
    ax.bar(x + 0.4, dev_scores, width=0.4, label='Development')

    ax.set_ylabel('R^2 Score')
    ax.set_title('Model Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, train_inputs, train_targets, feature_names):
    if isinstance(model, RandomForestRegressor):
        model.fit(train_inputs, train_targets)
        importances = model.feature_importances_
        result = permutation_importance(model, train_inputs, train_targets, n_repeats=10, random_state=0)
        importances_std = result.importances_std

        fig, ax = plt.subplots()
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=45)
        ax.bar(np.arange(len(feature_names)) - 0.15, importances, label='Random Forest Feature Importance', width=0.3)
        ax.bar(np.arange(len(feature_names)) + 0.15, result.importances_mean, yerr=importances_std, capsize=3, label='Random Forest Permutation Importance', width=0.3)

        ax.set_ylabel('Importance')
        ax.set_title('Feature Importance - Random Forest Regressor')
        ax.legend()

        plt.tight_layout()
        plt.show()

def get_top_N_features(importances, N):
    return np.argsort(importances)[-N:]

if __name__ == '__main__':
    current_dir = Path(__file__).parent
    filename = current_dir / 'GLODAPv2.2022_Atlantic_Ocean.csv'
    feature_names = ["G2Temperature", "G2 Salinity", "G2Oxygen", "G2aou", "G2talk", "G2cfc11", "G2cfc12", "G2phosphate", "G2pcfc12", "G2nitrate", "G2silicate"]
    N = 3

    inputs, outputs = preprocess_data(filename)
    train_inputs, dev_inputs, train_targets, dev_targets = split_data(inputs, outputs)

    linear_regression_model = LinearRegression()
    random_forest_regressor = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=3, n_jobs=-1)
    svm_model = SVR(kernel='linear')
    mlp_model = MLPRegressor(hidden_layer_sizes=(25, 50, 10), batch_size=64, learning_rate_init=0.001, max_iter=200, random_state=0)

    models = [linear_regression_model, random_forest_regressor, svm_model, mlp_model]
    model_names = ["Linear Regression", "Random Forest Regressor", "Support Vector Machine", "MLP Regressor"]
    train_scores = []
    dev_scores = []

    random_forest_regressor.fit(train_inputs, train_targets)
    importances = random_forest_regressor.feature_importances_
    top_features_indices = get_top_N_features(importances, N)
    top_train_inputs = train_inputs[:, top_features_indices]
    top_dev_inputs = dev_inputs[:, top_features_indices]

    for model, model_name in zip(models, model_names):
        if isinstance(model, RandomForestRegressor):
            train_model(model, top_train_inputs, train_targets)
            train_score = evaluate_model(model, top_train_inputs, train_targets, model_name)
            dev_score = evaluate_model(model, top_dev_inputs, dev_targets, model_name)
        else:
            train_model(model, top_train_inputs, train_targets)
            train_score = evaluate_model(model, top_train_inputs, train_targets, model_name)
            dev_score = evaluate_model(model, top_dev_inputs, dev_targets, model_name)

        train_scores.append(train_score)
        dev_scores.append(dev_score)
        plot_loss_curve(model, model_name)

    plot_model_performance(model_names, train_scores, dev_scores)
    plot_feature_importance(random_forest_regressor, train_inputs, train_targets, feature_names)
