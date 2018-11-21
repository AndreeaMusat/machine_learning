# Andreea Musat, November 2018

import numpy as np
import pandas as pd

from typing import List
from typing import Dict

from utils import compute_distance
from utils import get_data

def get_features(X: np.array):
    # Add bias term on the first column
    feats = np.ones((X.shape[0], 1 + X.shape[1]))
    feats[:, 1:] = X
    return feats


class LinearRegression:
    def __init__(self, reg: str = None, penalty: float = 0.0, lr: float = 0.1, dim: int = 1):
        self.regularization = reg
        self.penalty = penalty
        self.learning_rate = lr
        self.weights = np.random.rand(dim + 1, 1)
        self.num_iters = 1000
        self.training_loss = []

    def predict(self, X):
        return np.dot(get_features(X), self.weights)

    def error(self, X, y):
        sq_err = 0.5 / X.shape[0] * ((self.predict(X) - y)**2).sum()
        if self.regularization == None:
            return sq_err
        if self.regularization.lower() == 'ridge':
            return sq_err + self.penalty / 2 * (self.weights**2).sum()
        elif self.regularization.lower() == 'lasso':
            return sq_err + self.penalty / 2 * np.abs(self.weights).sum()

    def update_weights(self, X, y):
        d_weights = np.dot(get_features(X).T, self.predict(X) - y) / X.shape[0]
        self.weights -= (self.learning_rate * d_weights)

    def fit(self, X: np.array, y: np.array):
        for it in range(self.num_iters):
            idx = np.random.randint(X.shape[0], size=32)

            self.update_weights(X[idx, :], y[idx, :])
            self.training_loss.append(self.error(X, y))


class KNNRegression:
    def __init__(self, K=3):
        self.K = K
        
    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        dist = compute_distance(X, self.X)
        dist = [[tuple(pair) for pair in line] for line in dist.tolist()]
        dist = [sorted(line, key=lambda pair : pair[0]) for line in dist]
        neighs = np.matrix([[pair[1] for pair in line] for line in dist])
        dists = np.matrix([[pair[0] for pair in line] for line in dist])
        k_neighs = neighs[:, :self.K].astype(np.int)
        total_dist = np.sum(dists[:, :self.K], axis=1)
        weights = dists[:, :self.K] / total_dist
        neigh_vals = self.y[k_neighs]
        return (np.multiply(weights, np.squeeze(neigh_vals))).sum(axis=1)


def unnormalize(arr, mean, std):
    return arr * std + mean


def grid_search(data : pd.DataFrame, predicted_column : str, drop_columns : List, means: Dict,
                stds: Dict, regs: List, pens: List, lrs: List):
    train_X, train_y, test_X, test_y = get_data(data, predicted_column, drop_columns, 0.8)
    
    print('Train X shape: ', train_X.shape)
    print('Train y shape: ', train_y.shape)
    print('Test X shape: ', test_X.shape)
    print('Test y shape: ', test_y.shape)

    best_model = {}
    unnormalized_test_y = unnormalize(
        test_y, means[predicted_column], stds[predicted_column])

    # Grid-search for the best model
    for reg in regs:
        for pen in pens:
            for lr in lrs:
                regression_model = LinearRegression(
                    reg=reg, penalty=pen, lr=lr, dim=len(train_X[0]))
                regression_model.fit(train_X, train_y)
                predicted_y = regression_model.predict(test_X)
                unnormalized_predicted_y = unnormalize(
                    predicted_y, means[predicted_column], stds[predicted_column])
                squared_err, rms_err, l1_err = compute_error(
                    unnormalized_test_y, unnormalized_predicted_y)

                # Update the best model
                if best_model == {} or rms_err < best_model['rms_err']:
                    best_model['rms_err'] = rms_err
                    best_model['sq_err'] = squared_err
                    best_model['reg'] = reg
                    best_model['pen'] = pen
                    best_model['lr'] = lr
                    best_model['model'] = regression_model
                    best_model['l1_err'] = l1_err
                    print('Best model is {}'.format(best_model))


    return best_model


def fill_in_missing_values(data: pd.DataFrame, column_name: str):
    indices = data[data[column_name].isnull()].index.values
    data.at[indices, column_name] = data[data[column_name].notnull()][column_name].mean()


def compute_error(labels: np.array, predicted: np.array):
    N = predicted.shape[0]
    squared_err = (np.square(labels - predicted)).sum() / 2.0
    rms_err = np.sqrt(2.0 * squared_err / N)
    abs_err = np.abs(labels - predicted).mean()
    return squared_err, rms_err, abs_err


def test_knn_regr():
    x_train = np.random.rand(10, 3)
    y_train = np.random.rand(10)
    x_test = np.random.rand(4, 3)
    y_test = np.random.rand(4, 3)
    regr = KNNRegression(K=2)
    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_test)


test_knn_regr()