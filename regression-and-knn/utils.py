# Andreea Musat, November 2018

import numpy as np
import pandas as pd

from typing import List

def compute_distance(X1 : np.array, X2 : np.array, dist_name='euclidean'):
    """
        Compute the distance matrix dist[i, j] for all the pairwise 
        distances between i and j
        dist[i, j, 0] = dist(i, j)
        dist[i, j, 1] = j (used to later retrieve the index on the nearest neigh.)
    """
    dist = np.zeros((X1.shape[0], X2.shape[0], 2), dtype=np.float)
    dist[:, :, 1] = np.arange(X2.shape[0])

    if dist_name == 'euclidean':
        dist[:, :, 0] = np.sqrt(np.sum((X1[:, np.newaxis] - X2[np.newaxis, :])**2, axis=2))
    elif dist_name == 'manhattan':  # TODO
        dist[:, :, 0] = np.abs(X1[:, np.newaxis] - X2[np.newaxis, :]).sum(axis=2)

    return dist

def get_data(data: pd.DataFrame, label: str, drop_columns: List, train_percent: float):
    """
   	Drop the columns indicated by drop_columns. Return
    train_percent% of data as train data and the rest as test data. 
    label is the name of the column with the value that has to be predicted.
    label should not be included in drop_columns
    """
    # df = data.sample(frac=1).reset_index(drop=True)		# shuffle data
    # X = pd.get_dummies(df.drop(columns=drop_columns)).values
    df = data.copy()
    drop_columns.append(label)
    X = df.drop(columns=drop_columns).values

    y = data[label][:, np.newaxis]
    num_train_data = int(train_percent * len(X))
    train_idx = np.sort(np.random.choice(
        len(X), num_train_data, replace=False))
    test_idx = np.setdiff1d(np.arange(len(X)), train_idx)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    return X_train, y_train, X_test, y_test