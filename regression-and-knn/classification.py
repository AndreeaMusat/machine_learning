import numpy as np
import pandas as pd

from utils import compute_distance

class KNN:
    def __init__(self, K=3, dist_type='euclidean'):
        self.K = K
        self.dist_type = dist_type
        
    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        # dist.shape = (Num test data, num train data, 2)
        dist = compute_distance(X, self.X, self.dist_type)
        dist = [[tuple(pair) for pair in line] for line in dist.tolist()]
        dist = [sorted(line, key=lambda pair : pair[0]) for line in dist]
        neighs = np.matrix([[pair[1] for pair in line] for line in dist])
        k_neighs = neighs[:, :self.K].astype(np.int)
        neigh_vals = self.y[k_neighs][:, :, 0]
        counts = np.apply_along_axis(np.bincount, axis=1, arr=neigh_vals, minlength=np.max(neigh_vals) +1)
        winner_neigh_idx = np.argmax(counts, axis=1)   
        return winner_neigh_idx


def compute_accuracy(labels : np.array, predicted : np.array):
	N = np.max(labels) + 1
	accuracy = 0.0
	confusion_matrix = np.zeros((N, N))
	for i in range(len(labels)):
		confusion_matrix[labels[i], predicted[i]] += 1.0
		if labels[i] == predicted[i]:
			accuracy += 1
	return accuracy / len(labels), confusion_matrix