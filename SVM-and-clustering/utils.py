import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def split_data(corpus, authors, test_split=0.25):
    """
            corpus : 2D matrix of shape texts_per_author x num_authors
            authors : name of the authors
            test_split : percentage of data to be used for testing
    """
    num_test_split_per_author = int(test_split * corpus.shape[1])
    num_train_data = (corpus.shape[1] -
                      num_test_split_per_author) * corpus.shape[0]

    data = [(corpus[text_idx, author_idx], author_idx) for author_idx in range(
        len(authors)) for text_idx in range(corpus.shape[0])]
    np.random.shuffle(data)
    X, y = zip(*data)

    return np.array(X), np.array(y), num_train_data


def update_best_model(embed_type, embed_size, min_df, max_df, svm_params, score):
    best_model = {}
    best_model['embed_type'] = embed_type
    best_model['embed_size'] = embed_size
    best_model['min_df'] = min_df
    best_model['max_df'] = max_df
    best_model['gamma'] = svm_params['gamma']
    best_model['C'] = svm_params['C']
    best_model['score'] = score
    return best_model


def embed_documents(corpus, embed_size, method='tf', min_df=None, max_df=None):
    """
            text : the document to be embedded
            method : should be one of the following: 
                    'tf', 'tf-idf', 'word2vec', 'glove', 'use'
    """
    if method == 'tf':
        vectorizer = CountVectorizer(lowercase=True, stop_words='english', min_df=min_df,
                                     max_df=max_df, max_features=embed_size, token_pattern='[a-zA-Z][a-zA-Z]+')
        X = vectorizer.fit_transform(corpus)
        return X

    elif method == 'tf-idf':
        vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', min_df=min_df,
                                     max_df=max_df, max_features=embed_size, token_pattern='[a-zA-Z][a-zA-Z]+')
        X = vectorizer.fit_transform(corpus)
        return X


def plot_clusters(reduced_data, model, ax, y=None):
    x_min, x_max = reduced_data[:50, 0].min(
    ) - 1, reduced_data[:50, 0].max() + 1
    y_min, y_max = reduced_data[:50, 1].min(
    ) - 1, reduced_data[:50, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    clusters = model.cluster_centers_

    if clusters is not None:
        ax.scatter(clusters[:, 0], clusters[:, 1],
                   marker='x', linewidths=3, color='w')

    if y is not None:
        colors = ['red', 'green', 'blue', 'yellow']
        unique_y = np.unique(y)
        for i in range(len(unique_y)):
            idx = np.argwhere(y == unique_y[i])
            ax.plot(reduced_data[idx, 0], reduced_data[idx,
                                                       1], 'k.', markersize=5, color=colors[i])
    else:
        ax.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)

    ax.imshow(Z, interpolation='nearest',
              extent=(xx.min(), xx.max(), yy.min(), yy.max()),
              cmap=plt.cm.Paired,
              aspect='auto', origin='lower')
