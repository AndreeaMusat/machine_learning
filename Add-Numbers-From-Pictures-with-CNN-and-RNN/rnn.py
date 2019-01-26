# Andreea Musat, January 2019

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from functools import reduce
from operator import mul

from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.rnn import static_rnn, BasicLSTMCell
from tensorflow.contrib.rnn import MultiRNNCell

from utils import plot_acc_cost


def new_rnn_layer(prev_layer, weights, biases, hidden_size, timesteps, num_classes, name):
    X = tf.unstack(prev_layer, timesteps, 1)
    rnn_cell = BasicLSTMCell(hidden_size, forget_bias=1.0, name=name)
    outputs, states = static_rnn(rnn_cell, X, dtype=tf.float32)
    outputs = tf.reshape(tf.convert_to_tensor(
        outputs), shape=[-1, hidden_size])
    res = tf.matmul(outputs, weights) + biases
    res = tf.reshape(res, shape=[timesteps, -1, num_classes])
    return tf.transpose(res[(timesteps // 2):], [1, 0, 2])


def rnn_data_generator_batch(batch_size):
    x1 = np.random.randint(0, 256, size=(batch_size,))
    x2 = np.random.randint(0, 256, size=(batch_size,))
    y = [x + y for x, y in list(zip(x1, x2))]
    return list(zip(x1, x2)), y


def get_one_hot(X, num_classes):
    """
        X should be 2D (batch_size, timesteps)
    """
    X = X.astype(int)
    N = reduce(mul, X.shape)
    X_reshaped = np.reshape(X, (-1, ))
    X_one_hot = np.zeros((N, num_classes), dtype=np.float32)
    X_one_hot[np.arange(N), X_reshaped] = 1
    return X_one_hot.reshape((*X.shape, num_classes))


def get_rnn_feed_data(Xs, ys, num_classes):
    def num_to_digits(N): return [START_TOK] + list(
        map(int, list(('%3s' % N).replace(' ', '0')))) + [END_TOK]  # 10 = EOS, 11 = SOS
    X = [num_to_digits(Xs[i][0]) + num_to_digits(Xs[i][1])
         for i in range(len(Xs))]

    if len(ys) != 0:
        y = [num_to_digits(ys[i]) for i in range(len(ys))]
        return get_one_hot(np.array(X), num_classes), get_one_hot(np.array(y), num_classes)

    return get_one_hot(np.array(X), num_classes)


START_TOK, END_TOK = 11, 10


class RecurrentNet(object):
    def __init__(self, num_classes, hidden_size, timesteps=5):
        self.hidden_size = hidden_size
        self.num_classes = num_classes + 2  # 1 for EOS, 1 for SOS
        self.timesteps = timesteps
        self.batch_size = 64

        self.create_placeholders()
        self.create_layers()
        self.create_optimizer_and_compute_acc()
        self.train_network()

    def create_placeholders(self):
        self.X = tf.placeholder(
            tf.float32, [None, 2 * self.timesteps, self.num_classes])
        self.Y = tf.placeholder(
            tf.float32, [None, self.timesteps, self.num_classes])

    def create_layers(self):
        self.weights = tf.get_variable(shape=[self.hidden_size, self.num_classes],
                                       name='weighs-324etyzzr', initializer=xavier_initializer())
        self.biases = tf.get_variable(shape=[self.num_classes],
                                      name='biass-344eyztyr', initializer=tf.constant_initializer(0.01))
        self.logits = new_rnn_layer(self.X, self.weights, self.biases, self.hidden_size,
                                    2 * self.timesteps, self.num_classes, name='rn-344eyyr')
        self.prediction = tf.nn.softmax(self.logits)
        self.y_pred_cls = tf.argmax(self.prediction, axis=2)

    def create_optimizer_and_compute_acc(self):
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)
        self.correct_pred = tf.equal(
            tf.argmax(self.Y, axis=2)[:, :-1], tf.argmax(self.prediction, axis=2)[:, :-1])
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def get_feed_dict(self, batch_size):
        Xs, y = rnn_data_generator_batch(batch_size)
        feed_X, feed_y = get_rnn_feed_data(
            Xs, y, self.num_classes)
        return {self.X: feed_X, self.Y: feed_y}

    def train_network(self):
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        num_iteratiors, accs, costs = 10000, [], []
        for it in range(num_iteratiors):
            self.session.run(
                self.optimizer, feed_dict=self.get_feed_dict(self.batch_size))
            if it % 100 == 0:
                cost, acc = self.session.run(
                    [self.cost, self.accuracy], feed_dict=self.get_feed_dict(1000))
                costs, accs = costs + [cost], accs + [acc]
                print('accuracy: {}'.format(acc))
                self.saver.save(self.session, "./tmp/rnn_model.ckpt")

        plot_acc_cost(accs, costs, 'RNN')

    def predict(self, Xs):
        feed_X = get_rnn_feed_data(
            Xs, [], self.num_classes)
        return self.session.run(self.y_pred_cls, feed_dict={self.X: feed_X})
