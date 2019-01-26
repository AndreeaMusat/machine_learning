# Andreea Musat, January 2019

import numpy as np
import tensorflow as tf

from utils import new_conv_layer, new_fc_layer, new_flatten_layer, plot_acc_cost
from data_generator import training_generator, test_generator

from tensorflow.contrib.rnn import MultiRNNCell, BasicLSTMCell, RNNCell, static_rnn
from tensorflow.contrib.layers import xavier_initializer

START_TOK, END_TOK = 10, 11

# Similar to ConvNet, except we stop at the first fully connected layer
# and we do not train it separately


class ConvNetEmbedding(object):
    def __init__(self, X, img_size, num_channels, num_classes, name):
        self.X = X
        self.conv_filter_size1 = 5
        self.conv_filter_size2 = 5
        self.fc_size = 256
        self.num_filters1, self.num_filters2 = 32, 64
        self.img_size_flat = img_size * img_size * num_channels
        self.img_size, self.num_channels = img_size, num_channels
        self.num_classes = num_classes
        self.name = name

        self.create_placeholders()
        self.create_layers()

    def create_placeholders(self):
        self.X_img = tf.reshape(
            self.X, [-1, self.img_size, self.img_size, self.num_channels])

    def create_layers(self):
        self.layer_conv1, self.weights_conv1 = new_conv_layer(prev_layer=self.X_img,
                                                              in_channels=self.num_channels, out_channels=self.num_filters1,
                                                              filter_size=self.conv_filter_size1, name=self.name+'conv-1')
        self.layer_conv2, self.weights_conv2 = new_conv_layer(prev_layer=self.layer_conv1,
                                                              in_channels=self.num_filters1, out_channels=self.num_filters2,
                                                              filter_size=self.conv_filter_size2, name=self.name+'conv-2')
        self.flattened, self.num_features = new_flatten_layer(
            prev_layer=self.layer_conv2)
        self.fc_layer1 = new_fc_layer(prev_layer=self.flattened, num_inputs=self.num_features,
                                      num_outputs=self.fc_size, name=self.name+'fc-1', relu=True)


def get_one_hot(numbers, is_input, num_classes):
    def num_to_digits(n): return [START_TOK] + [int(digit)
                                                for digit in list(('%3s' % str(n)).replace(' ', '0'))] + [END_TOK]
    numbers_flattened = [item for sublist in numbers for item in sublist]
    digits = list(map(num_to_digits, numbers_flattened))
    one_hot = np.zeros((len(numbers), len(digits[0]), num_classes))
    for i in range(len(numbers)):
        one_hot[i, np.arange(len(digits[0])), digits[i]] = 1

    if is_input:
        return one_hot[:, :-1, :]
    return one_hot[:, 1:, :]


class End2EndAdditionNet(object):
    def __init__(self, img_size, num_channels, hidden_size=512, num_classes=12, timesteps=4):
        self.img_size = img_size
        self.num_channels = num_channels
        self.img_size_flat = img_size * img_size * num_channels
        self.timesteps = timesteps
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        self.create_placeholders()
        self.create_layers()
        self.create_optimizer_and_compute_acc()
        self.train_network()

    def create_placeholders(self):
        self.img1 = tf.placeholder(
            tf.float32, shape=[None, self.img_size_flat], name='x1')
        self.img2 = tf.placeholder(
            tf.float32, shape=[None, self.img_size_flat], name='x2')
        self.X1_img = tf.reshape(
            self.img1, [-1, self.img_size, self.img_size, self.num_channels])
        self.X2_img = tf.reshape(
            self.img2, [-1, self.img_size, self.img_size, self.num_channels])
        self.input_digits = tf.placeholder(
            tf.float32, shape=[None, self.timesteps, self.num_classes])
        self.y_true = tf.placeholder(
            tf.float32, shape=[None, self.timesteps, self.num_classes])

    def create_layers(self):
        self.img1_embed = ConvNetEmbedding(
            self.img1, self.img_size, self.num_channels, self.num_classes, 'img1-0')
        self.img2_embed = ConvNetEmbedding(
            self.img2, self.img_size, self.num_channels, self.num_classes, 'img2-0')

        self.img1_embed_norm = tf.layers.batch_normalization(
            self.img1_embed.fc_layer1, training=True)
        self.img2_embed_norm = tf.layers.batch_normalization(
            self.img2_embed.fc_layer1, training=True)

        self.weights = tf.get_variable(shape=[self.hidden_size, self.num_classes],
                                       name='weights-0', initializer=xavier_initializer())
        self.biases = tf.get_variable(shape=[self.num_classes],
                                      name='biases-0', initializer=tf.constant_initializer(0.01))

        self.concat = tf.concat(
            [self.img1_embed_norm, self.img2_embed_norm], axis=1)

        rnn_cell = BasicLSTMCell(self.hidden_size, name='rnn-0')
        outputs, states = static_rnn(rnn_cell, tf.unstack(
            self.input_digits, self.timesteps, 1), initial_state=[self.concat, self.concat])
        outputs = tf.reshape(tf.convert_to_tensor(
            outputs), shape=[-1, self.hidden_size])

        self.logits = tf.transpose(tf.reshape(tf.matmul(
            outputs, self.weights) + self.biases, shape=[self.timesteps, -1, self.num_classes]), [1, 0, 2])

        self.prediction = tf.nn.softmax(self.logits)
        self.y_pred_cls = tf.argmax(self.prediction, axis=2)

    def create_optimizer_and_compute_acc(self):
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits, labels=self.y_true))
        self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)
        self.correct_pred = tf.equal(
            tf.argmax(self.y_true, axis=2)[:, :-1], tf.argmax(self.prediction, axis=2)[:, :-1])
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def get_feed_dict(self, generator):
        X, numbers, numbers_sum = next(generator)
        img1 = np.reshape(X[:, 0, :, :], newshape=[-1, self.img_size_flat])
        img2 = np.reshape(X[:, 1, :, :], newshape=[-1, self.img_size_flat])
        input_digits = get_one_hot(
            numbers_sum, is_input=True, num_classes=self.num_classes)
        predicted_digits = get_one_hot(
            numbers_sum, is_input=False, num_classes=self.num_classes)

        return {self.img1: img1, self.img2: img2, self.input_digits: input_digits,
                self.y_true: predicted_digits}

    def train_network(self):
        num_epochs, batch_size, accs, costs = 10000, 64, [], []

        train_gen = training_generator(batch_size=batch_size)
        test_gen = test_generator(batch_size=1000)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        for ep in range(num_epochs):
            self.session.run(
                self.optimizer, feed_dict=self.get_feed_dict(train_gen))
            if ep % 100 == 0:
                cost, acc = self.session.run(
                    [self.cost, self.accuracy], feed_dict=self.get_feed_dict(test_gen))
                costs, accs = costs + [cost], accs + [acc]
                print('accuracy: {}'.format(acc))
                self.saver.save(self.session, './tmp/end2end_model.ckpt')
        plot_acc_cost(accs, costs, 'End to end model')

    def predict(self, X):
        """
        X is generated by test_generator
        """
        batch_size = len(X)
        img1 = np.reshape(X[:, 0, :, :], newshape=[-1, self.img_size_flat])
        img2 = np.reshape(X[:, 1, :, :], newshape=[-1, self.img_size_flat])
        digits = np.zeros((batch_size, self.timesteps,
                           self.num_classes), dtype=np.float32)
        digits[:, 0, START_TOK] = 1

        predicted = np.zeros((batch_size, self.timesteps - 1), dtype=np.int32)
        for i in range(1, self.timesteps):
            feed_dict = {self.img1: img1, self.img2: img2,
                         self.input_digits: digits}
            pred = self.session.run(self.y_pred_cls, feed_dict)
            predicted[:, i - 1] = pred[:, i - 1]
            digits[:, i, pred] = 1

        return predicted
