# Andreea Musat, January 2019

import numpy as np
import tensorflow as tf

from imgaug import augmenters as iaa
from keras import datasets
from tensorflow.contrib.layers import xavier_initializer

from utils import new_conv_layer, new_fc_layer, new_flatten_layer
from utils import plot_acc_cost

seq = iaa.Sequential([
    iaa.Affine(
        scale={"x": (0.8, 1), "y": (0.8, 1)},
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        rotate=(-15, 15),
        shear=(-5, 5),
        cval=(0, 0),
        mode='constant'
    )
])


class ConvNet(object):
    def __init__(self, img_size, num_channels, num_classes):
        self.conv_filter_size1 = 5
        self.conv_filter_size2 = 5
        self.fc_size = 128
        self.num_filters1, self.num_filters2 = 16, 32
        self.img_size_flat = img_size * img_size * num_channels
        self.img_size, self.num_channels = img_size, num_channels
        self.num_classes = num_classes
        self.batch_size = 16
        self.create_placeholders()
        self.create_layers()
        self.create_optimizer_and_compute_acc()
        self.train_network()

    def create_placeholders(self):
        self.X = tf.placeholder(
            tf.float32, shape=[None, self.img_size_flat], name='x')
        self.X_img = tf.reshape(
            self.X, [-1, self.img_size, self.img_size, self.num_channels])
        self.y_true = tf.placeholder(
            tf.float32, shape=[None, self.num_classes], name='y_true')
        self.y_true_cls = tf.argmax(self.y_true, axis=1)

    def create_layers(self):
        self.layer_conv1, self.weights_conv1 = new_conv_layer(prev_layer=self.X_img,
                                                              in_channels=self.num_channels, out_channels=self.num_filters1,
                                                              filter_size=self.conv_filter_size1, name='conv-1')
        self.layer_conv2, self.weights_conv2 = new_conv_layer(prev_layer=self.layer_conv1,
                                                              in_channels=self.num_filters1, out_channels=self.num_filters2,
                                                              filter_size=self.conv_filter_size2, name='conv-2')
        self.flattened, self.num_features = new_flatten_layer(
            prev_layer=self.layer_conv2)
        self.fc_layer1 = new_fc_layer(prev_layer=self.flattened, num_inputs=self.num_features,
                                      num_outputs=self.fc_size, name='fc-1', relu=True)
        self.fc_layer2 = new_fc_layer(prev_layer=self.fc_layer1, num_inputs=self.fc_size,
                                      num_outputs=self.num_classes, name='fc-2')
        self.prediction = tf.nn.softmax(self.fc_layer2)
        self.y_pred_cls = tf.argmax(self.prediction, axis=1)

    def create_optimizer_and_compute_acc(self):
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.fc_layer2, labels=self.y_true)
        self.cost = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=1e-3).minimize(self.cost)
        self.correct_pred = tf.equal(self.y_true_cls, self.y_pred_cls)
        self.acc = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def get_feed_dict(self, X, y, rand_batch_idx):
        batch_size = len(rand_batch_idx)
        x_batch = X[rand_batch_idx].reshape(-1, 784)
        y_true_batch = np.zeros((batch_size, self.num_classes))
        y_true_batch[np.arange(batch_size), y[rand_batch_idx]] = 1
        return {self.X: x_batch, self.y_true: y_true_batch}

    def train_network(self):
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        num_iterations, accs, costs = 55000 // self.batch_size, [], []

        for i in range(2):
            x_train = seq.augment_images(x_train)
            x_test = seq.augment_images(x_test)
            for it in range(num_iterations):
                rand_batch_idx = np.arange(
                    it * self.batch_size, (it + 1) * self.batch_size)
                feed_dict_train = self.get_feed_dict(
                    x_train, y_train, rand_batch_idx)
                self.session.run(self.optimizer, feed_dict=feed_dict_train)
                if it % 100 == 0:
                    feed_dict_test = self.get_feed_dict(
                        x_test, y_test, np.arange(1000))
                    cost, acc = self.session.run(
                        [self.cost, self.acc], feed_dict=feed_dict_test)
                    costs, accs = costs + [cost], accs + [acc]
                    self.saver.save(self.session, "./tmp/cnn_model.ckpt")

        plot_acc_cost(accs, costs, 'CNN')
        print('Final accuracy: {}'.format(acc))

    def predict(self, imgs):
        """
        imgs.shape = (Batch_size, 28, 28, 1)
        """
        test_dict = {self.X: imgs}
        y_pred = self.session.run(self.y_pred_cls, feed_dict=test_dict)
        return y_pred
