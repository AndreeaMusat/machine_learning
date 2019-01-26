# Andreea Musat, January 2019

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random

from functools import reduce
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.morphology import label
from skimage.filters import sobel
from scipy import ndimage

from tensorflow.contrib.layers import xavier_initializer

IMG_SIZE = 28
START_TOK, END_TOK = 10, 11


def watershed_segementation(img, show=True):
    elevation_map = sobel(img)
    markers = np.zeros_like(img)
    markers[img < 20] = 1
    markers[img > 150] = 2
    if show:
        plt.title('Markers')
        plt.imshow(markers)
        plt.show()
    segmentation = watershed(elevation_map, markers)
    segmentation = ndimage.binary_fill_holes(segmentation - 1)
    labels, _ = ndimage.label(segmentation)

    if show:
        plt.title('Labels')
        plt.imshow(labels)
        plt.show()

    return labels


def get_each_digit_as_image(img, show=True):
    """
    For a given image containing a number, return all its digits separately in 28x28 images.
    """
    labels_ws = watershed_segementation(img, show)

    # We are interested in the unique colors in the image when it's scanned from
    # left to right.
    unique_colors = [0]
    for i in range(img.shape[1]):
        crt_unique_colors = np.unique(
            labels_ws[:, i])  # pylint: disable=no-member
        for color in crt_unique_colors:
            if color not in unique_colors:
                unique_colors.append(color)

    digits = []
    prevs = []
    for color in unique_colors[1:]:
        color_cols = np.argwhere(np.any(labels_ws == color, axis=0))
        first_col, last_col = color_cols[0, 0], color_cols[-1, 0]

        if last_col - first_col <= 1:
            continue

        # Current image is completely found in anouther image
        skip = any([first_col > prev_first_col and last_col < prev_last_col
                    for prev_first_col, prev_last_col in prevs])
        if skip:
            continue
        # Current image overlaps with previously found image
        overlaps = [first_col < prev_last_col and last_col - first_col > prev_last_col - prev_first_col
                    for prev_first_col, prev_last_col in prevs]
        # Remove covered images
        new_prevs = [i for j, i in enumerate(prevs) if overlaps[j] == False]
        prevs = new_prevs

        # Add current image
        prevs.append((first_col, last_col))

    for first_col, last_col in prevs:
        curr_img = img[:, first_col:last_col+1]
        resized = np.zeros((IMG_SIZE, IMG_SIZE))
        begin = max(0, (IMG_SIZE - curr_img.shape[1]) // 2)
        end = begin + curr_img.shape[1]
        if end - begin > 28:
            # print('Not segmented properly?')
            continue
        resized[:, begin:end] = curr_img
        digits.append(resized)
        if show:
            plt.title('Segmented digit')
            plt.imshow(resized)
            plt.show()

    return digits


def get_numbers(cnn, imgs, show=True):
    """
    For each image in imgs, get the number in the image. We first
    segment the image to get each digit in a 28x28 image, then we
    classify each image individually.
    """

    cnt_label, labels, all_digits = 0, [], []
    for img in imgs:
        digits = get_each_digit_as_image(img, show=False)
        all_digits += digits
        labels += [cnt_label] * len(digits)
        cnt_label += 1

    labels = np.array(labels)
    all_digits_np = np.array(all_digits).reshape(len(all_digits), -1)
    y_pred = cnn.predict(all_digits_np)

    all_labels, cnt = [], 0
    for img in imgs:
        crt_digits = y_pred[labels == cnt]
        if len(crt_digits) == 0:
            # print('IDK, I am gonna predict some random number')
            all_labels.append(random.randint(0, 256))
            continue
        label = reduce(lambda x, y: 10 * x + y, crt_digits)
        if show:
            plt.imshow(img)
            plt.show()
            print(label)
        all_labels.append(label)
        cnt += 1

    return all_labels


def new_conv_layer(prev_layer, in_channels, out_channels, filter_size, name):
    # Create the filters (initialize them with random values)
    filter_shape = [filter_size, filter_size, in_channels, out_channels]
    filter_weights = tf.get_variable(
        shape=filter_shape, name=name+'-filters', initializer=xavier_initializer())
    biases = tf.Variable(tf.constant(0.01, shape=[out_channels]))
    conv_layer = tf.nn.conv2d(input=prev_layer, filter=filter_weights, strides=[
                              1, 1, 1, 1], padding='SAME')
    conv_layer += biases
    conv_layer = tf.nn.max_pool(value=conv_layer, ksize=[1, 2, 2, 1], strides=[
                                1, 2, 2, 1], padding='SAME')
    conv_layer = tf.nn.relu(conv_layer)
    conv_layer = tf.layers.batch_normalization(conv_layer, training=True)

    return conv_layer, filter_weights


def new_flatten_layer(prev_layer):
    num_features = prev_layer.get_shape()[1:4].num_elements()
    prev_layer_flat = tf.reshape(prev_layer, [-1, num_features])
    return prev_layer_flat, num_features


def new_fc_layer(prev_layer, num_inputs, num_outputs, name, relu=False):
    weights_shape = [num_inputs, num_outputs]
    weights = tf.get_variable(
        shape=weights_shape, name=name+'-weights', initializer=xavier_initializer())
    biases = tf.Variable(tf.constant(0.01, shape=[num_outputs]))
    layer = tf.matmul(prev_layer, weights) + biases
    if relu:
        layer = tf.nn.relu(layer)
    return layer


def plot_acc_cost(accs, costs, model_name):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 3))
    ax1.set_title(model_name + ' accuracy evolution')
    ax1.set_xlabel('Iteration (x100)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.plot(accs, color='blue')

    ax2.set_title(model_name + ' loss evolution')
    ax2.set_xlabel('Iteration (x100)')
    ax2.set_ylabel('Loss')
    ax2.plot(costs, color='red')


def compute_exact_match(y_true, y_pred):
    same = (np.array(y_true) == np.array(y_pred))
    exact_match_acc = np.count_nonzero(same) / len(y_true)
    return exact_match_acc


def compute_digit_accuracy(y_true, y_pred):
    def num_to_digits(N): return np.array(list(
        map(int, list(('%3s' % N).replace(' ', '0')))))
    matching = 0
    for i in range(len(y_true)):
        digits_true = num_to_digits(y_true[i])
        digits_pred = num_to_digits(y_pred[i])
        matching += np.count_nonzero(digits_true == digits_pred)
    return matching * 1.0 / len(y_true) / 3


def digits_to_number(digits):
    crt_pred = list(digits)
    if END_TOK in crt_pred:
        crt_pred = crt_pred[:crt_pred.index(END_TOK)]
    if START_TOK in crt_pred:
        crt_pred.remove(START_TOK)
    if len(crt_pred) == 0:
        print('EMPTY prediction')
        return 0
    return reduce(lambda x, y: x * 10 + y, crt_pred)
