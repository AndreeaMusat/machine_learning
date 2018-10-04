import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import tensorflow as tf

from sklearn.metrics import confusion_matrix
from sys import exit

"""
Returns (x_train, y_train), (x_test, y_test)

x_train.shape = (num_training_examples, img_w, img_h, num_channels) or
				(num_training_examples, img_w, img_h) if num_channels = 1
y_train.shape = (num_training_examples, )
"""
def load_data(dataset_name):
	if dataset_name == 'mnist':
		return tf.keras.datasets.mnist.load_data()
	elif dataset_name == 'cifar10':
		return tf.keras.datasets.cifar10.load_data()
	else:
		exit('[ERROR] Unknown dataset.')


if __name__ == '__main__':
	(x_train, y_train), (x_test, y_test) = load_data('mnist')

	# Number of unique labels in our training data.
	num_classes = np.unique(y_train).size
	img_shape = x_train.shape[1:]
	img_size_flat = np.prod(img_shape)

	# Make each input image a vector.
	x_train = x_train.reshape((x_train.shape[0], -1))
	x_test = x_test.reshape((x_test.shape[0], -1))

	# Make one hot vectors from the labels. (Use np.array for the test set).
	y_train_one_hot = np.zeros((y_train.shape[0], num_classes))
	y_train_one_hot[np.arange(y_train.shape[0]), y_train] = 1
	y_test_one_hot = np.zeros((y_test.shape[0], num_classes))
	y_test_one_hot[np.arange(y_test.shape[0]), y_test] = 1

	# Plot some sample images from the dataset.
	def plot_images(imgs, labels, num_rows, output_file):
		assert num_rows * num_rows < x_train.shape[0]
		img_batch_shape = (num_rows * num_rows, ) + img_shape 
		images = imgs[:num_rows*num_rows].reshape(img_batch_shape)
		plt.clf()
		fig, axes = plt.subplots(num_rows, num_rows)
		fig.subplots_adjust(hspace=0.5, wspace=0.5)
		for i, ax in enumerate(axes.flat):
			ax.imshow(images[i])
			xlabel = labels[i]
			ax.set_xlabel(xlabel)
			ax.set_xticks([])
			ax.set_yticks([])
		plt.savefig(output_file)

	plot_images(x_train, y_train, 3, 'training_examples.png')

	# Define a linear model.
	x = tf.placeholder(tf.float32, [None, img_size_flat])
	y_true = tf.placeholder(tf.float32, [None, num_classes])
	y_true_cls = tf.placeholder(tf.int64, [None])
	weights = tf.Variable(tf.random_uniform([img_size_flat, num_classes]), dtype=tf.float32)
	biases = tf.Variable(tf.zeros([num_classes]), dtype=tf.float32)

	# (batch_size, img_size_flat) * (img_size_flat, num_classes) ->
	# 		(batch_size, num_classes)
	logits = tf.matmul(x, weights) + biases
	y_pred = tf.nn.softmax(logits)
	y_pred_cls = tf.argmax(y_pred, axis=1)

	# Loss function, optimizer and model evaluation.
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_true)
	cost = tf.reduce_mean(cross_entropy)

	# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
	optimizer = tf.train.AdamOptimizer(learning_rate=0.03).minimize(cost)

	correct_prediction = tf.equal(y_pred_cls, y_true_cls)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# Train the model
	session = tf.Session()
	session.run(tf.global_variables_initializer())
	
	feed_dict_test = {x : x_test, y_true : y_test_one_hot, y_true_cls : y_test}

	def optimize(num_iterations, batch_size):
		for i in range(num_iterations):
			rand_batch_idx = np.random.choice(x_train.shape[0], batch_size).astype(np.int32)
			x_batch = x_train[rand_batch_idx]
			y_true_batch = y_train_one_hot[rand_batch_idx, :]
			feed_dict_train = {x : x_batch, y_true : y_true_batch}
			session.run(optimizer, feed_dict=feed_dict_train)

	def print_accuracy():
		feed_dict_test = {x : x_test, y_true : y_test_one_hot, y_true_cls : y_test}
		acc = session.run(accuracy, feed_dict=feed_dict_test)
		print("accuracy:", acc)
		return acc

	print_accuracy()

	def print_confusion_matrix():
		predicted_classes = session.run(y_pred_cls, feed_dict=feed_dict_test)
		cm = confusion_matrix(y_true=y_test, y_pred=predicted_classes)
		# print(cm)
		plt.clf()
		plt.imshow(cm, interpolation='nearest')
		plt.tight_layout()
		tick_marks = np.arange(num_classes)
		plt.xticks(tick_marks, range(num_classes))
		plt.yticks(tick_marks, range(num_classes))
		plt.xlabel('Predicted')
		plt.ylabel('True')
		plt.savefig('confusion_matrix.png')

	print_confusion_matrix()

	def plot_example_errors():
		correct, predicted = session.run([correct_prediction, y_pred_cls],
										  feed_dict=feed_dict_test)

		# Mask of all the incorrect predictions.
		incorrect = (correct == False)
		images, y_pred, y_true = x_test[incorrect], predicted[incorrect], y_test[incorrect]
		plot_images(images, y_pred, 3, 'incorrect_predictions.png')

	plot_example_errors()

	def plot_weights():
		W = session.run(weights)
		W_min, W_max = np.min(W), np.max(W)

		plt.clf()
		fig, axes = plt.subplots(2, 5)
		fig.subplots_adjust(hspace=0.25, wspace=0.25)

		for i, ax in enumerate(axes.flat):
			# Weights for the ith digit
			image = W[:, i].reshape(img_shape)
			ax.set_xlabel("Weights: {0}".format(i))
			ax.imshow(image, vmin=W_min, vmax=W_max)
		plt.savefig('weights.png')

	plot_weights()

	accuracies = []
	for i in range(100):
		optimize(num_iterations=1, batch_size=32)
		print_confusion_matrix()
		plot_weights()
		accuracies.append(print_accuracy())

	plt.clf()
	plt.plot(accuracies)
	plt.savefig('accuracy.png')