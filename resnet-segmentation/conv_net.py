import numpy as np
import pickle

from os.path import join
from matplotlib import pyplot as plt
from skimage.io import imsave

from layers_new import ReluLayer
from layers_new import ConvolutionalLayer
from layers_new import ResidualBlock
from data_provider import miniBatchGenerator
from sklearn.metrics import f1_score

EPS = 0.00001

def binary_crossentropy(my_preds, gt_preds):
	"""
		outputs.shape = (batch_size, IMG_SIZE, IMG_SIZE)
		targets.shape = (batch_suze, IMG_SIZE, IMG_SIZE)
	"""
	assert my_preds.shape == gt_preds.shape

	batch_size = my_preds.shape[0]
	IMG_SIZE = my_preds.shape[1]

	my_preds[my_preds == 0.0] = EPS
	my_preds[my_preds == 1.0] = 1.0 - EPS

	loss = - (gt_preds * np.log(my_preds))
	loss -= (1. - gt_preds) * np.log(1. - my_preds)
	loss = loss.sum() / (batch_size * IMG_SIZE * IMG_SIZE)

	return loss


class ConvNet(object):
	def __init__(self, layers, learning_rate, num_iterations):
		
		# last layer should have depth 2 (as we have 2 classes, 
		# foreground and background pixels)
		assert layers[-1].output_depth == 2

		print('INITIALIZED CONV NET with num layers=', len(layers))
		self.layers = layers
		self.learning_rate = learning_rate
		self.num_iterations = num_iterations

		self.train_loss = []
		self.test_loss = []
		self.batch_train_loss = []
		self.f1_scores = []

	def forward(self, X):
		out = X
		for layer in self.layers:
			out = layer.forward(out)
		return out

	def backward(self, output_errors):
		errs = output_errors
		for layer in reversed(self.layers):
			errs = layer.backward(errs)

	def update_parameters(self,):
		for layer in self.layers:
			layer.update_parameters(self.learning_rate)

	def softmax(self, X):
		max_val = np.max(X, axis=1)
		ex = np.exp(X - max_val[:, np.newaxis])
		probs = ex / np.sum(ex, keepdims=True, axis=1)
		return probs

	def get_probs(self, X, Y, IMG_SIZE):
		num_classes = 2
		batch_size = self.layers[0].batch_size

		Y_reshaped = np.reshape(Y, (-1,))
		out = self.forward(X)

		print('OUTPUTS=', out[0, :, 10:30, 23])
		
		my_probs = self.softmax(out)

		gt_probs = np.zeros((len(Y_reshaped), num_classes))
		gt_probs[range(len(Y_reshaped)), Y_reshaped] = 1
		gt_probs = np.reshape(gt_probs, \
					(batch_size, IMG_SIZE, IMG_SIZE, num_classes))
		gt_probs = np.transpose(gt_probs, (0, 3, 1, 2))

		return my_probs, gt_probs

	def train(self, data_path, IMG_SIZE):
		
		train_data_path = join(data_path, 'Train')
		test_data_path = join(data_path, 'Test')
		batch_size = self.layers[0].batch_size

		data_generator = miniBatchGenerator(train_data_path, \
											batch_size, IMG_SIZE, normalize=True, count=None)
		
		self.evaluate(test_data_path, IMG_SIZE)

		for it in range(self.num_iterations):

			total_loss = 0.0

			for i in range(data_generator.num_batches):
				X, Y = data_generator.next_batch()

				my_probs, gt_probs = self.get_probs(X, Y, IMG_SIZE)
				curr_loss = binary_crossentropy(my_probs[:, 0, :, :], 
												gt_probs[:, 0, :, :])
				total_loss += curr_loss
				output_errors = my_probs - gt_probs

				self.plot_batch_train_loss(curr_loss)
				self.backward(output_errors)
				self.update_parameters()

			data_generator.reset()
			total_loss /= data_generator.num_batches

			self.evaluate(test_data_path, IMG_SIZE)
			self.plot_loss(total_loss)

			pickle.dump(self, open('net.p', 'wb'))

	def plot_loss(self, loss):
		self.train_loss.append(loss)

		plt.figure()
		plt.plot(self.train_loss, color='r')
		plt.plot(self.test_loss, color='b')
		plt.savefig('train_loss.png')

	def plot_batch_train_loss(self, loss):
		self.batch_train_loss.append(loss)

		plt.figure()
		plt.plot(self.batch_train_loss)
		plt.savefig('batch_train_loss.png')


	def evaluate(self, test_data_path, IMG_SIZE):
		batch_size = self.layers[0].batch_size
		data_generator = miniBatchGenerator(test_data_path, batch_size, IMG_SIZE, True, 1000)

		avg_test_loss = 0.0
		avg_f1_score = 0.0
		saved = False
		for i in range(data_generator.num_batches):
			X, Y = data_generator.next_batch()

			my_probs, gt_probs = self.get_probs(X, Y, IMG_SIZE)
			my_pred = np.argmax(my_probs, axis=1)
			avg_test_loss += binary_crossentropy(my_probs[:, 0, :, :], 
												gt_probs[:, 0, :, :])

			print(my_pred)

			for k in range(batch_size):
				curr_f1_score = f1_score(Y[k].reshape(-1), my_pred[k].reshape(-1))
				print('CURR f1 score = ', curr_f1_score)	
				avg_f1_score += curr_f1_score

			if i in range(5):
				for k in range(batch_size):
					name = 'my_pred_' + str(i) + '_' + str(k) + '.png'
					imsave(name, (255 * my_pred[k]).astype(np.uint8).reshape(IMG_SIZE, IMG_SIZE))
					name = 'gt_pred_' + str(i) + '_' + str(k) + '.png'
					imsave(name, (255 * Y[k]).astype(np.uint8).reshape(IMG_SIZE, IMG_SIZE))
					name = 'orig_' + str(i) + '_' + str(k) + '.png'
					imsave(name, (255 * X[k]).astype(np.uint8).transpose(1, 2, 0))

		avg_f1_score /= (data_generator.num_batches * batch_size)
		avg_test_loss /= data_generator.num_batches

		self.f1_scores.append(avg_f1_score)
		self.test_loss.append(avg_test_loss)

		plt.figure()
		plt.plot(self.f1_scores)
		plt.savefig('f1_scores.png')

def test_convnet():
	batch_size = 2
	img_channels = 3
	num_filters1 = 64
	num_filters2 = 32
	num_filters3 = 16
	IMG_SIZE = 100
	filter_size = 7
	stride = 2

	data_path = join('..', 'data')

	layers = []

	layers.append(ConvolutionalLayer(batch_size, img_channels, IMG_SIZE, IMG_SIZE, num_filters1, 3, 1))
	layers.append(ReluLayer())

	res_layers = []
	res_layers.append(ConvolutionalLayer(batch_size, num_filters1, IMG_SIZE, IMG_SIZE, num_filters1, 3, 1))
	res_layers.append(ReluLayer())
	res_layers.append(ConvolutionalLayer(batch_size, num_filters1, IMG_SIZE, IMG_SIZE, num_filters1, 3, 1))
	res_layers.append(ReluLayer())
	layers.append(ResidualBlock(res_layers))

	layers.append(ConvolutionalLayer(batch_size, num_filters1, IMG_SIZE, IMG_SIZE, num_filters2, 3, 1))
	layers.append(ReluLayer())

	res_layers = []
	res_layers.append(ConvolutionalLayer(batch_size, num_filters2, IMG_SIZE, IMG_SIZE, num_filters2, 3, 1))
	res_layers.append(ReluLayer())
	res_layers.append(ConvolutionalLayer(batch_size, num_filters2, IMG_SIZE, IMG_SIZE, num_filters2, 3, 1))
	res_layers.append(ReluLayer())
	layers.append(ResidualBlock(res_layers))

	layers.append(ConvolutionalLayer(batch_size, num_filters2, IMG_SIZE, IMG_SIZE, num_filters3, 3, 1))
	layers.append(ReluLayer())

	res_layers = []
	res_layers.append(ConvolutionalLayer(batch_size, num_filters3, IMG_SIZE, IMG_SIZE, num_filters3, 3, 1))
	res_layers.append(ReluLayer())
	res_layers.append(ConvolutionalLayer(batch_size, num_filters3, IMG_SIZE, IMG_SIZE, num_filters3, 3, 1))
	res_layers.append(ReluLayer())
	layers.append(ResidualBlock(res_layers))

	layers.append(ConvolutionalLayer(batch_size, num_filters3, IMG_SIZE, IMG_SIZE, num_filters3, 3, 1))
	layers.append(ReluLayer())

	res_layers = []
	res_layers.append(ConvolutionalLayer(batch_size, num_filters3, IMG_SIZE, IMG_SIZE, num_filters3, 3, 1))
	res_layers.append(ReluLayer())
	res_layers.append(ConvolutionalLayer(batch_size, num_filters3, IMG_SIZE, IMG_SIZE, num_filters3, 3, 1))
	res_layers.append(ReluLayer())
	layers.append(ResidualBlock(res_layers))
	layers.append(ConvolutionalLayer(batch_size, num_filters3, IMG_SIZE, IMG_SIZE, 2, 3, 1))

	net = ConvNet(layers, 0.001, 10)
	net.train(data_path, IMG_SIZE)

if __name__ == '__main__':
	test_convnet()
