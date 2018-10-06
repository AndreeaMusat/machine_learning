import numpy as np

from math import floor
from utils import im2col
from utils import col2im

beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-8

class LayerInterface(object):
	def forward(self, inputs):
	    return self.outputs

	def backward(self, inputs, output_errors):
	    return None

	def update_parameters(self, learning_rate):
	    pass

	def to_string(self):
	    pass


class ReluLayer(LayerInterface):
	def __init__(self): pass

	def forward(self, X):
		self.cache = X
		out = np.maximum(X, 0)
		return out

	def backward(self, output_errors):
		dx = output_errors
		dx[self.cache < 0] = 0
		return dx

class ConvolutionalLayer(LayerInterface):
	def __init__(self, BS, D, H, W, K, F, S):
		self.batch_size = BS
		self.input_depth = D
		self.input_height = H
		self.input_width = W
		self.output_depth = K
		self.filter_size = F
		self.stride = S

		self.total_pad = (H - 1) * S + F - H
		print('total+pad', self.total_pad)
		self.pad1 = int(floor(self.total_pad / 2))
		self.pad2 = self.total_pad - self.pad1

		self.input_width += self.total_pad
		self.input_height += self.total_pad

		self.output_height = (H + self.total_pad - F) // S + 1
		self.output_width = (W + self.total_pad - F) // S + 1

		self.input_shape = (BS, D, self.input_height, self.input_width)
		self.output_shape = (BS, K, self.output_height, self.output_width)

		weights_mean = 0 
		weights_dev = np.sqrt(2.0 / float(2 * F + D + K))
		weights_shape = (K, F, F, D)

		bias_mean = 0
		bias_dev = np.sqrt(2.0 / (K + 1))
		bias_shape = (K, 1)

		self.W = np.random.normal(weights_mean, weights_dev, weights_shape)
		self.b = np.random.normal(bias_mean, bias_dev, bias_shape)

		self.mem_W, self.mem_W_sq = np.zeros(self.W.shape), np.zeros(self.W.shape)
		self.mem_b, self.mem_b_sq = np.zeros(self.b.shape), np.zeros(self.b.shape)
		self.t = 0

	def to_string(self):
		return "[C ((%s, %s, %s) -> (%s, %s) -> (%s, %s, %s)]" % \
			(self.input_depth, self.input_height, self.input_width, \
			 self.filter_size, self.stride, self.output_depth, \
			 self.output_height, self.output_width)

	def forward(self, X):

		# pad X so that X.shape = (batch_size, input_depth, input_height, input_width)
		X = np.pad(X, ((0,0), (0,0), (self.pad1,self.pad2), (self.pad1,self.pad2)), \
					mode='constant')
	
		assert X.shape == self.input_shape

		# W_blocks.shape = (output_depth, filter_size**2 * input_depth)
		# X_blocks.shape = (filter_size**2 * input_depth, 
		#				    output_height * output_width * batch_size)
		# outputs.shape = (output_depth, output_height * out_width * batch_size)
		W_blocks = np.reshape(self.W, (self.output_depth, -1))
		X_blocks = im2col(X, self.filter_size, self.filter_size, \
							 0, self.stride)
		outputs = np.dot(W_blocks, X_blocks) + self.b
		outputs = np.reshape(outputs, (self.output_depth, self.output_height, \
								  self.output_width, self.batch_size))
		outputs = outputs.transpose(3, 0, 1, 2)

		self.X_blocks = X_blocks

		print('X.shape=', X.shape, 'FILTER SIZE=', self.filter_size)
		print('H2=', self.output_height, 'W2=', self.output_width)
		print('STRIDE=', self.stride, 'total+pad=', self.total_pad)
		print('W_Blocks.shape=', W_blocks.shape)
		print('X_blocks.shape=', X_blocks.shape)
		print('outputs shape=', outputs.shape)

		return outputs

	def backward(self, output_errors):

		assert output_errors.shape == self.output_shape

		grad_b = np.sum(output_errors, axis=(0, 2, 3))
		grad_b = np.reshape(grad_b, (self.output_depth, 1))

		output_errors = np.transpose(output_errors, (1, 2, 3, 0))
		output_errors = np.reshape(output_errors, (self.output_depth, -1))
		
		W_reshaped = np.reshape(self.W, (self.output_depth, -1))
		grad_x_blocks = np.dot(output_errors.T, W_reshaped).T
		grad_x = col2im(grad_x_blocks, self.input_shape, self.filter_size, \
									   self.filter_size, padding=0, stride=self.stride)
		grad_x = grad_x[:,:,self.pad1:-self.pad2,self.pad1:-self.pad2]

		grad_W = np.dot(output_errors, self.X_blocks.T)
		grad_W = np.reshape(grad_W, self.W.shape)

		self.grad_W = grad_W
		self.grad_b = grad_b

		return grad_x

	def update_parameters(self, learning_rate):

		print('GRAD W = ', self.grad_W[0, 0, :10, 0])
		print('weights before:', self.W[0, 0, :10, 0])

		self.t += 1
		self.mem_W = beta_1 * self.mem_W + (1. - beta_1) * self.grad_W
		self.mem_W_sq = beta_2 * self.mem_W_sq + (1. - beta_2) * self.grad_W * self.grad_W
		
		self.mem_b = beta_1 * self.mem_b + (1. - beta_1) * self.grad_b
		self.mem_b_sq = beta_2 * self.mem_b_sq + (1. - beta_2) * self.grad_b * self.grad_b

		w_m_cap = self.mem_W / (1 - (beta_1**(self.t / 10)))
		w_sq_cap = self.mem_W_sq / (1 - (beta_2**(self.t / 10)))
		w_prev_weights = np.copy(self.W)

		b_m_cap = self.mem_b / (1 - (beta_1**(self.t / 10)))
		b_sq_cap = self.mem_b_sq / (1 - (beta_2**(self.t / 10)))

		self.W = self.W - (learning_rate * w_m_cap) / (np.sqrt(w_sq_cap) + epsilon)
		self.b = self.b - (learning_rate * b_m_cap) / (np.sqrt(b_sq_cap) + epsilon)
		
		if np.abs(self.W - w_prev_weights).sum() < 0.0001:
			import sys
			sys.exit(0)

		# Simple Gradient Descent.
		# self.W -= learning_rate * self.grad_W
		# self.b -= learning_rate * self.grad_b

		print('weights after:', self.W[0, 0, :10, 0])


class ResidualBlock(LayerInterface):
	def __init__(self, layers):
		self.layers = layers

	def forward(self, X):
		out = np.copy(X)
		for layer in self.layers[:-1]:
			out = layer.forward(out)
		out += X
		out = self.layers[-1].forward(out)
		return out

	def backward(self, output_errors):
		errs = self.layers[-1].backward(output_errors)
		for layer in reversed(self.layers[:-1]):
			errs = layer.backward(errs)
		errs += output_errors
		return errs

	def update_parameters(self, learning_rate):
		for layer in self.layers:
			layer.update_parameters(learning_rate)


def test_conv_layer():
	conv_layer = ConvolutionalLayer(1, 3, 6, 6, 12, 3, 2)
	print(conv_layer.to_string())
	X = np.random.random((1, 3, 6, 6))
	out = conv_layer.forward(X)
	errs = np.random.random((1, 12, 6, 6))
	conv_layer.backward(errs)
	conv_layer.update_parameters(0.01)

def test_residual_block():
	conv_layer1 = ConvolutionalLayer(100, 50, 6, 6, 50, 3, 2)
	relu_layer = ReluLayer()
	conv_layer2 = ConvolutionalLayer(100, 50, 6, 6, 30, 3, 2)

	residual_block = ResidualBlock([conv_layer1, relu_layer, conv_layer2])
	residual_block.forward(np.random.random((100, 50, 6, 6)))


if __name__ == '__main__':
	test_conv_layer()
	test_residual_block()