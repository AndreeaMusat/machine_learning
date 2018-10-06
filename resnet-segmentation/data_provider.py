import numpy as np
from sys import exit
from os import listdir
from os.path import isfile, join
from skimage.transform import resize
from skimage.io import imread, imsave


def print_exit(message, code):
	print(message)
	exit(code)


class miniBatchGenerator(object):

	def __init__(self, data_path, batch_size, IMG_SIZE, normalize, count):
		self.batch_size = batch_size
		self.imgs_path = join(data_path, 'Imgs')
		self.labels_path = join(data_path, 'GTs')
		self.img_idx = [f[:-4] for f in listdir(self.imgs_path) \
							if isfile(join(self.imgs_path, f))]

		if count is not None and count < len(self.img_idx):
			self.img_idx = self.img_idx[:count]

		self.N = len(self.img_idx)
		self.batch_start_idx = 0
		self.num_batches = self.N // self.batch_size
		self.IMG_SIZE = IMG_SIZE
		self.normalize = normalize

		if self.normalize:
			mean_img_name = 'mean_' + str(IMG_SIZE) + '.png'
			var_img_name = 'variance_' + str(IMG_SIZE) + '.png'
			self.mean_img = resize(imread(mean_img_name), (self.IMG_SIZE, self.IMG_SIZE, 3))
			self.variance_img = resize(imread(var_img_name), (self.IMG_SIZE, self.IMG_SIZE, 3))

		self.reset() 	# shuffle data

		print('num batches=', self.num_batches)
		print('[SUCCESS] Loaded data. N =', self.N)

	def next_batch(self):
		X = np.zeros((self.batch_size, 3, self.IMG_SIZE, self.IMG_SIZE), dtype=np.float)
		Y = np.zeros((self.batch_size, self.IMG_SIZE, self.IMG_SIZE), dtype=np.int)

		counter = 0
		saved = False

		while counter < self.batch_size:
			crt_idx = self.img_idx[self.batch_start_idx + counter]
			img_name = join(self.imgs_path, crt_idx + '.jpg')
			label_name = join(self.labels_path, crt_idx + '.png')
			img = resize(imread(img_name), (self.IMG_SIZE, self.IMG_SIZE, 3))

			label = resize(imread(label_name), (self.IMG_SIZE, self.IMG_SIZE))
			if self.normalize:
				img = (img - self.mean_img)
				min_val = np.min(img)
				max_val = np.max(img)
				img = (img - min_val) / (max_val - min_val)
			img = np.transpose(img, (2, 0, 1))
			X[counter], Y[counter] = img, label
			counter += 1

		self.batch_start_idx += self.batch_size

		return X, Y
	
	def reset(self):
		self.batch_start_idx = 0
		np.random.shuffle(self.img_idx)

def compute_mean_and_variance(IMG_SIZE):
	mean_file_name = 'mean_' + str(IMG_SIZE) + '.png'
	variance_file_name = 'variance_' + str(IMG_SIZE) + '.png'

	batch_size = 100
	generator = miniBatchGenerator(join('..', 'data', 'Train'), 
								   batch_size, 
								   IMG_SIZE, normalize=False) # batch size, img size
	
	mean = np.zeros((3, IMG_SIZE, IMG_SIZE))
	variance = np.zeros((3, IMG_SIZE, IMG_SIZE))
	
	for i in range(generator.num_batches):
		X, _ = generator.next_batch()
		curr = np.sum(X, axis=0)	# sum all the images in the batch
		mean += curr
	
	mean /= generator.num_batches * batch_size
	mean_img = mean.transpose((1, 2, 0))
	mean_img = (mean_img * 255).astype(np.uint8)
	imsave(mean_file_name, mean_img)

	generator.reset()
	for i in range(generator.num_batches):
		X, _ = generator.next_batch()
		X -= mean
		variance += (X ** 2).sum(axis=0)
	variance /= (generator.num_batches * batch_size - 1)
	variance_img = variance.transpose((1, 2, 0))
	variance_img = (variance_img * 255).astype(np.uint8)
	imsave(variance_file_name, variance_img)

def test():
	batch_size = 10
	IMG_SIZE = 250
	generator =  miniBatchGenerator(join('..', 'data', 'Test'), 
								   batch_size, 
								   IMG_SIZE, normalize=True) # batch size, img size

	for i in range(generator.num_batches):
		X, Y = generator.next_batch()
		img = (X[0] * 255).astype(np.uint8).transpose(1, 2, 0)
	
	generator.reset()
	print(generator.img_idx)


if __name__ == '__main__':
	test()