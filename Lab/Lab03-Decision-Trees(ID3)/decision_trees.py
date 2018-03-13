# Andreea Musat, March 2018

import numpy as np
import sys
import re
from collections import Counter
from copy import deepcopy

classes, attributes, data, attr_to_idx = None, None, None, None
class TreeNode(object):
	def __init__(self):
		self.children = {}		# access children using the value of the split attribute
		self.split_attribute = None
		self.is_leaf = False
		self.label = None

def build_tree(curr_data, curr_attributes, method='random', depth=None, print_debug=False):
	most_common_labels = Counter([li[-1] for li in curr_data]).most_common()
	
	# all elements in the same class or no more attributes
	if most_common_labels[0][1] == len(curr_data) or \
	   bool(curr_attributes) == False or \
	   depth == 0:
		# create leaf node with most frequent label
		leaf_node = TreeNode()
		leaf_node.is_leaf = True
		leaf_node.label = most_common_labels[0][0]
		return leaf_node

	else:
		best_attr, best_attr_idx = None, None

		if method == 'random':
			# choose a random attribute for branching
			best_attr = np.random.choice(list(curr_attributes.keys()))
			best_attr_idx = attr_to_idx[best_attr]

		elif method == 'id3':
			# choose attribute that maximizes the (informational) gain of the tree
			min_expected_info_attr, min_expected_info = None, None
			for attr_name in list(curr_attributes.keys()):

				attr_idx = attr_to_idx[attr_name]
				attr_vals_freqs = np.array([len(list(filter(lambda x : x[attr_idx]==attr_val, curr_data))) \
								for attr_val in curr_attributes[attr_name]])
				attr_vals_probs = attr_vals_freqs.astype(float) / len(curr_data)

				if print_debug:
					print("ATTRIBUTE=", attr_name)
					print("attr val probs", attr_vals_probs)

				# info[i, j] = percent of data entries that have label = classes[i] and value of 
				# attribute attr_name = curr_attributes[attr_name][j]
				info = [[len(list(filter(lambda x : x[-1]==classes[c_idx] and \
													x[attr_idx]==curr_attributes[attr_name][val_idx],curr_data))) \
					   		for val_idx in range(len(curr_attributes[attr_name]))] \
					   		for c_idx in range(len(classes))]

				if print_debug:
					print("info")
					print(info)
				
				 # don't divide by 0 ! 
				attr_vals_freqs[attr_vals_freqs==0] = 1
				info = np.array(info).astype(float) / attr_vals_freqs
				
				info_not_zero = np.copy(info)
				# don't divide by 0 ! 
				info_not_zero[info_not_zero==0] = 1e-50	
				entropies = -np.sum(np.multiply(info, np.log2(info_not_zero)), axis=0)
				
				if print_debug:
					print("entropies")
					print(entropies)

				# expected info = sum after all values 
				expected_info = np.dot(attr_vals_probs, entropies)
				
				if print_debug:
					print("expected info", expected_info)
					print()
					print()

				if min_expected_info == None or (min_expected_info != None and expected_info < min_expected_info):
					min_expected_info = expected_info
					min_expected_info_attr = attr_name

			if print_debug:
				print("CHOSEN ATTRIBUTE=", min_expected_info_attr)
			
			min_expected_info_attr_idx = attr_to_idx[min_expected_info_attr]
			
			best_attr = min_expected_info_attr
			best_attr_idx = min_expected_info_attr_idx

		curr_node = TreeNode()
		curr_node.split_attribute = best_attr
		curr_node.is_leaf = False
		curr_node.label = most_common_labels[0][0]

		for attr_val in curr_attributes[best_attr]:
			child_data = [d for d in curr_data if d[best_attr_idx]==attr_val]
			if child_data == []:
				continue
			child_attributes = deepcopy(curr_attributes)
			child_attributes.pop(best_attr)

			if depth is None:
				curr_node.children[attr_val] = build_tree(child_data, child_attributes, method='id3', print_debug=print_debug)
			else:
				curr_node.children[attr_val] = build_tree(child_data, child_attributes, method='id3', depth=depth-1, print_debug=print_debug)
		return curr_node

# just for testing
def print_tree(root, attr_value=None, num_tabs=0):
	if root.is_leaf == True:
		tab_str = ''.join(['\t'] * num_tabs) if num_tabs!=0 else ""
		attr_val_str = None
		if attr_val_str is not None:
			attr_val_str = "Attr value = " + attr_value
		else:
			attr_val_str = ""

		print(tab_str, attr_val_str, "Leaf with label:", root.label)
	else:
		tab_str = ''.join(['\t'] * num_tabs) if num_tabs!=0 else ""
		attr_val_str = None
		if attr_val_str is not None:
			attr_val_str = "Attr value = " + attr_value
		else:
			attr_val_str = ""

		print(tab_str, attr_val_str, "Int nd; split attr=", root.split_attribute)
		for attr_val in root.children:
			print_tree(root.children[attr_val], attr_value=attr_val, num_tabs=num_tabs+1)

def classify(node, data_entry):
	if node.is_leaf == True:
		return node.label
	else:
		# get the value of the split attribute for data_entry
		split_attr_idx = attr_to_idx[node.split_attribute]
		split_attr_val = data_entry[split_attr_idx]
		if split_attr_val not in node.children:
			return node.label
		return classify(node.children[split_attr_val], data_entry)

def tree_accuracy(data, tree, msg="Accuracy="):
	total_data = len(data)
	total_correct = 0
	
	for data_entry in data:
		tree_label = classify(tree, data_entry[:-1])
		if tree_label == data_entry[-1]:
			total_correct += 1

	accuracy = total_correct / total_data
	print(msg, accuracy)

def get_depth(tree):
	if tree is None or tree.is_leaf == True:
		return 0

	max_child_depth = 0
	for attr_val_child in tree.children:
		max_child_depth = max(max_child_depth, get_depth(tree.children[attr_val_child]))

	return 1 + max_child_depth

def test_rand_dec_tree(training_data, test_data, print_tree_var=False):
	rand_tree_depth = min(8, len(attributes.keys()))
	rand_root = build_tree(training_data, attributes, method='random', depth=rand_tree_depth)
	msg = "random decision tree with depth " + str(rand_tree_depth) + " test data accuracy "
	print()
	if print_tree_var:
		print_tree(rand_root)

	tree_accuracy(test_data, rand_root, msg)

	msg = "random decision tree with depth " + str(rand_tree_depth) + " training data accuracy "
	tree_accuracy(training_data, rand_root, msg)
	print()
	
def test_id3_dec_tree(training_data, test_data, print_tree_var=False):
	id3_root = build_tree(training_data, attributes, method='id3', print_debug=False)
	print()
	msg = "id3 decision tree with depth " + str(get_depth(id3_root)) + " test data accuracy "

	if print_tree_var:
		print_tree(id3_root)
		
	tree_accuracy(test_data, id3_root, msg)

	msg = "id3 decision tree with depth " + str(get_depth(id3_root)) + " training data accuracy "
	tree_accuracy(training_data, id3_root, msg)
	print()

def forest_accuracy(forest, data):
	total_correct = 0
	total_data = len(data)

	# classify each data entry with every tree; the final label should be the most frequent one
	for data_entry in data:
		label_freqs = {}
		for tree in forest:
			label = classify(tree, data_entry[:-1])
			if label in label_freqs:
				label_freqs[label] += 1
			else:
				label_freqs[label] = 1
		most_freq_label = max(label_freqs, key=lambda x : label_freqs[x])
		if most_freq_label == data_entry[-1]:
			total_correct += 1

	accuracy = total_correct / total_data
	return accuracy

def test_forest(no_trees, depth, data_percent, training_data, test_data, training_data_percent, use_random=True):
	num_trees = no_trees
	tree_depth = depth
	data_usage_percent = min(training_data_percent, data_percent)

	forest = []
	len_training = int(len(training_data) * data_usage_percent / training_data_percent)
	
	for i in range(num_trees):
		np.random.shuffle(training_data)
		curr_training_data = training_data[:len_training]
		if use_random:
			curr_tree = build_tree(curr_training_data, attributes, method='random', depth=tree_depth)
		else:
			curr_tree = build_tree(curr_training_data, attributes, method='id3', depth=tree_depth)
		forest.append(curr_tree)

	
	word = "Random" if use_random == True else "id3"
	print()
	print(word, "forest (%d trees of height %d) test dataset accuracy is: %.4f" %\
			 (num_trees, tree_depth, forest_accuracy(forest, test_data)))

	print(word, "forest (%d trees of height %d) training dataset accuracy is: %.4f" %\
	 		(num_trees, tree_depth, forest_accuracy(forest, training_data)))
	print()

def main(args):
	global classes, attributes, data, attr_to_idx

	# classes = list of classes
	classes = [x.strip() for x in open(args[1], 'r').read().split(',')]

	# attributes = dict: attribute -> its possible values
	attr_lines = [re.sub(' +', ' ', x).strip() for x in open(args[2], 'r').readlines()]
	attr_lines = [x.replace(',', ' ').split() for x in attr_lines]
	attributes = {attr_lines[i][0] : attr_lines[i][1:] for i in range(len(attr_lines))}
	attr_to_idx = {attr_lines[i][0] : i for i in range(len(attr_lines))}

	# data: list of attributes + last element = labeled class
	data = [x.strip().split(',') for x in open(args[3], 'r').readlines()]

	np.random.shuffle(data)
	training_data_percent = 0.8
	len_training_data = int(training_data_percent * len(data))
	training_data = data[:len_training_data]
	test_data = data[len_training_data:]

	if args[4] == "random_tree":
		test_rand_dec_tree(training_data, test_data)
	elif args[4] == "id3_tree":
		test_id3_dec_tree(training_data, test_data)
	elif args[4] == "random_forest":
		test_forest(80, 5, 0.6, training_data, test_data, training_data_percent)
	elif args[4] == "id3_forest":
		test_forest(150, 5, 0.55, training_data, test_data, training_data_percent, False)
	else:
		print("Unkown parameter: %s. Exiting...\n" % args[4])
		sys.exit(1)

if __name__ == "__main__":
	if len(sys.argv) != 5:
		print("Usage: python this.py classes_file attr_file data_file [random_tree|id3_tree|random_forest|id3_forest]")
		sys.exit(1)

	main(sys.argv)