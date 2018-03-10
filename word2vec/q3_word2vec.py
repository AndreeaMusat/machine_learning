import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function """
    # Implement a function that normalizes each row of a matrix to have unit length
    
    ### YOUR CODE HERE
    x = x / np.sqrt(np.sum(x ** 2, keepdims=True, axis=1))
    ### END YOUR CODE
    
    return x

def test_normalize_rows():
    print("Testing normalizeRows...")
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]])) 
    # the result should be [[0.6, 0.8], [0.4472, 0.8944]]
    print(x)
    assert (x.all() == np.array([[0.6, 0.8], [0.4472, 0.8944]]).all())
    print("")

def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models """
    
    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, assuming the softmax prediction function and cross      
    # entropy loss.                                                   
    
    # Inputs:                                                         
    # - predicted: numpy ndarray, predicted word vector (\hat{v} in 
    #   the written component or \hat{r} in an earlier version) # this is v_c
    # - target: integer, the index of the target word               
    # - outputVectors: "output" vectors (as rows) for all tokens # this is U     
    # - dataset: needed for negative sampling, unused here.         
    
    # Outputs:                                                        
    # - cost: cross entropy cost for the softmax word prediction    
    # - gradPred: the gradient with respect to the predicted word   
    #        vector    # dJ / dv_c                                            
    # - grad: the gradient with respect to all the other word        
    #        vectors     # dJ / dU                                          
    
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!                                                  
    
    ### YOUR CODE HERE

 	# all_probs = \hat(y)
    all_probs = softmax(np.dot(predicted, outputVectors.T))

    # J = - sum (y_i log(\hat(y_i))); y_i is 0 for every word besides the target
    # so basically J = - log(\hat(y_k)) where k is the index of the target
    cost = -np.log(all_probs[target])

    # dJ / dv_c = -u_o + sum_x(\hat(y_i) * u_x)
    gradPred = -outputVectors[target] + np.dot(all_probs, outputVectors)

    # dJ / du_k = \hat(y_k) * v_c for k != o (index of target word)
    # dJ / du_o = (\hat(y_k) - 1) * v_c for k = o 
    grad = all_probs
    grad[target] -= 1

    # dJ / dU should be a matrix of size n * m where n is the number or rows in U
    # (which is also the length of all probs) and m is the number of columns in U
    # (the number of features used)
    grad = np.reshape(grad, (grad.shape[0], 1)) * np.reshape(predicted, (1, predicted.shape[0]))

    ### END YOUR CODE
    
    return cost, gradPred, grad

def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, K=10):
	""" Negative sampling cost function for word2vec models """

	# Implement the cost and gradients for one predicted word vector  
	# and one target word vector as a building block for word2vec     
	# models, using the negative sampling technique. K is the sample  
	# size. You might want to use dataset.sampleTokenIdx() to sample  
	# a random word index. 
	# 
	# Note: See test_word2vec below for dataset's initialization.
	#                                       
	# Input/Output Specifications: same as softmaxCostAndGradient     
	# We will not provide starter code for this function, but feel    
	# free to reference the code you previously wrote for this        
	# assignment!

	### YOUR CODE HERE
	# sample K u-vector indices and store them:
	u_idx = []
	for i in range(K):
		sampled_idx = dataset.sampleTokenIdx()
		if sampled_idx == target:
			sampled_idx = dataset.sampleTokenIdx()
		u_idx.append(sampled_idx)

	# all_probs = sigmoid(-u_k * v_c) (for all the sampled output vectors)
	# target_prob = sigmoid(u_o * v_c)
	all_probs = sigmoid(-np.dot(outputVectors[u_idx], predicted))
	target_prob = sigmoid(np.dot(outputVectors[target].T, predicted))

	# J_neg_sample = - log(sigmoid(u_o.T * v_c)) - sum_k log(sigmoid(-u_k.T * v_c))
	# cost = J_neg_sample
	cost = -np.log(target_prob)
	cost -= np.sum(np.log(all_probs))

	# gradPred = dJ / dv_c
	# dJ / dv_c = (sigmoid(u_o.T * v_c) - 1) * u_o - sum_k (sigmoid(-u_k.T * v_c) - 1) * u_k
	gradPred = (target_prob - 1) * outputVectors[target]

	# wtf is wrong here ? np.sum gives an incorrect answer but sum is ok 
	# (numpy floating point precision problems?)
	gradPred -= sum((all_probs - 1)[:, np.newaxis] * outputVectors[u_idx])

	# grad = dJ / dU (has the same shape as U)
	# dJ / du_o = sigmoid(u_o * v_c)
	grad = np.zeros(outputVectors.shape)
	grad[target] = (sigmoid(np.dot(outputVectors[target], predicted)) - 1.0) * predicted

	# tried to vectorize this (only update the rows in my_grad without using for) but u_idx 
	# has duplicates (as we have more samples that the size of out dataset) so there might 
	# be rows that need to be updated twice or more
	for k in u_idx:
	    grad[k] += (1.0 - sigmoid(np.dot(-outputVectors[k].T, predicted))) * predicted

	### END YOUR CODE

	return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ Skip-gram model in word2vec """

    # Implement the skip-gram model in this function.

    # Inputs:                                                         
    # - currrentWord: a string of the current center word           
    # - C: integer, context size                                    
    # - contextWords: list of no more than 2*C strings, the context words                                               
    # - tokens: a dictionary that maps words to their indices in    
    #      the word vector list                                
    # - inputVectors: "input" word vectors (as rows) for all tokens   # V        
    # - outputVectors: "output" word vectors (as rows) for all tokens    # U     
    # - word2vecCostAndGradient: the cost and gradient function for 
    #      a prediction vector given the target word vectors,  
    #      could be one of the two cost functions you          
    #      implemented above

    # Outputs:                                                        
    # - cost: the cost function value for the skip-gram model       
    # - grad: the gradient with respect to the word vectors         
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    ### YOUR CODE HERE
    center_word_index = tokens[currentWord]
    v_c = inputVectors[center_word_index]

    # initialize the cost & gradients
    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    # for every context word, compute the loss and its gradients
    # and add them to the total cost and gradients
    for context_word in contextWords:
    	target = tokens[context_word]
    	curr_cost, curr_gradIn, curr_gradOut = word2vecCostAndGradient(v_c, target, outputVectors, dataset)
    	cost += curr_cost
    	# make sure we only update the gradient corresponding to v_c
    	gradIn[center_word_index] += curr_gradIn
    	gradOut += curr_gradOut

    ### END YOUR CODE
    
    return cost, gradIn, gradOut

def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ CBOW model in word2vec """

    # Implement the continuous bag-of-words model in this function.            
    # Input/Output specifications: same as the skip-gram model        
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    #################################################################
    # IMPLEMENTING CBOW IS EXTRA CREDIT, DERIVATIONS IN THE WRIITEN #
    # ASSIGNMENT ARE NOT!                                           #  
    #################################################################
    
    cost = 0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    # raise NotImplementedError
    # TODO
    ### END YOUR CODE
    
    return cost, gradIn, gradOut

#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:int(N/2),:]
    outputVectors = wordVectors[int(N/2):,:]
    for i in range(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)
        
        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1
        
        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:int(N/2), :] += gin / batchsize / denom
        grad[int(N/2):, :] += gout / batchsize / denom
        
    return cost, grad

def test_word2vec():
    # Interface to the dataset for negative sampling
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] \
           for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print("==== Gradient check for skip-gram ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
    print("\n==== Gradient check for CBOW      ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)

    print("\n=== Results ===")
    print(skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print(skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient))
    print(cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset))
    print(cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient))

if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()