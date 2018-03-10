import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 
    
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    # data.shape = (N, D_x)
    # W1.shape = (D_x, H)
    # b1.shape = (N, H)
    # W2.shape = (H, D_y)
    # b2.shape = (N, D_y)
    z1 = np.dot(data, W1) + b1          # (N, D_x) * (D_x, H) = (N, H)
    h = sigmoid(z1)                     # (N, H)
    z2 = np.dot(h, W2) + b2             # (N, H) * (H, D_y) = (N, D_y)
    scores = softmax(z2)                # (N, D_y)
    cost = -np.sum(labels * np.log(scores))
    ### END YOUR CODE
    ### YOUR CODE HERE: backward propagation
    grad_z2 = scores - labels           # (N, D_y)
    grad_W2 = np.dot(h.T, grad_z2)      # (H, N) * (N, D_y) = (H, D_y)
    grad_b2 = np.sum(grad_z2, axis=0)   # (N, D_y)    minibatch gd -> the gradient is computed for more examples at a time so we have to add them
    grad_h = np.dot(grad_z2, W2.T)      # (N, D_y) * (D_y, H) = (N, H)
    grad_z1 = np.multiply(grad_h, sigmoid_grad(h))     # Hadamard product; (N, H) prod (N, H) = (N, H) (elementwise)
    grad_W1 = np.dot(data.T, grad_z1)   # (D_x, N) * (N, H) = (D_x, H)
    grad_b1 = np.sum(grad_z1, axis=0)   # (N, H)
    ### END YOUR CODE
    
    ### Stack gradients (do not modify)
    grad = np.concatenate((grad_W1.flatten(), grad_b1.flatten(), 
        grad_W2.flatten(), grad_b2.flatten()))
    
    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using 
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1
    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
        dimensions), params)

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    # raise NotImplementedError
    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()