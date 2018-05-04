# Tudor Berariu, 2016

from numpy.linalg import norm

def close_enough(arr1, arr2, max_err = 0.0000001):
    assert(arr1.shape == arr2.shape)
    return norm(arr1.reshape(arr1.size) - arr2.reshape(arr2.size)) < max_err
