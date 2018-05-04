import numpy as np

from layer_interface import LayerInterface

class MaxPoolingLayer(LayerInterface):

    def __init__(self, stride):
        # Dimensions: stride   # assuming filter size = stride
        self.stride = stride
        # indexes of max activations
        self.switches = {}

    def forward(self, inputs):

        # TODO 3
        D, H, W = inputs.shape

        self.outputs = np.zeros((D, H // self.stride, W // self.stride))

        for i in range(D):
            for j in range(H // self.stride):
                for k in range(W // self.stride):
                    start_row, end_row = j * self.stride, (j + 1) * self.stride
                    start_col, end_col = k * self.stride, (k + 1) * self.stride
                    curr_slice = inputs[i, start_row:end_row, start_col:end_col]
                    self.outputs[i, j, k] = np.max(curr_slice)
                    flattened_idx = np.argmax(curr_slice)
                    delta_j, delta_k = flattened_idx // self.stride, flattened_idx % self.stride
                    self.switches[i, j * self.stride + delta_j, k * self.stride + delta_k] = 1

        # print(self.switches)
        return self.outputs

    def backward(self, inputs, output_errors):
        grad = np.zeros(inputs.shape)
        for i, j, k in self.switches:
            grad[i, j, k] = output_errors[i, j // self.stride, k // self.stride]
        return grad

    def to_string(self):
        return "[MP (%s x %s)]" % (self.stride, self.stride)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

from util import close_enough

def test_max_pooling_layer():

    l = MaxPoolingLayer(2)

    x = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]],
                  [[9, 10, 11, 12], [13, 14, 15, 16]],
                  [[17, 18, 19, 20], [21, 22, 23, 24]]])

    print("Testing forward computation...")
    output = l.forward(x)
    target = np.array([[[6, 8]],
                       [[14, 16]],
                       [[22, 24]]])
    assert (output.shape == target.shape), "Wrong output size"
    assert close_enough(output, target), "Wrong values in layer ouput"
    print("Forward computation implemented ok!")


    output_err = output

    print("Testing backward computation...")

    g = l.backward(x, output_err)
    print(g)


    print("Testing gradients")
    in_target = np.array([[[0, 0, 0, 0], [0, 6, 0, 8]],
                          [[0, 0, 0, 0], [0, 14, 0, 16]],
                          [[0, 0, 0, 0], [0, 22, 0, 24]]])

    assert (g.shape == in_target.shape), "Wrong size"
    assert close_enough(g, in_target), "Wrong values in gradients"
    print("     OK")

    print("Backward computation implemented ok!")


if __name__ == "__main__":
    test_max_pooling_layer()
