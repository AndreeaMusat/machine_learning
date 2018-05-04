import numpy as np

from layer_interface import LayerInterface

class LinearizeLayer(LayerInterface):

    def __init__(self, depth, height, width):
        # Dimensions: depth, height, width
        self.depth = depth
        self.height = height
        self.width = width


    def forward(self, inputs):
        assert(inputs.shape == (self.depth, self.height, self.width))

        # TODO 1
        # Reshape inputs- transform volume to column
        self.outputs = np.reshape(inputs, self.depth * self.height * self.width)[:, np.newaxis]
        return self.outputs

    def backward(self, inputs, output_errors):
        # unused argument - inputs
        assert(output_errors.shape == (self.depth * self.height * self.width, 1))

        # TODO 1
        # Reshape gradients - transform column to volume
        return np.reshape(output_errors, (self.depth, self.height, self.width))

    def to_string(self):
        return "[Lin ((%s, %s, %s) -> %s)]" % (self.depth, self.height, self.width, self.depth * self.height * self.width)


class LinearizeLayerReverse(LayerInterface):

    def __init__(self, depth, height, width):
        # Dimensions: depth, height, width
        self.depth = depth
        self.height = height
        self.width = width


    def forward(self, inputs):
        assert(inputs.shape == (self.depth * self.height * self.width, 1))

        # TODO 1
        # Reshape inputs - transform column to volume
        self.outputs = np.reshape(inputs, (self.depth, self.height, self.width))
        return self.outputs

    def backward(self, inputs, output_errors):
        # unused argument - inputs
        assert(output_errors.shape == (self.depth, self.height, self.width))

        # TODO 1
        # Reshape gradients - transform volume to column
        return np.reshape(output_errors, self.depth * self.height * self.width)[:, np.newaxis]

    def to_string(self):
        return "[Lin (%s -> (%s, %s, %s))]" % (self.depth * self.height * self.width, self.depth, self.height, self.width)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

from util import close_enough

def test_linearize_layer():

    l = LinearizeLayer(2, 3, 4)

    x = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                  [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]])

    print("Testing forward computation...")
    output = l.forward(x)
    target = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24]])
    assert (output.shape == target.shape), "Wrong output size"
    assert close_enough(output, target), "Wrong values in layer ouput"
    print("Forward computation implemented ok!")

    output_err = output

    print("Testing backward computation...")

    g = l.backward(x, output_err)

    print("Testing gradients")
    in_target = x

    assert (g.shape == in_target.shape), "Wrong size"
    assert close_enough(g, in_target), "Wrong values in gradients"
    print("     OK")

    print("Backward computation implemented ok!")


def test_linearize_layer_reverse():

    l = LinearizeLayerReverse(2, 3, 4)

    x = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24]])

    print("Testing forward computation...")
    output = l.forward(x)
    target = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                  [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]])
    assert (output.shape == target.shape), "Wrong output size"
    assert close_enough(output, target), "Wrong values in layer ouput"
    print("Forward computation implemented ok!")

    output_err = output

    print("Testing backward computation...")

    g = l.backward(x, output_err)

    print("Testing gradients")
    in_target = x

    assert (g.shape == in_target.shape), "Wrong size"
    assert close_enough(g, in_target), "Wrong values in gradients"
    print("     OK")

    print("Backward computation implemented ok!")


if __name__ == "__main__":
    test_linearize_layer()
    test_linearize_layer_reverse()