import numpy as np
from transfer_functions import relu, not_mine_relu

from layer_interface import LayerInterface

class ReluLayer(LayerInterface):

    def __init__(self):
        pass

    def forward(self, inputs):
        self.outputs = relu(inputs)
        return self.outputs


    def backward(self, inputs, output_errors):
        return output_errors * relu(inputs, True)

    def to_string(self):
        return "[Relu]"

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

from util import close_enough

def test_relu_layer():

    l = ReluLayer()

    x = np.array([
        -100.0, -10.0, -1.0, -.1, -.01, .0, .1, .01, .1, 1.0, 10.0, 100.0
    ])

    print("Testing forward computation...")
    output = l.forward(x)
    target = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.01, 0.1, 1.0, 10.0, 100.0
    ])
    assert (output.shape == target.shape), "Wrong output size"
    assert close_enough(output, target), "Wrong values in layer ouput"
    print("Forward computation implemented ok!")


    output_err = np.array([
        -760.0, -154.0, -145.0, -45.1, -.7601, .0, 3.1, 23.01, 1.1, 14.0, 150.0, 1.5
    ])

    print("Testing backward computation...")

    g = l.backward(x, output_err)

    print("Testing gradients")
    in_target = np.array([.0, .0, .0, .0, .0, .0, 3.1, 23.01, 1.1, 14., 150., 1.5 ])

    assert (g.shape == in_target.shape), "Wrong size"
    assert close_enough(g, in_target), "Wrong values in gradients"
    print("     OK")

    print("Backward computation implemented ok!")


if __name__ == "__main__":
    test_relu_layer()
