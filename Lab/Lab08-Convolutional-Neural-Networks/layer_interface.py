import numpy as np

class LayerInterface:

    def __init__(self, inputs_no, outputs_no, transfer_function):
        self.outputs = np.array([])

    def forward(self, inputs):
        return self.outputs

    def backward(self, inputs, output_errors):
        return None

    def update_parameters(self, learning_rate):
        pass

    def to_string(self):
        pass

