# Tudor Berariu, 2016

import numpy as np
from layer_interface import LayerInterface

class FeedForward:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        last_input = inputs
        for layer in self.layers:
            last_input = layer.forward(last_input)
        return last_input

    def backward(self, inputs, output_error):
        crt_error = output_error
        for layer_no in range(len(self.layers)-1, 0, -1):
            crt_layer = self.layers[layer_no]
            prev_layer = self.layers[layer_no - 1]
            crt_error = crt_layer.backward(prev_layer.outputs, crt_error)
        self.layers[0].backward(inputs, crt_error)

    def update_parameters(self, learning_rate):
        # TODO (3)
        for layer in self.layers:
            layer.update_parameters(learning_rate)

    def to_string(self):
        return " -> ".join(map(lambda l: l.to_string(), self.layers))
