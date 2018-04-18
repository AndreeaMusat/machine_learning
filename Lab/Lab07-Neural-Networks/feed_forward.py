# Tudor Berariu, 2016

import numpy as np
from layer import Layer

class FeedForward:
    def __init__(self, input_size, layers_info):
        self.layers = []
        last_size = input_size
        for layer_size, transfer_function in layers_info:
            self.layers.append(Layer(last_size, layer_size, transfer_function))
            last_size = layer_size

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
        ######################## TODO (3) ######################
        for layer in self.layers:
            # print('before: ', layer.weights[:5])
            layer.weights -= learning_rate * layer.g_weights
            layer.biases -= learning_rate * layer.g_biases
            # print('after: ', layer.weights[:5])
        ########################################################

    def to_string(self):
        return " -> ".join(map(lambda l: l.to_string(), self.layers))
