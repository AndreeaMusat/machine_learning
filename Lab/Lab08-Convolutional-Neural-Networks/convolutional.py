import numpy as np

from layer_interface import LayerInterface

class ConvolutionalLayer(LayerInterface):

    def __init__(self, inputs_depth, inputs_height, inputs_width, outputs_depth, k, stride):
        # Number of inputs, number of outputs, filter size, stride

        self.inputs_depth = inputs_depth
        self.inputs_height = inputs_height
        self.inputs_width = inputs_width

        self.k = k      # filter size
        self.stride = stride

        self.outputs_depth = outputs_depth
        self.outputs_height = int((self.inputs_height - self.k) / self.stride + 1)
        self.outputs_width = int((self.inputs_width - self.k) / self.stride + 1)

        # Layer's parameters
        self.weights = np.random.normal(
            0,
            np.sqrt(2.0 / float(self.outputs_depth + self.inputs_depth + self.k + self.k)),
            (self.outputs_depth, self.inputs_depth, self.k, self.k)
        )
        self.biases = np.random.normal(
            0,
            np.sqrt(2.0 / float(self.outputs_depth + 1)),
            (self.outputs_depth, 1)
        )

        # Computed values
        self.outputs = np.zeros((self.outputs_depth, self.outputs_height, self.outputs_width))

        # Gradients
        self.g_weights = np.zeros(self.weights.shape)
        self.g_biases = np.zeros(self.biases.shape)

    def forward(self, inputs):
        assert(inputs.shape == (self.inputs_depth, self.inputs_height, self.inputs_width))

        # TODO (4.a)
        # -> compute self.outputs
        
        # outputs method 0
        for i in range(self.outputs_height):
            for j in range(self.outputs_width):
                curr_slice = inputs[:, i*self.stride:i*self.stride+self.k, j*self.stride:j*self.stride+self.k]
                self.outputs[:, i, j] += (self.weights * curr_slice).sum(axis=-1).sum(axis=-2).sum(axis=1) + self.biases.T[0].T

        # outputs methods 1
        # for n in range(self.outputs_depth):
        #     for i in range(self.outputs_height):
        #         for j in range(self.outputs_width):

        #             for m in range(self.inputs_depth):
        #                 for p in range(self.k):
        #                     for q in range(self.k):
        #                         tmp = self.weights[n, m, p, q] * inputs[m, i*self.stride+p, j*self.stride+q]
        #                         self.outputs[n, i, j] += tmp
        #             self.outputs[n, i, j] += self.biases[n]

        return self.outputs

    def backward(self, inputs, output_errors):
        assert(output_errors.shape == (self.outputs_depth, self.outputs_height, self.outputs_width))

        # g_biases method 0
        self.g_biases += output_errors.sum(axis=-1).sum(axis=1)[:, np.newaxis]

        # g_biases method 1
        # for n in range(self.outputs_depth):
            # self.g_biases[n] += output_errors[n, :, :].sum()

        # g_biases method 2
        # for n in range(self.outputs_depth):
        #     for i in range(self.outputs_height):
        #         for j in range(self.outputs_width):
        #             self.g_biases[n] += output_errors[n][i][j]

        # TODO (4.b.ii)
        # Compute the gradients w.r.t. the weights (self.g_weights)

        # self.g_weights method 1
        for d in range(self.inputs_depth):
            for x in range(self.k):
                for y in range(self.k):
                    inputs_slice = inputs[d, x:self.outputs_height*self.stride+x, y:self.outputs_width*self.stride+y]
                    self.g_weights[:, d, x, y] += (inputs_slice * output_errors).sum(axis=-1).sum(axis=1)

        # self.g_weights method 2
        # for n in range(self.outputs_depth):
        #     for d in range(self.inputs_depth):
        #         for x in range(self.k):
        #             for y in range(self.k):
        #                 out_errors_slice = output_errors[n, :, :]
        #                 inputs_slice = inputs[d, x:self.outputs_height*self.stride+x, y:self.outputs_width*self.stride+y]
        #                 self.g_weights[n, d, x, y] += (inputs_slice * out_errors_slice).sum(axis=0).sum(axis=0)

        # self.g_weights method 3
        # for n in range(self.outputs_depth):
        #     for i in range(self.outputs_height):
        #         for j in range(self.outputs_width):
        #             for d in range(self.inputs_depth):
        #                 for x in range(self.k):
        #                     for y in range(self.k):
        #                         self.g_weights[n][d][x][y] += inputs[d][i * self.stride + x][j * self.stride + y] * \
        #                                                     output_errors[n][i][j]

        # TODO (4.b.iii)
        # Compute and return the gradients w.r.t the inputs of this layer

        delta = np.zeros(shape = inputs.shape)

        # delta method 1
        for n in range(self.outputs_depth):
            out_errors_slice = output_errors[n, :, :]
            for d in range(self.inputs_depth):
                for x in range(self.k):
                    for y in range(self.k):
                        tmp = self.weights[n, d, x, y] * out_errors_slice
                        delta[d, x:self.outputs_height*self.stride+x, y:self.outputs_width*self.stride+y] += tmp

        # delta method 2
        # for n in range(self.outputs_depth):
        #     for i in range(self.outputs_height):
        #         for j in range(self.outputs_width):
        #             for d in range(self.inputs_depth):
        #                 for x in range(self.k):
        #                     for y in range(self.k):
        #                         delta[d, i * self.stride + x, j * self.stride + y] += \
        #                             (self.weights[n, d, x, y] * output_errors[n, i, j])

        return delta

    def update_parameters(self, learning_rate):
        self.biases -= self.g_biases * learning_rate
        self.weights -= self.g_weights * learning_rate


    def to_string(self):
        return "[C ((%s, %s, %s) -> (%s, %s ) -> (%s, %s, %s)]" % (self.inputs_depth, self.inputs_height, self.inputs_width, self.k, self.stride, self.outputs_depth, self.outputs_height, self.outputs_width)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

from util import close_enough

def test_convolutional_layer():

    np.random.seed(0)

    l = ConvolutionalLayer(2, 3, 4, 3, 2, 1)


    l.weights = np.random.rand(3, 2, 2, 2)

    l.biases = np.random.rand(3, 1)

    x = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                  [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]])

    print("Testing forward computation...")
    output = l.forward(x)
    target = np.array([[[ 34.55043437, 38.95942899, 43.36842361],
                        [ 52.18641284, 56.59540746, 61.00440208]],
    [[ 30.72457988, 34.08923073, 37.45388158],
     [ 44.18318328, 47.54783413, 50.91248498]],
     [[ 28.2244684, 31.30220961, 34.37995083],
      [ 40.53543326, 43.61317448, 46.69091569]]])
    assert (output.shape == target.shape), "Wrong output size"
    assert close_enough(output, target), "Wrong values in layer ouput"
    print("Forward computation implemented ok!")

    output_err = np.random.rand(3, 2, 3)

    print("Testing backward computation...")

    g = l.backward(x, output_err)
#    print(l.g_biases)
#    print(l.g_weights)
#    print(g)

    print("    i. testing gradients w.r.t. the bias terms...")
    gbias_target =  np.array([[ 2.4595299 ],
                              [ 3.86207926],
                              [ 1.17504241]])

    assert (l.g_biases.shape == gbias_target.shape), "Wrong size"
    assert close_enough(l.g_biases, gbias_target), "Wrong values"
    print("     OK")

    print("   ii. testing gradients w.r.t. the weights...")
    gweights_target = np.array(
            [[[[ 12.19071134, 14.65024124],
               [ 22.02883093, 24.48836083]],
    [[ 41.70507011, 44.1646    ],
     [ 51.54318969, 54.00271959]]],

    [[[ 17.14269456, 21.00477382],
      [ 32.59101161, 36.45309087]],

  [[ 63.4876457 , 67.34972496],
   [ 78.93596275, 82.79804201]]],

 [[[  5.38434096,  6.55938337],
   [ 10.08451061, 11.25955302]],

  [[ 19.4848499 , 20.65989231],
   [ 24.18501955, 25.36006196]]]]
    )

    assert (l.g_weights.shape == gweights_target.shape), "Wrong size"
    assert close_enough(l.g_weights, gweights_target), "Wrong values"
    print("     OK")


    print("  iii. testing gradients w.r.t. the inputs...")
    in_target = np.array(
[[[ 0.1873886 , 1.2046128 , 1.46196328, 0.55403977],
  [ 1.60530819, 2.33479767, 2.57498862, 1.37639801],
  [ 1.04216109, 0.97715783, 1.17609438, 0.71793208]],

 [[ 0.1055839 , 0.66864782, 0.87158109, 0.36204295],
  [ 0.96613275, 1.97814629, 2.12435555, 0.97276618],
  [ 1.26778987, 1.01822369, 1.45453542, 0.45243098]]]
    )

    assert (g.shape == in_target.shape), "Wrong size"
    assert close_enough(g, in_target), "Wrong values in input gradients"
    print("     OK")

    print("Backward computation implemented ok!")


if __name__ == "__main__":
    test_convolutional_layer()
