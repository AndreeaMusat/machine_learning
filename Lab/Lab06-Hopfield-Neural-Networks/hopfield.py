import numpy
import random
from PIL import Image

INF = int(1e15)

class HopfieldNetwork:
    CHAR_TO_INT = { "_": -1, "X": 1 }
    INT_TO_CHAR = { -1: "_", 1: "X" }

    # Initalize a Hopfield Network with N neurons
    def __init__(self, neurons_no):
        self.neurons_no = neurons_no
        self.state = numpy.ones((self.neurons_no), dtype=int)
        self.weights = numpy.zeros((self.neurons_no, self.neurons_no))
        self.convergence_test = False


# ------------------------------------------------------------------------------

    # Learn some patterns
    def learn_patterns(self, patterns, learning_rate):
        ############################### TASK 1 ################################
        bin_patterns = [list(map(lambda x : -1 if x == '_' else 1, pattern)) for pattern in patterns]
        # self.weights = numpy.zeros((self.neurons_no, self.neurons_no))
        for pattern in bin_patterns:
            self.weights += learning_rate * (numpy.outer(pattern, pattern) - numpy.eye(len(pattern)))
        #######################################################################

    # Compute the energy of the current configuration
    def energy(self):
        ############################### TASK 1 ################################
        H = -1 / 2 * numpy.dot(numpy.dot(self.state.T, self.weights), self.state)
        return H
        #######################################################################

    # Update a single random neuron
    def single_update(self):
        ############################### TASK 1 ################################
        neuron_changed = False
        for rand_neuron_idx in numpy.random.permutation(self.neurons_no):
            weight_rand_neuron = self.weights[rand_neuron_idx, :]
            new_val = 1 if (weight_rand_neuron * self.state).sum() > 0 else -1
            
            if (new_val != self.state[rand_neuron_idx]):
                self.state[rand_neuron_idx] = new_val
                neuron_changed = True
                break

        if neuron_changed:
            self.convergence_test = False
        else:
            self.convergence_test = True
        #######################################################################

    # Check if energy is minimal
    def is_energy_minimal(self):
        ############################### TASK 1 ################################
        return self.convergence_test
        #######################################################################

    # --------------------------------------------------------------------------

    # Approximate the distribution of final states
    # starting from @samples_no random states.
    def get_final_states_distribution(self, samples_no=10):
    	######################### TASK 3 ######################################
        distribution = {}
        
        for i in range(samples_no):
            self.random_reset()
            
            cnt = 0
            self.convergence_test = False
            while not self.is_energy_minimal():
                print('Energy', self.energy(), 'updating i ', i, ' cnt ', cnt)
                cnt += 1
                self.single_update()

            pattern = self.get_pattern()
            if pattern in distribution:
                distribution[pattern] += 1
            else:
                distribution[pattern] = 1.0

        for patt in distribution.keys():
            distribution[patt] /= samples_no

        return distribution
        #######################################################################
    # -------------------------------------------------------------------------


    # Unlearn some patterns
    def unlearn_patterns(self, patterns, learning_rate, unlearn_no):
       	######################### TASK 4 (BONUS) ##############################
        patterns_to_unlearn = []
        for i in range(unlearn_no):
            self.random_reset()
            self.convergence_test = False
            while not self.is_energy_minimal():
                self.single_update()

            pattern = self.get_pattern()
            if pattern not in patterns:
                print("PATTERN ", pattern)
                patterns_to_unlearn.append(pattern)

        print('patterns to unlearn', patterns_to_unlearn)
        self.learn_patterns(patterns_to_unlearn, -learning_rate)
       	######################################################################
    # -------------------------------------------------------------------------


    # Get the pattern of the state as string
    def get_pattern(self):
        return "".join([HopfieldNetwork.INT_TO_CHAR[n] for n in self.state])

    # Reset the state of the Hopfield Network to a given pattern
    def reset(self, pattern):
        assert(len(pattern) == self.neurons_no)
        for i in range(self.neurons_no):
            self.state[i] = HopfieldNetwork.CHAR_TO_INT[pattern[i]]

    # Reset the state of the Hopfield Network to a random pattern
    def random_reset(self):
        for i in range(self.neurons_no):
            self.state[i] = 1 - 2* numpy.random.randint(0, 2)

    def to_string(self):
        return HopfieldNetwork.state_to_string(self.state)

    @staticmethod
    def state_to_string(state):
        return "".join([HopfieldNetwork.INT_TO_CHAR[c] for c in state])

    @staticmethod
    def state_from_string(str_state):
        return numpy.array([HopfieldNetwork.CHAR_TO_INT[c] for c in str_state])

    # display the current state of the HopfieldNetwork
    def display_as_matrix(self, rows_no, cols_no, msg=None):
        assert(rows_no * cols_no == self.neurons_no)
        if msg:
            print(msg)
        HopfieldNetwork.display_state_as_matrix(self.state, rows_no, cols_no)

    # display the current state of the HopfieldNetwork
    def display_as_image(self, rows_no, cols_no):
        assert(rows_no * cols_no == self.neurons_no)
        
        pixels = [1 if s > 0 else 0 for s in self.state]
        img = Image.new('1', (rows_no, cols_no))
        img.putdata(pixels)
        img.show()

    @staticmethod
    def display_state_as_matrix(state, rows_no, cols_no):
        assert(state.size == rows_no * cols_no)
        print("")
        for i in range(rows_no):
            print("".join([HopfieldNetwork.INT_TO_CHAR[state[i*cols_no+j]]
                           for j in range(cols_no)]))
        print("")
