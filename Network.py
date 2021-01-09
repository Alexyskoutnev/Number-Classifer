import numpy as np
import random
class Network(object):

    def __init__(self, sizes):
        self.num_layer = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, v):
        """
        :param v: input vector of size (n,1)
        :return: output vector of neural network
        """
        for b, w in zip(self.biases, self.weights):
            v = self.sigmoid(np.dot(w, v) + b)
        return v

    def SGN(self, training_data, epochs, mini_sample_size, eta, test_data = None):
        '''
        :param training_data: the training data-set
        :param epochs: Number of epochs to train for
        :param mini_sample_size: The size of sample backpropagation
        :param eta: learning rate
        :param test_data: the test data-set
        :return: finds exterma of cost function
        '''
        if test_data:
            n_test = len(test_data)
        n_train = len(training_data)
        for i in range(epochs):
            random.shuffle(training_data)
            mini_samples = [training_data[j:j+mini_sample_size] for j in range(0, n_test, mini_sample_size)]
            for sample in mini_samples:
                self.update_mini_sample(sample, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(i, self.evualate(test_data)/ n_test))
            else:
                print("Epoch {0} complete".format(i))

    def update_mini_sample(self, sample, eta):
        """
        :param sample: sample from mini sample
        :param eta: learning rate
        :return: updates biases and weights of each node
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in sample:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta/len(sample))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(sample)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        :param x:
        :param y:
        :return:
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        :param test_data:
        :return:
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        :param output_activations:
        :param y:
        :return:
        """
        return (output_activations - y)

    def sigmoid(self, z):
        '''
        :param z: input numpy array of size (n,1)K
        :return: output sigmoidized numpy array of size (n,1)
        '''
        return 1/(1 + np.exp(-z))

    def sigmoid_prime(self, z):
        """
        :param z: input numpy of size (n,1)
        :return: output derivative of sigmoidized numpy array of size (n,1)
        """
        return self.sigmoid(z)*(1 - self.sigmoid(z))

network = Network([2,3,2])
# print(network.biases)
# print(network.weights)
print(network.sigmoid(np.array([1,2,3,4])))