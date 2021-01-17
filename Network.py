import numpy as np
import random

class CrossEntropyCost():

    @staticmethod
    def cost(a, y):
        """
        :param a: Result output
        :param y: Actual output
        :return: Cost between Result and Actual
        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(self, a, y):
        """
        :param a: Result output
        :param y: Actual output
        :return: Error Delta
        """
        return (a-y)

class QuadraticCost():

    @staticmethod
    def cost(a, y):
        """
        :param a: Result output
        :param y: Actual output
        :return: Cost between Result and Actual
        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(self, z, a, y):
        """
        :param z: Activation input
        :param a: Result output
        :param y: Actual output
        :return: Error Delta
        """
        return (a-y) * self.sigmoid_prime(z)

class Network:

    def __init__(self, sizes, cost = CrossEntropyCost):
        self.num_layer = len(sizes)
        self.sizes = sizes
        self.Default_Weight_Initializer()
        self.cost = cost


    def Default_Weight_Initializer(self):
        """
        Uses a Gaussian distribution with mean 0 and std of 1/sqrt(n_in) (input weights)
        to find the random weights and biases
        :return: weights and biases
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def Large_Weight_Initializer(self):
        """
        Uses a Gaussian distribution with mean 0 and std of 1
        to find the random weights and biases
        :return: weights and biases
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, v):
        """
        :param v: input vector of size (n,1)
        :return: output vector of neural network
        """
        for b, w in zip(self.biases, self.weights):
            v = sigmoid(np.dot(w, v) + b)
        return v

    def SGN(self, training_data, epochs, mini_sample_size, eta, lmda = 0, test_data = None):
        '''
        :param training_data: the training data-set
        :param epochs: Number of epochs to train for
        :param mini_sample_size: The size of sample backpropagation
        :param eta: learning rate
        :param test_data: the test data-set
        :return: finds exterma of cost function
        '''
        lmda = 0
        if test_data:
            n_test = len(test_data)
        n_train = len(training_data)
        for i in range(epochs):
            random.shuffle(training_data)
            mini_samples = [training_data[j:j+mini_sample_size] for j in range(0, n_test, mini_sample_size)]
            for sample in mini_samples:
                self.update_mini_sample(sample, eta, lmda, len(training_data))
            if test_data:
                print("Epoch {0}: {1:.02f}%".format(i, (self.evaluate(test_data)/n_test*100)))
            else:
                print("Epoch {0} complete".format(i))

    def update_mini_sample(self, sample, eta, lmda, n):
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
        self.weights = [(1-eta*(lmda/n))*w - (eta/len(sample)) for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(sample))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        :param x: Training Data
        :param y: Training Output
        :return: Partial derivatives of the biases and weights
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layer):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        :param test_data: Tests the Neural Network using the each data point in test data
        :return: returns the number of correctly identified data points
        """
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

#Helper Functions
def sigmoid(z):
    '''
    :param z: input numpy array of size (n,1)K
    :return: output sigmoidized numpy array of size (n,1)
    '''
    return 1/(1 + np.exp(-z))

def sigmoid_prime(z):
    """
    :param z: input numpy of size (n,1)
    :return: output derivative of sigmoidized numpy array of size (n,1)
    """
    return sigmoid(z)*(1 - sigmoid(z))

# network = Network([10,10,10])
# print(network.feedforward([1,2,3,4,5,6,7,8,9,10]))