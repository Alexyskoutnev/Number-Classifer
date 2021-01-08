import numpy as np
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
    def SGN(self):
        pass
    def sigmoid(self, z):
        '''
        :param z: input numpy array of size (n,1)
        :return: output sigmoidized numpy array of size (n,1)
        '''
        return 1/(1 + np.exp(-z))

network = Network([2,3,2])
# print(network.biases)
# print(network.weights)
print(network.sigmoid(np.array([1,2,3,4])))