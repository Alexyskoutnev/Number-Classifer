from Network import Network
from mnist_reader import load_data_wrapper
from Read_Digit import read


network = Network([784, 50, 10])
training_data, validation_data, test_data = load_data_wrapper()
network.SGN(training_data, 2, 10, 1, test_data= test_data)
read(network)