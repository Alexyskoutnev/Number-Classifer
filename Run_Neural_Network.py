from Network import Network
from mnist_reader import load_data_wrapper
from Read_Digit import read


network = Network([784, 100, 50, 25, 10])
training_data, validation_data, test_data = load_data_wrapper()
network.SGN(training_data, 30, 10, 2, test_data= test_data)
read(network)