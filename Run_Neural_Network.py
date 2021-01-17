from Network import Network
from mnist_reader import load_data_wrapper
from Read_Digit import read


network = Network([784, 30, 10])
training_data, validation_data, test_data = load_data_wrapper()
network.SGN(training_data, 30, 10, .5, lmda= 5.0, test_data= test_data)
read(network)