from Network import Network
from mnist_reader import load_data_wrapper


network = Network([784, 50, 10])
training_data, validation_data, test_data = load_data_wrapper()
network.SGN(training_data, 40, 15, 1, test_data= test_data)