from Network import Network
from mnist_reader import load_data_wrapper


network = Network([784, 100, 10])
training_data, validation_data, test_data = load_data_wrapper()
network.SGN(training_data, 25, 10, 3, test_data= test_data)