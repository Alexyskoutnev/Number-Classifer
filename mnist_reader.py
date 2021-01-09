import pickle
import gzip
import numpy as np

def read_data():
    """
    Reads the mnist file containing handwritten samples of number 0-9
    :return: training data, validation data, and test data
    """
    file = gzip.open("../data/mnist.pkl.gz", "rb")
    training_data, validation_data, test_data = pickle.load(file)
    file.close()
    return training_data, validation_data, test_data

def load_data_wrapper():
    """
    Uses read_data to transform data into tuple for convenient use in neural network
    :return: Returns the data in tuple format
    """
    train, validation, test = read_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in train[0]]
    training_outputs = [vectorized(y) for y in train[1]]
    training_data = zip(training_inputs, training_outputs)
    validation_inputs = [np.reshape(x, (784, 1)) for x in validation[0]]
    validation_data = zip(validation_inputs, validation[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in test[0]]
    test_data = zip(test_inputs, test[1])
    return training_data, validation_data, test_data

def vectorized(j):
    """
    return a (10,1) numpy array with the jth position equal to 1 (the actual digit)
    and zero else where.
    :param y: Handwritten Digit
    :return: Numpy vectorized array
    """
    e = np.zeros((10,1))
    e[j] = 1.0
    return e