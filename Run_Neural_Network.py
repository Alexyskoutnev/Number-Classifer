from Network import Network
from mnist_reader import load_data_wrapper
from Read_Digit import read

import argparse
import sys
import os




def get_args():
    parser = argparse.ArgumentParser(description='Train the Perception network on gridmap and features')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1.0e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--dataset-size', '-db', dest='db_limit', type=int, default=6, help='Number of datafiles imported to observations')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

network = Network([784, 30, 10])
training_data, validation_data, test_data = load_data_wrapper()
network.SGN(training_data, args.epochs, args.batch_size, .5, lmda= 5.0, test_data= test_data)
read(network)

