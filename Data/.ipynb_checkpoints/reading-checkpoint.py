# code from https://www.kaggle.com/code/hojjatk/read-mnist-dataset
# reading "MNIST Dataset"

import numpy as np # linear algebra
import struct
from array import array
from os.path  import join
import os
import random
import matplotlib.pyplot as plt

#os.chdir('C:\\Users\\malvarez\\Documents\\GitHub\\model_LeNET5\\Data\\')
#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = 'train-images.idx3-ubyte'
        self.training_labels_filepath = 'train-labels.idx1-ubyte'
        self.test_images_filepath = 't10k-images.idx3-ubyte'
        self.test_labels_filepath = 't10k-labels.idx1-ubyte'
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)  

    def sampling (self, tr_samples, ts_samples, x_train, y_train, x_test, y_test):
        X_train = []
        X_test = []
        Y_train = []
        Y_test = []
    
        for i in range(0, tr_samples):
            r = random.randint(0, 60000)
            X_train.append(x_train[r])
            Y_train.append(y_train[r])
        
        for i in range(0, ts_samples):
            r = random.randint(1, 10000)
            X_test.append(x_test[r])
            Y_test.append(y_test[r])
        return X_train, Y_train, X_test, Y_test