
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

class MnistData(object):
    def __init__(self):
        # Download images and labels into
        # mnist.test (10K images+labels)
        # and mnist.train (60K images+labels)
        self.mnist = read_data_sets("data", one_hot=True,
                                    reshape=False,
                                    validation_size=0)
        # input X: 28x28 grayscale images,
        # the first dimension (None) will index the images in the mini-batch
        self.test_X = self.mnist.test.images
        self.test_Y = self.mnist.test.labels

    def next_batch(self, n):
        return self.mnist.train.next_batch(100)

    def size(self):
        return self.mnist.train.images.shape[0]+1

