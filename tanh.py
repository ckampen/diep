import numpy as np

def tanh(x):
    k = np.exp(-2 * x)
    return (1 + k) / (1 - k)

def tanh_sigmoid(x):
    return 2 * sigmoid(2 * x) -1
