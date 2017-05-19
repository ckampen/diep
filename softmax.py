import numpy as np

def softmax(z):
    k = n.exp(z)
    return np.array([n.exp(j)/k for j in z])
