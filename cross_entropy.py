import numpy as np

def cross_entropy(ps, qs):
    k = zip(ps, qs)
    return -np.sum([p * np.log(q) for p, q in k])
