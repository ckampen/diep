import numpy as np

#Multiply Vector w with Matrix x and add Vector b
def calc_layer(w, x, b):
    return np.dot(w,x) + b
