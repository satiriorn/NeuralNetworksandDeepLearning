import math
import numpy as np

def basic_sigmoid(x):
    s = 1/(1 + math.exp(-x))
    return s
result = basic_sigmoid(10)
print(result)

xv = np.array([1, 2, 3])
print(np.exp(xv))
print(xv + 3)

def sigmoid(x):
    s = 1/(1 + np.exp(-x))
    return s

print(sigmoid(xv))