import math
import numpy as np
import math

def basic_sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x)* (1 - sigmoid(x))

def image2vector(image):
    return image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)

def normalizeRows(x):
    return x/np.linalg.norm(x, axis=1, keepdims=True)

def softmax(x):
    x_exp = np.exp(x)
    return x_exp/np.sum(x_exp, axis=1, keepdims=True)

def L2(yhat, y):
    return np.sum(np.power((y - yhat), 2))