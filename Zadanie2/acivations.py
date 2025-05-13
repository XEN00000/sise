import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(sig_x):
    return sig_x * (1 - sig_x)
