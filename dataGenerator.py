import numpy as np


def normal(mean, var, size):
    U = np.random.uniform(0, 1, size)
    V = np.random.uniform(0, 1, size)
    X = np.sqrt(var) * (np.sqrt(-2 * np.log(U)) * np.cos(2 * np.pi * V)) + mean
    return X
