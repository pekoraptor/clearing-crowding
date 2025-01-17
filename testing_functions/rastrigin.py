import numpy as np


def rastrigin(x, a=10):
    return a * len(x) + sum(xi**2 - a * np.cos(2 * np.pi * xi) for xi in x)
