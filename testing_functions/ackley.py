import numpy as np


def ackley(x, a=20, b=0.2, c=2 * np.pi):
    return (
        -a * np.exp(-b * np.sqrt(1 / len(x) * sum(xi**2 for xi in x)))
        - np.exp(1 / len(x) * sum(np.cos(c * xi) for xi in x))
        + a
        + np.exp(1)
    )
