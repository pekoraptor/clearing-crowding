import numpy as np


def ackley(x, a=20, b=0.2, c=2 * np.pi):
    return (
        -a * np.exp(-b * np.sqrt(1 / len(x) * sum(xi**2 for xi in x)))
        - np.exp(1 / len(x) * sum(np.cos(c * xi) for xi in x))
        + a
        + np.exp(1)
    )


def rosenbrock(x, a=1, b=100):
    return sum(
        b * (x[i + 1] - x[i] ** 2) ** 2 + (a - x[i]) ** 2 for i in range(0, len(x) - 1)
    )


def rastrigin(x, a=10):
    return a * len(x) + sum(xi**2 - a * np.cos(2 * np.pi * xi) for xi in x)


if __name__ == "__main__":
    rosenbrock([1])
