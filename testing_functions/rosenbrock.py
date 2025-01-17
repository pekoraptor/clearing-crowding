def rosenbrock(x, a=1, b=100):
    return sum(
        b * (x[i + 1] - x[i] ** 2) ** 2 + (a - x[i]) ** 2 for i in range(0, len(x) - 1)
    )
