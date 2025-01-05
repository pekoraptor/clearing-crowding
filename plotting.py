import numpy as np
import matplotlib.pyplot as plt
from testing_functions import *


def plt_function(f, width=10, additionalPoints=None, colors=None):
    x = np.linspace(-width, width, 100)
    y = f([x])

    if additionalPoints:
        for index, setOfPoints in enumerate(additionalPoints):
            additional_x, additional_y = zip(*setOfPoints)
            plt.scatter(
                additional_x,
                additional_y,
                c=colors[index % len(colors)],
                marker="o",
                label=f"x0 = {additional_x[0]}",
                s=15,
            )

    plt.plot(x, y)
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.show()


def plt_function3D(
    func, additionalPoints=None, pointsColors=None, pltCmap="PiYG", visibility=1
):
    x1 = np.linspace(-2, 2, 100)
    x2 = np.linspace(-2, 2, 100)
    x1, x2 = np.meshgrid(x1, x2)
    y = np.zeros_like(x1)

    for i in range(x1.shape[0]):
        for j in range(x1.shape[1]):
            y[i, j] = func([x1[i, j], x2[i, j]])

    figure = plt.figure()
    plot = figure.add_subplot(111, projection="3d")
    plot.plot_surface(x1, x2, y, cmap=pltCmap, alpha=visibility)

    if additionalPoints:
        for index, setOfPoints in enumerate(additionalPoints):
            additionalX, additionalY = zip(*setOfPoints)
            additionalX1, additionalX2 = zip(*additionalX)
            plot.scatter(
                additionalX1,
                additionalX2,
                additionalY,
                c=pointsColors[index % len(pointsColors)],
                s=15,
                label=f"x1, x2 = {additionalX[0]}",
            )

    plot.set_xlabel("x1")
    plot.set_ylabel("x2")
    plot.set_zlabel("g(x1, x2)")
    plot.set_title("3D Plot of g(x1, x2)")

    plt.legend()
    plt.show()


def plt_heatmap(
    func,
    additionalPoints=None,
    pointsColors=None,
    pltCmap="viridis",
    visibility=1,
):
    x1 = np.linspace(-2, 2, 100)
    x2 = np.linspace(-2, 2, 100)
    x1, x2 = np.meshgrid(x1, x2)

    z = np.zeros_like(x1)
    for i in range(x1.shape[0]):
        for j in range(x1.shape[1]):
            point = [x1[i, j], x2[i, j]]
            z[i, j] = func(point)

    figure = plt.figure()

    plot = figure.add_subplot(111)
    heatmap = plot.pcolormesh(x1, x2, z, cmap=pltCmap, shading="auto", alpha=visibility)
    figure.colorbar(heatmap, ax=plot, label="f(x1, x2)")

    if additionalPoints:
        for point in additionalPoints:
            plot.scatter(point[0], point[1], c="red", s=15, marker="o")

    plot.set_xlabel("x")
    plot.set_ylabel("y")
    plot.set_title("2D Plot")

    plt.show()


if __name__ == "__main__":
    plt_function3D(ackley)
    # plt_heatmap(ackley)
