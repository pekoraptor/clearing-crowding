from testing_functions import *
from genetic_algorithm import gaSolver
from clearing import clearing
from crowding import crowding
import numpy as np
import matplotlib.pyplot as plt
import copy
import pandas as pd


def plt_function(f, ax, width=10, additionalPoints=None, colors=None):
    x = np.linspace(-width, width, 100)
    y = f([x])

    if additionalPoints:
        for index, setOfPoints in enumerate(additionalPoints):
            additional_x, additional_y = zip(*setOfPoints)
            ax.scatter(
                additional_x,
                additional_y,
                c=colors[index % len(colors)],
                marker="o",
                label=f"x0 = {additional_x[0]}",
                s=15,
            )

    ax.plot(x, y)
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Plot of f(x)")
    ax.grid(True)
    ax.legend()


def plt_function3D(
    func, ax, additionalPoints=None, pointsColors=None, pltCmap="PiYG", visibility=1
):
    x1 = np.linspace(-3, 3, 100)
    x2 = np.linspace(-3, 3, 100)
    x1, x2 = np.meshgrid(x1, x2)
    y = np.zeros_like(x1)

    for i in range(x1.shape[0]):
        for j in range(x1.shape[1]):
            y[i, j] = func([x1[i, j], x2[i, j]])

    ax.plot_surface(x1, x2, y, cmap=pltCmap, alpha=visibility)

    if additionalPoints:
        for index, setOfPoints in enumerate(additionalPoints):
            additionalX, additionalY = zip(*setOfPoints)
            additionalX1, additionalX2 = zip(*additionalX)
            ax.scatter(
                additionalX1,
                additionalX2,
                additionalY,
                c=pointsColors[index % len(pointsColors)],
                s=15,
                label=f"x1, x2 = {additionalX[0]}",
            )

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x1, x2)")
    ax.set_title("Plot of f(x1, x2)")


def plt_heatmap(
    func,
    ax,
    fig,
    additionalPoints=None,
    pointsColors=None,
    pltCmap="viridis",
    visibility=1,
):
    x1 = np.linspace(-3, 3, 100)
    x2 = np.linspace(-3, 3, 100)
    x1, x2 = np.meshgrid(x1, x2)

    z = np.zeros_like(x1)
    for i in range(x1.shape[0]):
        for j in range(x1.shape[1]):
            point = [x1[i, j], x2[i, j]]
            z[i, j] = func(point)

    heatmap = ax.pcolormesh(x1, x2, z, cmap=pltCmap, shading="auto", alpha=visibility)

    if additionalPoints:
        for point in additionalPoints:
            ax.scatter(point[0], point[1], c="red", s=15, marker="o")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Top down view of f(x1, x2)")
    fig.colorbar(heatmap, ax=ax, label="f(x1, x2)")


def plt_all(func, plot3D=True, heatmap=True, plot2D=True):
    fig = plt.figure(figsize=(18, 6))

    if plot3D:
        ax1 = fig.add_subplot(131, projection="3d")
        plt_function3D(func, ax=ax1)

    if heatmap:
        ax2 = fig.add_subplot(132)
        plt_heatmap(func, ax=ax2, fig=fig)

    if plot2D:
        ax3 = fig.add_subplot(133)
        plt_function(func, ax=ax3)

    plt.tight_layout()
    plt.show()


def plt_multiple_plots(ax, data_arr, labels, title):
    for data in data_arr:
        ax.plot(data)
    ax.legend(labels)
    ax.set_title(title)
    return ax


def plt_multiple_plots_side_by_side(data_arr, labels, titles):
    fig, axs = plt.subplots(1, len(data_arr), figsize=(18, 5))
    for i, data in enumerate(data_arr):
        axs[i] = plt_multiple_plots(axs[i], data, labels, titles[i])
    plt.show()


def compare_evolutionary_parameter(
    problem,
    parameter_values,
    param_name,
    func,
    mut_prob=0.05,
    cross_prob=0.1,
):
    ga = gaSolver(mut_prob=mut_prob, cross_prob=cross_prob)
    population = ga.init_population()
    diversities = []
    averages = []
    best_scores = []
    best_individuals = []

    for param_value in parameter_values:
        kwargs = {param_name: param_value}

        best_individuals.append(
            ga.solve(
                problem,
                **kwargs,
                crowding_func=(
                    func if param_name in ["cf_size", "crowding_threshold"] else None
                ),
                clearing_func=(
                    func
                    if param_name in ["clearing_radius", "niche_capacity"]
                    else None
                ),
                pop0=copy.deepcopy(population),
                is_measuring_diversity=True,
                is_measuring_avg=True,
            )
        )

        diversities.append(copy.deepcopy(ga.population_diversities))
        averages.append(copy.deepcopy(ga.average_scores))
        best_scores.append(copy.deepcopy(ga.best_scores))

    return diversities, averages, best_scores, best_individuals


def plt_compare_diversity(problem, mut_prob=0.05, cross_prob=0.1):
    ga = gaSolver(mut_prob=mut_prob, cross_prob=cross_prob)
    ga.solve(problem, is_measuring_diversity=True)
    base_diversities = copy.deepcopy(ga.population_diversities)

    ga.solve(problem, clearing_func=clearing, is_measuring_diversity=True)
    clearing_diversities = copy.deepcopy(ga.population_diversities)

    ga.solve(problem, crowding_func=crowding, is_measuring_diversity=True)
    crowding_diversities = copy.deepcopy(ga.population_diversities)

    ga.solve(
        problem,
        clearing_func=clearing,
        crowding_func=crowding,
        is_measuring_diversity=True,
    )
    clearing_crowding_diversities = copy.deepcopy(ga.population_diversities)

    plt_multiple_plots(
        [
            base_diversities,
            clearing_diversities,
            crowding_diversities,
            clearing_crowding_diversities,
        ],
        ["Base", "Clearing", "Crowding", "Clearing and Crowding"],
    )


def table_compare_best(individual_list, labels):
    table_data = {}
    for i, individual in enumerate(individual_list):
        table_data[labels[i]] = {
            "x": individual.traits[0],
            "y": individual.traits[1],
            "score": individual.score,
        }
    df = pd.DataFrame(table_data, index=["x", "y", "score"])
    return df
