from testing_functions import *
from genetic_algorithm.genetic_algorithm import gaSolver
from genetic_algorithm.clearing import clearing
from genetic_algorithm.crowding import crowding
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
    x1 = np.linspace(-3, 3, 1000)
    x2 = np.linspace(-3, 3, 1000)
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


def average_results(n, track_best_individuals=True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            all_diversities = []
            all_averages = []
            all_best_scores = []
            best_individuals_per_param = None

            for _ in range(n):
                diversities, averages, best_scores, best_individuals = func(
                    *args, **kwargs
                )
                all_diversities.append(diversities)
                all_averages.append(averages)
                all_best_scores.append(best_scores)

                if track_best_individuals:
                    if best_individuals_per_param is None:
                        best_individuals_per_param = best_individuals
                    else:
                        for i, score in enumerate(best_scores):
                            if score > max(all_best_scores, key=lambda x: x[i])[i]:
                                best_individuals_per_param[i] = best_individuals[i]

            avg_diversities = np.mean(all_diversities, axis=0)
            avg_averages = np.mean(all_averages, axis=0)
            avg_best_scores = np.mean(all_best_scores, axis=0)

            return (
                avg_diversities,
                avg_averages,
                avg_best_scores,
                best_individuals_per_param,
            )

        return wrapper

    return decorator


@average_results(n=10)
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
                is_measuring_best=True,
            )
        )

        diversities.append(copy.deepcopy(ga.population_diversities))
        averages.append(copy.deepcopy(ga.average_scores))
        best_scores.append(copy.deepcopy(ga.best_scores))

    return diversities, averages, best_scores, best_individuals


@average_results(n=10, track_best_individuals=False)
def compare_diversity_best(problem, n_dims=2):
    ga = gaSolver(individual_size=n_dims)
    pop0 = ga.init_population()

    ga.solve(
        problem,
        pop0=copy.deepcopy(pop0),
        is_measuring_diversity=True,
        is_measuring_best=True,
    )
    base_diversities = copy.deepcopy(ga.population_diversities)
    base_best_scores = copy.deepcopy(ga.best_scores)

    ga.solve(
        problem,
        pop0=copy.deepcopy(pop0),
        clearing_func=clearing,
        is_measuring_diversity=True,
        is_measuring_best=True,
    )
    clearing_diversities = copy.deepcopy(ga.population_diversities)
    clearing_best_scores = copy.deepcopy(ga.best_scores)

    ga.solve(
        problem,
        pop0=copy.deepcopy(pop0),
        crowding_func=crowding,
        is_measuring_diversity=True,
        is_measuring_best=True,
    )
    crowding_diversities = copy.deepcopy(ga.population_diversities)
    crowding_best_scores = copy.deepcopy(ga.best_scores)

    averages = [0] * len(base_diversities)
    best_individuals = [None] * len(base_best_scores)

    return (
        (
            base_diversities,
            clearing_diversities,
            crowding_diversities,
        ),
        (
            averages,
            averages,
            averages,
        ),
        (
            base_best_scores,
            clearing_best_scores,
            crowding_best_scores,
        ),
        best_individuals,
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
