import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from genetic_algorithm.genetic_algorithm import gaSolver


def plt_heatmap_animation(
    func, ax, additionalPoints=None, pltCmap="viridis", visibility=1
):
    x1 = np.linspace(-10, 10, 1000)
    x2 = np.linspace(-10, 10, 1000)
    x1, x2 = np.meshgrid(x1, x2)

    z = np.zeros_like(x1)
    for i in range(x1.shape[0]):
        for j in range(x1.shape[1]):
            point = [x1[i, j], x2[i, j]]
            z[i, j] = func(point)

    # Plot the heatmap
    heatmap = ax.pcolormesh(x1, x2, z, cmap=pltCmap, shading="auto", alpha=visibility)

    # Add a colorbar
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Top down view of f(x1, x2)")
    return heatmap


def animate_genetic_algo(path, func):
    fig, ax = plt.subplots(figsize=(6, 6))

    plt_heatmap_animation(func, ax)

    scatter = ax.scatter([], [], c="red", s=8, marker="o")

    def update(frame):
        positions = path[frame]
        x = [p[0] for p in positions]
        y = [p[1] for p in positions]
        scatter.set_offsets(np.c_[x, y])
        ax.set_title(f"Generation {frame + 1}")
        return (scatter,)

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(path),
        interval=1000 / 120,
        repeat=False,
        blit=True,
    )

    plt.close()
    return anim


def generate_animation(problem, title, func=None):
    ga = gaSolver(population_size=100, possible_genes=[-10, 10])
    ga.solve(
        problem,
        crowding_func=(
            func
            if func and func.__name__ in ["crowding", "clearing_crowding"]
            else None
        ),
        clearing_func=(
            func
            if func and func.__name__ in ["clearing", "clearing_crowding"]
            else None
        ),
        is_saving_path=True,
    )
    ani = animate_genetic_algo(ga.path, problem)
    plt.show()

    ani.save(title + ".gif", writer="pillow")
