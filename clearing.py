import numpy as np


def clearing(
    population: list,
    clearing_radius: float = 4,
    niche_capacity: int = 5,
    cleared_score=10000,
):
    population = sorted(population, key=lambda x: x.score)[::-1]

    for i, niche_winner in enumerate(population):
        if niche_winner.score != cleared_score:
            capacity = niche_capacity
            for individual in population[i + 1 :]:
                if (
                    np.linalg.norm(individual.traits - niche_winner.traits)
                    < clearing_radius
                ):
                    individual.score = cleared_score
                    capacity -= 1
                    if capacity == 0:
                        break

    return population
