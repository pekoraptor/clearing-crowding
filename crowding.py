import numpy as np
import random
import copy


def crowding(offspring, population, cf_size=5, distance_threshold=1):
    closest_individual = None
    min_distance = float("inf")
    selected_individuals = random.sample(population, k=cf_size)

    for individual in selected_individuals:
        distance = np.linalg.norm(offspring.traits - individual.traits)
        if distance < min_distance:
            min_distance = distance
            closest_individual = individual

    if closest_individual.score < offspring.score and min_distance < distance_threshold:
        return copy.deepcopy(closest_individual)

    return copy.deepcopy(offspring)
