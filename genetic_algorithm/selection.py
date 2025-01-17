import copy
import random


def tournament_selection(population, tournament_size=3, diversity_factor=0.1):
    new_population = []

    for _ in range(len(population)):
        tournament = random.sample(population, tournament_size)
        tournament.sort(key=lambda individual: individual.score)

        if random.random() < diversity_factor:
            winner = random.choice(tournament)
        else:
            winner = tournament[0]

        new_population.append(copy.deepcopy(winner))

    return new_population
