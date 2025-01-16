from individual import Individual, RealValuedIndividual
from selection import tournament_selection
import random
from collections.abc import Callable
import heapq
import itertools
import numpy as np
import copy


class gaSolver:
    def __init__(
        self,
        selection_func: Callable[[list, list, float], list] = tournament_selection,
        individual_class: Individual = RealValuedIndividual,
        max_it: int = 1000,
        cross_prob: float = 0.01,
        mut_prob: float = 0.01,
        population_size: int = 100,
        individual_size: int = 2,
        elite_size: int = 1,
        possible_genes: list = (-5, 5),
    ):
        self.selection = selection_func
        self.individual_class = individual_class
        self.max_it = max_it
        self.cross_prob = cross_prob
        self.mut_prob = mut_prob
        self.population_size = population_size
        self.individual_size = individual_size
        self.elite_size = elite_size
        self.possible_genes = possible_genes
        self.average_scores = []
        self.best_scores = []
        self.population_diversities = []
        self.best_individual = None

    def population_avg(self, population):
        return sum(individual.score for individual in population) / len(population)

    def population_diversity(self, population):
        pair_distances = [
            np.linalg.norm(ind1.traits - ind2.traits)
            for ind1, ind2 in itertools.combinations(population, 2)
        ]
        avg_distance = np.mean(pair_distances)
        return avg_distance

    def get_parameters(self):
        return {
            "max_it": self.max_it,
            "cross_prob": self.cross_prob,
            "mut_prob": self.mut_prob,
            "population_size": self.population_size,
            "individual_size": self.individual_size,
        }

    def init_population(self):
        population = []
        for _ in range(self.population_size):
            population.append(
                self.individual_class(self.individual_size, self.possible_genes)
            )
        return population

    def evaluation(self, cost_func, population):
        for individual in population:
            individual.evaluate(cost_func)

    def find_best(self, population, n_best=1):
        return heapq.nsmallest(1, population, key=lambda individual: individual.score)[
            0
        ]

    def mutation(self, population):
        for individual in population:
            individual.mutate(self.mut_prob)

    def measure_population(
        self,
        population,
        is_measuring_avg,
        is_measuring_diversity,
        is_measuring_best,
        is_saving_path,
    ):
        if is_measuring_avg:
            self.average_scores.append(self.population_avg(population))
        if is_measuring_diversity:
            self.population_diversities.append(self.population_diversity(population))
        if is_measuring_best:
            self.best_scores.append(self.best_individual.score)
        if is_saving_path:
            self.path.append([ind.traits for ind in population])

    def apply_crowding(
        self, new_population, old_population, crowding_func, cf_size, cf_threshold
    ):
        if not crowding_func:
            return new_population

        result_population = copy.deepcopy(old_population)

        for offspring in new_population:
            replaced_individual = crowding_func(
                offspring, result_population, cf_size, cf_threshold
            )
            closest_idx = min(
                range(len(result_population)),
                key=lambda i: np.linalg.norm(
                    result_population[i].traits - replaced_individual.traits
                ),
            )
            result_population[closest_idx] = copy.deepcopy(offspring)

        return result_population

    def apply_clearing(
        self, population, clearing_func, clearing_radius, niche_capacity
    ):
        if clearing_func:
            return clearing_func(population, clearing_radius, niche_capacity)
        return population

    def crossover(self, population):
        new_population = []
        for i in range(len(population)):
            if random.random() < self.cross_prob:
                cross_index = random.randint(0, len(population) - 1)
                new_population.append(population[i] + population[cross_index])
            else:
                new_population.append(population[i])
        return new_population

    def clear_measurements(self):
        self.average_scores = []
        self.best_scores = []
        self.population_diversities = []
        self.path = []
        self.best_individual = None

    def solve(
        self,
        problem,
        pop0=None,
        cf_size=3,
        crowding_threshold=1,
        clearing_func=None,
        clearing_radius=1,
        niche_capacity=5,
        crowding_func=None,
        is_measuring_diversity=False,
        is_measuring_avg=False,
        is_measuring_best=False,
        is_saving_path=False,
    ):
        self.clear_measurements()

        if pop0 is None:
            pop0 = self.init_population()

        population = pop0
        self.evaluation(problem, population)
        self.best_individual = self.find_best(population)

        if is_measuring_avg or is_measuring_diversity:
            self.measure_population(
                pop0,
                is_measuring_avg,
                is_measuring_diversity,
                is_measuring_best,
                is_saving_path,
            )

        for _ in range(self.max_it):
            selected = self.selection(population)
            crossed = self.crossover(selected)
            self.mutation(crossed)

            self.evaluation(problem, crossed)

            population = self.apply_crowding(
                crossed, population, crowding_func, cf_size, crowding_threshold
            )
            population = self.apply_clearing(
                population, clearing_func, clearing_radius, niche_capacity
            )

            new_best_individual = self.find_best(population)
            if new_best_individual.score < self.best_individual.score:
                self.best_individual = copy.deepcopy(new_best_individual)

            self.measure_population(
                population,
                is_measuring_avg,
                is_measuring_diversity,
                is_measuring_best,
                is_saving_path,
            )

        return self.best_individual
