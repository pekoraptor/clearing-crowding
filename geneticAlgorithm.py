import random
from collections.abc import Callable


def mutation(population, mut_prob, possible_genes):
    mutated_population = []

    for individual in population:
        mutated_individual = ""
        for trait in individual:
            if random.random() < mut_prob:
                possible_genes = possible_genes.remove(trait)
                mutated_individual += random.choice(possible_genes)
            else:
                mutated_individual += trait
        mutated_population.append(mutated_individual)

    return mutated_population


def crossover(
    population,
    cross_prob,
):
    new_population = []
    is_pair = False
    left_individual = ""
    individual_size = len(population[0])

    for individual in population:
        if is_pair:
            drawn_num = random.random()
            if drawn_num < cross_prob:
                drawn_point = random.randint(1, individual_size - 2)
            else:
                drawn_point = individual_size
            new_population.append(
                left_individual[:drawn_point] + individual[drawn_point:]
            )
            new_population.append(
                individual[:drawn_point] + left_individual[drawn_point:]
            )
        else:
            left_individual = individual

        is_pair = not is_pair

    return new_population


class gaSolver:
    def __init__(
        self,
        mutation_func: Callable[[list, float, list], list],
        crossover_func: Callable[[list, float], list],
        max_it: int = 1000,
        cross_prob: float = 0.01,
        mut_prob: float = 0.01,
        population_size: int = 100,
        individual_size: int = 200,
        possible_genes: list = ["0", "1"],
    ):
        self.max_it = max_it
        self.cross_prob = cross_prob
        self.mut_prob = mut_prob
        self.population_size = population_size
        self.individual_size = individual_size
        self.possible_genes = possible_genes
        self.mutation = mutation_func
        self.crossover = crossover_func

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
                "".join(
                    random.choice(self.possible_genes)
                    for _ in range(self.individual_size)
                )
            )
        return population

    def evaluation(self, cost_func, population):
        scores = []
        for individual in population:
            scores.append(cost_func(individual))

        return scores

    def find_best(self, population, scores):
        best_id = scores.index(max(scores))
        return population[best_id], scores[best_id]

    def selection(self, population, scores, scaling_const=0):
        intervals_prob = []
        new_population = []
        last_score = 0

        for score in scores:
            intervals_prob.append(score + scaling_const + last_score)
            last_score += score + scaling_const

        for _ in range(len(population)):
            drawn_num = random.randint(0, last_score)
            left_end = intervals_prob[0]

            if drawn_num <= left_end:
                new_population.append(population[0])

            else:
                for i in range(1, len(population)):
                    if drawn_num > left_end and drawn_num <= intervals_prob[i]:
                        new_population.append(population[i])
                        break
                    left_end = intervals_prob[i]

        return new_population

    def solve(self, problem, pop0, scaling_const=0):
        best_scores = []
        avg_hist = []

        scores = self.evaluation(problem, pop0)
        best_individual, best_score = self.find_best(pop0, scores)
        population = pop0

        for _ in range(self.max_it):
            selected = self.selection(population, scores, scaling_const)
            mutated = self.mutation(
                self.crossover(selected, self.cross_prob),
                self.mut_prob,
                self.possible_genes,
            )
            scores = self.evaluation(problem, mutated)

            new_best_individual, new_best_score = self.find_best(mutated, scores)
            if new_best_score > best_score:
                best_score = new_best_score
                best_individual = new_best_individual

            population = mutated

        return best_individual, best_score
