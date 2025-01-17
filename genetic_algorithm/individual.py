import random
import numpy as np
from collections.abc import Callable
from abc import ABC


class Individual(ABC):
    def __init__(self):
        self.score = 0

    def mutate(self):
        pass

    def evaluate(self, costFunction: Callable[[list], float]) -> float:
        pass

    def __lt__(self, other: "Individual") -> bool:
        return self.score < other.score

    def __gt__(self, other: "Individual") -> bool:
        return self.score > other.score

    def __eq__(self, other: "Individual") -> bool:
        return self.score == other.score

    def __repr__(self):
        return f"Individual with score {self.score}"

    def __len__(self):
        return len(self.traits)

    def __getitem__(self, key):
        return self.traits[key]


class RealValuedIndividual(Individual):
    def __init__(self, size: int, possible_genes: tuple[float, float], traits=None):
        super().__init__()
        self.possible_genes = possible_genes
        if traits is not None:
            self.traits = traits
        else:
            self.traits = np.array([self.random_trait() for _ in range(size)])

    def random_trait(self) -> float:
        return random.uniform(self.possible_genes[0], self.possible_genes[1])

    def mutate(self, mut_prob: float, sigma: float = 1):
        for id in range(len(self.traits)):
            if random.random() < mut_prob:
                self.traits[id] += random.gauss(0, sigma)
                self.traits[id] = max(
                    self.possible_genes[0], min(self.traits[id], self.possible_genes[1])
                )

    def crossover(
        self, other: "RealValuedIndividual", offset=0.1
    ) -> "RealValuedIndividual":
        child_traits = np.array(
            [
                random.uniform(0.5 - offset, 0.5 + offset) * left
                + random.uniform(0.5 - offset, 0.5 + offset) * right
                for left, right in zip(self.traits, other.traits)
            ]
        )

        return RealValuedIndividual(
            len(self.traits), self.possible_genes, traits=child_traits
        )

    def evaluate(self, costFunction: Callable[[list], float]):
        self.score = costFunction(self.traits)

    def __add__(self, other: "RealValuedIndividual") -> "RealValuedIndividual":
        return self.crossover(other)

    def __repr__(self):
        return super().__repr__() + f" and traits: {self.traits}"
