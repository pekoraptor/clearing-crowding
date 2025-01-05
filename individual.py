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


class RealValuedIndividual(Individual):
    def __init__(self, size: int, possibleGenes: tuple[float, float]):
        super().__init__()
        self.traits = [self.random_trait() for _ in range(size)]
        self.possibleGenes = possibleGenes

    def random_trait(self) -> float:
        return random.uniform(self.possibleGenes[0], self.possibleGenes[1])


def mutate(self, mut_prob: float, sigma: float = 0.1):
    for id in range(len(self.traits)):
        if random.random() < mut_prob:
            self.traits[id] += random.gauss(0, sigma)
            self.traits[id] = max(
                self.possibleGenes[0], min(self.traits[id], self.possibleGenes[1])
            )

    def evaluate(self, costFunction: Callable[[list], float]):
        self.score = costFunction(self.traits)

    def __repr__(self):
        return super().__repr__() + f" and traits: {self.traits}"
