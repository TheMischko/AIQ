import random

from agents.neural_utils.genetic.IGenomeGenerator import IGenomeGenerator


class TestGenerator(IGenomeGenerator):

    def set_generators(self):
        part_generators = [
            lambda: (random.random()-0.5) * 10.,
            lambda: (random.random()-0.5) * 2.,
            lambda: random.random() * 10,
            lambda: random.randint(50, 99),
        ]
        return part_generators
