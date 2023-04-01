import random

from agents.neural_utils.genetic.IGenomeGenerator import IGenomeGenerator


class C51WeightGenerator(IGenomeGenerator):
    def set_generators(self):
        part_generator = [
            # Alpha - learning rate in range [0.05, 0.005]
            lambda: float(random.randint(5, 50)) / 1000,
            # Gamma - discount rate in range [0.01, 0.99]
            lambda: float(random.randint(0, 98)+1) / 100,
            # Batch size in size of 2^n for n [2, 9] generating numbers
            # 4, 8, 16, 32, 64, 128, 256, 512
            lambda: 2 ** (random.randint(2, 9)),
            # Epsilon - action randomness in range [0.05, 0.5]
            lambda: float(random.randint(5, 50)) / 100,
            # Grid size of distributions
            lambda: 51,
            # Tau in range [0, 1]
            lambda: float(random.randint(0, 100)) / 100,
            # Variance setting bounds for distribution in range [1, 15]
            lambda: random.randint(1, 15),
            # Neural net hidden layer size
            lambda: random.randint(1, 24) * 16,

        ]
        return part_generator
