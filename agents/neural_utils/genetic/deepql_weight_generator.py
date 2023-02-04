import random

from agents.neural_utils.genetic.IGenomeGenerator import IGenomeGenerator


class DeepQLWeightGenerator(IGenomeGenerator):
    def set_generators(self):
        part_generator = [
            # Learning rate in range [0.005, 0.0005]
            lambda: float(random.randint(50, 500)) / 100000,
            # Gamma - discount rate in range [0.1, 0.9]
            lambda: float(random.randint(10, 90)) / 100,
            # Starting epsilon in range [0.1, 1]
            lambda: float(random.randint(10, 100)) / 100,
            # Batch size in size of 2^n for n [2, 9] generating numbers
            # 4, 8, 16, 32, 64, 128, 256, 512
            lambda: 2**(random.randint(2, 9)),
            # Epsilon decay length - number of steps till epsilon is min value
            # values are in range [200, 1000]
            lambda: random.randint(20, 100) * 10,
            # Tau in range [0, 1]
            lambda: float(random.randint(0, 100)) / 100,
            # Update interval length in length [10, 1000]
            lambda: random.randint(1, 100) * 10

        ]
        return part_generator
