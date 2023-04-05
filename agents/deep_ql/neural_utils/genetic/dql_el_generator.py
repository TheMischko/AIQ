import random

from agents.deep_ql.neural_utils.genetic.IGenomeGenerator import IGenomeGenerator


class DQlElGenerator(IGenomeGenerator):
    def set_generators(self):
        part_generator = [
            # Learning rate in range [0.01, 0.0001] or [0.005, 0.0005]
            lambda: float(random.randint(1, 100)) / 1000 if random.random() > 0.4 else float(
                random.randint(50, 500)) / 100000,
            # Gamma - discount rate in range [0.05, 0.99]
            lambda: float(random.randint(1, 99)) / 100,
            # Batch size in size of 2^n for n [2, 9] generating numbers
            # 4, 8, 16, 32, 64, 128, 256, 512
            lambda: 2 ** (random.randint(2, 9)),
            # Epsilon decay length - number of steps till epsilon is min value
            # values are in range [200, 1000] or [1000, 4000]
            lambda: random.randint(4, 20) * 50 if random.random() > 0.5 else random.randint(10, 40) * 100,
            # Size of neural net layer 1
            lambda: random.randint(1, 16) * 16,
            # Size of neural net layer 2
            lambda: random.randint(1, 24) * 16,
            # Size of neural net layer 3 (can be zero)
            lambda: random.randint(1, 16) * 16 if random.random() > 0.35 else 0,
            # Tau in range [0, 1]
            lambda: float(random.randint(0, 100)) / 100,
            # Update interval length in length [10, 1000]
            lambda: random.randint(2, 50) * 20,
            # Lambda
            lambda: random.randint(4, 19) * 0.05,
            # Eligiblity method
            lambda: random.choice([0, 1, 2])

        ]
        return part_generator
