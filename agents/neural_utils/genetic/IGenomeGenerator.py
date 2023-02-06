import copy
import random


class IGenomeGenerator:
    """
    Abstract class for generation of genome for individual problem.
    """

    def __init__(self):
        self.part_generators = self.set_generators()

    def set_generators(self):
        raise NotImplementedError()

    def generate_genome(self):
        """
        Generates a genome with values within domain of given problem.
        :return: List of values for an individual.
        """
        genome = list()
        for i in range(len(self.part_generators)):
            genome.append(self.part_generators[i]())

        return genome

    def mutate_genome(self, genome, n_changes):
        """
        Mutates some parts of genome to new random values.
        :param genome: Original list of values.
        :param n_changes: Number of values that will be changed.
        :return:
        """
        if n_changes > len(self.part_generators):
            raise Exception("Number of changes cannot be bigger than size of the genome.")
        new_genome = copy.deepcopy(genome)
        indices = [i for i in range(len(self.part_generators))]
        indices_to_change = random.sample(indices, n_changes)
        for i in indices_to_change:
            new_genome[i] = self.part_generators[i]()
        return new_genome

    def crossover_genomes(self, genome_1, genome_2, n_changes):
        if n_changes > len(self.part_generators):
            raise Exception("Number of changes cannot be bigger than size of the genome.")
        new_genome = copy.deepcopy(genome_1)
        indices = [i for i in range(len(self.part_generators))]
        indices_to_change = random.sample(indices, n_changes)
        for i in indices_to_change:
            new_genome[i] = genome_2[i]
        return new_genome
