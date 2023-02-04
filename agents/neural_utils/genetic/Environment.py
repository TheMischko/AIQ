import math
import random
import time

import numpy as np
import multiprocessing as mp

from agents.neural_utils.genetic.Individual import Individual
from agents.neural_utils.genetic.IGenomeGenerator import IGenomeGenerator


class Environment:
    def __init__(self, genome_generator, scoring_function, pop_size, num_select_best, iterations):
        """
        :param IGenomeGenerator genome_generator:
        :param scoring_function:
        :param pop_size:
        :param num_select_best:
        :param iterations:
        """
        self.genome_generator = genome_generator
        self.scoring_function = scoring_function
        self.pop_size = pop_size
        self.iterations = iterations
        self.num_select_best = num_select_best
        self.num_mutations = math.floor((self.pop_size - self.num_select_best)/2)
        self.num_crossover = self.pop_size - self.num_select_best - self.num_mutations
        pass

    def simulate(self, log=False):
        population = self.create_population(self.pop_size)
        best_individuals = None
        for i in range(self.iterations):
            start_time = time.time()
            scores = self.evalute_population(population)
            best_indices = self.select_best_individuals(scores).tolist()
            best_individuals = [population[j] for j in best_indices ]
            if log:
                print("%d. Iteration" % i)
                self.log_results(best_individuals)
            population.clear()
            # Add best to new generation
            population.extend(best_individuals)

            # Mutate best found genomes
            for j in range(self.num_mutations):
                genome_to_mutate = random.randint(0, self.num_select_best-1)
                mutated_genome = self.genome_generator.mutate_genome(best_individuals[genome_to_mutate].get_genome(), 2)
                population.append(
                    Individual(
                        mutated_genome,
                        self.scoring_function)
                )

            # Create children from best found genomes
            for j in range(self.num_crossover):
                genome1_to_mutate = random.randint(0, self.num_select_best-1)
                genome2_to_mutate = random.randint(0, self.num_select_best-1)
                while genome2_to_mutate is not genome1_to_mutate:
                    genome2_to_mutate = random.randint(0, self.num_select_best - 1)
                population.append(
                    Individual(
                        self.genome_generator.crossover_genomes(
                            best_individuals[genome1_to_mutate].get_genome(),
                            best_individuals[genome2_to_mutate].get_genome(),
                            2
                        ),
                        self.scoring_function
                    )
                )
                population = self.regenerate_same_genomes(population)
            if log:
                print("Took time %f" % (time.time()-start_time))
                print("___________________________________________")


        value_list = [best_individuals[i].eval() for i in range(len(best_individuals))]
        best_index = np.argmax(np.array(value_list))
        return best_individuals[best_index]

    def create_population(self, size):
        population = list()
        for i in range(size):
            genome = self.genome_generator.generate_genome()
            population.append(Individual(genome, self.scoring_function))
        return population

    def evalute_population(self, population):
        pool = mp.Pool(mp.cpu_count())
        # pop = 10 # cores = 4 #then do 4 + 4 + 2    10/4 = 2.5 -> 2
        # 0 1 2 3 -> 0 * 4 + i
        # 4 5 6 7 -> 1 * 4 + i
        # 8 9
        iterations = math.floor(self.pop_size/mp.cpu_count())
        values = list()
        for i in range(iterations):
            values.extend([pool.apply(population[i*mp.cpu_count() + j].eval) for j in range(mp.cpu_count())])
        values.extend(
            [
                pool.apply(
                    population[iterations * mp.cpu_count() + j].eval
                ) for j in range(self.pop_size % mp.cpu_count())
            ]
        )
        pool.close()
        return np.array(values, dtype=np.float)

    def select_best_individuals(self, scores):
        partition = np.argpartition(scores, -self.num_select_best)
        return partition[-self.num_select_best:]

    def regenerate_same_genomes(self, population):
        """
        Goes through whole generation and replaces all same genomes with random ones.
        :param list population:
        :return: Population without same valued genomes.
        """
        for i in range(len(population)):
            if population[i] is None:
                continue
            for j in range(0, len(population)):
                if i == j:
                    continue
                if population[j] is None:
                    continue
                if population[i].get_genome() == population[j].get_genome():
                    population[j] = None
        for i in range(0, len(population), 1):
            if population[i] is None:
                population[i] = Individual(
                    self.genome_generator.generate_genome(),
                    self.scoring_function
                )
        return population

    def log_results(self, individuals):
        for i in range(len(individuals)):
            print("  %2d:   %s   with value: %5.2f" %(i, individuals[i], individuals[i].eval()))
        print()