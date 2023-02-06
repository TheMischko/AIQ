import copy
import math
import random
import time

import numpy as np
import multiprocessing as mp

from agents.neural_utils.genetic.Individual import Individual
from agents.neural_utils.genetic.IGenomeGenerator import IGenomeGenerator

import datetime


class Environment:
    def __init__(self, genome_generator, scoring_function, pop_size, num_select_best, iterations, seed_genomes = None):
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
        self.log_file = "logs/%s-genetic-log.txt" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.num_changes = 4
        self.seed_genomes = seed_genomes

    def simulate(self, log=False):
        if self.seed_genomes is None:
            population = self.create_population(self.pop_size)
        else:
            population = self.create_seeded_population(self.seed_genomes)
        best_individuals = None
        for i in range(self.iterations):
            start_time = time.time()
            scores = self.evalute_population(population)
            best_indices = self.select_best_individuals(scores)
            best_individuals = [population[j] for j in best_indices]
            if log:
                self.log_results(population, i, start_time, willPrint=False, saveLog=True)
            # Add best to new generation
            population.clear()
            population.extend(best_individuals)

            # Mutate best found genomes
            for j in range(self.num_mutations):
                genome_to_mutate = random.randint(0, len(best_individuals)-1)
                mutated_genome = self.genome_generator.mutate_genome(best_individuals[genome_to_mutate].get_genome(),
                                                                     self.num_changes)
                population.append(
                    Individual(
                        mutated_genome,
                        self.scoring_function)
                )

            # Create children from best found genomes
            for j in range(self.num_crossover):
                genome1_to_mutate = random.randint(0, len(best_individuals)-1)
                genome2_to_mutate = random.randint(0, len(best_individuals)-1)
                while genome2_to_mutate == genome1_to_mutate:
                    genome2_to_mutate = random.randint(0, len(best_individuals)-1)
                population.append(
                    Individual(
                        self.genome_generator.crossover_genomes(
                            best_individuals[genome1_to_mutate].get_genome(),
                            best_individuals[genome2_to_mutate].get_genome(),
                            self.num_changes
                        ),
                        self.scoring_function
                    )
                )
                population = self.regenerate_same_genomes(population)
            if log:
                self.log_results(best_individuals, i, start_time, willPrint=True, saveLog=False)

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
        allowed_threads = 4
        pool = mp.Pool(allowed_threads)
        iterations = math.floor(self.pop_size / allowed_threads)
        values = list()
        for i in range(iterations):
            results = pool.map(evaluate_individual, [population[i + index] for index in range(allowed_threads)])
            values.extend(results)
        results = pool.map(evaluate_individual, [population[index + (allowed_threads * iterations)] for index in
                                                 range(self.pop_size % allowed_threads)])
        values.extend(results)
        pool.close()
        return np.array(values, dtype=np.float)

    def select_best_individuals(self, scores):
        partition = np.argpartition(scores, -self.num_select_best)
        return partition[-self.num_select_best:].tolist()

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

    def log_results(self, individuals, iteration, start_time, willPrint=True, saveLog=False):
        log_string = "\n"
        log_string += "%d. Generation \n" % (int(iteration)+1)
        for i in range(len(individuals)):
            log_string += "  %2d:   %s   with value: %5.2f\n" %(i, individuals[i], individuals[i].eval())
        log_string += "Took time %f \n" % (time.time() - start_time)
        log_string += "___________________________________________\n"
        if saveLog:
            with open(self.log_file, "a") as file:
                file.write(log_string)
        if willPrint:
            print(log_string)

    def create_seeded_population(self, seed_genomes):
        population = list()
        for i in range(len(seed_genomes)):
            if i >= self.pop_size:
                break
            if len(seed_genomes[i]) == len(self.genome_generator.part_generators):
                population.append(Individual(seed_genomes[i], self.scoring_function))

        num_new_needed = self.pop_size-len(population)
        if num_new_needed <= 0:
            return population
        for i in range(num_new_needed):
            genome = self.genome_generator.generate_genome()
            population.append(Individual(genome, self.scoring_function))
        return population


def evaluate_individual(individual):
    return individual.eval()
