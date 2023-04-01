import math
import random
import time

import numpy as np
import multiprocessing as mp

from agents.neural_utils.genetic.EvalProcess import EvalProcess
from agents.deep_ql.neural_utils.genetic.Individual import Individual

import datetime


class Environment:
    def __init__(self, genome_generator, scoring_function, pop_size, num_select_best, iterations, num_agents, scoring_params=None,
                 seed_genomes=None):
        """
        :param IGenomeGenerator genome_generator:
        :param scoring_function:
        :param pop_size:
        :param num_select_best:
        :param iterations:
        """
        self.genome_generator = genome_generator
        self.scoring_function = scoring_function
        self.scoring_params = scoring_params
        self.pop_size = pop_size
        self.iterations = iterations
        self.num_sim_agents = num_agents
        self.num_select_best = num_select_best
        self.num_mutations = math.floor((self.pop_size - self.num_select_best)/2)
        self.num_crossover = self.pop_size - self.num_select_best - self.num_mutations
        self.log_file = "logs/%s-genetic-log.txt" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.num_changes = 4
        self.seed_genomes = seed_genomes
        self.starting_time = None
        self.debug = False

        f = open(self.log_file, "x", encoding="UTF-8")
        f.close()

    def simulate(self, log=False, debug=False):
        self.starting_time = time.time()
        self.debug = debug

        # Create population
        if self.seed_genomes is None:
            self.debug_print("Creating new population.")
            population = self.create_population(self.pop_size)
        else:
            self.debug_print("Creating new population with seeded individuals.")
            population = self.create_seeded_population(self.seed_genomes)

        # Main Loop
        best_individuals = None
        for i in range(self.iterations):
            self.debug_print("Starting iteration %d." % (i+1))
            start_time = time.time()
            self.debug_print("Starting population evaluation.")
            population, scores = self.evaluate_population(population)

            self.debug_print("Selecting best individuals.")
            best_indices = self.select_best_individuals(scores)
            best_individuals = [population[j] for j in best_indices]
            if log:
                self.debug_print("Logging best individuals.")
                self.log_results(population, i, start_time, willPrint=False, saveLog=True)

            # Add best to new generation
            population.clear()
            population.extend(best_individuals)
            self.debug_print("Mutating best genomes to create new ones.")

            # Mutate best found genomes
            for j in range(self.num_mutations):
                genome_to_mutate = random.randint(0, len(best_individuals)-1)
                mutated_genome = self.genome_generator.mutate_genome(best_individuals[genome_to_mutate].get_genome(),
                                                                     self.num_changes)
                population.append(
                    Individual(
                        mutated_genome,
                        self.scoring_function,
                        self.scoring_params
                    )
                )
            self.debug_print("Applying crossover.")

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
                        self.scoring_function,
                        self.scoring_params
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
            population.append(Individual(genome, self.scoring_function, self.scoring_params))
        return population

    def evaluate_population(self, population):
        allowed_threads = self.num_sim_agents
        # Tells how many there will be allowed_threads-times threads running
        iterations = math.floor(self.pop_size / allowed_threads)

        population_queue = mp.Queue()
        evaluated_queue = mp.Queue()

        # Fill queue
        for individual in population:
            population_queue.put(individual)

        for i in range(iterations):
            self.debug_print("Evaluating genomes %d-%d." % ((i*allowed_threads), (i*allowed_threads)+allowed_threads-1))
            processes = list()
            for t in range(allowed_threads):
                processes.append(EvalProcess(evaluated_queue, population_queue))
            for process in processes:
                process.start()
            for process in processes:
                process.join()

        # Evaluate rest of the genomes
        self.debug_print("Evaluating rest of the genomes.")
        processes = list()
        num_rest = self.pop_size - (iterations * allowed_threads)
        for t in range(num_rest):
            processes.append(EvalProcess(evaluated_queue, population_queue))
        for process in processes:
            process.start()
        for process in processes:
            process.join()

        pop = list()
        values = list()
        for i in range(self.pop_size):
            individual = evaluated_queue.get()
            pop.append(individual)
            values.append(individual.eval())

        return pop, values

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
                    self.scoring_function,
                    self.scoring_params
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
                population.append(Individual(seed_genomes[i], self.scoring_function, self.scoring_params))

        num_new_needed = self.pop_size-len(population)
        if num_new_needed <= 0:
            return population
        for i in range(num_new_needed):
            genome = self.genome_generator.generate_genome()
            population.append(Individual(genome, self.scoring_function, self.scoring_params))
        return population

    def parse_time_to_str(self, time_val):
        return "%02d:%02d:%02d" % (math.floor(time_val / 3600), math.floor(time_val / 60), time_val % 60)

    def debug_print(self, msg):
        if not self.debug:
            return
        time_from_start = time.time() - self.starting_time
        print("[%s]   %s" % (self.parse_time_to_str(time_from_start), msg))
