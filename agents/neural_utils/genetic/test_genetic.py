import os
import sys
import argparse

sys.path.append(os.getcwd().split("AIQ")[0] + "AIQ")

from agents.neural_utils.genetic.Environment import Environment
from agents.neural_utils.genetic.deepql_weight_generator import DeepQLWeightGenerator
from agents.neural_utils.genetic.deepql_weight_scoring import eval_weights



def print_header(pop_size, num_select, epochs, iterations, samples, agents, threads):
    print("-----------------------------------------------------------------------")
    print("-------Starting Genetic algorithm for finding best DeepQL setup--------")
    print("-----------------------------------------------------------------------")
    print("Parameters:")
    print("   Population size: %d" % pop_size)
    print("   Number of best individuals for next gen: %d" % num_select)
    print("   Number of epochs: %d" % epochs)
    print("   Number of iterations for AIQ test: %d" % iterations)
    print("   Number of samples for AIQ test: %d" % samples)
    print("   Number of simultaneously running agents: %d" % agents)
    print("   Number of thread each agent will use in AIQ: %d" % threads)
    print("-----------------------------------------------------------------------")

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--pop_size", help="Size of population for each iteration.")
    argParser.add_argument("-n", "--num_select",
                           help="Set size of how many best individuals will be moved to next generation.")
    argParser.add_argument("-e", "--epochs", help="Number of epochs for genetic algorithm.")
    argParser.add_argument("-i", "--iterations", help="Iterations for AIQ test.")
    argParser.add_argument("-s", "--samples", help="Number of samples for AIQ test.")
    argParser.add_argument("-a", "--agents", help="Set how many agents will be run at single time.")
    argParser.add_argument("-t", "--threads", help="Set how many threads will be used for single agents in AIQ test.")
    args = argParser.parse_args()

    pop_size = int(args.pop_size or 10)
    num_select = int(args.num_select or 2)
    epochs = int(args.epochs or 10)
    iterations = int(args.iterations or 1000)
    samples = int(args.samples or 100)
    agents = int(args.agents or 2)
    threads = int(args.threads or 2)

    gen_env = Environment(DeepQLWeightGenerator(), eval_weights, pop_size, num_select, epochs, agents, scoring_params={
        "iterations": iterations,
        "samples": samples,
        "threads": threads
    }, seed_genomes=[
        [0.00134, 0.83, 32, 520, 256, 248, 0, 0.25, 60],
        [0.00200, 0.75, 64, 1000, 64, 128, 64, 0.8, 100]
    ])
    print_header(pop_size, num_select, epochs, iterations, samples, agents, threads)
    result = gen_env.simulate(log=True)
    print()
    print("Best found individual is %s with value %5.2f" % (result, result.eval()))
    print()
