import os
import sys
import argparse

sys.path.append(os.getcwd().split("AIQ")[0] + "AIQ")

from agents.deep_ql.neural_utils.genetic.Environment import Environment
from agents.deep_ql.neural_utils.genetic.deepql_weight_generator import DeepQLWeightGenerator
from agents.deep_ql.neural_utils.genetic import deepql_weight_scoring, dql_el_scoring
from agents.deep_ql.neural_utils.genetic import c51_weight_scoring
from agents.deep_ql.neural_utils.genetic.c51_weight_generator import C51WeightGenerator
from agents.deep_ql.neural_utils.genetic.dql_el_generator import DQlElGenerator



def print_header(agent_type, pop_size, num_select, epochs, iterations, samples, agents, threads):
    print("-----------------------------------------------------------------------")
    print("-------Starting Genetic algorithm for finding best %s setup--------" % agent_type)
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
    argParser.add_argument("-x", "--agent_type", help="Set a name of the agent for testing. (Values are DeepQL or C51)")
    argParser.add_argument("-d", "--debug", help="Turns on the debug messages to print", action="store_true")
    args = argParser.parse_args()

    pop_size = int(args.pop_size or 10)
    num_select = int(args.num_select or 2)
    epochs = int(args.epochs or 10)
    iterations = int(args.iterations or 1000)
    samples = int(args.samples or 100)
    agents = int(args.agents or 2)
    threads = int(args.threads or 2)
    agent_type = args.agent_type
    debug = args.debug

    param_generator = None
    eval_weights = None
    if agent_type == "C51":
        param_generator = C51WeightGenerator()
        eval_weights = c51_weight_scoring.eval_weights
    elif agent_type == "DQL_EL":
        param_generator = DQlElGenerator()
        eval_weights = dql_el_scoring.eval_weights
    else:
        param_generator = DeepQLWeightGenerator()
        eval_weights = deepql_weight_scoring.eval_weights

    gen_env = Environment(param_generator, eval_weights, pop_size, num_select, epochs, agents, scoring_params={
        "iterations": iterations,
        "samples": samples,
        "threads": threads
    })

    print_header(agent_type, pop_size, num_select, epochs, iterations, samples, agents, threads)
    result = gen_env.simulate(log=True, debug=True)
    print()
    print("Best found individual is %s with value %5.2f" % (result, result.eval()))
    print()
