import os

from agents.neural_utils.genetic.Environment import Environment
from agents.neural_utils.genetic.deepql_weight_generator import DeepQLWeightGenerator
from agents.neural_utils.genetic.deepql_weight_scoring import eval_weights



if __name__ == '__main__':
    gen_env = Environment(DeepQLWeightGenerator(), eval_weights, 8, 3, 25, seed_genomes=[
        [0.00108, 0.13, 32, 290, 200, 208, 112, 0.05, 10],
        [0.0034, 0.55, 32, 970, 64, 232, 64, 0.25, 220],
        [0.00132, 0.21, 16, 280, 152, 184, 0, 0.65, 190],
        [0.00434, 0.39, 32, 400, 48, 192, 0, 0.62, 220]
    ])
    result = gen_env.simulate(log=True)
    print()
    print("Best found individual is %s with value %5.2f" % (result, result.eval()))
    print()