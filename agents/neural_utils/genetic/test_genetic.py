import os

from agents.neural_utils.genetic.Environment import Environment
from agents.neural_utils.genetic.deepql_weight_generator import DeepQLWeightGenerator
from agents.neural_utils.genetic.deepql_weight_scoring import eval_weights
from agents.neural_utils.genetic.test_generator import TestGenerator
from agents.neural_utils.genetic.test_scoring import eval_score



if __name__ == '__main__':
    gen_env = Environment(DeepQLWeightGenerator(), eval_weights, 6, 2, 10)
    result = gen_env.simulate(log=True)
    print()
    print("Best found individual is %s with value %5.2f" % (result, result.eval()))
    print()