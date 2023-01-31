import numpy as np


def eval_score(genome):
    val_generic = [((-1)^i)*(genome[0]+genome[1])*genome[2] for i in range(genome[3])]
    values = np.array(val_generic)
    sum = np.sum(values)
    return sum
