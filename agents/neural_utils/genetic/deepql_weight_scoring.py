import os

ITERATION_COUNT = 1000
BATCH_SIZE = 100

def eval_weights(genome):
    output = os.popen(
        "python E:\VSE_2022Z\DP\AIQ\AIQ.py -r BF -a DeepQL,%f,%f,%f,%d,%d,%f,%d -l %d -s %d" % (
            genome[0], genome[1], genome[2], genome[3], genome[4], genome[5], genome[6], ITERATION_COUNT, BATCH_SIZE
        )).read()
    lines = output.split("\n")
    result_line = lines[len(lines)-2]
    split_result_line = result_line.split(" ")
    values = list()
    for parts in split_result_line:
        if parts is None or parts == '':
            continue
        values.append(parts)
    result = float(values[1])-float(values[3])
    return result
