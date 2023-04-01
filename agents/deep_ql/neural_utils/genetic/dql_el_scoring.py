import os

ITERATION_COUNT = 2000
BATCH_SIZE = 150

def eval_weights(genome, eval_params):
    try:
        iterations = eval_params["iterations"] or ITERATION_COUNT
        samples = eval_params["samples"] or BATCH_SIZE
        threads = eval_params["threads"] or 2
        path = os.getcwd().split("AIQ")[0] + "AIQ/AIQ.py"
        output = os.popen(
            "python %s -r BF -a DQL_Dual_ET_Decay,%f,%f,%d,%d,%d,%d,%d,%f,%d,%f,%d -l %d -s %d -t %d" % (
                path,
                genome[0], genome[1], genome[2], genome[3], genome[4], genome[5], genome[6], genome[7], genome[8],
                genome[9], genome[10], iterations, samples, threads
            )).read()
        lines = output.split("\n")
        result_line = lines[len(lines)-2]
        split_result_line = result_line.split(" ")
        values = list()
        for parts in split_result_line:
            if parts is None or parts == '':
                continue
            values.append(parts)
        if len(values) > 4:
            return -9e10
        if values[1] == "nan" or values[3] == "nan":
            return -9e10
        result = float(values[1])-float(values[3])
        if result == float("nan"):
            result = -9e10
        return result
    except Exception:
        return -9e10
