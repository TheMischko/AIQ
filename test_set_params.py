import os

epsilons = [0.005, 0.01, 0.02, 0.03, 0.04]
gamma = [0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995]

if __name__ == "__main__":
    for eps in epsilons:
        for gam in gamma:
                script = "python AIQ.py -r BF -a DeepQLNoDecay,0.00134,%f,32,%f,256,248,0,0.25,60 -l 10000 -s 200 -t 6" % (gam, eps)
                output = os.popen(script).read()
                print(output)
