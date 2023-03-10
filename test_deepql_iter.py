import os

iters = [100, 300, 500, 1000]
deepql_decay_on_10k = 300
agent_script_strings = {
    "DeepQL": "python AIQ.py --log -r BF -l %i -s 100 -t 6 -a DeepQL,0.00468,0.33,32,%i,64,224,176,0.25,60",
    "Q_l": "python AIQ.py --log -r BF -l %i -s 100 -t 6 -a Q_l,0.0,0.0,0.5,0.005,0.95 %i"
}
test_agents = ["DeepQL", "Q_l"]

if __name__ == "__main__":
    for agent in test_agents:
        for iter in iters:
            deepql_decay = deepql_decay_on_10k if iter < 10000 else deepql_decay_on_10k/10000*iter
            script = agent_script_strings[agent] % (iter, deepql_decay)
            output = os.popen(script).read()
            print(output)