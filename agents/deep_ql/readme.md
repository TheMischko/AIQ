# Module DeepQ-learning
This directory contains few agents implementing variants of DeepQ-learning [[Mnih el al.]](https://arxiv.org/pdf/1312.5602.pdf).
These agents are imported into rest of the AIQ agents and are ready to use in the AIQ test.


| Agent | Replay memory  | Dual net  | Epsilon decay  | Eligibility Traces |
| --- |:--------------:|:---------:|:--------------:|:------------------:|
| [DQL.py](DQL.py) |       ✅        |     ❌     |       ❌        |           ❌        |
| [DQL_Decay.py](DQL_Decay.py) |       ✅        |     ❌     |       ✅       |           ❌        |
| [DQL_Dual.py](DQL_Dual.py) |       ✅        |     ✅    |       ❌        |           ❌        |
| [DQL_Dual_Decay.py](DQL_Dual_Decay.py) |       ✅        |     ✅     |       ✅        |           ❌        |
| [DQL_Dual_ET.py](DQL_Dual_ET.py) |       ✅        |     ✅     |       ❌        |           ✅        |
| [DQL_Dual_ET_Decay.py](DQL_Dual_ET_Decay.py) |       ✅        |     ✅     |       ✅        |           ✅        |

Note that all DQL agents implement Interface [IDeepQLAgent](neural_utils/IDeepQLAgent.py).

This interface fetches some basic configuration from [dql_config.py](dql_config.py) 

## Vanilla DeepQ-learning
- [DQL.py](DQL.py)
- Example of use:
```
python AIQ.py -r BF -a DQL,0.0025,0.7,16,0.2,256,256,0 -l 1000 -s 100
```
- This agent implements:
  - Replay memory
  - Steady epsilon value
  - Single neural network
- Its parameters are:
  1. learning rate
  2. gamma
  3. batch size
  4. epsilon
  5. size of layer 1
  6. size of layer 2
  7. size of layer 3 (can be zero)

## Vanilla DeepQ-learning with epsilon decay
- [DQL_Decay.py](DQL_Decay.py)
- Example of use:
```
python AIQ.py -r BF -a DQL_Decay,0.0025,0.7,16,100,256,256,0 -l 1000 -s 100
```
- This agent implements:
  - Replay memory
  - Linear epsilon decay
  - Single neural network
- Its parameters are:
  1. learning rate
  2. gamma
  3. batch size
  4. count of iterations till epsilon will be reduced to minimal value
  5. size of layer 1
  6. size of layer 2
  7. size of layer 3 (can be zero)

## Dualnet DeepQ-learning
- Source: [Mnih et al.](https://www.nature.com/articles/nature14236)
- [DQL_Dual.py](DQL_Dual.py)
- Example of use:
```
python AIQ.py -r BF -a DQL_Dual,0.04,0.34,16,0.995,200,144,0,0.05,10 -l 1000 -s 100
```
- This agent implements:
  - Replay memory
  - Steady epsilon value
  - Target neural network
  - Policy neural network
- Its parameters are:
  1. learning rate
  2. gamma
  3. batch size
  4. epsilon
  5. size of layer 1
  6. size of layer 2
  7. size of layer 3 (can be zero)
  8. tau (parameter for factor of policy net weights copied to target net)
  9. target network update interval

## Dualnet DeepQ-learning with epsilon decay
- [DQL_Dual_Decay.py](DQL_Dual_Decay.py)
- Example of use:
```
python AIQ.py -r BF -a DQL_Dual_Decay,0.04,0.34,16,290,200,144,0,0.05,10 -l 1000 -s 100
```
- This agent implements:
  - Replay memory
  - Linear epsilon decay
  - Target neural network
  - Policy neural network
- Its parameters are:
  1. learning rate
  2. gamma
  3. batch size
  4. count of iterations till epsilon will be reduced to minimal value
  5. size of layer 1
  6. size of layer 2
  7. size of layer 3 (can be zero)
  8. tau (parameter for factor of policy net weights copied to target net)
  9. target network update interval

## Dualnet DeepQ-learning with Eligibility Traces
- Source: [Mousavi et al.](https://www.researchgate.net/publication/326927951_Applying_Ql-learning_in_Deep_Reinforcement_Learning_to_Play_Atari_Games)
- [DQL_Dual_ET.py](DQL_Dual_ET.py)
- Example of use:
```
python AIQ.py -r BF -a DQL_Dual_ET,0.04,0.34,16,0.3,200,144,0,0.05,10,0.7,2 -l 1000 -s 100
```
- This agent implements:
  - Replay memory
  - Steady epsilon value
  - Target neural network
  - Policy neural network
  - Eligibility traces
- Its parameters are:
  1. learning rate
  2. gamma
  3. batch size
  4. epsilon
  5. size of layer 1
  6. size of layer 2
  7. size of layer 3 (can be zero)
  8. tau (parameter for factor of policy net weights copied to target net)
  9. target network update interval
  10. lambda
  11. [eligibility traces method](neural_utils/traces.py) (0 - replacing, 1 - accumulating, 2 - dutch)

## Dualnet DeepQ-learning with Eligibility Traces and epsilon decay
- [DQL_Dual_ET_Decay.py](DQL_Dual_ET_Decay.py)
- Example of use:
```
python AIQ.py -r BF -a DQL_Dual_ET_Decay,0.04,0.34,16,290,200,144,0,0.05,10,0.7,0 -l 1000 -s 100
```
- This agent implements:
  - Replay memory
  - Steady epsilon value
  - Target neural network
  - Policy neural network
  - Eligibility traces
- Its parameters are:
  1. learning rate
  2. gamma
  3. batch size
  4. count of iterations till epsilon will be reduced to minimal value
  5. size of layer 1
  6. size of layer 2
  7. size of layer 3 (can be zero)
  8. tau (parameter for factor of policy net weights copied to target net)
  9. target network update interval
  10. lambda
  11. [eligibility traces method](neural_utils/traces.py) (0 - replacing, 1 - accumulating, 2 - dutch)