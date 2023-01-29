# Rozšíření AIQ testu o agenty využívající metodu DeepQ Learning
Do AIQ testu byli přidáni dva agenti:
- Vanilla DeepQL agent, který je implementován podle https://arxiv.org/abs/1312.5602
- DeepQL agent využívající dvě neuronové sítě jak zmiňuje například https://towardsdatascience.com/deep-q-network-dqn-ii-b6bf911b6b2c
### Vanilla DeepQL agenta
[Implementace](agents/VanillaDeepQL.py)
Příklad použití:
```
python AIQ.py -r BF -a VanillaDeepQL,0.002,1,16,0.9,1000 -l 10000 -s 100 -t 4 -d 1
```

### DeepQL agent s použitím dvou neuronových sítí
[Implementace](agents/DeepQL.py)
Příklad použití:
```
python AIQ.py -r BF -a DeepQL,0.002,1,16,0.9 -l 10000 -s 100 -t 4 -d 1
```