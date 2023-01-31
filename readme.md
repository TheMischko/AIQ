# Rozšíření AIQ testu o agenty využívající metodu DeepQ Learning
Do AIQ testu byli přidáni dva agenti:
- Vanilla DeepQL agent, který je implementován podle https://arxiv.org/abs/1312.5602
- DeepQL agent využívající dvě neuronové sítě jak zmiňuje například https://towardsdatascience.com/deep-q-network-dqn-ii-b6bf911b6b2c
### Vanilla DeepQL agenta
- [Implementace](agents/VanillaDeepQL.py)
- Příklad použití:
```
python AIQ.py -r BF -a VanillaDeepQL,0.0025,1,16,1000 -l 2000 -s 200 -t 4 -d 1 -d 1
```
- parametry:
  - learning rate,
  - počáteční epsilon
  - velikost učícího batche
  - počet učících kroků po kterých bude epsilon na minimu

### DeepQL agent s použitím dvou neuronových sítí
- [Implementace](agents/DeepQL.py)
- Příklad použití:
```
python AIQ.py -r BF -a DeepQL,0.0025,1,16,1000,0.9,100 -l 10000 -s 200 -t 4 -d 0.95
```
- parametry:
  - learning rate,
  - počáteční epsilon
  - velikost učícího batche
  - počet učících kroků po kterých bude epsilon na minimu
  - tau (poměr vah při učení target network)
  - počet učících kroků, po kterých budou překopírovány váhy z policy network do target network v poměru tau