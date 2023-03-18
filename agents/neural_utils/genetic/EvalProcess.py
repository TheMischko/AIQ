from multiprocessing import Process, Queue

from agents.neural_utils.genetic.Individual import Individual


class EvalProcess(Process):
    def __init__(self, queue: Queue, individual_queue: Queue):
        Process.__init__(self)
        self.queue = queue
        self.individual = individual_queue.get()

    def run(self):
        self.individual.eval()
        self.queue.put(self.individual)
