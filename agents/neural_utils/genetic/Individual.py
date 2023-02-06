class Individual:
    def __init__(self, initial_genome, eval_fnc):
        self.genome = initial_genome
        self.eval_fnc = eval_fnc
        self.value = None

    def eval(self):
        if self.value is None:
            self.value = self.eval_fnc(self.genome)
            return self.value
        else:
            return self.value

    def change_genome(self, new_genome):
        self.genome = new_genome

    def get_genome(self):
        return self.genome

    def __str__(self):
        return "[{0}]".format(", ".join(str(e) for e in self.genome))

