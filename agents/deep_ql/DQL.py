from agents.deep_ql.DQL_Decay import DQL_Decay


class DQL(DQL_Decay):
    def __init__(self, refm, disc_rate, learning_rate, gamma, batch_size, epsilon, neural_size_l1,
                 neural_size_l2, neural_size_l3):
        super().__init__(refm, disc_rate, learning_rate, gamma, batch_size, 0, neural_size_l1,
                 neural_size_l2, neural_size_l3)
        self.epsilon = epsilon
        self.starting_epsilon = epsilon

    def decrement_epsilon(self):
        return

    def __str__(self):
        return "DQL(%.4f,%.2f,%d,%.3f,%d,%d,%d)" % (
            self.learning_rate,
            self.gamma,
            self.batch_size,
            self.epsilon,
            self.neural_size_l1,
            self.neural_size_l2,
            self.neural_size_l3
        )
