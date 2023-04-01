from agents.deep_ql.DQL_Dual_ET_Decay import DQL_Dual_ET_Decay


class DQL_Dual_ET(DQL_Dual_ET_Decay):
    def __init__(self, refm, disc_rate, learning_rate, gamma, batch_size, epsilon, neural_size_l1,
                 neural_size_l2, neural_size_l3, tau, update_interval_length, lambda_val, traces_method):
        super().__init__(refm, disc_rate, learning_rate, gamma, batch_size, 0, neural_size_l1,
                         neural_size_l2, neural_size_l3, tau, update_interval_length, lambda_val, traces_method)
        self.epsilon = epsilon
        self.starting_epsilon = epsilon

    def decrement_epsilon(self):
        return

    def __str__(self):
        return "DQL_Dual_Decay(%.4f,%.2f,%d,%.3f,%d,%d,%d,%.2f,%d,%.2f,%s)" % (
            self.learning_rate,
            self.gamma,
            self.batch_size,
            self.epsilon,
            self.neural_size_l1,
            self.neural_size_l2,
            self.neural_size_l3,
            self.tau,
            self.update_interval_length,
            self.lambda_val,
            self.trace_method
        )