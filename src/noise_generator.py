import numpy as np

class IndependentMarginals:
    def __init__(self, data, alphabet_size, alpha = 0.1, seed=123):
        N, n_vars = data.shape
        self.n_vars = n_vars
        self.alphabet_size = alphabet_size
        self.rng = np.random.default_rng(seed=seed)

        # compute marginals
        self.marginal_probs = np.zeros((n_vars,alphabet_size))
        for i in range(n_vars):
            for j in range(alphabet_size):
                self.marginal_probs[i,j] = np.sum(data[:,i] == j)
        self.marginal_probs += alpha
        self.marginal_probs /= self.marginal_probs.sum(axis=1, keepdims=True)

    def generate_samples(self, n_samples):
        samples = np.zeros((n_samples, self.n_vars), dtype=int)
        for i in range(self.n_vars):
            samples[:,i] = self.rng.choice(self.alphabet_size, size = n_samples, p = self.marginal_probs[i,:])
        return samples

    def log_prob(self, data):
        # data has shape (N, n_vars)
        log_q = np.zeros(data.shape[0])
        for i in range(self.n_vars):
            log_q += np.log(self.marginal_probs[i,data[:,i]])
        return log_q