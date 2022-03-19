"""Building an importance sampling engine from scratch for
familiarizing with the technique.
"""

import numpy as np
import scipy.stats as st

class Prior:
    def __init__(self, start, stop, D):
        self.start = start
        self.stop = stop
        self.D = D
        self.rv = st.uniform(loc=start, scale=stop-start)

    def pdf(self, x):
        assert x.shape[1] == self.D # x: (BS, D)
        return np.prod(self.rv.pdf(x), axis=1)

    def sample(self, batch_size, seed=None):
        return self.rv.rvs([batch_size, self.D], random_state=seed)

class Proposal:
    def __init__(self, start, stop, D):
        self.start = start
        self.stop = stop
        self.D = D
        self.rv = st.uniform(loc=start, scale=stop-start)

    def pdf(self, x):
        assert x.shape[1] == self.D # x: (BS, D)
        return np.prod(self.rv.pdf(x), axis=1)

    def sample(self, batch_size, seed=None):
        return self.rv.rvs([batch_size, self.D], random_state=seed)

class Likelihood:
    def __init__(self):
        pass

    def pdf(self, X, Y, theta):
        assert X.ndim == 2 # (N, D)
        assert Y.ndim == 1 # (N,)
        assert theta.ndim == 2 # (BS, D)

        lik = []
        for i in range(theta.shape[0]):
            Y_mean = np.matmul(X, theta[i])
            self.rv = st.norm(loc=Y_mean, scale=1)
            lik.append(np.prod(self.rv.pdf(Y)))
        return np.array(lik)

class ImportanceSampling:
    def __init__(self, prior, proposal, likelihood):
        self.prior = prior
        self.proposal = proposal
        self.likelihood = likelihood

        self.weights_unn = None
        self.samples = None

    def sample(self, N):
        th = self.proposal.sample(N)
        p = self.prior.pdf(th)
        q = self.proposal.pdf(th)
        lik = self.likelihood.pdf(X, Y, th)

        # weights
        if self.weights_unn is not None:
            self.weights_unn = np.concatenate([self.weights_unn, lik*p/q])
        else:
            self.weights_unn = lik*p/q

        if self.samples is not None:
            self.samples = np.concatenate([self.samples, th])
        else:
            self.samples = th

        # normalize and sort
        weights_norm = self.weights_unn / np.sum(self.weights_unn)
        ind = np.argsort(-weights_norm)
        w = weights_norm[ind]
        th = self.samples[ind]
        return w, th

# data
D = 4
theta_star = np.array([-9, 9, 0 ,1])
X = np.array([[1, 2, 3 , 4], [2, 3, 4, 5]])
Y = np.matmul(X, theta_star)

# modeling
prior = Prior(start=-10, stop=10, D=D)
proposal = Proposal(start=-10, stop=10, D=D)
likelihood = Likelihood()

# sample
sampling = ImportanceSampling(prior, proposal, likelihood)
w, th = sampling.sample(N=10000)
