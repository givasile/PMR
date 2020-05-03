"""Example illustrating how to train a parameterized distribution from some observed data.

The observed data correspond to the observable variables. The rest are the unobserved. The goal is to define
a prior distribution over the unobservable (latent) variables and try to find the posterior over them.
"""

import pyro
import torch
import torch.distributions
import numpy as np
import matplotlib.pyplot as plt
pyro.set_rng_seed(101)

# The ground truth parameters
theta_gt = 0.5
m0_gt = 25.
s0_gt = 7.
m1_gt = 15.
s1_gt = 10.


def generative_process():
    """
    Generates a single sample.

    :return: list, the generated sample
    """
    x1 = torch.distributions.Bernoulli(theta_gt).sample()

    if x1 == 0:
        x2 = torch.distributions.Normal(m0_gt, s0_gt).sample()
    else:
        x2 = torch.distributions.Normal(m1_gt, s1_gt).sample()
    return [x1, x2]


def driver(data):
    """
    (i)  generates samples from the unobserved variables m0, m1 ~ P(m0, m1) = P(m0)*P(m1)
    (ii) generates samples from the observed variables x2^(i) ~ P(x2^(i)|m0,m1,x1^(i)) for all i

    The name of the samples of the unobserved variables (m0, m1) must match with the corresponding name
    in the parameterized function afterwards.

    Intuition: This is the posterior distribution.

    :param data: the set of observed data points
    """
    # (i) generates samples from the prior distributions p(m0), p(m1)
    m0 = pyro.sample("m0", fn=pyro.distributions.Normal(0, 2))
    m1 = pyro.sample("m1", fn=pyro.distributions.Normal(0, 2))

    # (ii) generates samples from p(x2|x1,m0,m1), pointing that those points are observed
    for i in range(data.shape[0]):
        if data[i, 0] == 0:
            x2 = pyro.sample(name='x2_%04d' % i,
                             fn=pyro.distributions.Normal(m0, s0_gt),
                             obs=data[i][1])
        else:
            x2 = pyro.sample(name='x2_%04d' % i,
                             fn=pyro.distributions.Normal(m1, s1_gt),
                             obs=data[i][1])


def parameterized(data):
    """
    (i)  Initializes the parameters that will be optimized through the training process.
    (ii) Generates samples from the parameterized distribution

    Intuition: This is a parameterized distribution that will try to match the posterior.

    :param data: the set of observed data points
    """

    # Parameters initialization
    a0 = pyro.param('a0', torch.tensor(20.))
    a1 = pyro.param('a1', torch.tensor(20.))
    b0 = pyro.param('b0', torch.tensor(10.))
    b1 = pyro.param('b1', torch.tensor(10.))

    # Generation of samples
    m0 = pyro.sample("m0", pyro.distributions.Normal(a0, b0))
    m1 = pyro.sample("m1", pyro.distributions.Normal(a1, b1))


# defines the svi training mechanism
svi = pyro.infer.SVI(model=driver,
                     guide=parameterized,
                     optim=pyro.optim.SGD({"lr": 0.01, "momentum": 0.1}),
                     loss=pyro.infer.Trace_ELBO())


# generate data
tmp = []
for i in range(1000):
    tmp.append(generative_process())
data = np.array(tmp)

# training process
pyro.clear_param_store()
loss, a0, a1, b0, b1 = [], [], [], [], []
for _ in range(100):
    loss.append(svi.step(data))
    a0.append(pyro.param("a0").item())
    a1.append(pyro.param("a1").item())
    b0.append(pyro.param("b0").item())
    b1.append(pyro.param("b1").item())

# Plot the loss curve
plt.figure()
plt.title("Loss curve")
plt.plot(loss, 'ro')
plt.show()

# Plot the learned parameters
plt.figure()
plt.title('Parameters')
plt.plot(a0, 'r', label='a0 = 25')
plt.plot(a1, 'b', label='a1 = 15')
plt.plot(b0, 'y', label='b0')
plt.plot(b1, 'g', label='b1')
plt.legend()
plt.show()