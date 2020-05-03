"""Example illustrating how to train a parameterized distribution to follow a driver distribution.

x1: r.v. binary (1=cloudy, 0=sunny) whether the day is sunny or cloudy
x2: r.v. real (degrees) the temperature of the specific day

Driver (ground truth) distribution: P(x2|x1) = N(x2; 15,10) if x1=1 (cloudy)
                                               N(x2; 25,7)  if x1=0 (sunny)

Parameterized distribution:         P(x2|x1) = N(x2; m0,s0) if x1=1 (cloudy)
                                               N(x2; m1,s1) if x1=0 (sunny)

For generating the data we also use P(x1) = Bernoulli(x1; 0.3). So the joint (generative) distribution
is: P(x1,x2) = P(x2|x1)P(x1)

The target of the problem is to make the parameterized distribution follow the driver.
"""

import pyro
import torch
import numpy as np
import matplotlib.pyplot as plt
pyro.set_rng_seed(101)

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
    x1 = pyro.sample(name='x1', fn=pyro.distributions.Bernoulli(theta_gt))

    if x1 == 0:
        x2 = pyro.sample(name='x2', fn=pyro.distributions.Normal(m0_gt, s0_gt))
    else:
        x2 = pyro.sample(name='x2', fn=pyro.distributions.Normal(m1_gt, s1_gt))
    return [x1, x2]


def driver_dist(x1):
    """
    Returns a sample from the driver distribution P(x2|x1).

    :param x1: binary variable
    :return: x2
    """
    if x1 == 0:
        x2 = pyro.sample(name='x2', fn=pyro.distributions.Normal(m0_gt, s0_gt))
    else:
        x2 = pyro.sample(name='x2', fn=pyro.distributions.Normal(m1_gt, s1_gt))
    return x2


def parameterized_dist(x1):
    """
    Returns a sample from the parameterized distribution P(x2|x1;m0,s0,m1,s1).

    :param x1: binary variable
    :return: x2
    """
    m0 = pyro.param('m0', torch.tensor(10.))
    m1 = pyro.param('m1', torch.tensor(10.))

    s0 = pyro.param('s0', torch.tensor(1.))
    s1 = pyro.param('s1', torch.tensor(1.))

    if x1 == 0:
        x2 = pyro.sample(name='x2', fn=pyro.distributions.Normal(m0, s0))
    else:
        x2 = pyro.sample(name='x2', fn=pyro.distributions.Normal(m1, s1))
    return x2


# defines the svi training mechanism
svi = pyro.infer.SVI(model=driver_dist,
                     guide=parameterized_dist,
                     optim=pyro.optim.SGD({"lr": 0.01, "momentum": 0.1}),
                     loss=pyro.infer.Trace_ELBO())


# generate data
tmp = []
for i in range(1000):
    tmp.append(generative_process())
x = np.array(tmp)

# training process
pyro.clear_param_store()
loss, m0_tr, m1_tr, s0_tr, s1_tr = [], [], [], [], []
for _ in range(100000):
    point = generative_process()
    loss.append(svi.step(point[0]))
    m0_tr.append(pyro.param("m0").item())
    m1_tr.append(pyro.param("m1").item())
    s0_tr.append(pyro.param("s0").item())
    s1_tr.append(pyro.param("s1").item())

plt.figure()
plt.title("Loss curve")
plt.plot(loss, 'ro')
plt.show()

plt.figure()
plt.title('Parameters')
plt.plot(m0_tr, 'r', label='m0=' + str(m0_gt))
plt.plot(m1_tr, 'b', label='m1=' + str(m1_gt))
plt.plot(s0_tr, 'y', label='s0=' + str(s0_gt))
plt.plot(s1_tr, 'g', label='s1=' + str(s1_gt))
plt.legend()
plt.show()
