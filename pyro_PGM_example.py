"""
x1: grade in first degree
x2: grade in the first semester
x3: provisional final grade
x4: final grade in the MSc
"""
import pyro
import pyro.distributions as pdist
import numpy as np
import torch
import matplotlib.pyplot as plt


def restrict(x):
    if x >= 10:
        x = 10
    elif x <= 0:
        x = 0
    return x


def generative_process():
    x1 = restrict(torch.distributions.Normal(loc=8, scale=1).sample())
    x2 = restrict(torch.distributions.Normal(loc=7, scale=1.5).sample())

    w1 = 0.3
    w2 = 0.7
    x3 = restrict(torch.distributions.Normal(loc=w1*x1 + w2*x2, scale=2).sample())

    a = -0.3
    x4 = restrict(torch.distributions.Normal(loc=x3-a, scale=0.5).sample())
    return (x1, x2, x3, x4)


def driver(data):
    # the prior on the unobserved vars (model parameters) that remain unchanged for all observations
    w1 = pyro.sample(name="w1", fn=pdist.Normal(loc=0.5, scale=0.5))
    w2 = pyro.sample(name="w2", fn=pdist.Normal(loc=0.5, scale=0.5))
    a = pyro.sample(name="a", fn=pdist.Normal(loc=-0.2, scale=0.3))

    for i in range(data.shape[0]):
        x1 = pyro.sample(name='x1_%04d' % i,
                         fn=pyro.distributions.Normal(8, 1),
                         obs=data[i][0])
        x2 = pyro.sample(name='x2_%04d' % i,
                         fn=pyro.distributions.Normal(7, 1.5),
                         obs=data[i][1])
        x3 = pyro.sample(name='x3_%04d' % i,
                         fn=pyro.distributions.Normal(w1*x1+w2*x2, 2),
                         )# obs=data[i][2])
        x4 = pyro.sample(name='x4_%04d' % i,
                         fn=pyro.distributions.Normal(x3-a, 1),
                         obs=data[i][3])

def parameterized(data):
    # Parameters initialization
    w1_m = pyro.param('w1_m', torch.tensor(.5), constraint=pdist.constraints.positive)
    w2_m = pyro.param('w2_m', torch.tensor(.5), constraint=pdist.constraints.positive)
    a_m = pyro.param('a_m', torch.tensor(1.))
    # x3_m = pyro.param('x3_m', torch.tensor(7.))

    w1_s = pyro.param('w1_s', torch.tensor(0.5), constraint=pdist.constraints.positive)
    w2_s = pyro.param('w2_s', torch.tensor(.5), constraint=pdist.constraints.positive)
    a_s = pyro.param('a_s', torch.tensor(1.), constraint=pdist.constraints.positive)
    # x3_s = pyro.param('x3_s', torch.tensor(1.5))

    # Generation of samples
    w1 = pyro.sample(name="w1", fn=pdist.Normal(loc=w1_m, scale=w1_s))
    w2 = pyro.sample(name="w2", fn=pdist.Normal(loc=w2_m, scale=w2_s))
    a = pyro.sample(name="a", fn=pdist.Normal(loc=a_m, scale=a_s))
    # x3 = pyro.sample(name="x3", fn=pdist.Normal(loc=x3_m, scale=x3_s))



# defines the svi training mechanism
svi = pyro.infer.SVI(model=driver,
                     guide=parameterized,
                     optim=pyro.optim.SGD({"lr": 0.0001, "momentum": 0.01}),
                     loss=pyro.infer.Trace_ELBO())


# generate data
tmp = []
for i in range(500):
    tmp.append(generative_process())
data = np.array(tmp)

# training process
pyro.clear_param_store()
loss, w1_m, w2_m, a_m = [], [], [], []
for _ in range(200):
    loss.append(svi.step(data))
    w1_m.append(pyro.param("w1_m").item())
    w2_m.append(pyro.param("w2_m").item())
    a_m.append(pyro.param("a_m").item())

# Plot the loss curve
plt.figure()
plt.title("Loss curve")
plt.plot(loss, 'ro')
plt.show()

# Plot the learned parameters
plt.figure()
plt.title('Parameters')
plt.plot(w1_m, 'r', label='w1 = 0.3')
plt.plot(w2_m, 'b', label='w2 = 0.7')
plt.plot(a_m, 'y', label='a = -0.3')
plt.legend()
plt.show()