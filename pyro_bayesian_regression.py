"""Simple implementation of a Bayesian Regression task.

Data is generated through: y = cos(x) + eps, where: eps ~ N(0,0.1)

The model used to fit is: y = w1*sigmoid(w0*x + b0) + b1 + eps, where eps ~ N(0,0.1)

The weights w0, w1, b0, b1 are not learned as point estimates, but through Bayesian Inference.
Hence, a helping distribution q(theta|params) is used to approximate the real posterior P(theta|data)

The set of weights is sampled from the approximate posterior q(theta|params).
"""
import pyro
from pyro import distributions as dist
import torch
import torch.distributions
import numpy as np
import matplotlib.pyplot as plt
pyro.set_rng_seed(101)


N = 100


def generate_data(N):
    x = np.linspace(-6, 1, num=N)
    y = np.cos(x) + np.random.normal(0, 0.1, size=N)
    x = x.astype(np.float32).reshape((N, 1))
    y = y.astype(np.float32).reshape((N, 1))
    return x, y


X, Y = generate_data(N)

def driver(X, Y):
    X = torch.tensor(X)
    Y = torch.tensor(Y)

    # priors
    w0 = pyro.sample(name="w0", fn=dist.Normal(loc=torch.zeros(1,2), scale=torch.ones(1,2)))
    w1 = pyro.sample(name="w1", fn=dist.Normal(loc=torch.zeros(2,1), scale=torch.ones(2,1)))
    b0 = pyro.sample(name="b0", fn=dist.Normal(loc=torch.zeros(2), scale=torch.ones(2)))
    b1 = pyro.sample(name="b1", fn=dist.Normal(loc=torch.zeros(1), scale=torch.ones(1)))

    # conditional
    for i in range(X.shape[0]):
        pyro.sample(name='y_%04d' % i,
                    fn=dist.Normal(loc=torch.matmul(torch.sigmoid((torch.matmul(X[i], w0) + b0)), w1) + b1, scale=.1),
                    obs=Y[i])


def parameterized(X, Y):
    # define params for mean
    qW0_loc = pyro.param(name="qW0_loc", init_tensor=torch.ones(1,2)*.01)
    qW1_loc = pyro.param(name="qW1_loc", init_tensor=torch.ones(2,1)*.01)
    qb0_loc = pyro.param(name="qb0_loc", init_tensor=torch.ones(2)*.01)
    qb1_loc = pyro.param(name="qb1_loc", init_tensor=torch.ones(1)*.01)

    # define params for std
    qW0_scale = pyro.param(name="qW0_scale", init_tensor=torch.ones(1,2), constraint=dist.constraints.positive)
    qW1_scale = pyro.param(name="qW1_scale", init_tensor=torch.ones(2,1), constraint=dist.constraints.positive)
    qb0_scale = pyro.param(name="qb0_scale", init_tensor=torch.ones(2), constraint=dist.constraints.positive)
    qb1_scale = pyro.param(name="qb1_scale", init_tensor=torch.ones(1), constraint=dist.constraints.positive)

    # sample from posterior
    w0 = pyro.sample(name="w0", fn=dist.Normal(loc=qW0_loc, scale=qW0_scale))
    w1 = pyro.sample(name="w1", fn=dist.Normal(loc=qW1_loc, scale=qW1_scale))
    b0 = pyro.sample(name="b0", fn=dist.Normal(loc=qb0_loc, scale=qb0_scale))
    b1 = pyro.sample(name="b1", fn=dist.Normal(loc=qb1_loc, scale=qb1_scale))

    return w0, w1, b0, b1


svi = pyro.infer.SVI(model=driver, guide=parameterized,
                     optim=pyro.optim.SGD({'lr':0.0001, "momentum":0.1}),
                     loss=pyro.infer.Trace_ELBO())

# training process
pyro.clear_param_store()
loss, qW0_loc, qW1_loc, qb0_loc, qb1_loc = [], [], [], [], []
for _ in range(1000):
    loss.append(svi.step(X, Y))
    qW0_loc.append(pyro.param("qW0_loc").detach().numpy())
    qW1_loc.append(pyro.param("qW1_loc").detach().numpy())
    qb0_loc.append(pyro.param("qb0_loc").detach().numpy())
    qb1_loc.append(pyro.param("qb1_loc").detach().numpy())

# sample a model
def sample_model():
    w0, w1, b0, b1 = parameterized(None, None)
    w0 = w0.detach()
    w1 = w1.detach()
    b0 = b0.detach()
    b1 = b1.detach()
    return w0, w1, b0, b1


def predict(x, w0, w1, b0, b1):
    return torch.matmul(torch.sigmoid((torch.matmul(x, w0) + b0)), w1) + b1


# visualize data points
plt.figure()
plt.title("data points")
plt.plot(X, Y, 'rx')
x1 = torch.linspace(-10, 10, 100).unsqueeze(-1)
for i in range(5):
    w0, w1, b0, b1 = sample_model()
    plt.plot(x1, predict(x1, w0, w1, b0, b1), 'o')
plt.show()
