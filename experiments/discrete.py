import numpy as np
from scipy.stats.distributions import rv_discrete

from randomvars._discrete import Disc

x = [0, 1, 2]
prob = [0.1, 0.2, 0.7]
vals = np.array([-1, 0, 0 + 1e-8, 1, 2, 2 + 1e-8]).reshape(2, 3)

my_disc = Disc(x, prob)

my_disc.pmf(vals)
my_disc.cdf(vals)
my_disc.ppf(vals)
my_disc.rvs(size=(3, 2))

my_disc.mean()
my_disc.support()
my_disc.interval(0.5)

# Example of problems due to floating point representation
rv = Disc([0.5, 1, 3], [0.1, 0.2, 0.7])
rv.pmf(0.5)

rv_2 = rv_discrete(values=([0.5, 1, 3], [0.1, 0.2, 0.7]))
rv_2.pmf(0.5)


def pmf_scipy(vals, x_ref, prob):
    return np.select(
        [vals == k for k in x_ref],
        [np.broadcast_arrays(p, vals)[0] for p in prob],
        0,
    )


def pmf_my(vals, x_ref, prob):
    inds = np.searchsorted(x_ref, vals)
    inds = np.clip(inds, 0, len(x_ref) - 1)
    # return np.where(np.isclose(x_ref[inds], vals, rtol=0), prob[inds], 0)
    return np.where(x_ref[inds] == vals, prob[inds], 0)


vals = np.array([0, 0.5, 1, 1 + 1e-7])
x_ref = np.array([0.5, 1])
prob = np.array([0.1, 0.9])

pmf_scipy(vals, x_ref, prob)
pmf_my(vals, x_ref, prob)

n = 1000
x_ref = np.sort(np.random.rand(n))
prob = np.random.rand(n)
prob = prob / np.sum(prob)
vals = np.concatenate([x, x + 1e-7])

# %timeit pmf_scipy(vals, x_ref, prob)
# %timeit pmf_my(vals, x_ref, prob)
