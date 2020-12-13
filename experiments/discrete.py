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


# %% Probability mass function
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


# %% Cdf function
import randomvars._utils as utils


def cdf_my(rv, x):
    inds = np.searchsorted(rv.x, x, side="right")
    res = np.ones_like(x, dtype="float64")
    res = np.where(inds == 0, 0.0, rv._cump[inds - 1])

    return utils._copy_nan(fr=x, to=res)


def cdf_scipy(rv, x):
    xx, xxk = np.broadcast_arrays(x[:, None], rv.x)
    indx = np.argmax(xxk > xx, axis=-1) - 1
    return rv._cump[indx]


n = 1000
x = np.sort(np.random.rand(n))
prob = np.random.rand(n)
prob = prob / np.sum(prob)
rv = Disc(x=x, prob=prob)
vals = np.concatenate([x, x + 1e-7, x - 1e-7])

# %timeit cdf_scipy(rv, x)
# %timeit cdf_my(rv, x)


# %% Quantile function
x = np.array([0.5, 1, 3])
prob = np.array([0.1, 0.2, 0.7])

h = 1e-12
q_vec = np.array([0, 0.1 - h, 0.1, 0.1 + h, 0.3 - h, 0.3, 0.3 + h, 1 - h, 1])

rv_scipy = rv_discrete(values=(x, prob))
rv_scipy.ppf(q_vec)
rv_scipy.ppf(np.array([-np.inf, -h, np.nan, 1 + h, np.inf]))

rv_my = Disc(x, prob)
rv_my.ppf(q_vec)
rv_my.ppf(np.array([-np.inf, -h, np.nan, 1 + h, np.inf]))


# %% from_rv
import scipy.stats as ss

from randomvars import Disc
import randomvars.options as op


def disc_from_rv(rv):
    # Get options
    small_prob = op.get_option("small_prob")

    # Construct values
    x = []
    prob = []
    tot_prob = 0.0

    while tot_prob < 1 - small_prob:
        cur_x = rv.ppf(tot_prob + small_prob)
        cur_tot_prob = rv.cdf(cur_x)

        if cur_tot_prob <= tot_prob:
            raise ValueError(
                "`Disc.from_rv`: Couldn't get increase in total probability. "
                "Check corretness of `ppf` and `cdf` methods."
            )

        x.append(cur_x)
        prob.append(cur_tot_prob - tot_prob)

        tot_prob = cur_tot_prob

    return Disc(x=x, prob=prob)


rv = ss.poisson(mu=10)
rv_disc = disc_from_rv(rv)

rv.pmf(rv_disc.x) - rv_disc.pmf(rv_disc.x)
1 - rv.cdf(np.max(rv_disc.x))
