import numpy as np
from scipy.stats.distributions import norm, beta
from scipy.integrate import quad
import matplotlib.pyplot as plt

from randomvars import Cont
from randomvars.options import config

# %% `from_sample()` from `Cont`
def sklearn_estimator_cont(*args, **kwargs):
    from sklearn.neighbors import KernelDensity

    def estimator_cont(x):
        dens = KernelDensity(*args, **kwargs)
        dens.fit(x.reshape(-1, 1))

        def res(x):
            x = np.asarray(x).reshape(-1, 1)
            return np.exp(dens.score_samples(x))

        return res

    return estimator_cont


def statsmodels_estimator_cont(*args, **kwargs):
    import statsmodels.api as sm

    def estimator_cont(x):
        density_class = sm.nonparametric.KDEUnivariate(x)
        density_class.fit()

        def res(x):
            return density_class.evaluate(x)

        return res

    return estimator_cont


def describe_output(rv, sample, name):
    estimator_cont = config.estimator_cont
    density = estimator_cont(sample)
    integral = quad(density, rv.x[0], rv.x[-1])[0]
    print(
        f"""
    {name}:
        Grid number of elements = {len(rv.x)}
        Integral coverage = {integral}
        Density range = {rv.x[0], rv.x[-1]}
    """
    )


np.random.seed(101)

# x = norm().rvs(size=10000)
x = np.concatenate([norm().rvs(size=500), norm(loc=100).rvs(size=500)])
# x = np.concatenate([norm().rvs(size=50), norm(loc=100).rvs(size=50)])
# x = beta(a=99, b=1).rvs(size=10000)
# beta1 = beta(a=10, b=20)
# beta2 = beta(a=40, b=10)
# x = np.concatenate([beta1.rvs(size=500), beta2.rvs(size=500)])
# true_pdf = lambda x: 0.5 * beta1.pdf(x) + 0.5 * beta2.pdf(x)

config.reset("estimator_cont")
rv_scipy = Cont.from_sample(x)
describe_output(rv_scipy, x, "Scipy")

with config.context({"estimator_cont": sklearn_estimator_cont()}):
    rv_sklearn = Cont.from_sample(x)
    describe_output(rv_sklearn, x, "Sklearn")

with config.context({"estimator_cont": statsmodels_estimator_cont()}):
    rv_statsmodels = Cont.from_sample(x)
    describe_output(rv_statsmodels, x, "Statsmodels")

plt.plot(rv_scipy.x, rv_scipy.y, "-k")
plt.plot(rv_sklearn.x, rv_sklearn.y, "-b")
plt.plot(rv_statsmodels.x, rv_statsmodels.y, "-m")
# plt.plot(rv_scipy.x, true_pdf(rv_scipy.x), "-r")
plt.show()


# %% Stress testing
import time

import scipy.stats.distributions as distrs


DISTRIBUTIONS = {
    # Common distributions
    "beta": distrs.beta(a=10, b=20),
    "chi_sq": distrs.chi2(df=10),
    "expon": distrs.expon(),
    "f": distrs.f(dfn=20, dfd=20),
    "gamma": distrs.gamma(a=10),
    "lognorm": distrs.lognorm(s=0.5),
    "norm": distrs.norm(),
    "norm2": distrs.norm(loc=10),
    "norm3": distrs.norm(scale=0.1),
    "norm4": distrs.norm(scale=10),
    "norm5": distrs.norm(loc=10, scale=0.1),
    "t": distrs.t(df=10),
    "uniform": distrs.uniform(),
    "uniform2": distrs.uniform(loc=10, scale=0.1),
    "weibull_max": distrs.weibull_max(c=2),
    "weibull_min": distrs.weibull_min(c=2),
    # Distributions with infinite density
    "inf_beta_both": distrs.beta(a=0.4, b=0.6),
    "inf_beta_left": distrs.beta(a=0.5, b=2),
    "inf_beta_right": distrs.beta(a=2, b=0.5),
    "inf_chi_sq": distrs.chi2(df=1),
    "inf_weibull_max": distrs.weibull_max(c=0.5),
    "inf_weibull_min": distrs.weibull_min(c=0.5),
    # Distributions with heavy tails
    "heavy_cauchy": distrs.cauchy(),
    "heavy_lognorm": distrs.lognorm(s=1),
    "heavy_t": distrs.t(df=2),
}
distr_names = np.array(list(DISTRIBUTIONS.keys()))


def test_from_sample_accuracy(rng, low, high):
    distr_name = rng.choice(distr_names, size=1)[0]
    size = rng.integers(low=low, high=high, size=1)[0]
    x = DISTRIBUTIONS[distr_name].rvs(size=size, random_state=rng)

    time_start = time.time()
    rv = Cont.from_sample(x)
    time_end = time.time()

    density = config.estimator_cont(x)
    max_diff = np.max(np.abs(density(rv.x) - rv.y)) * 10 ** 5

    print(
        f"Distr: {distr_name:15} Sample size: {len(x):3} "
        f"Grid size: {len(rv.x):4} Max. diff.: {max_diff:3.0f}e-5 "
        f"Duration: {(time_end-time_start)*1000:4.0f} ms"
    )


rng = np.random.default_rng(1001)

for _ in range(100):
    test_from_sample_accuracy(rng, low=2, high=10)

for _ in range(100):
    test_from_sample_accuracy(rng, low=100, high=1001)

for _ in range(100):
    test_from_sample_accuracy(rng, low=1001, high=1002)

for _ in range(100):
    test_from_sample_accuracy(rng, low=10001, high=10002)
