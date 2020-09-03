import numpy as np
from scipy.stats.distributions import norm, beta
from scipy.integrate import quad
import matplotlib.pyplot as plt

from randomvars import rv_piecelin
import randomvars.options as op

np.random.seed(101)

# x = norm().rvs(size=10000)
x = np.concatenate([norm().rvs(size=500), norm(loc=100).rvs(size=500)])
# x = np.concatenate([norm().rvs(size=50), norm(loc=100).rvs(size=50)])
# x = beta(a=99, b=1).rvs(size=10000)
# beta1 = beta(a=10, b=20)
# beta2 = beta(a=40, b=10)
# x = np.concatenate([beta1.rvs(size=500), beta2.rvs(size=500)])
# true_pdf = lambda x: 0.5 * beta1.pdf(x) + 0.5 * beta2.pdf(x)


# %% `from_sample()` from `rv_piecelin`
def sklearn_density_estimator(*args, **kwargs):
    from sklearn.neighbors import KernelDensity

    def density_estimator(x):
        dens = KernelDensity(*args, **kwargs)
        dens.fit(x.reshape(-1, 1))

        def res(x):
            x = np.asarray(x).reshape(-1, 1)
            return np.exp(dens.score_samples(x))

        return res

    return density_estimator


def statsmodels_density_estimator(*args, **kwargs):
    import statsmodels.api as sm

    def density_estimator(x):
        density_class = sm.nonparametric.KDEUnivariate(x)
        density_class.fit()

        def res(x):
            return density_class.evaluate(x)

        return res

    return density_estimator


def describe_output(rv, sample, name):
    density_estimator = op.get_option("density_estimator")
    density = density_estimator(sample)
    integral = quad(density, rv.x[0], rv.x[-1])[0]
    print(
        f"""
    {name}:
        Grid number of elements = {len(rv.x)}
        Integral coverage = {integral}
        Density range = {rv.x[0], rv.x[-1]}
    """
    )


op.reset_option("density_estimator")
rv_scipy = rv_piecelin.from_sample(x)
describe_output(rv_scipy, x, "Scipy")

with op.option_context({"density_estimator": sklearn_density_estimator()}):
    rv_sklearn = rv_piecelin.from_sample(x)
    describe_output(rv_sklearn, x, "Sklearn")

with op.option_context({"density_estimator": statsmodels_density_estimator()}):
    rv_statsmodels = rv_piecelin.from_sample(x)
    describe_output(rv_statsmodels, x, "Statsmodels")

plt.plot(rv_scipy.x, rv_scipy.y, "-k")
plt.plot(rv_sklearn.x, rv_sklearn.y, "-b")
plt.plot(rv_statsmodels.x, rv_statsmodels.y, "-m")
# plt.plot(rv_scipy.x, true_pdf(rv_scipy.x), "-r")
plt.show()
