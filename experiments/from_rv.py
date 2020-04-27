import os
import sys

import numpy as np
import scipy.stats.distributions as distrs
import matplotlib.pyplot as plt

# This uses "development" version of `rv_pieceline`. instead of "installed in
# current virtual environment" version.
sys.path.insert(0, os.path.abspath("../randomvars"))
from rv_piecelin import rv_piecelin

from regrid_maxtol import regrid_maxtol


#%% Functions
def plot_pdf(self):
    fig, ax = plt.subplots()
    ax.plot(self.x, self.y)

    plt.show()


def mean_piecelin(self):
    x_lag = self._x[:-1]
    x_lead = self._x[1:]
    y_lag = self._y[:-1]
    y_lead = self._y[1:]
    # x_lag = x[:-1]
    # x_lead = x[1:]
    # y_lag = y[:-1]
    # y_lead = y[1:]
    y_sum = y_lag + y_lead
    x_mass = (x_lag * (y_lag + y_sum) + x_lead * (y_lead + y_sum)) / (3 * y_sum)

    prob = np.diff(self._p)
    # prob = np.diff(_trapez_integral_cum(x, y))

    return np.nansum(x_mass * prob)


setattr(rv_piecelin, "plot", plot_pdf)
setattr(rv_piecelin, "mean", mean_piecelin)


def max_finite(x):
    return np.max(x[np.isfinite(x)])


def min_finite(x):
    return np.min(x[np.isfinite(x)])


def science_notation(x):
    return f"{x:.2E}"


def from_rv_equi(rv, supp=None, n_grid=1001, integr_tol=1e-4, *args, **kwargs):
    left, right = get_rv_supp(rv, supp, *args, **kwargs)
    x = np.linspace(left, right, n_grid)
    # Using `edge_order=2` in `np.gradient()` improves approximation on edges
    # Using `np.clip()` to avoid possible negative values of `1e-16` order of magnitude
    y = np.clip(np.gradient(rv.cdf(x), x, edge_order=2), 0, np.inf)

    x, y = regrid_maxtol(x, y, integr_tol / (right - left))

    return rv_piecelin(x, y)


def from_rv_double(
    rv, supp=None, n_grid=1001, integr_tol=1e-4, double_pass=True, *args, **kwargs
):
    x_left, x_right = get_rv_supp(rv, supp, *args, **kwargs)
    x_equi = np.linspace(x_left, x_right, n_grid)

    prob_left, prob_right = rv.cdf([x_left, x_right])
    prob_equi = np.linspace(prob_left, prob_right, n_grid)
    x_quan = rv.ppf(prob_equi)

    x = combine_x(x_equi, x_quan)
    y = np.clip(np.gradient(rv.cdf(x), x, edge_order=2), 0, np.inf)

    x, y = regrid_maxtol(x, y, integr_tol / (x[-1] - x[0]), double_pass=double_pass)

    return rv_piecelin(x, y)


def combine_x(x1, x2, min_diff=1e-13):
    x = np.concatenate([x1, x2])
    x = np.sort(x)
    x_is_good = np.concatenate([[True], np.diff(x) > min_diff])
    return x[x_is_good]


def get_rv_supp_probs(rv, tail_prob=1e-6):
    if np.isfinite(rv.ppf(0)):
        left = 0
    else:
        left = tail_prob

    if np.isfinite(rv.ppf(1)):
        right = 1
    else:
        right = 1 - tail_prob

    return left, right


def augment_grid(x, n_inner_points):
    test_arr = [
        np.linspace(x[i], x[i + 1], n_inner_points + 1, endpoint=False)
        for i in np.arange(len(x) - 1)
    ]
    test_arr.append([x[-1]])
    return np.concatenate(test_arr)


def get_rv_supp(rv, supp, tail_prob=1e-6, *args, **kwargs):
    if supp is None:
        supp = [None, None]

    if supp[0] is None:
        left = rv.ppf(0)
        if np.isneginf(left):
            left = rv.ppf(tail_prob)
    else:
        left = supp[0]

    if supp[1] is None:
        right = rv.ppf(1)
        if np.isposinf(right):
            right = rv.ppf(1 - tail_prob)
    else:
        right = supp[1]

    return left, right


class RVDiff:
    def __init__(self, rv_base, rv_test, n_inner_points=10):
        self.rv_base = rv_base
        self.rv_test = rv_test

        x, y = rv_test.x, rv_test.y
        self.x_test = self._augment_grid(x, n_inner_points)
        self.p_test = self._augment_grid(rv_test.cdf(x), n_inner_points)

    def pdf_diff(self):
        x = self.x_test
        diff = self.rv_base.pdf(x) - self.rv_test.pdf(x)
        return x, diff

    def cdf_diff(self):
        x = self.x_test
        diff = self.rv_base.cdf(x) - self.rv_test.cdf(x)
        return x, diff

    def ppf_diff(self):
        p = self.p_test
        diff = self.rv_base.ppf(p) - self.rv_test.ppf(p)
        return p, diff

    def diff_summary(self, use_abs=True):
        diff_dict = {
            "Density": self.pdf_diff()[1],
            "CDF": self.cdf_diff()[1],
            "Quantile": self.ppf_diff()[1],
        }

        if use_abs:
            diff_dict = {key: np.abs(val) for key, val in diff_dict.items()}

        return {key: self._vec_summary(val) for key, val in diff_dict.items()}

    def plot_diff(self, use_abs=False):
        diff_dict = {
            "Density": self.pdf_diff(),
            "CDF": self.cdf_diff(),
            "Quantile": self.ppf_diff(),
        }

        if use_abs:
            diff_dict = {
                key: (val[0], np.abs(val[1])) for key, val in diff_dict.items()
            }

        fig, axs = plt.subplots(3, 1)

        axs[0].axhline(y=0, color="red", lw=0.4)
        axs[0].plot(*diff_dict["Density"])
        axs[0].set_title("Density errors")

        axs[1].axhline(y=0, color="red", lw=0.4)
        axs[1].plot(*diff_dict["CDF"])
        axs[1].set_title("CDF errors")

        axs[2].axhline(y=0, color="red", lw=0.4)
        axs[2].plot(*diff_dict["Quantile"])
        axs[2].set_title("Quantile function errors")

        plt.show()

    @staticmethod
    def _augment_grid(x, n_inner_points=10):
        test_arr = [
            np.linspace(x[i], x[i + 1], n_inner_points + 1, endpoint=False)
            for i in np.arange(len(x) - 1)
        ]
        test_arr.append([x[-1]])
        return np.concatenate(test_arr)

    @staticmethod
    def _vec_summary(diff):
        diff_finite = diff[~np.isinf(diff)]
        quans = np.quantile(diff, [0.25, 0.5, 0.75])

        return {
            "min": np.min(diff),
            "min_finite": np.min(diff_finite),
            "Q0.25": quans[0],
            "median": quans[1],
            "mean": np.mean(diff),
            "Q0.75": quans[2],
            "max_finite": np.max(diff_finite),
            "max": np.max(diff),
        }


def compute_max_abserrors(rv_base, rv_test):
    rv_diff = RVDiff(rv_base, rv_test)

    pdf_diff = rv_diff.pdf_diff()[1]
    density_res = max_finite(pdf_diff)

    cdf_diff = rv_diff.cdf_diff()[1]
    cdf_res = max_finite(cdf_diff)

    return {
        "Grid Size": rv_diff.rv_test.x.size,
        "Density": science_notation(density_res),
        "CDF": science_notation(cdf_res),
    }


#%% Experiments
args = tuple()
kwargs = dict()


distr_dict = {
    "Normal": distrs.norm(),
    "Normal2": distrs.norm(loc=100, scale=0.1),
    "Beta": distrs.beta(a=10, b=10),
    "Beta_inf": distrs.beta(a=0.5, b=0.7),
    "Chi2": distrs.chi2(df=30),
    "Chi2_inf": distrs.chi2(df=1),
    "Student": distrs.t(df=30),
    "Cauchy": distrs.cauchy(),
    "Pareto": distrs.pareto(b=1),
}

{
    key: compute_max_abserrors(rv, from_rv_equi(rv, n_grid=2001))
    for key, rv in distr_dict.items()
}

{
    key: compute_max_abserrors(rv, from_rv_double(rv, n_grid=1001))
    for key, rv in distr_dict.items()
}

# Test for approximating discrete distributions
rv = distrs.poisson(mu=10)
# rv_test = from_rv_double(rv)
rv_test = from_rv_equi(rv)
rv_test.plot()

x_test = np.arange(20)
x_test_cdf = np.concatenate([[-np.inf], 0.5 * (x_test[:-1] + x_test[1:]), [np.inf]])
print(rv.pmf(x_test) - np.diff(rv_test.cdf(x_test_cdf)))
## CONCLUSION: only equidistant grid approximates discrete RV, doubly
## equidistant - not yet

# Time measurements
rv = distrs.norm()
%timeit from_rv_equi(rv)
%timeit from_rv_equi(rv, n_grid=2001)
%timeit from_rv_double(rv)
%timeit from_rv_equi(rv, integr_tol=1e-3)
%timeit from_rv_double(rv, integr_tol=1e-3)


#%% Effect of double pass in `regrid_maxtol()`
distrs_piecelin = {
    key: (from_rv_double(rv, double_pass=False), from_rv_double(rv, double_pass=True))
    for key, rv in distr_dict.items()
}

# Comparisons
for rv_name, (single, double) in distrs_piecelin.items():
    print(rv_name)
    print("  Grid size")
    print(
        f"    Single pass: {single.x.size}, double pass: {double.x.size}"
    )
    print("  Mean")
    print(f"    Single pass: {single.mean()}, double pass: {double.mean()}")

{key: rv.mean() for key, rv in distrs_piecelin.items()}


# Timings
n_points = 10000
np.random.RandomState(101)
x = np.unique(np.round(np.sort(np.random.randn(n_points)), decimals=6))
y = np.random.randn(len(x))

%timeit regrid_maxtol(x, y, tol=1e-1, double_pass=False)
%timeit regrid_maxtol(x, y, tol=1e-1, double_pass=True)
