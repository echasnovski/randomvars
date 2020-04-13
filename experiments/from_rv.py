import os
import sys

import numpy as np
import scipy.stats.distributions as distrs
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("../randomvars"))
from rv_piecelin import rv_piecelin

from regrid_maxtol import regrid_maxtol


def plot_pdf(self):
    fig, ax = plt.subplots()
    ax.plot(*self.pdf_grid())

    plt.show()


setattr(rv_piecelin, "plot", plot_pdf)


def from_rv(rv, supp=None, n_grid=10001, integr_tol=1e-4, *args, **kwargs):
    left, right = get_rv_supp(rv, supp, *args, **kwargs)
    x = np.linspace(left, right, n_grid)
    y = np.clip(np.gradient(rv.cdf(x), x), 0, np.inf)

    x, y = regrid_maxtol(x, y, integr_tol / (right - left))

    return rv_piecelin(x, y)


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

        x, y = rv_test.pdf_grid()
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
            np.linspace(x[i], x[i + 1], n_inner_points + 2)
            for i in np.arange(len(x) - 1)
        ]
        return np.unique(np.concatenate(test_arr))

    @staticmethod
    def _vec_summary(diff):
        diff_noninf = diff[~np.isinf(diff)]
        quans = np.quantile(diff, [0.25, 0.5, 0.75])

        return {
            "min": np.min(diff),
            "min_noninf": np.min(diff_noninf),
            "Q0.25": quans[0],
            "median": quans[1],
            "mean": np.mean(diff),
            "Q0.75": quans[2],
            "max_noninf": np.max(diff_noninf),
            "max": np.max(diff),
        }


rv = distrs.norm()
args = tuple()
kwargs = dict()

# from_rv(rv, tail_prob=0.1).plot()
# from_rv(rv, supp=[-2, None]).plot()
# from_rv(rv, supp=[-2, None]).pdf_grid()

rv_diff = RVDiff(rv, from_rv(rv))
rv_diff.diff_summary()

# rv = distrs.norm()
# rv = distrs.beta(a=10, b=10)
# rv = distrs.chi2(df=2)
rv = distrs.poisson(mu=10)
rv_diff = RVDiff(rv, from_rv(rv, n_grid=100001, integr_tol=1e-5))
rv_diff.plot_diff()
rv_diff.rv_test.pdf_grid()[0].size
rv_diff.diff_summary()

# Test for approximating discrete distributions
rv = distrs.poisson(mu=10)
rv_test = from_rv(rv)
rv_test.plot()
## It does approximates pretty good!
