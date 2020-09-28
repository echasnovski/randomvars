""" Code for discrete random variable
"""

import numpy as np
from scipy.stats.distributions import rv_discrete

import randomvars.options as op
import randomvars._utils as utils


class Disc(rv_discrete):
    """Discrete random variable

    Class for discrete random variable with **finite number of (finite) values**.
    It is similar to `rv_sample` class from `scipy.stats.distributions`, but works with
    float numbers as distributions values (opposite to only integers in `rv_sample`).

    Main way to create instance of `Disc` is to directly supply values (`x`)
    and probabilities (`prob`) of distribution:
    ```
        my_disc = Disc(x=[1.618, 2.718, 3.141], prob=[0.1, 0.2, 0.7])
        my_disc.pmf([1.618, 1.619])
    ```
    """

    def __new__(cls, x, prob, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, x, prob, *args, **kwargs):
        x, prob = self._impute_xprob(x, prob)

        self._x = x
        self._prob = prob
        self._p = np.cumsum(prob)

        # Not using `values` argument, as currently it doesn't seem to be supported
        # in `__init__`. As usage of `values` results into object of class
        # `rv_sample` (which is not publicaly exported), core functionality is
        # explicitly taken from it.
        super(Disc, self).__init__(a=x[0], b=x[-1], *args, **kwargs)

    @staticmethod
    def _impute_xprob(x, prob):
        x = utils._as_1d_finite_float(x, "x")
        prob = utils._as_1d_finite_float(prob, "prob")

        x, prob = utils._sort_parallel(x, prob, y_name="prob", warn=True)

        utils._assert_positive(prob, "prob")

        prob = prob / np.sum(prob)

        return x, prob

    @property
    def x(self):
        """Return values of discrete distribution"""
        return self._x

    @property
    def prob(self):
        """Return probabilities of discrete distribution"""
        return self._prob

    @property
    def p(self):
        """Return cumulative probabilities of discrete distribution"""
        return self._p

    def _pmf(self, x):
        """Probability mass function

        Return values of probability mass function at points `x`.

        **Note** that probability is taken from object probabilities if input
        value is "close enough" to the corresponding value of object's `x`.
        Function `numpy.isclose()` is used for that, with relative and absolute
        tolerance values taken from `tolerance` package option. See
        documentation of `randomvars.options.get_option()` for more
        information.

        Parameters
        ----------
        x : array_like with numeric values

        Returns
        -------
        pmf_vals : ndarray with shape inferred from `x`
        """
        rtol, atol = op.get_option("tolerance")

        inds = utils._find_nearest_ind(x, self.x)

        x_is_matched = np.isclose(x, self.x[inds], rtol=rtol, atol=atol)

        res = np.where(x_is_matched, self.prob[inds], 0)
        return utils._copy_nan(fr=x, to=res)

    # Override default `rv_discrete`'s `_pmf` to `pmf` transition to properly
    # work with "tolerance matching"
    pmf = _pmf

    def _cdf(self, x):
        """Cumulative distribution function

        Return values of cumulative distribution function at points `x`.

        Parameters
        ----------
        x : array_like with numeric values

        Returns
        -------
        cdf_vals : ndarray with shape inferred from `x`
        """
        inds = np.searchsorted(self.x, x, side="right")
        # This is needed to avoid possible confusion at index 0 when subsetting
        # `self.p`
        inds_clipped = np.maximum(inds, 1)

        res = np.ones_like(x, dtype=np.float64)
        res = np.where(inds == 0, 0.0, self.p[inds_clipped - 1])

        return utils._copy_nan(fr=x, to=res)

    # Override default `rv_discrete`'s `_cdf` to `cdf` transition for speed reasons
    cdf = _cdf

    def _ppf(self, q):
        """Percent point (quantile, inverse of cdf) function

        Return values of percent point (quantile, inverse of cdf) function at
        cumulative probabilities `q`.

        Parameters
        ----------
        q : array_like with numeric values

        Returns
        -------
        ppf_vals : ndarray with shape inferred from `q`
        """
        q_inds = np.searchsorted(self.p, q, side="left")
        # This is needed to avoid `IndexError` in later `np.where()` call
        q_inds_clipped = np.minimum(q_inds, len(self.p) - 1)

        res = np.empty_like(q, dtype=np.float64)
        res = np.where(q_inds != len(self.p), self.x[q_inds_clipped], res)
        res[(q < 0) | (q > 1)] = np.nan

        return utils._copy_nan(fr=q, to=res)

    # Override default `rv_discrete`'s `_ppf` to `ppf` transition to change behavior
    # of `ppf(0)` (which should return minimum element of distribution `x_min` and not
    # `x_min - 1`)
    ppf = _ppf

    def _rvs(self, size=None, random_state=None):
        """Random number generation

        Generate random numbers into array of desired size.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Defining number of random variates (default is 1).
        random_state : `None` or RandomState, optional
            Source of uniform random number generator. If `None`,
            `numpy.random.uniform()` is used.
        """
        if random_state is None:
            U = np.random.uniform(size=size)
        else:
            U = random_state.uniform(size=size)
        return self._ppf(U)

    # Override default `rv_discrete`'s `_rvs` to `rvs` transition to make it possible to
    # work with non-integers. `scipy`'s version produces only integer output.
    rvs = _rvs
