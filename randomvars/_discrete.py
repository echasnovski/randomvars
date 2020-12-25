""" Code for discrete random variable
"""

import numpy as np
from scipy.stats.distributions import rv_frozen

from randomvars._random import Rand
import randomvars.options as op
import randomvars._utils as utils


class Disc(Rand):
    """Discrete random variable

    Class for discrete random variable with **finite number of (finite and
    unique) values**. Defined by xp-grid of probability mass function: values
    in x-grid and their probabilities in p-grid. It is similar to (unexported)
    `rv_sample` class from `scipy.stats.distributions`, but works with float
    numbers as distribution values (opposite to focusing on integers in
    `rv_sample`).

    There are three ways to create instance of `Disc` class:

    1. Directly supply parts of xp-grid (`x` for x-grid and `p` for p-grid):
    ```
        my_disc = Disc(x=[1.618, 2.718, 3.141], p=[0.1, 0.2, 0.7])
        my_disc.pmf([1.618, 1.619])
    ```
    2. Use `Disc.from_rv()` to create approximation of some existing discrete
    random variable (object with methods `cdf()` and `ppf()`):
    ```
        from scipy.stats import binom
        rv_binom = binom(n=10, p=0.5)
        my_binom = Disc.from_rv(rv_binom)
        rv_binom.pmf([0, 5, 10])
        my_binom.pmf([0, 5, 10])

        # In general, `Disc` represents approximation to input random variable
        # as it might not detect x-values with small probabilities (see
        # documentation of `Disc.from_rv` for more information)
        rv_binom_wide = binom(n=100, p=0.5)
        my_binom_wide = Disc.from_rv(rv_binom_wide)
        ## Values in tails are not detected as they have low probabilities
        my_binom_wide.x
    ```
    3. Use `Disc.from_sample()` to create estimation based on some existing sample:
    ```
        from scipy.stats import binom
        sample = binom(n=10, p=0.1).rvs(size=100, random_state=101)
        my_rv = Disc.from_sample(sample)
        my_rv.pmf([0, 1, 10])
    ```
    """

    def __init__(self, x, p):
        x, p = self._impute_init_args(x, p)

        # User-facing attributes
        self._x = x
        self._p = p
        self._a = x[0]
        self._b = x[-1]

        # Private attributes
        self._cump = np.cumsum(p)

        super().__init__()

    @staticmethod
    def _impute_init_args(x, p):
        x = utils._as_1d_numpy(x, "x", chkfinite=True, dtype="float64")
        p = utils._as_1d_numpy(p, "p", chkfinite=True, dtype="float64")

        x, p = utils._sort_parallel(x, p, y_name="p", warn=True)

        if not np.all(np.diff(x) > 0):
            x, p = utils._unique_parallel(x, p, warn=True)

        utils._assert_positive(p, "p")

        p = p / np.sum(p)

        return x, p

    def __str__(self):
        x_len = len(self._x)
        s = "s" if x_len > 1 else ""
        return f"Discrete RV with {x_len} value{s} (support: [{self._a}, {self._b}])"

    @property
    def x(self):
        """Return x-grid (values of discrete distribution)"""
        return self._x

    @property
    def p(self):
        """Return p-grid (probabilities of discrete distribution)"""
        return self._p

    @property
    def a(self):
        """Return left edge of support"""
        return self._a

    @property
    def b(self):
        """Return right edge of support"""
        return self._b

    # `support()` is inherited from `Rand`

    @classmethod
    def from_rv(cls, rv):
        """Create discrete RV from general RV

        Discrete RV with finite number of values is created by iteratively
        searching for x-values with positive probability. This is done by
        "stepping" procedure with step size equal to `small_prob` (package
        option). It uses combination of `cdf()` (cumulative distribution
        function) and `ppf()` (quantile function) methods to walk across [0, 1]
        interval of cumulative probability.

        Single step tracks current cumulative probability `tot_prob` and has
        the following algorithm:
        - **Find next x-value `new_x`** as value of `ppf()` at `tot_prob +
          small_prob` ("make `small_prob` step"). **Note** that this means
          possibly skipping x-values with small probability.
        - **Find total probability `new_tot_prob` at x-value** as value of
          `cdf(new_x)`. This will usually be bigger than `tot_prob +
          small_prob`.
        - **Compute probability at new x-value** as difference `new_tot_prob -
          tot_prob`. If there are skipped x-values with small probabilities,
          those are automatically "squashed" to new x-value.
        - **Make `tot_prob` equal to `new_tot_prob`**.

        Iterations start with total probability being zero and end when it
        surpasses `1 - small_prob`.

        **Notes**:
        - If `rv` is already an object of class `Disc`, it is returned
          untouched.
        - By the nature of "stepping" procedure, output random variable will
          automatically have "trimmed tails" if they consist from x-values
          with small probabilities. This might result into fewer elements in
          output than there is in input. For example, binomial distribution
          with `n=100` and `p=0.5` by default will not have all elements from 0
          to 100, but only the ones close enough to 50.
        - It can take much time to complete if there are many points with
          positive probability, because it will result into many calls of
          `cdf()` and `ppf()` methods.

        Relevant package options: `small_prob`. See documentation of
        `randomvars.options.get_option()` for more information. To temporarily
        set options use `randomvars.options.option_context()` context manager.

        Parameters
        ----------
        rv : Object with methods `cdf()` and `ppf()`
            Methods `cdf()` and `ppf()` should implement functions for
            cumulative distribution and quantile functions respectively.
            Recommended to be an object of class
            `scipy.stats.distributions.rv_frozen` (`rv_discrete` with all
            hyperparameters defined).

        Returns
        -------
        rv_out : Disc
            Discrete random variable with **finite number of (finite and
            unique) values** which approximates probability distribution of
            input `rv`.
        """
        # Make early return
        if isinstance(rv, Disc):
            return rv

        # Check input
        rv_dir = dir(rv)
        if not all(method in rv_dir for method in ["cdf", "ppf"]):
            raise ValueError("`rv` should have methods `cdf()` and `ppf()`.")

        # Get options
        small_prob = op.get_option("small_prob")

        if (small_prob <= 0) or (small_prob >= 1):
            raise ValueError(
                "Option `small_prob` in `Disc.from_rv` should be "
                "bigger than 0 and smaller than 1."
            )

        # Find values with non-zero probability mass
        x = []
        p = []
        tot_prob = 0.0

        while tot_prob <= 1 - small_prob:
            cur_x = rv.ppf(tot_prob + small_prob)
            cur_tot_prob = rv.cdf(cur_x)

            # Try to guard from infinite loop
            if cur_tot_prob <= tot_prob:
                raise ValueError(
                    "Couldn't get increase of total probability in `Disc.from_rv`. "
                    "Check correctness of `ppf` and `cdf` methods."
                )

            x.append(cur_x)
            p.append(cur_tot_prob - tot_prob)

            tot_prob = cur_tot_prob

        return cls(x=x, p=p)

    @classmethod
    def from_sample(cls, sample):
        """Create discrete RV from sample

        Discrete RV is created by the following algorithm:
        - **Estimate distribution** with discrete estimator (taken from package
          option "discrete_estimator") in the form `estimate =
          discrete_estimator(sample)`. If `estimator` is object of class
          `Disc`, it is returned untouched. If it is an object of
          `scipy.stats.distributions.rv_frozen` (`rv_discrete` with all
          hyperparameters defined), it is forwarded to `Disc.from_rv()`.
        - **Create random variable** with `Disc(x=x, p=p)`, where `x` and `p`
          are first and second values of `estimate`.

        Relevant package options: `discrete_estimator`. See documentation of
        `randomvars.options.get_option()` for more information. To temporarily
        set options use `randomvars.options.option_context()` context manager.

        Parameters
        ----------
        sample : 1d array-like
            This should be a valid input to `np.asarray()` so that its output
            is numeric and has single dimension.

        Returns
        -------
        rv_out : Disc
            Discrete random variable with **finite number of (finite) values**
            which is an estimate based on input `sample`.
        """
        # Check and prepare input
        sample = utils._as_1d_numpy(sample, "sample", chkfinite=False, dtype="float64")

        # Get options
        discrete_estimator = op.get_option("discrete_estimator")

        # Estimate distribution
        estimate = discrete_estimator(sample)

        # Make early return if `estimate` is random variable
        if isinstance(estimate, Disc):
            return estimate
        if isinstance(estimate, rv_frozen):
            return Disc.from_rv(estimate)

        return cls(x=estimate[0], p=estimate[1])

    def pdf(self, x):
        raise AttributeError(
            "`Disc` doesn't have probability density function. Use `pmf()` instead."
        )

    def logpdf(self, x):
        raise AttributeError(
            "`Disc` doesn't have probability density function. Use `logpmf()` instead."
        )

    def pmf(self, x):
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
        x = np.asarray(x, "float64")

        rtol, atol = op.get_option("tolerance")

        inds = utils._find_nearest_ind(x, self._x)

        x_is_matched = np.isclose(x, self._x[inds], rtol=rtol, atol=atol)

        res = np.where(x_is_matched, self._p[inds], 0)

        return np.asarray(utils._copy_nan(fr=x, to=res), dtype="float64")

    # `logpmf()` is inherited from `Rand`

    def cdf(self, x):
        """Cumulative distribution function

        Return values of cumulative distribution function at points `x`.

        Parameters
        ----------
        x : array_like with numeric values

        Returns
        -------
        cdf_vals : ndarray with shape inferred from `x`
        """
        x = np.asarray(x, dtype="float64")

        inds = np.searchsorted(self._x, x, side="right")
        # This is needed to avoid possible confusion at index 0 when subsetting
        # `self._cump`
        inds_clipped = np.maximum(inds, 1)

        res = np.ones_like(x, dtype="float64")
        res = np.where(inds == 0, 0.0, self._cump[inds_clipped - 1])

        return np.asarray(utils._copy_nan(fr=x, to=res), dtype="float64")

    # `logcdf()` is inherited from `Rand`

    # `sf()` is inherited from `Rand`

    def ppf(self, q):
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
        q = np.asarray(q, dtype="float64")

        q_inds = np.searchsorted(self._cump, q, side="left")
        # This is needed to avoid `IndexError` in later `np.where()` call
        q_inds_clipped = np.minimum(q_inds, len(self._cump) - 1)

        res = np.empty_like(q, dtype="float64")
        res = np.where(q_inds != len(self._cump), self._x[q_inds_clipped], res)
        res[(q < 0) | (q > 1)] = np.nan

        return np.asarray(utils._copy_nan(fr=q, to=res), dtype="float64")

    # `rvs()` is inherited from `Rand`

    @property
    def _cdf_spline(self):
        cdf_tck = (self._x, self._cump[:-1], 0)
        return utils.BSplineConstExtrapolate(
            left=0, right=1, t=cdf_tck[0], c=cdf_tck[1], k=cdf_tck[2]
        )

    def integrate_cdf(self, a, b):
        """Efficient version of CDF integration"""
        return self._cdf_spline.integrate(a=a, b=b)
