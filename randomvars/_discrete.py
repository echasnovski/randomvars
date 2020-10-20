""" Code for discrete random variable
"""

import numpy as np
from scipy.stats.distributions import rv_frozen

import randomvars.options as op
import randomvars._utils as utils


class Disc:
    """Discrete random variable

    Class for discrete random variable with **finite number of (finite and
    unique) values**. It is similar to (unexported) `rv_sample` class from
    `scipy.stats.distributions`, but works with float numbers as distribution
    values (opposite to focusing on integers in `rv_sample`).

    There are three ways to create instance of `Disc` class:

    1. Directly supply x-values (`x`) and their probabilities (`prob`):
    ```
        my_disc = Disc(x=[1.618, 2.718, 3.141], prob=[0.1, 0.2, 0.7])
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

    def __init__(self, x, prob):
        x, prob = self._impute_xprob(x, prob)

        self._x = x
        self._prob = prob
        self._p = np.cumsum(prob)
        self._a = x[0]
        self._b = x[-1]

    def __str__(self):
        x_len = len(self.x)
        s = "s" if x_len > 1 else ""
        return f"Discrete RV with {x_len} value{s} (support: [{self.a}, {self.b}])"

    @staticmethod
    def _impute_xprob(x, prob):
        x = utils._as_1d_numpy(x, "x", chkfinite=True, dtype="float64")
        prob = utils._as_1d_numpy(prob, "prob", chkfinite=True, dtype="float64")

        x, prob = utils._sort_parallel(x, prob, y_name="prob", warn=True)

        if not np.all(np.diff(x) > 0):
            x, prob = utils._unique_parallel(x, prob, warn=True)

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

    @property
    def a(self):
        """Return left edge of support"""
        return self._a

    @property
    def b(self):
        """Return right edge of support"""
        return self._b

    def support(self):
        """Return support of random variable"""
        return (self.a, self.b)

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
        prob = []
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
            prob.append(cur_tot_prob - tot_prob)

            tot_prob = cur_tot_prob

        return cls(x=x, prob=prob)

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
        - **Create random variable** with `Disc(x=x, prob=prob)`, where `x` and `prob`
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

        return cls(x=estimate[0], prob=estimate[1])

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
        rtol, atol = op.get_option("tolerance")

        inds = utils._find_nearest_ind(x, self.x)

        x_is_matched = np.isclose(x, self.x[inds], rtol=rtol, atol=atol)

        res = np.where(x_is_matched, self.prob[inds], 0)
        return utils._copy_nan(fr=x, to=res)

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
        inds = np.searchsorted(self.x, x, side="right")
        # This is needed to avoid possible confusion at index 0 when subsetting
        # `self.p`
        inds_clipped = np.maximum(inds, 1)

        res = np.ones_like(x, dtype=np.float64)
        res = np.where(inds == 0, 0.0, self.p[inds_clipped - 1])

        return utils._copy_nan(fr=x, to=res)

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
        q_inds = np.searchsorted(self.p, q, side="left")
        # This is needed to avoid `IndexError` in later `np.where()` call
        q_inds_clipped = np.minimum(q_inds, len(self.p) - 1)

        res = np.empty_like(q, dtype=np.float64)
        res = np.where(q_inds != len(self.p), self.x[q_inds_clipped], res)
        res[(q < 0) | (q > 1)] = np.nan

        return utils._copy_nan(fr=q, to=res)

    def rvs(self, size=None, random_state=None):
        """Random number generation

        Generate random numbers into array of desired size.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Defining number of random variates (default is 1).
        random_state : `None`, int, or RandomState, optional
            Source of uniform random number generator. If `None`, it is
            initiated as `numpy.random.RandomState()`. If integer,
            `numpy.random.RandomState(seed=random_state)` is used.
        """
        if random_state is None:
            random_state = np.random.RandomState()
        elif isinstance(random_state, int):
            random_state = np.random.RandomState(seed=random_state)

        U = random_state.uniform(size=size)

        return self.ppf(U)
