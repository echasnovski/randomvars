""" Code for boolean random variable
"""

import numpy as np
from scipy.stats.distributions import rv_frozen

from randomvars._discrete import Disc
import randomvars.options as op
import randomvars._utils as utils


class Bool:
    """Boolean random variable

    Class for boolean random variable which is equivalent to a discrete random
    variable with only two values: `False` and `True`.

    There are three ways to create instance of `Bool` class:

    1. Directly supply probability of `True` (`prob_true`):
    ```
        my_bool = Bool(prob_true=0.75)
        my_bool.pmf([False, True])
        # Also works with values convertible to boolean
        my_bool.pmf([0, 1])
        my_bool.pmf([lambda x: x, {"a": 1}, {}])
    ```
    2. Use `Bool.from_rv()` to create approximation of some existing random
    variable (object with method `cdf()`):
    ```
        from scipy.stats import bernoulli, binom
        rv_bernoulli = bernoulli(p=0.75)
        my_bernoulli = Bool.from_rv(rv_bernoulli)
        rv_bernoulli.pmf([0, 1])
        my_bernoulli.pmf([False, True])

        # In general, `Bool` takes into account Python notion of "being true"
        # (see documentation of `Bool.from_rv` for more information)
        rv_binom = binom(n=3, p=0.5)
        my_binom = Bool.from_rv(rv_binom)
        ## Probability of `True` is equal to one minus probability of zero
        my_binom.prob_true
        1 - rv_binom.pmf(0)
    ```
    3. Use `Bool.from_sample()` to create estimation based on some existing sample:
    ```
        from scipy.stats import bernoulli
        sample = bernoulli(p=0.1).rvs(size=100, random_state=101).astype("bool")
        my_rv = Bool.from_sample(sample)
        my_rv.prob_true
    ```
    """

    def __init__(self, prob_true):
        try:
            if (prob_true < 0) or (prob_true > 1):
                raise ValueError("`prob_true` should be between 0 and 1 (inclusively).")
        except TypeError:
            raise ValueError("`prob_true` should be a number.")

        self.prob_true = float(prob_true)
        self.prob_false = 1.0 - prob_true

    @classmethod
    def from_rv(cls, rv):
        """Create boolean RV from general RV

        Boolean random variable is created by inferring probability of `True`
        value from input random variable. Following general Python agreement,
        probability of `True` is computed as probability of all non-zero
        elements, which in turn is one minus probability of zero. Probability
        of zero is computed using `tolerance` package option by calculating
        difference between values of cumulative distribution function at zero
        and `-atol` (minus second element of `tolerance` option).

        **Notes**:
        - If `rv` is already an object of class `Bool`, it is returned
          untouched.
        - If `rv` represents continuous random variable, output might have a
          very small probability of `False`, which doesn't quite align with
          expected theoretical result of 0.

        Relevant package options: `tolerance`. See documentation of
        `randomvars.options.get_option()` for more information. To temporarily
        set options use `randomvars.options.option_context()` context manager.

        Parameters
        ----------
        rv : Object with `cdf()` method
            Method `cdf()` should implement cumulative distribution function.
            Recommended to be an object of class `rv_frozen` (`rv_discrete`
            with all hyperparameters defined).

        Returns
        -------
        rv_out : Bool
            Boolean random variable which approximates probability distribution
            of input `rv`.
        """
        # Make early return
        if isinstance(rv, Bool):
            return rv

        # Check input
        if not ("cdf" in dir(rv)):
            raise ValueError("`rv` should have method `cdf()`.")

        # Get options
        _, atol = op.get_option("tolerance")

        # Compute probability of `False`
        prob_false = rv.cdf(0) - rv.cdf(-atol)

        return cls(prob_true=1 - prob_false)

    @classmethod
    def from_sample(cls, sample):
        """Create boolean RV from sample

        Boolean RV is created by the following algorithm:
        - **Estimate distribution** with boolean estimator (taken from package
          option "boolean_estimator") in the form `estimate =
          boolean_estimator(sample)`. If `estimator` is object of class `Bool`,
          it is returned untouched. If it is object of `Disc` or `rv_frozen`
          (`rv_discrete` with all hyperparameters defined), it is forwarded to
          `Bool.from_rv()`.
        - **Create random variable** with `Bool(prob_true=estimate)`.

        Relevant package options: `boolean_estimator`. See documentation of
        `randomvars.options.get_option()` for more information. To temporarily
        set options use `randomvars.options.option_context()` context manager.

        Parameters
        ----------
        sample : 1d array-like
            This should be a valid input to `np.asarray()` so that its output
            is boolean and has single dimension.

        Returns
        -------
        rv_out : Bool
            Boolean random variable which is an estimate based on input
            `sample`.
        """
        # Check and prepare input
        sample = utils._as_1d_numpy(sample, "sample", chkfinite=False, dtype="bool")

        # Get options
        boolean_estimator = op.get_option("boolean_estimator")

        # Estimate distribution
        estimate = boolean_estimator(sample)

        # Make early return if `estimate` is random variable
        if isinstance(estimate, Bool):
            return estimate
        if isinstance(estimate, (Disc, rv_frozen)):
            return Bool.from_rv(estimate)

        return cls(prob_true=estimate)

    def pmf(self, x):
        """Probability mass function

        Return values of probability mass function at **boolean** points `x`.

        Parameters
        ----------
        x : array_like with boolean values
            It is first converted to boolean numpy array.

        Returns
        -------
        pmf_vals : ndarray with shape inferred from `x`
        """
        x = np.asarray(x, dtype="bool")
        res = np.full(shape=x.shape, fill_value=self.prob_false)
        res[x] = self.prob_true
        return res

    def cdf(self, x):
        """Cumulative distribution function

        Return values of cumulative distribution function at **boolean** points
        `x`. **Note** that, following Python agreement, `False` is less than
        `True`. So `cdf(False)` is probability of `False` and `cdf(True)` is 1.

        Parameters
        ----------
        x : array_like with boolean values
            It is first converted to boolean numpy array.

        Returns
        -------
        cdf_vals : ndarray with shape inferred from `x`
        """
        x = np.asarray(x, dtype="bool")
        res = np.full(shape=x.shape, fill_value=self.prob_false)
        res[x] = 1.0
        return res

    def ppf(self, q):
        """Percent point (quantile, inverse of cdf) function

        Return **boolean** values of percent point (quantile, inverse of cdf)
        function at cumulative probabilities `q`. **Note** that output for
        invalid `q` values (outside of [0; 1] segment and `numpy.nan`) is
        `True`, which is aligned with how Numpy converts `numpy.nan` to boolean
        dtype.

        Parameters
        ----------
        q : array_like with numeric values

        Returns
        -------
        ppf_vals : ndarray with "bool" dtype and shape inferred from `q`
        """
        q = np.asarray(q, dtype="float64")
        res = np.full(shape=q.shape, fill_value=True)
        res[(0 <= q) & (q <= self.prob_false)] = False
        return res

    def rvs(self, size=None, random_state=None):
        """Random boolean generation

        Generate random boolean values into array of desired size.

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
