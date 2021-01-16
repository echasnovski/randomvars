""" Code for boolean random variable
"""

import numpy as np
from scipy.stats.distributions import rv_frozen

from randomvars._random import Rand
import randomvars.options as op
import randomvars._utils as utils


class Bool(Rand):
    """Boolean random variable

    Class for boolean random variable which is equivalent to a discrete random
    variable with only two values: `False` (with probability `prob_false`) and
    `True` (with probability `prob_true`.

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
        # User-facing attributes
        self._prob_true = self._impute_init_args(prob_true)
        self._prob_false = 1.0 - self._prob_true

        super().__init__()

    @staticmethod
    def _impute_init_args(prob_true):
        try:
            prob_true = float(prob_true)
        except ValueError:
            raise TypeError("`prob_true` should be a number.")

        if (prob_true < 0) or (prob_true > 1):
            raise ValueError("`prob_true` should be between 0 and 1 (inclusively).")

        return prob_true

    def __str__(self):
        return f"Boolean RV with {self._prob_true} probability of True"

    @property
    def prob_true(self):
        """Return the probability of `True`"""
        return self._prob_true

    @property
    def prob_false(self):
        """Return the probability of `False`"""
        return self._prob_false

    @property
    def a(self):
        """Return left edge of support"""
        return False

    @property
    def b(self):
        """Return right edge of support"""
        return True

    # `support()` is inherited from `Rand`

    @classmethod
    def from_rv(cls, rv):
        """Create boolean RV from general RV

        Boolean random variable is created by inferring probability of `True`
        value from input random variable. Following general Python agreement,
        probability of `True` is computed as probability of all non-zero
        elements, which in turn is one minus probability of zero. Probability
        of zero is computed using `base_tolerance` package option by
        calculating difference between values of cumulative distribution
        function at zero and `-base_tolerance` (minus value of `base_tolerance`
        option).

        **Notes**:
        - If `rv` is an object of class `Rand`, it is converted to
          `Bool` via `rv.convert("Bool")`.
        - If `rv` represents continuous random variable, output might have a
          very small probability of `False`, which doesn't quite align with
          expected theoretical result of 0.

        Relevant package options: `base_tolerance`. See documentation of
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
        if isinstance(rv, Rand):
            return rv.convert("Bool")

        # Check input
        if not ("cdf" in dir(rv)):
            raise ValueError("`rv` should have method `cdf()`.")

        # Get options
        base_tol = op.get_option("base_tolerance")

        # Compute probability of `False`
        prob_false = rv.cdf(0) - rv.cdf(-base_tol)

        return cls(prob_true=1 - prob_false)

    @classmethod
    def from_sample(cls, sample):
        """Create boolean RV from sample

        Boolean RV is created by the following algorithm:
        - **Estimate distribution** with boolean estimator (taken from package
          option "boolean_estimator") in the form `estimate =
          boolean_estimator(sample)`. If `estimator` is an object of class
          `Rand` or `rv_frozen` (`rv_discrete` with all hyperparameters
          defined), it is forwarded to `Bool.from_rv()`.
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
        if isinstance(estimate, (Rand, rv_frozen)):
            return Bool.from_rv(estimate)

        return cls(prob_true=estimate)

    def pdf(self, x):
        raise AttributeError(
            "`Bool` doesn't have probability density function. Use `pmf()` instead."
        )

    def logpdf(self, x):
        raise AttributeError(
            "`Bool` doesn't have probability density function. Use `logpmf()` instead."
        )

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
        res = np.full(shape=x.shape, fill_value=self._prob_false)
        res[x] = self._prob_true
        return np.asarray(res, dtype="float64")

    # `logpmf()` is inherited from `Rand`

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
        res = np.full(shape=x.shape, fill_value=self._prob_false)
        res[x] = 1.0
        return np.asarray(res, dtype="float64")

    # `logcdf()` is inherited from `Rand`

    # `sf()` is inherited from `Rand`

    # `logsf()` is inherited from `Rand`

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
        res[(0 <= q) & (q <= self._prob_false)] = False
        return np.asarray(res, dtype="bool")

    # `isf()` is inherited from `Rand`

    # `rvs()` is inherited from `Rand`

    def integrate_cdf(self, a, b):
        """Efficient version of CDF integration"""
        cdf_spline = utils.BSplineConstExtrapolate(
            left=0, right=1, t=[0, 1], c=[self._prob_false], k=0
        )

        return cdf_spline.integrate(a=a, b=b)

    def convert(self, to_class=None):
        """Convert to different RV class

        Conversion is done by the following logic depending on the value of
        `to_class`:
        - If it is `None` or `"Bool"`, `self` is returned.
        - If it is `"Cont"`, continuous RV is returned. Input is first
          converted to discrete RV, which then is converted to continuous RV.
        - If it is `"Disc"`, discrete RV is returned. Its x-grid is equal to
          `[0, 1]` and p-grid is `[self.prob_false, self.prob_true]`.
        - If it is `"Mixt"`, mixture RV with only discrete component (equal to
          conversion of `self` to discrete RV) is returned.

        Parameters
        ----------
        to_class : string or None, optional
            Name of target class. Can be one of: `"Bool"`, `"Cont"`, `"Disc"`,
            `"Mixt"`, or `None`.

        Raises
        ------
        ValueError:
            In case not supported `to_class` is given.
        """
        # Use inline `import` statements to avoid circular import problems
        if (to_class == "Bool") or (to_class is None):
            return self
        elif to_class == "Cont":
            return self.convert("Disc").convert("Cont")
        elif to_class == "Disc":
            import randomvars._discrete as disc

            return disc.Disc(x=[0, 1], p=[self._prob_false, self._prob_true])
        elif to_class == "Mixt":
            import randomvars._mixture as mixt

            # Output is a degenerate mixture with only continuous component
            return mixt.Mixt(disc=self.convert("Disc"), cont=None, weight_cont=0.0)
        else:
            raise ValueError(
                '`metric` should be one of "Bool", "Cont", "Disc", or "Mixt".'
            )
