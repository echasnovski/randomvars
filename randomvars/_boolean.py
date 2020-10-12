""" Code for boolean random variable
"""

import numpy as np

from randomvars._discrete import Disc
import randomvars.options as op


class Bool(Disc):
    def __new__(cls, prob_true, *args, **kwargs):
        try:
            if (prob_true < 0) or (prob_true > 1):
                raise ValueError("`prob_true` should be between 0 and 1 (inclusively).")
        except TypeError:
            raise ValueError("`prob_true` should be a number.")

        return super().__new__(
            cls, x=[0, 1], prob=[1 - prob_true, prob_true], *args, **kwargs
        )

    def __init__(self, prob_true):
        super().__init__(x=[0, 1], prob=[1 - prob_true, prob_true])
        self.prob_true = prob_true
        self.prob_false = 1 - prob_true

    # Because of integer nature of `False` and `True` in Python, these
    # attributes should work as expected when booleans are supplied:
    # - Properties `x`, `prob`, and `p`.
    # - Methods `pmf()` and `cdf()`.

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
        res = super().ppf(q)
        return res.astype("bool")

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
        res = super().rvs(size=size, random_state=random_state)
        return res.astype("bool")
