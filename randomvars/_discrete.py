""" Code for discrete random variable
"""

import numpy as np
from scipy.stats.distributions import rv_discrete

import randomvars.options as op
import randomvars._utils as utils


class Disc(rv_discrete):
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
