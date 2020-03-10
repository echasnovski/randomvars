""" Code for random variable with piecewise-linear density
"""
import numpy as np
from scipy.stats.distributions import rv_continuous


class rv_piecelin(rv_continuous):
    """ Random variable with piecewise-linear density
    """

    def __init__(self, density, *args, **kwargs):
        if len(density) != 2:
            raise ValueError("Expected length 2 for parameter `den`")
        if len(density[0]) != len(density[1]):
            raise ValueError(
                "Number of elements of `x` and `y` components do not match"
            )

        self._x = np.asarray(density[0])
        self._y = np.asarray(density[1]) / trapez_integral(density[0], density[1])
        self._cumprob = trapez_integral_cum(self._x, self._y)

        # Set support
        kwargs["a"] = self.a = self._x[0]
        kwargs["b"] = self.b = self._x[-1]

        super(rv_piecelin, self).__init__(*args, **kwargs)

    def _pdf(self, x, *args):
        """ Implementation of probability density function
        """
        return np.interp(x, self._x, self._y)

    def _coeffs_by_ind(self, ind):
        """ Compute density coefficients based on index of interval.

        Index `i` corresponds to coefficients from interval with endpoints
        `self._x[i-1]` and `self._x[i]`. Which interval values of `self._x`
        represent should be decided before calling this function during
        computation of `ind`.  Indexes 0 and `len(self._x)` result in zero
        coefficients.
        """

        inter = np.zeros_like(ind)
        slope = np.zeros_like(ind)

        ind_is_in = np.logical_and(ind > 0, ind < len(self._x))
        inds_in = ind[ind_is_in]

        slope[ind_is_in] = (self._y[inds_in - 1] - self._y[inds_in]) / (
            self._x[inds_in - 1] - self._x[inds_in]
        )
        inter[ind_is_in] = (
            self._y[inds_in - 1] - slope[ind_is_in] * self._x[inds_in - 1]
        )

        return (inter, slope)

    def coeffs(self, x):
        """ Compute density coefficients based on `x`.

        For density x-grid elements coefficients are computed from the right
        interval, except for last element, for which its left (i.e. last)
        interval is used.

        Returns
        -------
        coeffs : tuple of coefficients
            Tuple with two arrays (both same length as `x`) representing
            intercept and slope.
        """

        ind = np.searchsorted(self._x, x, side="right")
        # Include right-most edge of density x-grid to return coefficients from first
        # interval
        ind[x == self._x[-1]] = len(self._x) - 1
        return self._coeffs_by_ind(ind)


def trapez_integral(x, y):
    """ Compute integral with trapezoidal formula.
    """
    return np.sum(0.5 * np.diff(x) * (y[:-1] + y[1:]))


def trapez_integral_cum(x, y):
    """ Compute cumulative integral with trapezoidal formula.

    Element of output represents cumulative probability **before** its left "x"
    edge.
    """

    res = np.cumsum(0.5 * np.diff(x) * (y[:-1] + y[1:]))
    return np.concatenate([[0], res[:-1]])
