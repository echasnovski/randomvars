""" Code for random variable with piecewise-linear density
"""
import numpy as np
from scipy.stats.distributions import rv_continuous


class rv_piecelin(rv_continuous):
    """ Random variable with piecewise-linear density
    """

    def __init__(self, x, y, *args, **kwargs):
        if len(x) != len(y):
            raise ValueError("Number of elements of `x` and `y` do not match")

        self._x = np.asarray(x)
        self._y = np.asarray(y)
        self._y = self._y / _trapez_integral(self._x, self._y)
        self._cumprob = _trapez_integral_cum(self._x, self._y)

        # Set support
        kwargs["a"] = self.a = self._x[0]
        kwargs["b"] = self.b = self._x[-1]

        super(rv_piecelin, self).__init__(*args, **kwargs)

    def get_grid(self):
        """Get grid (tuple with `x` and `y`) defining piecewise-linear density
        """
        return (self._x, self._y)

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


def _trapez_integral(x, y):
    """ Compute integral with trapezoidal formula.

    >>> _trapez_integral(np.array([0, 1]), np.array([1, 1]))
    1.0
    >>> _trapez_integral(np.array([-1, 0, 1]), np.array([0, 10, 5]))
    12.5
    """
    return np.sum(0.5 * np.diff(x) * (y[:-1] + y[1:]))


def _trapez_integral_cum(x, y):
    """ Compute cumulative integral with trapezoidal formula.

    Element of output represents cumulative probability **before** its left "x"
    edge.

    >>> _trapez_integral_cum(np.array([0, 1]), np.array([1, 1]))
    array([0., 1.])
    >>> _trapez_integral_cum(np.array([-1, 0, 1]), np.array([0, 10, 5]))
    array([ 0. ,  5. , 12.5])
    """
    res = np.cumsum(0.5 * np.diff(x) * (y[:-1] + y[1:]))
    return np.concatenate([[0], res])
