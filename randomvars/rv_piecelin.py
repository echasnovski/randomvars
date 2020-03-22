""" Code for random variable with piecewise-linear density
"""
import warnings

import numpy as np
from scipy.stats.distributions import rv_continuous


class rv_piecelin(rv_continuous):
    """ Random variable with piecewise-linear density
    """

    def __init__(self, x, y, *args, **kwargs):
        x, y = self._impute_xy(x, y)

        self._x = x
        self._y = y
        self._cumprob = _trapez_integral_cum(self._x, self._y)

        # Set support
        kwargs["a"] = self.a = self._x[0]
        kwargs["b"] = self.b = self._x[-1]

        super(rv_piecelin, self).__init__(*args, **kwargs)

    @staticmethod
    def _impute_xy(x, y):
        try:
            x = np.asarray_chkfinite(x, dtype=np.float64)
        except:
            raise ValueError(
                "`x` is not convertable to numeric numpy array with finite values."
            )
        if len(x.shape) > 1:
            raise ValueError("`x` is not a 1d array.")

        try:
            y = np.asarray_chkfinite(y, dtype=np.float64)
        except:
            raise ValueError(
                "`y` is not convertable to numeric numpy array with finite values."
            )
        if len(y.shape) > 1:
            raise ValueError("`y` is not a 1d array.")

        if len(x) != len(y):
            raise ValueError("Lengths of `x` and `y` do not match.")

        if not np.all(np.diff(x) > 0):
            warnings.warn(
                "`x` is not sorted. `x` and `y` are rearranged so as `x` is sorted."
            )
            x_argsort = np.argsort(x)
            x = x[x_argsort]
            y = y[x_argsort]

        if np.any(y < 0):
            raise ValueError("`y` has negative values.")
        if not np.any(y > 0):
            raise ValueError("`y` has no positive values.")

        y = y / _trapez_integral(x, y)

        return x, y

    def pdf_grid(self):
        """Get density grid

        Returns
        -------
        grid : tuple with 2 numeric arrays of same length.
            Two components are `x` and `y` arrays defining piecewise-linear density.
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


if __name__ == "__main__":
    import doctest

    doctest.testmod()
